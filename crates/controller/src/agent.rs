mod hosted;

use crate::{
    App,
    hf_tokens::PrivateFile,
    idempotency::Replay,
    mcp::Server,
    web::{ApiError, read_hf_token_request, read_submission},
};
use axum::{
    Extension, Json, Router,
    extract::{DefaultBodyLimit, Path, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post, put},
};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json, value::RawValue};
use std::{
    fs,
    os::unix::fs::DirBuilderExt as _,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    sync::watch,
    time::{Instant, sleep_until},
};
use uuid::Uuid;

const MAX_KEY: usize = 1024;
const MAX_GOAL: usize = 4096;
const MAX_TOOL_RESULT: usize = 64 * 1024;
const MAX_CONTEXT: usize = 1024 * 1024;
const AUTH_DIRECTORY: &str = "agent-credentials";
type Result<T> = std::result::Result<T, ApiError>;
type Payload<T> = std::result::Result<Json<T>, JsonRejection>;
type AppState = State<Arc<App>>;

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Anthropic,
    Openai,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Provider {
    kind: Kind,
    base_url: String,
    model: String,
    max_tokens: u64,
    max_steps: u64,
    max_session_tokens: u64,
    timeout_seconds: u64,
    request_timeout_seconds: u64,
}

#[derive(Serialize)]
struct Settings {
    provider: Option<Provider>,
    key_configured: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Key {
    key: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Start {
    goal: String,
    model_id: String,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Running,
    Completed,
    Cancelled,
    Failed,
    BudgetExhausted,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct Summary {
    pub(crate) id: String,
    goal: String,
    model_id: String,
    status: Status,
    steps: u64,
    tokens: u64,
    scratch_ids: Vec<String>,
    message: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct Session {
    #[serde(flatten)]
    pub(crate) info: Summary,
    entries: Vec<Entry>,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Entry {
    Assistant { text: String },
    ToolCall { name: String, arguments: Value },
    ToolResult { name: String, result: Value },
}

pub(crate) struct Service {
    auth: Option<PathBuf>,
    // Keep this guard after cancel until the admitted tool (if any) returns.
    active: Mutex<Option<(String, watch::Sender<()>)>>,
}

impl Service {
    pub(crate) fn new(data: &FsPath, dist: &FsPath) -> std::io::Result<Self> {
        let root = data.canonicalize()?.join(AUTH_DIRECTORY);
        let dist = dist.canonicalize().or_else(|_| std::path::absolute(dist))?;
        if root.starts_with(&dist) || dist.starts_with(&root) {
            return Err(std::io::Error::other(
                "The web asset directory must not expose agent credential storage",
            ));
        }
        let auth = (|| {
            match fs::DirBuilder::new().mode(0o700).create(&root) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(_) => return None,
            }
            let path = root.join("key");
            PrivateFile::open(&path).ok()?;
            Some(path)
        })();
        Ok(Self {
            auth,
            active: Mutex::new(None),
        })
    }

    fn secret(&self) -> Result<PrivateFile> {
        let path = self.auth.as_ref().ok_or_else(secret_unavailable)?;
        PrivateFile::open(path).map_err(|_| secret_unavailable())
    }

    pub(crate) fn protected_path(&self, path: &FsPath) -> bool {
        self.auth.as_ref().is_some_and(|key| {
            let root = key.parent().unwrap();
            let contains = |path: &FsPath| root.starts_with(path) || path.starts_with(root);
            contains(path) || path.canonicalize().is_ok_and(|path| contains(&path))
        })
    }
}

impl Provider {
    fn validate(&mut self) -> Result<()> {
        let bad_url =
            || ApiError::bad("Use an HTTPS base URL without credentials, a query, or a fragment");
        let url = Url::parse(&self.base_url).map_err(|_| bad_url())?;
        let loopback = url
            .host_str()
            .and_then(|host| {
                host.trim_matches(['[', ']'])
                    .parse::<std::net::IpAddr>()
                    .ok()
            })
            .is_some_and(|ip| ip.is_loopback());
        let path = url.path().trim_end_matches('/');
        if !(url.scheme() == "https" || (url.scheme() == "http" && loopback))
            || url.host().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || self.base_url.len() > 512
            || path.ends_with("/chat/completions")
            || path.ends_with("/messages")
        {
            return Err(bad_url());
        }
        self.base_url = url.as_str().trim_end_matches('/').to_owned();
        self.model = self.model.trim().to_owned();
        if !(1..=200).contains(&self.model.len()) || self.model.chars().any(char::is_control) {
            return Err(ApiError::bad("Enter a model ID of 1 to 200 bytes"));
        }
        if !(1..=16384).contains(&self.max_tokens)
            || !(1..=32).contains(&self.max_steps)
            || !(1..=250000).contains(&self.max_session_tokens)
            || self.max_tokens > self.max_session_tokens
            || !(1..=900).contains(&self.timeout_seconds)
            || !(1..=120).contains(&self.request_timeout_seconds)
        {
            return Err(ApiError::bad(
                "Set provider request and session limits in range",
            ));
        }
        Ok(())
    }

    fn same_endpoint(&self, other: &Self) -> bool {
        self.kind == other.kind && self.base_url == other.base_url
    }
}

pub(crate) fn router(app: Arc<App>, mcp: Arc<Server>) -> Router {
    Router::new()
        .route("/settings/agent-provider", get(settings).put(configure))
        .route(
            "/settings/agent-provider/key",
            put(set_key).delete(remove_key),
        )
        .route("/agent/sessions", get(list).post(start))
        .route("/agent/sessions/{id}", get(detail))
        .route("/agent/sessions/{id}/cancel", post(cancel))
        .layer(DefaultBodyLimit::max(8 * 1024))
        .layer(Extension(mcp))
        .with_state(app)
}

fn secret_unavailable() -> ApiError {
    ApiError(
        StatusCode::SERVICE_UNAVAILABLE,
        "Private provider key storage is not available".into(),
    )
}

fn invalid_secret() -> ApiError {
    ApiError::bad("Enter a valid provider API key")
}

fn valid_secret(key: &str) -> bool {
    (8..=MAX_KEY).contains(&key.len()) && key.bytes().all(|b| b.is_ascii_graphic())
}

async fn settings(State(app): AppState) -> Result<Json<Settings>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    let provider = store.agent_provider().map_err(ApiError::storage)?;
    let key_configured = provider.is_some()
        && app
            .agent
            .secret()?
            .configured()
            .map_err(|_| secret_unavailable())?;
    Ok(Json(Settings {
        provider,
        key_configured,
    }))
}

async fn configure(
    State(app): AppState,
    payload: Payload<Box<RawValue>>,
) -> Result<Json<Settings>> {
    let mut provider: Provider = read_hf_token_request(payload)?;
    provider.validate()?;
    let store = app.store.lock().map_err(ApiError::storage)?;
    no_active_session(&app)?;
    let secret = app.agent.secret()?;
    let key = secret
        .read(MAX_KEY as u64)
        .map_err(|_| secret_unavailable())?;
    if key.as_deref().is_some_and(|key| {
        valid_secret(key) && (provider.base_url.contains(key) || provider.model.contains(key))
    }) {
        return Err(ApiError::bad("Keep the API key out of the provider fields"));
    }
    let old = store.agent_provider().map_err(ApiError::storage)?;
    let key_configured = if old.is_some_and(|old| provider.same_endpoint(&old)) {
        key.is_some()
    } else {
        // Remove first. A failed settings write must not bind an old key to a new URL.
        secret.change(None).map_err(|_| secret_unavailable())?;
        false
    };
    store
        .put_agent_provider(&provider)
        .map_err(ApiError::storage)?;
    Ok(Json(Settings {
        provider: Some(provider),
        key_configured,
    }))
}

async fn set_key(State(app): AppState, payload: Payload<Box<RawValue>>) -> Result<Json<Settings>> {
    let Key { key } = read_hf_token_request(payload)?;
    if !valid_secret(&key) {
        return Err(invalid_secret());
    }
    change_key(&app, Some(&key))
}

async fn remove_key(State(app): AppState) -> Result<Json<Settings>> {
    change_key(&app, None)
}

fn change_key(app: &App, key: Option<&str>) -> Result<Json<Settings>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    no_active_session(app)?;
    let provider = store.agent_provider().map_err(ApiError::storage)?;
    if let Some(key) = key {
        let provider = provider.as_ref().ok_or_else(|| {
            ApiError(
                StatusCode::CONFLICT,
                "Save a provider before you set its key".into(),
            )
        })?;
        if provider.base_url.contains(key) || provider.model.contains(key) {
            return Err(ApiError::bad("Keep the API key out of the provider fields"));
        }
    }
    let key_configured = app
        .agent
        .secret()?
        .change(key)
        .map_err(|_| secret_unavailable())?;
    Ok(Json(Settings {
        provider,
        key_configured,
    }))
}

fn no_active_session(app: &App) -> Result<()> {
    if app
        .agent
        .active
        .lock()
        .map_err(ApiError::storage)?
        .is_some()
    {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "Wait for the agent session and its last tool to stop".into(),
        ));
    }
    Ok(())
}

async fn list(State(app): AppState) -> Result<Json<Vec<Summary>>> {
    Ok(Json(
        app.store
            .lock()
            .map_err(ApiError::storage)?
            .agent_sessions()
            .map_err(ApiError::storage)?,
    ))
}

async fn detail(State(app): AppState, Path(id): Path<Uuid>) -> Result<Json<Session>> {
    Ok(Json(read_session(&app, &id.to_string())?))
}

fn read_session(app: &App, id: &str) -> Result<Session> {
    app.store
        .lock()
        .map_err(ApiError::storage)?
        .agent_session(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("agent session"))
}

async fn start(
    State(app): AppState,
    Extension(mcp): Extension<Arc<Server>>,
    headers: HeaderMap,
    payload: Payload<Box<RawValue>>,
) -> Result<Response> {
    let Json(raw) =
        payload.map_err(|e| ApiError(e.status(), "Enter a valid session request".into()))?;
    let input: Start = serde_json::from_str(raw.get()).map_err(|_| {
        ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Enter a valid session request".into(),
        )
    })?;
    let (input, submission) = read_submission(&headers, "agent_sessions", Ok(Json(input)))?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(submission) = &submission {
        match store.replay(submission).map_err(ApiError::storage)? {
            Replay::Matched(id) => {
                let row = store
                    .agent_session(&id)
                    .map_err(ApiError::storage)?
                    .ok_or_else(|| ApiError::not_found("agent session"))?;
                return Ok(Json(row).into_response());
            }
            Replay::Conflict => {
                return Err(ApiError(
                    StatusCode::CONFLICT,
                    "Idempotency-Key already used for a different request".into(),
                ));
            }
            Replay::New => {}
        }
    }
    if input.goal.trim().is_empty() || input.goal.len() > MAX_GOAL {
        return Err(ApiError::bad("Enter a goal of 1 to 4096 bytes"));
    }
    if store
        .model(&input.model_id)
        .map_err(ApiError::storage)?
        .is_none()
    {
        return Err(ApiError::not_found("Library model"));
    }
    let missing = || {
        ApiError(
            StatusCode::CONFLICT,
            "Set a provider and its API key in Settings".into(),
        )
    };
    let provider = store
        .agent_provider()
        .map_err(ApiError::storage)?
        .ok_or_else(missing)?;
    let key = app
        .agent
        .secret()?
        .read(MAX_KEY as u64)
        .map_err(|_| secret_unavailable())?
        .filter(|key| valid_secret(key))
        .ok_or_else(missing)?;
    if input.goal.contains(&key)
        || input.model_id.contains(&key)
        || headers
            .get("idempotency-key")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.contains(&key))
    {
        return Err(ApiError::bad(
            "Do not include the provider key in a session",
        ));
    }
    let mut active = app.agent.active.lock().map_err(ApiError::storage)?;
    if active.is_some() {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "An agent session or its last tool is still running".into(),
        ));
    }
    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .no_proxy()
        .timeout(Duration::from_secs(provider.request_timeout_seconds))
        .build()
        .map_err(|_| {
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "The provider client is not available".into(),
            )
        })?;
    let id = Uuid::new_v4().to_string();
    let row = Session {
        info: Summary {
            id: id.clone(),
            goal: input.goal,
            model_id: input.model_id,
            status: Status::Running,
            steps: 0,
            tokens: 0,
            scratch_ids: vec![],
            message: None,
        },
        entries: vec![],
    };
    store
        .add_agent_session(&row, submission.as_ref())
        .map_err(ApiError::storage)?;
    let (tx, rx) = watch::channel(());
    *active = Some((id.clone(), tx));
    let app = app.clone();
    tokio::spawn(async move {
        if run(&app, &mcp, &id, provider, key, client, rx)
            .await
            .is_err()
        {
            let _ = update(&app, &id, |row| {
                finish(row, Status::Failed, "The session could not continue.");
            });
        }
        if let Ok(mut active) = app.agent.active.lock() {
            active.take();
        }
    });
    Ok((StatusCode::CREATED, Json(row)).into_response())
}

async fn cancel(State(app): AppState, Path(id): Path<Uuid>) -> Result<Json<Session>> {
    let id = id.to_string();
    let store = app.store.lock().map_err(ApiError::storage)?;
    store.stop_agent_session(&id).map_err(ApiError::storage)?;
    let active = app.agent.active.lock().map_err(ApiError::storage)?;
    if let Some((running, tx)) = active.as_ref()
        && *running == id
    {
        let _ = tx.send(());
    }
    let row = store
        .agent_session(&id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("agent session"))?;
    Ok(Json(row))
}

fn update<T>(app: &App, id: &str, write: impl FnOnce(&mut Session) -> T) -> Result<T> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    let mut row = store
        .agent_session(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("agent session"))?;
    let value = write(&mut row);
    store.put_agent_session(&row).map_err(ApiError::storage)?;
    Ok(value)
}

fn finish(row: &mut Session, status: Status, message: &str) {
    if row.info.status == Status::Running {
        row.info.status = status;
        row.info.message = Some(message.into());
    }
}

fn has_secret(value: &Value, secret: &str) -> bool {
    match value {
        Value::String(value) => value.contains(secret),
        Value::Array(items) => items.iter().any(|value| has_secret(value, secret)),
        Value::Object(items) => items
            .iter()
            .any(|(key, value)| key.contains(secret) || has_secret(value, secret)),
        _ => false,
    }
}

const TIME_LIMIT: &str = "The session reached its time limit.";

#[allow(clippy::too_many_arguments)]
async fn run(
    app: &App,
    mcp: &Server,
    id: &str,
    provider: Provider,
    key: String,
    client: Client,
    mut cancel: watch::Receiver<()>,
) -> Result<()> {
    let start = read_session(app, id)?.info;
    let deadline = Instant::now() + Duration::from_secs(provider.timeout_seconds);
    let mut hosted = hosted::Conversation::new(client, provider, key, &mcp.list(), &start);
    let mut calls = 0;
    loop {
        let usage = update(app, id, |row| {
            if row.info.status != Status::Running {
                return None;
            }
            if row.info.steps >= hosted.provider.max_steps
                || row.info.tokens >= hosted.provider.max_session_tokens
            {
                finish(
                    row,
                    Status::BudgetExhausted,
                    "The session reached its request or token limit.",
                );
                return None;
            }
            if Instant::now() >= deadline {
                finish(row, Status::BudgetExhausted, TIME_LIMIT);
                return None;
            }
            row.info.steps += 1;
            Some(row.info.tokens)
        })?;
        let Some(tokens) = usage else { return Ok(()) };
        let turn = tokio::select! {
            biased;
            _ = cancel.changed() => return Ok(()),
            _ = sleep_until(deadline) => {
                update(app, id, |row| finish(row, Status::BudgetExhausted, TIME_LIMIT))?;
                return Ok(());
            }
            reply = hosted.turn(tokens) => reply,
        };
        let turn = match turn {
            Ok(turn) => turn,
            Err(hosted::Failure::Context) => {
                update(app, id, |row| {
                    finish(
                        row,
                        Status::BudgetExhausted,
                        "The session reached its context size limit.",
                    )
                })?;
                return Ok(());
            }
            Err(hosted::Failure::Provider(message)) => {
                update(app, id, |row| finish(row, Status::Failed, message))?;
                return Ok(());
            }
        };
        if !update(app, id, |row| {
            row.info.tokens = row.info.tokens.saturating_add(turn.tokens);
            if row.info.status != Status::Running {
                return false;
            }
            for text in turn.texts {
                row.entries.push(Entry::Assistant { text });
            }
            if row.info.tokens > hosted.provider.max_session_tokens
                || (!turn.calls.is_empty() && row.info.tokens >= hosted.provider.max_session_tokens)
            {
                finish(
                    row,
                    Status::BudgetExhausted,
                    "The session reached its token limit.",
                );
                return false;
            }
            if turn.calls.is_empty() {
                if turn.limited {
                    finish(
                        row,
                        Status::BudgetExhausted,
                        "The provider reached its response token limit.",
                    );
                } else {
                    finish(
                        row,
                        Status::Completed,
                        "The session is complete. Only an operator can approve publish requests.",
                    );
                }
                return false;
            }
            true
        })? {
            return Ok(());
        }
        let mut results = Vec::new();
        for call in turn.calls {
            if !update(app, id, |row| {
                if row.info.status != Status::Running {
                    return false;
                }
                if Instant::now() >= deadline || calls >= hosted.provider.max_steps {
                    let message = if calls >= hosted.provider.max_steps {
                        "The session reached its tool call limit."
                    } else {
                        TIME_LIMIT
                    };
                    finish(row, Status::BudgetExhausted, message);
                    return false;
                }
                row.entries.push(Entry::ToolCall {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
                true
            })? {
                return Ok(());
            }
            calls += 1;
            let input = json!({"name":call.name,"arguments":call.arguments});
            let pending = mcp.call_as(&input, crate::findings::Caller::Builtin(id.to_owned()));
            tokio::pin!(pending);
            let reply = tokio::select! {
                biased;
                _ = cancel.changed() => return Ok(()),
                _ = sleep_until(deadline) => {
                    update(app, id, |row| finish(row, Status::BudgetExhausted, TIME_LIMIT))?;
                    return Ok(());
                },
                // Once polled, let controller writes finish even if the session stops.
                first = poll_once(pending.as_mut()) => match first {
                    Some(reply) => reply,
                    None => tokio::select! {
                        _ = cancel.changed() => pending.await,
                        _ = sleep_until(deadline) => {
                            update(app, id, |row| finish(row, Status::BudgetExhausted, TIME_LIMIT))?;
                            pending.await
                        },
                        reply = &mut pending => reply,
                    }
                }
            };
            let (mut value, mut error) = match reply {
                Ok(mut value) => (value["structuredContent"].take(), value["isError"] == true),
                Err(message) => (json!({"error":message}), true),
            };
            let scratch = !error
                && value["qualification"] == "not release-qualified"
                && matches!(
                    value["state"].as_str(),
                    Some("scratch" | "publish_requested")
                );
            let scratch = if scratch {
                value["id"]
                    .as_str()
                    .filter(|id| Uuid::parse_str(id).is_ok())
                    .map(str::to_owned)
            } else {
                None
            };
            if hosted.contains_secret(&value) {
                value = json!({"error":"The tool result contains private data. It was not sent to the provider."});
                error = true;
            } else if serde_json::to_vec(&value).map_or(true, |v| v.len() > MAX_TOOL_RESULT) {
                value = json!({"error":"The result is too large for an agent turn. Request a smaller result."});
                error = true;
            }
            let text = value.to_string();
            let running = update(app, id, |row| {
                if let Some(scratch) = scratch
                    && !row.info.scratch_ids.contains(&scratch)
                {
                    row.info.scratch_ids.push(scratch);
                }
                row.entries.push(Entry::ToolResult {
                    name: call.name,
                    result: value,
                });
                row.info.status == Status::Running
            })?;
            if !running {
                return Ok(());
            }
            results.push((call.id, text, error));
        }
        hosted.add_results(results);
    }
}

async fn poll_once<F: Future>(mut future: std::pin::Pin<&mut F>) -> Option<F::Output> {
    std::future::poll_fn(|cx| {
        std::task::Poll::Ready(match future.as_mut().poll(cx) {
            std::task::Poll::Ready(value) => Some(value),
            std::task::Poll::Pending => None,
        })
    })
    .await
}
