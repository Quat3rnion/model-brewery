use crate::{
    App,
    conversion::{Conversion, NewConversion, StartError},
    dedupe,
    domain::{
        Envelope, Job, KINDS, NewJob, NewModel, NewPrompt, Reason, SIMPLE_WORKFLOWS,
        State as JobState, candidates, input_blocker, manual_registration_blocker,
        standard_job_host_blocker,
    },
    eligibility, feed, hf_tokens,
    idempotency::{self, Replay, Submission},
    store::Store,
    suggestions::{self, SuggestError, Suggestions},
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, Request, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header},
    middleware::{self, Next},
    response::{IntoResponse, Response, Sse, sse::KeepAlive},
    routing::{delete, get, post},
};
use brewery_worker::trace::{self, ReadReply, Slice, View};
use serde_json::{Value, json, value::RawValue};
use std::{collections::BTreeMap, net::SocketAddr, path::PathBuf, sync::Arc};
use tower_http::services::ServeDir;

type Shared = Arc<App>;
type Result<T> = std::result::Result<T, ApiError>;

pub fn router(app: Shared, listen: SocketAddr, dist: PathBuf) -> Router {
    let api = Router::new()
        .route(
            "/openapi.json",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "application/json")],
                    include_str!("../../../api/openapi.json"),
                )
            }),
        )
        .route("/state", get(state))
        .route("/conversions/eligibility", post(conversion_eligibility))
        .route("/settings", get(get_settings))
        .route(
            "/settings/{host}",
            axum::routing::put(put_settings).layer(DefaultBodyLimit::max(64 * 1024)),
        )
        .route("/settings/hf-tokens", get(get_hf_tokens))
        .route(
            "/settings/{host}/hf-token",
            axum::routing::put(set_hf_token)
                .delete(remove_hf_token)
                .layer(DefaultBodyLimit::max(2048)),
        )
        .route("/events", get(events))
        .route(
            "/suggestions/paths",
            post(suggest_paths).layer(DefaultBodyLimit::max(32 * 1024)),
        )
        .route("/models", get(list_models).post(add_model))
        .route(
            "/models/weight-calibration",
            post(crate::backend_calibration::register),
        )
        .route("/models/{id}", delete(delete_model))
        .route("/prompts", post(add_prompt))
        .route("/prompts/{id}", delete(delete_prompt).put(update_prompt))
        .route("/jobs", post(add_job))
        .route("/jobs/{id}", get(job))
        .route("/jobs/{id}/trace", get(trace_manifest))
        .route("/jobs/{id}/trace/tensor", get(trace_tensor))
        .route("/jobs/{id}/trace/logit-lens", get(trace_logit_lens))
        .route("/jobs/{id}/trace/model-logits", get(trace_model_logits))
        .route(
            "/jobs/{id}/trace/routing-atlas",
            get(crate::routing_atlas::get),
        )
        .route("/jobs/{id}/cancel", post(cancel))
        .route("/conversions", get(list_conversions).post(add_conversion))
        .route("/conversions/{id}", get(get_conversion))
        .route("/conversions/{id}/cancel", post(cancel_conversion))
        .route(
            "/profiles",
            get(list_profiles)
                .post(save_profile)
                .layer(DefaultBodyLimit::max(512 * 1024)),
        )
        .route(
            "/profiles/{revision}/export-options",
            get(profile_export_options),
        )
        .route(
            "/profiles/catalog",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "application/json")],
                    crate::profiles::CATALOG_JSON,
                )
            }),
        )
        .route(
            "/profiles/gguf-catalog",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "application/json")],
                    crate::gguf::CATALOG_JSON,
                )
            }),
        )
        .merge(crate::admin::router())
        .merge(crate::interventions::router())
        .merge(crate::sae::router())
        .merge(crate::scratch::router())
        .merge(crate::diff::router())
        .merge(crate::infra_hosts::router())
        .fallback(|| async { ApiError(StatusCode::NOT_FOUND, "API endpoint not found".into()) })
        .method_not_allowed_fallback(|| async {
            ApiError(
                StatusCode::METHOD_NOT_ALLOWED,
                "HTTP method not allowed".into(),
            )
        })
        .layer(DefaultBodyLimit::max(256 * 1024))
        .with_state(app.clone());
    let mcp = Arc::new(crate::mcp::Server::new(api.clone()));
    Router::new()
        .nest("/api", api.merge(crate::agent::router(app, mcp.clone())))
        .merge(crate::mcp::router(mcp))
        .fallback_service(ServeDir::new(dist))
        .layer(middleware::from_fn_with_state(listen, guard))
}

async fn guard(State(listen): State<SocketAddr>, req: Request, next: Next) -> Response {
    let headers = req.headers();
    let rejected = || {
        ApiError(
            StatusCode::FORBIDDEN,
            "host or origin is not allowed".into(),
        )
        .into_response()
    };
    let Some(host) = single(headers, "host").ok().flatten() else {
        return rejected();
    };
    let numeric = listen.to_string();
    let localhost = format!("localhost:{}", listen.port());
    let default_port_host =
        listen.port() == 80 && (host == "localhost" || numeric.strip_suffix(":80") == Some(host));
    if host != numeric && host != localhost && !default_port_host {
        return rejected();
    }
    // Request-target authority must agree with Host if an absolute-form URI was supplied.
    if req.uri().authority().is_some_and(|a| a.as_str() != host) {
        return rejected();
    }
    let origin = match single(headers, "origin") {
        Ok(value) if value.is_none_or(|value| value.strip_prefix("http://") == Some(host)) => value,
        _ => return rejected(),
    };
    let path = req.uri().path();
    let api = path == "/mcp" || path == "/api" || path.starts_with("/api/");
    if api {
        let fetch = match single(headers, "sec-fetch-site") {
            Ok(value @ (Some("same-origin" | "none") | None)) => value,
            _ => return rejected(),
        };
        // A no-body cancel is a simple HTML form POST. Require a browser-controlled
        // same-origin signal or the non-simple JSON media type for CLI requests.
        if req.method() == Method::POST
            && origin.is_none()
            && fetch != Some("same-origin")
            && !has_json(headers)
        {
            return ApiError(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "use application/json for this POST".into(),
            )
            .into_response();
        }
    }
    let mut response = next.run(req).await;
    let headers = response.headers_mut();
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"),
    );
    if api {
        headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    response
}

pub(crate) fn single<'a>(
    headers: &'a HeaderMap,
    name: &str,
) -> std::result::Result<Option<&'a str>, ()> {
    let all = headers.get_all(name);
    let mut values = all.iter();
    let value = values
        .next()
        .map(|value| value.to_str().map_err(|_| ()))
        .transpose()?;
    if values.next().is_some() {
        return Err(());
    }
    Ok(value)
}

pub(crate) fn has_json(headers: &HeaderMap) -> bool {
    single(headers, "content-type")
        .ok()
        .flatten()
        .and_then(|s| s.split(';').next())
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("application/json"))
}

async fn list_models(State(app): State<Shared>) -> Result<Json<Vec<crate::domain::Model>>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(store.models().map_err(ApiError::storage)?))
}

async fn suggest_paths(
    State(app): State<Shared>,
    payload: std::result::Result<Json<suggestions::Query>, JsonRejection>,
) -> Result<Json<Suggestions>> {
    let query = read(payload)?;
    let hosts = app.effective_hosts().map_err(ApiError::storage)?;
    if query.host == "auto" || !hosts.iter().any(|host| host.id == query.host) {
        return Err(ApiError::bad("unknown worker host"));
    }
    if !query.valid_prefix() {
        return Err(ApiError::bad(
            "prefix must be empty or a normalized absolute path of at most 4096 bytes",
        ));
    }
    let dispatch_config = app
        .ssh_hosts
        .config_for(&[&query.host])
        .map_err(|message| ApiError(StatusCode::CONFLICT, message.into()))?;
    app.suggestions
        .get(query, dispatch_config)
        .await
        .map(Json)
        .map_err(|error| {
            let (status, message) = match error {
                SuggestError::Busy => {
                    (StatusCode::SERVICE_UNAVAILABLE, "path suggestions are busy")
                }
                SuggestError::Timeout => {
                    (StatusCode::GATEWAY_TIMEOUT, "path suggestions timed out")
                }
                SuggestError::Unavailable => {
                    (StatusCode::BAD_GATEWAY, "path suggestions unavailable")
                }
            };
            ApiError(status, message.into())
        })
}

async fn state(State(app): State<Shared>) -> Result<Json<Value>> {
    // Every controller mutation takes this mutex, and its feed entry commits
    // in the same SQLite transaction. Read the cursor while still holding it.
    let store = app.store.lock().map_err(ApiError::storage)?;
    let mut hosts = app.effective_hosts().map_err(ApiError::storage)?;
    for host in &mut hosts {
        host.refresh_capability_reasons();
    }
    let mut models = store.models().map_err(ApiError::storage)?;
    let jobs = store.jobs().map_err(ApiError::storage)?;
    app.model_availability.refresh(&hosts, &jobs, &mut models);
    let capabilities: BTreeMap<_, _> = KINDS
        .iter()
        .map(|&kind| (kind, hosts.iter().any(|host| host.capabilities[kind])))
        .collect();
    Ok(Json(json!({
        "hosts": hosts,
        "models": models,
        "prompts": store.prompts().map_err(ApiError::storage)?,
        "jobs": jobs,
        "capabilities": capabilities,
        "eligibility": eligibility::snapshot(&hosts),
        "event_cursor": store.feed_cursor().map_err(ApiError::storage)?,
    })))
}

async fn get_settings(State(app): State<Shared>) -> Result<Json<Value>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    let hosts = app.defined_hosts().map_err(ApiError::storage)?;
    let effective = app.effective_hosts().map_err(ApiError::storage)?;
    let views: Vec<_> = hosts
        .iter()
        .map(|host| {
            app.settings.view(
                &store,
                host,
                effective.iter().find(|entry| entry.id == host.id),
                app.admin.is_some(),
            )
        })
        .collect::<std::result::Result<_, _>>()
        .map_err(ApiError::storage)?;
    Ok(Json(json!({"hosts": views, "trace":app.trace_policy})))
}

async fn put_settings(
    State(app): State<Shared>,
    Path(id): Path<String>,
    payload: std::result::Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>> {
    use crate::settings::PutError;
    let Json(mut body) =
        payload.map_err(|error| ApiError(error.status(), "invalid settings request".into()))?;
    let binding = body
        .as_object_mut()
        .and_then(|body| body.remove("profile_binding"));
    let binding = binding
        .as_ref()
        .map(|binding| {
            binding
                .as_str()
                .ok_or_else(|| ApiError::bad("invalid settings profile binding"))
        })
        .transpose()?;
    let settings =
        serde_json::from_value::<brewery_worker::operational::Settings>(body).map_err(|_| {
            ApiError(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid settings request".into(),
            )
        })?;
    let store = app.store.lock().map_err(ApiError::storage)?;
    let hosts = app.defined_hosts().map_err(ApiError::storage)?;
    let host = hosts
        .iter()
        .find(|host| host.id == id)
        .ok_or_else(|| ApiError::not_found("host"))?;
    let native = app.settings.native(&id);
    if native.managed()
        && let Some(admin) = &app.admin
        && !admin
            .effective_hosts(std::slice::from_ref(host), &app)
            .map_err(ApiError::storage)?
            .iter()
            .any(|active| active.id == id)
    {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "worker disabled by infrastructure administration".into(),
        ));
    }
    if native.managed() && store.host_has_jobs(&id).map_err(ApiError::storage)? {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "worker has queued or running jobs".into(),
        ));
    }
    app.settings
        .put(&store, host, native, settings, binding)
        .map_err(|err| match err {
            PutError::NotReady(reason) => ApiError(StatusCode::SERVICE_UNAVAILABLE, reason.into()),
            PutError::Storage(error) => ApiError::storage(error),
            PutError::Conflict => {
                ApiError(StatusCode::CONFLICT, "settings revision conflict".into())
            }
            PutError::Invalid(error) => ApiError::bad(error),
        })?;
    let effective = app.effective_hosts().map_err(ApiError::storage)?;
    let view = app
        .settings
        .view(
            &store,
            host,
            effective.iter().find(|entry| entry.id == id),
            app.admin.is_some(),
        )
        .map_err(ApiError::storage)?;
    Ok(Json(json!(view)))
}

async fn get_hf_tokens(State(app): State<Shared>) -> Result<Json<Value>> {
    let _store = app.store.lock().map_err(ApiError::storage)?;
    let hosts = app.effective_hosts().map_err(ApiError::storage)?;
    let mut statuses = Vec::new();
    for host in &hosts {
        if host.capabilities["hf_download"]
            && let Some(path) = app.hf_tokens.file(&host.id)
            && let Ok(configured) =
                hf_tokens::PrivateFile::open(path).and_then(|file| file.configured())
        {
            statuses.push(hf_tokens::Status {
                id: &host.id,
                configured,
            });
        }
    }
    Ok(Json(json!({"hosts":statuses})))
}

async fn set_hf_token(
    State(app): State<Shared>,
    Path(id): Path<String>,
    payload: std::result::Result<Json<Box<RawValue>>, JsonRejection>,
) -> Result<Json<Value>> {
    let set: hf_tokens::Set = read_hf_token_request(payload)?;
    if !brewery_worker::hf_download::valid_token(&set.token) {
        return Err(ApiError::bad("invalid Hugging Face token format"));
    }
    change_hf_token(&app, &id, Some(&set.token))
}

async fn remove_hf_token(
    State(app): State<Shared>,
    Path(id): Path<String>,
    payload: std::result::Result<Json<Box<RawValue>>, JsonRejection>,
) -> Result<Json<Value>> {
    let _: hf_tokens::Remove = read_hf_token_request(payload)?;
    change_hf_token(&app, &id, None)
}

pub(crate) fn read_hf_token_request<T: serde::de::DeserializeOwned>(
    payload: std::result::Result<Json<Box<RawValue>>, JsonRejection>,
) -> Result<T> {
    let invalid = |status| ApiError(status, "invalid token request".into());
    let Json(raw) = payload.map_err(|error| invalid(error.status()))?;
    if !raw.get().starts_with('{') {
        return Err(invalid(StatusCode::UNPROCESSABLE_ENTITY));
    }
    serde_json::from_str(raw.get()).map_err(|_| invalid(StatusCode::UNPROCESSABLE_ENTITY))
}

fn change_hf_token(app: &App, id: &str, token: Option<&str>) -> Result<Json<Value>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    if !app.hosts.iter().any(|host| host.id == id) {
        return Err(ApiError::not_found("host"));
    }
    let effective = app.effective_hosts().map_err(ApiError::storage)?;
    let path = app
        .hf_tokens
        .file(id)
        .filter(|_| {
            effective
                .iter()
                .any(|host| host.id == id && host.capabilities["hf_download"])
        })
        .ok_or_else(|| {
            ApiError(
                StatusCode::CONFLICT,
                "worker is not enabled for local Hugging Face token settings".into(),
            )
        })?;
    if store.host_has_jobs(id).map_err(ApiError::storage)? {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "worker has queued or running jobs".into(),
        ));
    }
    let configured = hf_tokens::PrivateFile::open(path)
        .and_then(|file| file.change(token))
        .map_err(|_| {
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "Hugging Face token storage unavailable".into(),
            )
        })?;
    Ok(Json(json!(hf_tokens::Status { id, configured })))
}

async fn events(
    State(app): State<Shared>,
    method: Method,
    headers: HeaderMap,
    uri: Uri,
) -> Result<impl IntoResponse> {
    // Axum implicitly routes HEAD through GET, but HEAD must not subscribe.
    if method != Method::GET {
        return Err(ApiError(
            StatusCode::METHOD_NOT_ALLOWED,
            "HTTP method not allowed".into(),
        ));
    }
    #[derive(serde::Deserialize)]
    struct Cursor {
        after: Option<String>,
    }
    let bad_cursor = || ApiError::bad("event cursor must be a non-negative decimal integer");
    let after = match single(&headers, "last-event-id").map_err(|_| bad_cursor())? {
        Some(header) => Some(header.to_owned()),
        None => {
            let Query(cursor) = Query::<Cursor>::try_from_uri(&uri).map_err(|_| bad_cursor())?;
            cursor.after
        }
    };
    let after = match after {
        Some(value) if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) => {
            value.parse::<i64>().map_err(|_| bad_cursor())?
        }
        Some(_) => return Err(bad_cursor()),
        None => app
            .store
            .lock()
            .map_err(ApiError::storage)?
            .feed_cursor()
            .map_err(ApiError::storage)?,
    };
    let permit = app
        .feed_subscribers
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "event stream capacity reached".into(),
            )
        })?;
    Ok(Sse::new(feed::subscribe(app, permit, after))
        .keep_alive(KeepAlive::new().interval(feed::HEARTBEAT).text("heartbeat")))
}

async fn add_model(
    State(app): State<Shared>,
    headers: HeaderMap,
    payload: std::result::Result<Json<NewModel>, JsonRejection>,
) -> Result<Response> {
    let (request, submission) = read_submission(&headers, "models", payload)?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, submission.as_ref())? {
        return Ok(response);
    }
    let model = request.validate().map_err(ApiError::bad)?;
    let path = std::path::Path::new(&model.path);
    if store.is_scratch_path(path) {
        return Err(ApiError::bad("scratch files require operator approval"));
    }
    if app.agent.protected_path(path) {
        return Err(ApiError::bad(
            "private credential storage cannot enter the Library",
        ));
    }
    if store
        .is_private_editor_path(path)
        .map_err(ApiError::storage)?
    {
        return Err(ApiError::bad(
            "private scratch job output cannot enter the Library",
        ));
    }
    if let Some(reason) = manual_registration_blocker(&model.host, &model.kind) {
        return Err(ApiError::bad(reason.text));
    }
    if !app
        .effective_hosts()
        .map_err(ApiError::storage)?
        .iter()
        .any(|host| host.id == model.host)
    {
        return Err(ApiError::bad("unknown worker host"));
    }
    store
        .add_model(&model, submission.as_ref())
        .map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(model)).into_response())
}

async fn delete_model(State(app): State<Shared>, Path(id): Path<String>) -> Result<StatusCode> {
    if !app
        .store
        .lock()
        .map_err(ApiError::storage)?
        .delete_model(&id)
        .map_err(ApiError::storage)?
    {
        return Err(ApiError::not_found("model"));
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn add_prompt(
    State(app): State<Shared>,
    headers: HeaderMap,
    payload: std::result::Result<Json<NewPrompt>, JsonRejection>,
) -> Result<Response> {
    let (request, submission) = read_submission(&headers, "prompts", payload)?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, submission.as_ref())? {
        return Ok(response);
    }
    let prompt = request.validate().map_err(ApiError::bad)?;
    store
        .add_prompt(&prompt, submission.as_ref())
        .map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(prompt)).into_response())
}

async fn update_prompt(
    State(app): State<Shared>,
    Path(id): Path<String>,
    payload: std::result::Result<Json<NewPrompt>, JsonRejection>,
) -> Result<Json<crate::domain::Prompt>> {
    let prompt = read(payload)?.validate_with_id(id).map_err(ApiError::bad)?;
    if !app
        .store
        .lock()
        .map_err(ApiError::storage)?
        .update_prompt(&prompt)
        .map_err(ApiError::storage)?
    {
        return Err(ApiError::not_found("prompt"));
    }
    Ok(Json(prompt))
}

async fn delete_prompt(State(app): State<Shared>, Path(id): Path<String>) -> Result<StatusCode> {
    use crate::store::PromptDeletion;
    match app
        .store
        .lock()
        .map_err(ApiError::storage)?
        .delete_prompt(&id)
        .map_err(ApiError::storage)?
    {
        PromptDeletion::Deleted => Ok(StatusCode::NO_CONTENT),
        PromptDeletion::Missing => Err(ApiError::not_found("prompt")),
        PromptDeletion::InUse(job) => Err(ApiError(
            StatusCode::CONFLICT,
            format!("Job {job} uses this prompt."),
        )),
    }
}

async fn add_job(
    State(app): State<Shared>,
    headers: HeaderMap,
    payload: std::result::Result<Json<NewJob>, JsonRejection>,
) -> Result<Response> {
    let (request, submission) = read_submission(&headers, "jobs", payload)?;
    let fingerprint = dedupe::Request::new("jobs", &request).map_err(ApiError::storage)?;
    let (job, mut envelope, intervention) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        if let Some(response) = replay(&store, submission.as_ref())? {
            return Ok(response);
        }
        let tx = store.submission_transaction().map_err(ApiError::storage)?;
        if let Some(id) = find_duplicate(&headers, &tx, &fingerprint, submission.as_ref())? {
            tx.commit().map_err(ApiError::storage)?;
            return Ok(duplicate_reply(&id, json!({"id": id})));
        }
        let (mut job, mut envelope) = prepare_job(&app, &store, request)?;
        let intervention = if job.kind == trace::JOB {
            crate::interventions::TracePlan::prepare(
                &store,
                &mut job,
                &mut envelope,
                app.trace_policy,
            )?
        } else {
            None
        };
        let Some(intervention) = intervention else {
            return enqueue_job(
                &app,
                &store,
                tx,
                &fingerprint,
                job,
                envelope,
                submission.as_ref(),
            );
        };
        (job, envelope, intervention)
    };
    // A patch read can wait on a remote worker. Release the transaction and lock.
    let resolved = intervention.resolve(&app, &mut envelope).await;
    let store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, submission.as_ref())? {
        return Ok(response);
    }
    let tx = store.submission_transaction().map_err(ApiError::storage)?;
    if let Some(id) = find_duplicate(&headers, &tx, &fingerprint, submission.as_ref())? {
        tx.commit().map_err(ApiError::storage)?;
        return Ok(duplicate_reply(&id, json!({"id": id})));
    }
    resolved?;
    let input = &envelope.inputs[0];
    if let Some(digest) = envelope.scratch_pins.get(&input.path) {
        let (_, current) = crate::scratch::jobs::trace_input(&store, &input.id)?;
        if digest != &current {
            return Err(ApiError::bad("scratch GGUF changed during admission"));
        }
    } else if store.model(&input.id).map_err(ApiError::storage)?.is_none() {
        return Err(ApiError::bad("unknown model ID"));
    }
    enqueue_job(
        &app,
        &store,
        tx,
        &fingerprint,
        job,
        envelope,
        submission.as_ref(),
    )
}

fn prepare_job(app: &App, store: &Store, request: NewJob) -> Result<(Job, Envelope)> {
    if request.input_ids.len() > 32 || !KINDS.contains(&request.kind.as_str()) {
        return Err(ApiError::bad("unknown job kind or too many inputs"));
    }
    let mut scratch_pins = BTreeMap::new();
    let inputs = request
        .input_ids
        .iter()
        .map(|id| {
            let model = if let Some(model) = store.model(id).map_err(ApiError::storage)? {
                model
            } else if request.kind == trace::JOB
                && request.input_ids.len() == 1
                && store.scratch(id).map_err(ApiError::storage)?.is_some()
            {
                let (model, sha) = crate::scratch::jobs::trace_input(store, id)?;
                scratch_pins.insert(model.path.clone(), sha);
                model
            } else {
                return Err(ApiError::bad("unknown model ID"));
            };
            if !app
                .effective_hosts()
                .map_err(ApiError::storage)?
                .iter()
                .any(|host| host.id == model.host)
            {
                return Err(ApiError::bad("input model's host is no longer configured"));
            }
            Ok(model)
        })
        .collect::<Result<Vec<_>>>()?;
    let host = if request.kind == "hf_download" && request.host.is_none() {
        if !app.backend_download {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "Backend Hugging Face downloads are unavailable".into(),
            ));
        }
        crate::backend_hf::ID
    } else {
        request
            .host
            .as_deref()
            .or_else(|| {
                if matches!(
                    request.kind.as_str(),
                    "cache_cleanup" | "transfer" | "transfer_preview" | "benchmark"
                ) {
                    None
                } else {
                    inputs.first().map(|m| m.host.as_str())
                }
            })
            .ok_or_else(|| ApiError::bad("a destination host is required"))?
    };
    let host = host.to_owned();
    if inputs
        .iter()
        .any(|model| model.host == crate::backend_hf::ID)
    {
        if host != crate::backend_hf::ID
            || !matches!(
                request.kind.as_str(),
                "weight_inspect" | "weight_propose" | "weight_apply" | "weight_tune"
            )
            || inputs.len() != if request.kind == "weight_tune" { 3 } else { 1 }
            || inputs.iter().any(|m| m.host != crate::backend_hf::ID)
        {
            return Err(ApiError::bad("backend files require a local weight job"));
        }
        if !store
            .backend_publication(&inputs[0])
            .map_err(ApiError::storage)?
        {
            return Err(ApiError::bad("backend input has no verified publication"));
        }
        if !matches!(inputs[0].kind.as_str(), "gguf" | "hf" | "safetensors") {
            return Err(ApiError::bad(
                "backend editing requires GGUF, SafeTensors or an indexed Hugging Face checkpoint",
            ));
        }
        if !app.backend_edit_formats.contains(&inputs[0].kind) {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "backend editor does not support this model format".into(),
            ));
        }
        if inputs[1..]
            .iter()
            .any(|m| !crate::backend_calibration::is_managed(app.backend_calibration.as_deref(), m))
        {
            return Err(ApiError::bad(
                "backend tuning requires registered calibration inputs",
            ));
        }
    }
    if host != "auto" || request.kind != "convert" {
        if !app
            .defined_hosts()
            .map_err(ApiError::storage)?
            .iter()
            .any(|entry| entry.id == host)
        {
            return Err(ApiError::bad("unknown worker host"));
        }
        let effective = app.effective_hosts().map_err(ApiError::storage)?;
        let entry = effective
            .iter()
            .find(|effective| effective.id == host)
            .ok_or_else(|| ApiError(StatusCode::CONFLICT, "worker unavailable".into()))?;
        if !entry.capabilities[&request.kind] {
            let message = if SIMPLE_WORKFLOWS.contains(&request.kind.as_str()) {
                entry.capability_reason(&request.kind).text
            } else {
                entry.reasons[&request.kind].clone()
            };
            return Err(ApiError(StatusCode::CONFLICT, message));
        }
        if request.kind == "benchmark" {
            entry
                .check_benchmark(&request.options)
                .map_err(ApiError::bad)?;
        }
        if let Some(source) = request.options.get("baseline_source")
            && !entry
                .baseline_sources
                .iter()
                .any(|entry| source.as_str() == Some(&entry.id))
        {
            return Err(ApiError::bad(
                "baseline_source is not configured on this host",
            ));
        }
    }
    let prompt = match (&request.prompt_id, &request.prompt) {
        (Some(_), Some(_)) => return Err(ApiError::bad("provide prompt_id or prompt, not both")),
        (Some(id), None) => Some(
            store
                .prompt(id)
                .map_err(ApiError::storage)?
                .ok_or_else(|| ApiError::bad("unknown prompt ID"))?
                .text,
        ),
        (None, prompt) => prompt.clone(),
    };
    let (job, mut envelope) = request.build(host, inputs, prompt).map_err(ApiError::bad)?;
    envelope.scratch_pins = scratch_pins;
    Ok((job, envelope))
}

fn enqueue_job(
    app: &App,
    store: &Store,
    tx: rusqlite::Transaction<'_>,
    fingerprint: &dedupe::Request,
    job: Job,
    mut envelope: Envelope,
    submission: Option<&Submission<'_>>,
) -> Result<Response> {
    let job = if job.kind == trace::JOB && !envelope.scratch_pins.is_empty() {
        crate::scratch::jobs::prepare_enqueue(app, job, &envelope)?
    } else {
        admit_job(app, store, job, &mut envelope)?
    };
    Store::enqueue(&tx, &job, &envelope, submission).map_err(ApiError::storage)?;
    fingerprint
        .record(&tx, &job.id)
        .map_err(ApiError::storage)?;
    tx.commit().map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(job)).into_response())
}

fn admit_job(app: &App, store: &Store, mut job: Job, envelope: &mut Envelope) -> Result<Job> {
    let hosts = app.effective_hosts().map_err(ApiError::storage)?;
    if crate::weights::is_kind(&job.kind)
        && let Some(entry) = hosts.iter().find(|entry| entry.id == job.host)
    {
        job.local_edit = entry.local_edit;
        job.remote_editor = entry.remote_editor;
    }
    if job.kind == "weight_apply" {
        let prior_id = job.options["proposal_job_id"]
            .as_str()
            .expect("validated proposal ID");
        let prior = store
            .job(prior_id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("proposal job"))?;
        let saved = store.envelope(prior_id).map_err(ApiError::storage)?;
        if prior.local_edit != job.local_edit || prior.remote_editor != job.remote_editor {
            return Err(ApiError::bad(
                "make a new proposal on the current weight editor",
            ));
        }
        let result = store
            .job_result(prior_id)
            .map_err(ApiError::storage)?
            .unwrap_or(Value::Null);
        let (proposal, checks) =
            crate::weights::apply_inputs(&envelope.inputs[0], &prior, &saved, &result)
                .map_err(ApiError::bad)?;
        envelope.inputs.push(proposal);
        envelope.options = checks.as_object().expect("apply checks").clone();
    }
    if candidates(&job, envelope, &hosts).is_empty() {
        let reason = if SIMPLE_WORKFLOWS.contains(&job.kind.as_str()) {
            hosts
                .iter()
                .find(|host| host.id == job.host)
                .and_then(|host| standard_job_host_blocker(host, &job.kind))
                .or_else(|| input_blocker(&job, envelope, &hosts))
        } else {
            None
        };
        return Err(ApiError(
            StatusCode::CONFLICT,
            reason.map_or_else(
                || "no eligible worker for the requested job options".into(),
                |reason| reason.text,
            ),
        ));
    }
    if job.local_edit {
        job.directory_revision = app
            .editor_grants(&job, envelope)
            .map_err(ApiError::storage)?
            .ok_or(ApiError(
                StatusCode::CONFLICT,
                "A model directory is not authorized for this editor".into(),
            ))?
            .revision;
    }
    Ok(job)
}

fn find_duplicate(
    headers: &HeaderMap,
    tx: &rusqlite::Transaction<'_>,
    fingerprint: &dedupe::Request,
    submission: Option<&Submission<'_>>,
) -> Result<Option<String>> {
    if single(headers, "brewery-allow-duplicate") == Ok(Some("1")) {
        return Ok(None);
    }
    let matched = fingerprint.find(tx).map_err(ApiError::storage)?;
    if let Some(id) = &matched {
        // Bind a fresh key even when no new work is queued. A lost match
        // response can then replay the same ID after the duplicate window.
        idempotency::record(tx, submission, id).map_err(ApiError::storage)?;
    }
    Ok(matched)
}

fn duplicate_reply(id: &str, mut body: Value) -> Response {
    body["duplicate_of"] = json!(true);
    ([("brewery-duplicate", id)], Json(body)).into_response()
}

pub(crate) fn read_submission<'a, T: serde::Serialize>(
    headers: &'a HeaderMap,
    scope: &'static str,
    payload: std::result::Result<Json<T>, JsonRejection>,
) -> Result<(T, Option<Submission<'a>>)> {
    let invalid = || ApiError::bad("invalid Idempotency-Key");
    let key = single(headers, "idempotency-key").map_err(|_| invalid())?;
    if key.is_some_and(|key| !idempotency::valid_key(key)) {
        return Err(invalid());
    }
    let request = read(payload)?;
    let submission = key
        .map(|key| Submission::new(scope, key, &request))
        .transpose()
        .map_err(ApiError::storage)?;
    Ok((request, submission))
}

pub(crate) fn replay(
    store: &Store,
    submission: Option<&Submission<'_>>,
) -> Result<Option<Response>> {
    let Some(submission) = submission else {
        return Ok(None);
    };
    match store.replay(submission).map_err(ApiError::storage)? {
        Replay::New => Ok(None),
        Replay::Matched(id) => Ok(Some(Json(json!({"id": id})).into_response())),
        Replay::Conflict => Err(ApiError(
            StatusCode::CONFLICT,
            "Idempotency-Key already used for a different request".into(),
        )),
    }
}

async fn job(State(app): State<Shared>, Path(id): Path<String>) -> Result<Json<Value>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(
        store
            .detail(&id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("job"))?,
    ))
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceQuery {
    #[serde(default)]
    prompt: usize,
    layer: Option<usize>,
    role: Option<trace::Role>,
    start: Option<usize>,
    end: Option<usize>,
}

type TraceQueryResult =
    std::result::Result<Query<TraceQuery>, axum::extract::rejection::QueryRejection>;

async fn trace_manifest(
    State(app): State<Shared>,
    Path(id): Path<String>,
    query: TraceQueryResult,
) -> Result<Json<Value>> {
    trace_slice(app, id, query, View::Manifest).await
}

async fn trace_tensor(
    State(app): State<Shared>,
    Path(id): Path<String>,
    query: TraceQueryResult,
) -> Result<Json<Value>> {
    trace_slice(app, id, query, View::Tensor).await
}

async fn trace_logit_lens(
    State(app): State<Shared>,
    Path(id): Path<String>,
    query: TraceQueryResult,
) -> Result<Json<Value>> {
    trace_slice(app, id, query, View::LogitLens).await
}

async fn trace_model_logits(
    State(app): State<Shared>,
    Path(id): Path<String>,
    query: TraceQueryResult,
) -> Result<Json<Value>> {
    trace_slice(app, id, query, View::ModelLogits).await
}

async fn trace_slice(
    app: Shared,
    id: String,
    query: TraceQueryResult,
    view: View,
) -> Result<Json<Value>> {
    let Query(TraceQuery {
        prompt,
        layer,
        role,
        start,
        end,
    }) = query.map_err(|_| ApiError::bad("invalid trace query"))?;
    let slice = Slice {
        view,
        prompt,
        layer,
        role,
        start,
        end,
    };
    slice.check().map_err(|e| ApiError::bad(e.to_string()))?;
    let job = app
        .store
        .lock()
        .map_err(ApiError::storage)?
        .job(&id)
        .map_err(ApiError::storage)?
        .filter(|j| j.kind == trace::JOB && j.state == JobState::Succeeded)
        .ok_or_else(|| ApiError::not_found("trace result"))?;
    match app.dispatcher.trace_read(&job, &slice).await.map_err(|e| {
        eprintln!("Trace read failed for {id}: {e}");
        ApiError(
            StatusCode::BAD_GATEWAY,
            "Trace worker could not read the result".into(),
        )
    })? {
        ReadReply::Ok(data) => Ok(Json(data)),
        ReadReply::Missing => Err(ApiError::not_found("trace slice")),
        ReadReply::TooLarge => Err(ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Select a smaller trace range".into(),
        )),
    }
}

async fn list_profiles(State(app): State<Shared>) -> Result<Json<Vec<crate::profiles::Profile>>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(store.profiles().map_err(ApiError::storage)?))
}

async fn profile_export_options(
    State(app): State<Shared>,
    Path(revision): Path<String>,
) -> Result<Json<Value>> {
    use crate::gguf::Format;

    if !crate::profiles::valid_revision(&revision) {
        return Err(ApiError::bad("invalid profile revision"));
    }
    let encoder = app
        .store
        .lock()
        .map_err(ApiError::storage)?
        .profile(&revision)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("profile revision"))?
        .format();

    // This describes paths, not a capability check, a proof, or a job request.
    let options = [Format::Hgn, Format::Gguf].map(|format| {
        if format == encoder {
            json!({
                "format": format,
                "admission": "standard_workflow",
                "stock_preservation": "by_design",
                "extension_requirement": "not_required",
                "runtime_proof": "not_assessed"
            })
        } else {
            json!({
                "format": format,
                "admission": "unavailable",
                "stock_preservation": "not_established",
                "extension_requirement": "not_established",
                "experimental_extension": "brewery.cross-container.v1",
                "runtime_proof": "not_assessed"
            })
        }
    });
    Ok(Json(
        json!({"revision": revision, "encoder": encoder, "options": options}),
    ))
}

async fn save_profile(
    State(app): State<Shared>,
    payload: std::result::Result<Json<crate::profiles::NewProfile>, JsonRejection>,
) -> Result<(StatusCode, Json<crate::profiles::Profile>)> {
    let profile = read(payload)?.save().map_err(ApiError::bad)?;
    let store = app.store.lock().map_err(ApiError::storage)?;
    if store.save_profile(&profile).map_err(ApiError::storage)? {
        return Ok((StatusCode::CREATED, Json(profile)));
    }
    let existing = store
        .profile(profile.revision())
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::storage("profile revision missing after save"))?;
    Ok((StatusCode::OK, Json(existing)))
}

async fn add_conversion(
    State(app): State<Shared>,
    headers: HeaderMap,
    payload: std::result::Result<Json<NewConversion>, JsonRejection>,
) -> Result<Response> {
    let (request, submission) = read_submission(&headers, "conversions", payload)?;
    let store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(submission) = &submission {
        match store.replay(submission).map_err(ApiError::storage)? {
            Replay::New => {}
            Replay::Matched(id) => {
                let row = store
                    .conversion(&id)
                    .map_err(ApiError::storage)?
                    .ok_or_else(|| ApiError::storage("idempotent conversion is missing"))?;
                return Ok(Json(row).into_response());
            }
            Replay::Conflict => {
                return Err(ApiError(
                    StatusCode::CONFLICT,
                    "Idempotency-Key already used for a different request".into(),
                ));
            }
        }
    }
    let fingerprint = dedupe::Request::new("conversions", &request).map_err(ApiError::storage)?;
    let tx = store.submission_transaction().map_err(ApiError::storage)?;
    if let Some(id) = find_duplicate(&headers, &tx, &fingerprint, submission.as_ref())? {
        let row = store
            .conversion(&id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::storage("duplicate conversion is missing"))?;
        tx.commit().map_err(ApiError::storage)?;
        return Ok(duplicate_reply(
            &id,
            serde_json::to_value(row).map_err(ApiError::storage)?,
        ));
    }
    let active = store
        .by_state(crate::domain::State::Running)
        .map_err(ApiError::storage)?;
    let (row, job, envelope) = admit_conversion(&app, &store, request, &active)?
        .map_err(|reason| ApiError(StatusCode::CONFLICT, reason.text))?;
    Store::create_conversion(&tx, &row, &job, &envelope, submission.as_ref())
        .map_err(ApiError::storage)?;
    fingerprint
        .record(&tx, &row.id)
        .map_err(ApiError::storage)?;
    tx.commit().map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(row)).into_response())
}

async fn conversion_eligibility(
    State(app): State<Shared>,
    payload: std::result::Result<Json<NewConversion>, JsonRejection>,
) -> Result<Json<Value>> {
    let request = read(payload)?;
    let store = app.store.lock().map_err(ApiError::storage)?;
    let decision = match admit_conversion(&app, &store, request, &[]) {
        Ok(Ok(_)) => eligibility::Entry::new(vec![]),
        Ok(Err(reason)) => eligibility::Entry::new(vec![reason]),
        Err(error) if error.0.is_client_error() => {
            eligibility::Entry::new(vec![Reason::link(&error.1, "#/convert")])
        }
        Err(error) => return Err(error),
    };
    Ok(Json(json!(decision)))
}

fn admit_conversion(
    app: &App,
    store: &Store,
    request: NewConversion,
    active: &[Job],
) -> Result<std::result::Result<(Conversion, Job, Envelope), Reason>> {
    if request.options.contains_key(brewery_worker::hy4::OPTION) {
        return Err(ApiError::bad(
            "Hy4 jobs require a single-stage /api/jobs request",
        ));
    }
    if !(1..=5).contains(&request.input_ids.len()) {
        return Err(ApiError::bad(
            "conversion requires one source and up to four auxiliary inputs",
        ));
    }
    let hosts = app.effective_hosts().map_err(ApiError::storage)?;
    if let Some(host) = request.host.as_deref()
        && host != "auto"
        && !hosts.iter().any(|entry| entry.id == host)
    {
        return Err(ApiError::bad("unknown worker host"));
    }
    let inputs = request
        .input_ids
        .iter()
        .map(|id| {
            let model = store
                .model(id)
                .map_err(ApiError::storage)?
                .ok_or_else(|| ApiError::bad("unknown model ID"))?;
            if !hosts.iter().any(|host| host.id == model.host) {
                return Err(ApiError::bad("input model's host is no longer configured"));
            }
            Ok(model)
        })
        .collect::<Result<Vec<_>>>()?;
    let profile = request
        .profile_revision
        .as_deref()
        .map(|revision| {
            if !crate::profiles::valid_revision(revision) {
                return Err(ApiError::bad("invalid profile revision"));
            }
            store
                .profile(revision)
                .map_err(ApiError::storage)?
                .ok_or_else(|| ApiError::not_found("profile revision"))
        })
        .transpose()?;
    match Conversion::start(request, inputs, &hosts, active, profile.as_ref()) {
        Ok(admitted) => Ok(Ok(admitted)),
        Err(StartError::Invalid(message)) => Err(ApiError::bad(message)),
        Err(StartError::Blocked(reason)) => Ok(Err(reason)),
    }
}

async fn list_conversions(State(app): State<Shared>) -> Result<Json<Vec<Conversion>>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(store.conversions().map_err(ApiError::storage)?))
}

async fn get_conversion(
    State(app): State<Shared>,
    Path(id): Path<String>,
) -> Result<Json<Conversion>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(
        store
            .conversion(&id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("conversion"))?,
    ))
}

async fn cancel_conversion(
    State(app): State<Shared>,
    Path(id): Path<String>,
) -> Result<Json<Conversion>> {
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(
        store
            .cancel_conversion(&id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("conversion"))?,
    ))
}

async fn cancel(State(app): State<Shared>, Path(id): Path<String>) -> Result<impl IntoResponse> {
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(
        store
            .cancel(&id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("job"))?,
    ))
}

pub(crate) fn read<T>(payload: std::result::Result<Json<T>, JsonRejection>) -> Result<T> {
    payload
        .map(|Json(body)| body)
        .map_err(|e| ApiError(e.status(), e.body_text()))
}

pub(crate) struct ApiError(pub(crate) StatusCode, pub(crate) String);
impl ApiError {
    pub(crate) fn bad(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, msg.into())
    }
    pub(crate) fn not_found(object: &str) -> Self {
        Self(StatusCode::NOT_FOUND, format!("{object} not found"))
    }
    pub(crate) fn storage(err: impl std::fmt::Display) -> Self {
        eprintln!("controller storage error: {err}");
        Self(
            StatusCode::INTERNAL_SERVER_ERROR,
            "controller storage error".into(),
        )
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.0,
            [("brewery-error", "1")],
            Json(json!({"error":self.1})),
        )
            .into_response()
    }
}
