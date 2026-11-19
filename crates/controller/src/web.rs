use crate::{
    App,
    conversion::{Conversion, NewConversion},
    domain::{KINDS, NewJob, NewModel, NewPrompt, candidates},
    feed, hf_tokens,
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
        .route("/models", post(add_model))
        .route("/models/{id}", delete(delete_model))
        .route("/prompts", post(add_prompt))
        .route("/jobs", post(add_job))
        .route("/jobs/{id}", get(job))
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
        .fallback(|| async { ApiError(StatusCode::NOT_FOUND, "API endpoint not found".into()) })
        .method_not_allowed_fallback(|| async {
            ApiError(
                StatusCode::METHOD_NOT_ALLOWED,
                "HTTP method not allowed".into(),
            )
        })
        .layer(DefaultBodyLimit::max(256 * 1024));
    Router::new()
        .nest("/api", api)
        .fallback_service(ServeDir::new(dist))
        .with_state(app)
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
    let api = req.uri().path() == "/api" || req.uri().path().starts_with("/api/");
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

fn single<'a>(headers: &'a HeaderMap, name: &str) -> std::result::Result<Option<&'a str>, ()> {
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

fn has_json(headers: &HeaderMap) -> bool {
    single(headers, "content-type")
        .ok()
        .flatten()
        .and_then(|s| s.split(';').next())
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("application/json"))
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
    app.suggestions.get(query).await.map(Json).map_err(|error| {
        let (status, message) = match error {
            SuggestError::Busy => (StatusCode::SERVICE_UNAVAILABLE, "path suggestions are busy"),
            SuggestError::Timeout => (StatusCode::GATEWAY_TIMEOUT, "path suggestions timed out"),
            SuggestError::Unavailable => (StatusCode::BAD_GATEWAY, "path suggestions unavailable"),
        };
        ApiError(status, message.into())
    })
}

async fn state(State(app): State<Shared>) -> Result<Json<Value>> {
    // Every controller mutation takes this mutex, and its feed entry commits
    // in the same SQLite transaction. Read the cursor while still holding it.
    let store = app.store.lock().map_err(ApiError::storage)?;
    let hosts = app.effective_hosts().map_err(ApiError::storage)?;
    let capabilities: BTreeMap<_, _> = KINDS
        .iter()
        .map(|&kind| (kind, hosts.iter().any(|host| host.capabilities[kind])))
        .collect();
    Ok(Json(json!({
        "hosts": hosts,
        "models": store.models().map_err(ApiError::storage)?,
        "prompts": store.prompts().map_err(ApiError::storage)?,
        "jobs": store.jobs().map_err(ApiError::storage)?,
        "capabilities": capabilities,
        "event_cursor": store.feed_cursor().map_err(ApiError::storage)?,
    })))
}

async fn get_settings(State(app): State<Shared>) -> Result<Json<Value>> {
    let _store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(
        json!({"hosts":app.hosts.iter().map(|host| app.settings.view(host)).collect::<Vec<_>>()}),
    ))
}

async fn put_settings(
    State(app): State<Shared>,
    Path(id): Path<String>,
    payload: std::result::Result<Json<brewery_worker::operational::Settings>, JsonRejection>,
) -> Result<Json<Value>> {
    use crate::settings::PutError;
    let Json(settings) =
        payload.map_err(|error| ApiError(error.status(), "invalid settings request".into()))?;
    let store = app.store.lock().map_err(ApiError::storage)?;
    let host = app
        .hosts
        .iter()
        .find(|host| host.id == id)
        .ok_or_else(|| ApiError::not_found("host"))?;
    if let Some(admin) = &app.admin
        && !admin
            .effective_hosts(std::slice::from_ref(host))
            .map_err(ApiError::storage)?
            .iter()
            .any(|active| active.id == id)
    {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "worker disabled by infrastructure administration".into(),
        ));
    }
    if app.settings.enrolled(&id) && store.host_has_jobs(&id).map_err(ApiError::storage)? {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "worker has queued or running jobs".into(),
        ));
    }
    let view = app.settings.put(host, settings).map_err(|err| match err {
        PutError::Locked => ApiError(StatusCode::CONFLICT, "worker settings are locked".into()),
        PutError::NotReady => ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker settings unavailable".into(),
        ),
        PutError::Conflict => ApiError(StatusCode::CONFLICT, "settings revision conflict".into()),
        PutError::Invalid(error) => ApiError::bad(error),
    })?;
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

fn read_hf_token_request<T: serde::de::DeserializeOwned>(
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

async fn add_job(
    State(app): State<Shared>,
    headers: HeaderMap,
    payload: std::result::Result<Json<NewJob>, JsonRejection>,
) -> Result<Response> {
    let (request, submission) = read_submission(&headers, "jobs", payload)?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, submission.as_ref())? {
        return Ok(response);
    }
    if request.input_ids.len() > 32 || !KINDS.contains(&request.kind.as_str()) {
        return Err(ApiError::bad("unknown job kind or too many inputs"));
    }
    let inputs = request
        .input_ids
        .iter()
        .map(|id| {
            let model = store
                .model(id)
                .map_err(ApiError::storage)?
                .ok_or_else(|| ApiError::bad("unknown model ID"))?;
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
    if host != "auto" || request.kind != "convert" {
        let entry = app
            .hosts
            .iter()
            .find(|entry| entry.id == host)
            .ok_or_else(|| ApiError::bad("unknown worker host"))?;
        let effective = app.effective_hosts().map_err(ApiError::storage)?;
        let entry = effective
            .iter()
            .find(|effective| effective.id == entry.id)
            .ok_or_else(|| ApiError(StatusCode::CONFLICT, "worker unavailable".into()))?;
        if !entry.capabilities[&request.kind] {
            return Err(ApiError(
                StatusCode::CONFLICT,
                entry.reasons[&request.kind].clone(),
            ));
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
    if job.kind == "weight_apply" {
        let prior_id = job.options["proposal_job_id"]
            .as_str()
            .expect("validated proposal ID");
        let prior = store
            .job(prior_id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("proposal job"))?;
        let saved = store.envelope(prior_id).map_err(ApiError::storage)?;
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
    let hosts = app.effective_hosts().map_err(ApiError::storage)?;
    if candidates(&job, &envelope, &hosts).is_empty() {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "no eligible worker for the requested job options".into(),
        ));
    }
    store
        .enqueue(&job, &envelope, submission.as_ref())
        .map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(job)).into_response())
}

fn read_submission<'a, T: serde::Serialize>(
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

fn replay(store: &Store, submission: Option<&Submission<'_>>) -> Result<Option<Response>> {
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
) -> Result<(StatusCode, Json<Conversion>)> {
    let (request, submission) = read_submission(&headers, "conversions", payload)?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(submission) = &submission {
        match store.replay(submission).map_err(ApiError::storage)? {
            Replay::New => {}
            Replay::Matched(id) => {
                let row = store
                    .conversion(&id)
                    .map_err(ApiError::storage)?
                    .ok_or_else(|| ApiError::storage("idempotent conversion is missing"))?;
                return Ok((StatusCode::OK, Json(row)));
            }
            Replay::Conflict => {
                return Err(ApiError(
                    StatusCode::CONFLICT,
                    "Idempotency-Key already used for a different request".into(),
                ));
            }
        }
    }
    if !(1..=5).contains(&request.input_ids.len()) {
        return Err(ApiError::bad(
            "conversion requires one source and up to four auxiliary inputs",
        ));
    }
    if let Some(host) = request.host.as_deref()
        && host != "auto"
        && !app
            .effective_hosts()
            .map_err(ApiError::storage)?
            .iter()
            .any(|entry| entry.id == host)
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
    let active = store
        .by_state(crate::domain::State::Running)
        .map_err(ApiError::storage)?;
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
    let hosts = app.effective_hosts().map_err(ApiError::storage)?;
    let (row, job, envelope) =
        Conversion::start(request, inputs, &hosts, &active, profile.as_ref())
            .map_err(ApiError::bad)?
            .ok_or_else(|| {
                ApiError(
                    StatusCode::CONFLICT,
                    "no eligible worker supports all conversion stages and requested options"
                        .into(),
                )
            })?;
    store
        .create_conversion(&row, &job, &envelope, submission.as_ref())
        .map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(row)))
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

fn read<T>(payload: std::result::Result<Json<T>, JsonRejection>) -> Result<T> {
    payload
        .map(|Json(body)| body)
        .map_err(|e| ApiError(e.status(), e.body_text()))
}

struct ApiError(StatusCode, String);
impl ApiError {
    fn bad(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, msg.into())
    }
    fn not_found(object: &str) -> Self {
        Self(StatusCode::NOT_FOUND, format!("{object} not found"))
    }
    fn storage(err: impl std::fmt::Display) -> Self {
        eprintln!("controller storage error: {err}");
        Self(
            StatusCode::INTERNAL_SERVER_ERROR,
            "controller storage error".into(),
        )
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error":self.1}))).into_response()
    }
}
