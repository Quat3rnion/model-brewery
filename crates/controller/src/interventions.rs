use crate::{
    App,
    domain::{id, now},
    idempotency::{Replay, Submission},
    store::Store,
    web::{ApiError, read_submission},
};
use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header},
    routing::get,
};
use brewery_worker::{is_digest, trace::Rule};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

type Result<T> = std::result::Result<T, ApiError>;
type Payload<T> = std::result::Result<Json<T>, JsonRejection>;
type Mutation = Result<(StatusCode, Json<Artifact>)>;

mod control_vector;
mod trace_plan;
pub use trace_plan::TracePlan;

#[derive(Serialize, Deserialize)]
pub struct Binding {
    pub id: String,
    pub name: String,
    pub model_id: String,
    pub model_sha256: String,
}

#[derive(Serialize, Deserialize)]
pub struct Summary {
    #[serde(flatten)]
    pub binding: Binding,
    pub latest_version: u32,
}

#[derive(Serialize, Deserialize)]
pub struct Artifact {
    #[serde(flatten)]
    pub binding: Binding,
    pub version: u32,
    pub created_at_ms: u64,
    pub model_hint: String,
    pub interventions: Vec<Rule>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Create {
    name: String,
    model_id: String,
    model_sha256: String,
    interventions: Vec<Rule>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NewVersion {
    interventions: Vec<Rule>,
}

pub fn router() -> Router<Arc<App>> {
    Router::new()
        .route("/interventions", get(list).post(create))
        .route(
            "/interventions/{id}/versions",
            axum::routing::post(new_version),
        )
        .route("/interventions/{id}/versions/{version}", get(get_version))
        .route("/interventions/{id}/versions/{version}/gguf", get(export))
}

async fn list(State(app): State<Arc<App>>) -> Result<Json<Vec<Summary>>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(store.interventions().map_err(ApiError::storage)?))
}

async fn get_version(
    State(app): State<Arc<App>>,
    Path((id, version)): Path<(String, u32)>,
) -> Result<Json<Artifact>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(artifact(&store, &id, version)?))
}

pub fn artifact(store: &Store, id: &str, version: u32) -> Result<Artifact> {
    store
        .intervention(id, version)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("intervention artifact version"))
}

async fn export(
    State(app): State<Arc<App>>,
    Path((id, version)): Path<(String, u32)>,
) -> Result<([(header::HeaderName, &'static str); 2], Vec<u8>)> {
    let artifact = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        artifact(&store, &id, version)?
    };
    Ok((
        [
            (header::CONTENT_TYPE, "application/octet-stream"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"control-vector.gguf\"",
            ),
        ],
        control_vector::encode(&artifact)?,
    ))
}

async fn create(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    payload: Payload<Create>,
) -> Mutation {
    let (body, submission) = read_submission(&headers, "interventions-create", payload)?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(prior) = replay(&store, submission.as_ref())? {
        return Ok((StatusCode::OK, Json(prior)));
    }
    let name = body.name.trim();
    if name.is_empty() || name.len() > 160 || name.chars().any(char::is_control) {
        return Err(ApiError::bad("Invalid intervention artifact name"));
    }
    if !is_digest(&body.model_sha256) {
        return Err(ApiError::bad("Expected a lowercase model SHA-256"));
    }
    let model = store
        .model(&body.model_id)
        .map_err(ApiError::storage)?
        .filter(|model| model.kind == "gguf");
    let name_hint = match model {
        Some(model) => model.name,
        None => {
            let (model, sha) = crate::scratch::jobs::trace_input(&store, &body.model_id)
                .map_err(|_| ApiError::bad("Intervention artifacts need a Library or private scratch GGUF"))?;
            if sha != body.model_sha256 {
                return Err(ApiError::bad("Intervention artifact must match the private GGUF hash"));
            }
            model.name
        }
    };
    save(
        &mut store,
        Binding {
            id: id(),
            name: name.into(),
            model_id: body.model_id,
            model_sha256: body.model_sha256,
        },
        1,
        name_hint,
        body.interventions,
        submission.as_ref(),
    )
}

async fn new_version(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<NewVersion>,
) -> Mutation {
    let body = crate::web::read(payload)?;
    let (_, submission) =
        read_submission(&headers, "interventions-version", Ok(Json((&id, &body))))?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(prior) = replay(&store, submission.as_ref())? {
        return Ok((StatusCode::OK, Json(prior)));
    }
    let latest = store
        .latest_intervention(&id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("intervention artifact"))?;
    let version = latest
        .version
        .checked_add(1)
        .ok_or_else(|| ApiError::storage("intervention version overflow"))?;
    save(
        &mut store,
        latest.binding,
        version,
        latest.model_hint,
        body.interventions,
        submission.as_ref(),
    )
}

fn save(
    store: &mut Store,
    binding: Binding,
    version: u32,
    model_hint: String,
    interventions: Vec<Rule>,
    submission: Option<&Submission<'_>>,
) -> Mutation {
    if interventions.is_empty() {
        return Err(ApiError::bad(
            "An intervention artifact needs at least one rule",
        ));
    }
    Rule::check_all(&interventions).map_err(|e| ApiError::bad(e.to_string()))?;
    let value = Artifact {
        binding,
        version,
        created_at_ms: now(),
        model_hint,
        interventions,
    };
    store
        .insert_intervention(&value, submission)
        .map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(value)))
}

fn replay(store: &Store, submission: Option<&Submission<'_>>) -> Result<Option<Artifact>> {
    let Some(submission) = submission else {
        return Ok(None);
    };
    match store.replay(submission).map_err(ApiError::storage)? {
        Replay::New => Ok(None),
        Replay::Matched(key) => {
            let (id, version) = key
                .split_once('/')
                .and_then(|(id, version)| Some((id, version.parse().ok()?)))
                .ok_or_else(|| ApiError::storage("invalid intervention version replay"))?;
            artifact(store, id, version).map(Some)
        }
        Replay::Conflict => Err(ApiError(
            StatusCode::CONFLICT,
            "Idempotency-Key already used for a different request".into(),
        )),
    }
}
