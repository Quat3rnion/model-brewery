//! An immutable direction artifact is stored atomically as its job result.
use crate::{App, domain::State as JobState, store::Store, web::ApiError};
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::get,
};
use brewery_worker::behavior_direction as direction;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

pub fn router() -> Router<Arc<App>> {
    Router::new().route("/directions/{id}/versions/{version}", get(get_version))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    layer: Option<usize>,
}

async fn get_version(
    State(app): State<Arc<App>>,
    Path((id, version)): Path<(String, u32)>,
    Query(selection): Query<Selection>,
) -> Result<Json<Value>, ApiError> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    let mut result = load(&store, &id, version)?;
    if let Some(layer) = selection.layer {
        let row = result["fit"]["layers"]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["layer"] == layer))
            .ok_or_else(|| ApiError::not_found("direction layer"))?;
        result["fit"]["layers"] = json!([row]);
    }
    Ok(Json(
        json!({"id":id, "version":version, "direction":result}),
    ))
}

pub(crate) fn load(store: &Store, id: &str, version: u32) -> Result<Value, ApiError> {
    if !matches!(version, 1 | direction::VERSION) {
        return Err(ApiError::not_found("direction version"));
    }
    let job = store
        .job(id)
        .map_err(ApiError::storage)?
        .filter(|job| job.kind == direction::JOB && job.state == JobState::Succeeded)
        .ok_or_else(|| ApiError::not_found("direction"))?;
    let result = store
        .job_result(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("direction result"))?;
    if result["schema_version"] != version {
        return Err(ApiError::not_found("direction version"));
    }
    let request = direction::Request::parse(&json!(job.options)).map_err(ApiError::storage)?;
    let envelope = store.envelope(id).map_err(ApiError::storage)?;
    let input = envelope
        .inputs
        .first()
        .ok_or_else(|| ApiError::storage("Direction input is missing"))?;
    request
        .verify_version(&result, &input.id, &request.model_sha256, version)
        .map_err(ApiError::storage)?;
    Ok(result)
}
