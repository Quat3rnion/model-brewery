//! An immutable direction artifact is stored atomically as its job result.
use crate::{App, domain::State as JobState, web::ApiError};
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
    if version != 1 {
        return Err(ApiError::not_found("direction version"));
    }
    let store = app.store.lock().map_err(ApiError::storage)?;
    let job = store
        .job(&id)
        .map_err(ApiError::storage)?
        .filter(|job| job.kind == direction::JOB && job.state == JobState::Succeeded)
        .ok_or_else(|| ApiError::not_found("direction"))?;
    let mut result = store
        .job_result(&id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("direction result"))?;
    let request = direction::Request::parse(&json!(job.options)).map_err(ApiError::storage)?;
    let envelope = store.envelope(&id).map_err(ApiError::storage)?;
    let input = envelope
        .inputs
        .first()
        .ok_or_else(|| ApiError::storage("Direction input is missing"))?;
    request
        .verify_result(&result, &input.id, &request.model_sha256)
        .map_err(ApiError::storage)?;
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
