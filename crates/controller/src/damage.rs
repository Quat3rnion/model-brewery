use crate::{App, domain::State as JobState, web::ApiError};
use axum::{
    Json,
    extract::{Path, State},
};
use brewery_worker::damage;
use serde_json::Value;
use std::sync::Arc;

pub async fn get(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    if !store
        .job(&id)
        .map_err(ApiError::storage)?
        .is_some_and(|j| j.kind == damage::JOB && j.state == JobState::Succeeded)
    {
        return Err(ApiError::not_found("quantization damage artifact"));
    }
    store
        .job_result(&id)
        .map_err(ApiError::storage)?
        .map(Json)
        .ok_or_else(|| ApiError::not_found("quantization damage artifact"))
}
