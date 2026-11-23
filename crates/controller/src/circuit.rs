use crate::{
    App,
    domain::{Envelope, State as JobState},
    web::ApiError,
};
use axum::{
    Json,
    extract::{Path, State},
};
use brewery_worker::circuit::{self, Artifact, Request};
use serde_json::Value;
use std::sync::Arc;

pub fn check(envelope: &Envelope, receipt: Value) -> Result<Artifact, crate::Error> {
    let request = Request::parse(&Value::Object(envelope.options.clone()))?;
    let artifact: Artifact = serde_json::from_value(receipt)?;
    let [model] = envelope.inputs.as_slice() else {
        return Err("circuit job must have exactly one input".into());
    };
    artifact.check(&request, &model.id)?;
    Ok(artifact)
}

pub async fn get(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    if !store
        .job(&id)
        .map_err(ApiError::storage)?
        .is_some_and(|job| job.kind == circuit::JOB && job.state == JobState::Succeeded)
    {
        return Err(ApiError::not_found("circuit artifact"));
    }
    store
        .job_result(&id)
        .map_err(ApiError::storage)?
        .map(Json)
        .ok_or_else(|| ApiError::not_found("circuit artifact"))
}
