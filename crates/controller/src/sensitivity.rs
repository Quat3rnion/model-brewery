use crate::{
    App,
    domain::{Envelope, Job, State as JobState},
    web::ApiError,
};
use axum::{
    Json,
    extract::{Path, State},
};
use brewery_worker::sensitivity::{self, Artifact, Request};
use serde_json::Value;
use std::sync::Arc;

pub fn prepare(envelope: &mut Envelope, revision: Option<&str>) -> Result<(), ApiError> {
    let req = Request::parse(&Value::Object(envelope.options.clone()))
        .map_err(|e| ApiError::bad(e.to_string()))?;
    let model = &envelope.inputs[0];
    if let Some(sha) = envelope
        .scratch_pins
        .get(&model.path)
        .or(model.sha256.as_ref())
    {
        req.check_model(sha)
            .map_err(|e| ApiError::bad(e.to_string()))?;
    }
    if let Some(revision) = revision {
        envelope.options.insert(
            sensitivity::REVISION_PIN.into(),
            Value::String(revision.into()),
        );
    }
    Ok(())
}

pub fn check(job: &Job, envelope: &Envelope, receipt: Value) -> Result<Artifact, crate::Error> {
    let (req, revision) = Request::resolved(&Value::Object(envelope.options.clone()))?;
    let artifact: Artifact = serde_json::from_value(receipt)?;
    artifact.check(&req, &envelope.inputs[0].id, &revision)?;
    if job.scratch_job
        && envelope.scratch_pins.get(&envelope.inputs[0].path) != Some(&artifact.model_sha256)
    {
        return Err("sensitivity artifact does not match the private GGUF pin".into());
    }
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
        .is_some_and(|job| job.kind == sensitivity::JOB && job.state == JobState::Succeeded)
    {
        return Err(ApiError::not_found("sensitivity artifact"));
    }
    store
        .job_result(&id)
        .map_err(ApiError::storage)?
        .map(Json)
        .ok_or_else(|| ApiError::not_found("sensitivity artifact"))
}
