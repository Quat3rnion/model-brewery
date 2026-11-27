use crate::{
    App,
    diff::trace_read,
    domain::{Job, State as JobState},
    web::ApiError,
};
use axum::{
    Json,
    extract::{Path, Query, State, rejection::QueryRejection},
    http::StatusCode,
};
use brewery_worker::{
    is_digest,
    trace::{self, Slice, View},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

type Result<T> = std::result::Result<T, ApiError>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Selection {
    #[serde(default)]
    prompt: usize,
    layer: usize,
    start: usize,
    end: usize,
    ablation_job_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Evidence {
    job_id: String,
    #[serde(default)]
    prompt: usize,
    layer: usize,
    start: usize,
    end: usize,
}

fn invalid(err: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::UNPROCESSABLE_ENTITY, err.to_string())
}

pub(crate) async fn get(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    query: std::result::Result<Query<Selection>, QueryRejection>,
) -> Result<Json<Value>> {
    let Query(q) = query.map_err(|_| ApiError::bad("invalid routing atlas query"))?;
    Ok(Json(report(&app, &id, q).await?))
}

pub(crate) async fn evidence(
    app: &App,
    request: Evidence,
    model: &str,
    digest: &str,
) -> Result<Value> {
    let atlas = report(
        app,
        &request.job_id,
        Selection {
            prompt: request.prompt,
            layer: request.layer,
            start: request.start,
            end: request.end,
            ablation_job_id: None,
        },
    )
    .await?;
    if atlas["model_id"] != model || atlas["model_sha256"] != digest {
        return Err(invalid(
            "routing evidence does not match the REAP source model and SHA-256",
        ));
    }
    // Keep the exact REAP map and its hash unchanged. This empirical observation
    // is supplemental; top-k routing is not an estimate of REAP saliency.
    Ok(json!({
        "job_id":atlas["job_id"],"manifest_sha256":atlas["manifest_sha256"],
        "model_sha256":atlas["model_sha256"],"prompt":request.prompt,"layer":request.layer,
        "start":request.start,"end":request.end,"positions":atlas["positions"],
        "mean_entropy_nats":atlas["mean_entropy_nats"],
        "experts":atlas["experts"].as_array().into_iter().flatten().map(|e|
            json!({"expert":e["expert"],"selected":e["selected"],"load":e["load"]})).collect::<Vec<_>>(),
        "method":"observed routing; not REAP saliency"
    }))
}

async fn report(app: &App, id: &str, q: Selection) -> Result<Value> {
    let slice = Slice {
        view: View::RoutingInput,
        prompt: q.prompt,
        layer: Some(q.layer),
        role: None,
        start: Some(q.start),
        end: Some(q.end),
    };
    slice.check().map_err(|e| ApiError::bad(e.to_string()))?;
    if q.ablation_job_id.as_deref() == Some(id) {
        return Err(ApiError::bad("select a separate expert ablation job"));
    }
    let permit = app.diffs.acquire()?;
    let (base, edit) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let job = |id| -> Result<Job> {
            store
                .job(id)
                .map_err(ApiError::storage)?
                .filter(|j| j.kind == trace::JOB && j.state == JobState::Succeeded)
                .ok_or_else(|| ApiError::not_found("trace result"))
        };
        let (base, edit) = (job(id)?, q.ablation_job_id.as_deref().map(job).transpose()?);
        if let Some(edit) = &edit {
            let prompt = |id| -> Result<Option<String>> {
                let env = store.envelope(id).map_err(ApiError::storage)?;
                Ok(if q.prompt == 0 && env.prompt.is_some() {
                    env.prompt
                } else {
                    env.options["prompts"]
                        .get(q.prompt)
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
            };
            let text = prompt(base.id.as_str())?;
            if text.is_none() || text != prompt(edit.id.as_str())? {
                return Err(invalid("paired trace prompt text differs"));
            }
        }
        (base, edit)
    };
    let (a, b) = tokio::try_join!(trace_read(app, &base, &slice), async {
        match &edit {
            Some(job) => trace_read(app, job, &slice).await.map(Some),
            None => Ok(None),
        }
    })?;
    for data in std::iter::once(&a).chain(&b) {
        if data["lanternfish_commit"] != trace::ROUTING_COMMIT
            || !data["manifest_sha256"].as_str().is_some_and(is_digest)
            || !data["model_sha256"].as_str().is_some_and(is_digest)
            || data["model_id"].as_str().is_none_or(str::is_empty)
            || data["model_sha256"] != a["model_sha256"]
        {
            return Err(invalid(
                "routing trace provenance or paired source digest does not match",
            ));
        }
    }
    if a["model_id"].as_str() != base.input_ids.first().map(String::as_str)
        || b.as_ref().zip(edit.as_ref()).is_some_and(|(data, job)| {
            data["model_id"].as_str() != job.input_ids.first().map(String::as_str)
        })
    {
        return Err(invalid(
            "routing receipt model ID does not match the job input",
        ));
    }
    let meta = json!({
        "job_id":base.id,"model_id":a["model_id"],"model_sha256":a["model_sha256"],
        "manifest_sha256":a["manifest_sha256"],"prompt":q.prompt,"start":q.start,"end":q.end,
        "qualification":"not release-qualified",
        "lanternfish_commit":trace::ROUTING_COMMIT,
        "ablation_job_id":q.ablation_job_id,"ablation_manifest_sha256":b.as_ref().map(|v| &v["manifest_sha256"])
    });
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut result = trace::routing::summarize(a, b, q.layer).map_err(invalid)?;
        result
            .as_object_mut()
            .unwrap()
            .extend(meta.as_object().unwrap().clone());
        Ok(result)
    })
    .await
    .map_err(ApiError::storage)?
}
