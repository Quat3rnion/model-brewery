use super::*;
use crate::domain::NewJob;
use axum::{
    body::{Body, to_bytes},
    http::Request,
    response::Response,
};
use std::time::Duration;
use tokio::{sync::Semaphore, time::Instant};
use tower::ServiceExt;

static SLOT: Semaphore = Semaphore::const_new(1);

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RequestBody {
    model_id: String,
    model_sha256: String,
    client_id: Option<String>,
}

pub(super) async fn start(
    State(app): State<Arc<App>>,
    caller: Option<Extension<Caller>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<RequestBody>,
) -> Mutation {
    let request = read(payload)?;
    let actor = Agent::resolve(caller, request.client_id.as_deref())?;
    let key = submission(&headers, "findings-replay", &(&actor, &id, &request))?;
    let (original, draft, slot) = {
        let mut store = app.store.lock().map_err(ApiError::storage)?;
        if let Some(row) = prior(&store, &key)? {
            return Ok((StatusCode::OK, Json(row)));
        }
        let original = lookup(&store, &id)?;
        let mut row = original.next(provenance(actor))?;
        let target = bind(&store, request.model_id, request.model_sha256)?;
        if target.id == original.summary.model.id || original.experiments.is_empty() {
            return Err(ApiError::bad(
                "use another model revision and a finding with evidence",
            ));
        }
        input_hash(&store, &target)?;
        for experiment in &original.experiments {
            match &experiment.spec {
                Spec::Job { request } if request["kind"] == "trace" => {
                    let options = &request["options"];
                    if [
                        "feature_steering",
                        "intervention_artifact",
                        "activation_patches",
                        "sae",
                    ]
                    .into_iter()
                    .any(|key| options.get(key).is_some())
                    {
                        return Err(conflict(
                            "this trace has a model-bound dependency; record a portable trace to replay",
                        ));
                    }
                }
                Spec::Job { .. } => {}
                Spec::TokenLoss { preparation, .. } => {
                    if !matches!(preparation.edit.as_str(), "prune" | "edits")
                        || preparation.arguments["pruning"]["mode"] == "wanda_mlp"
                    {
                        return Err(conflict(
                            "replay needs a direct edit or a prune without a calibration trace",
                        ));
                    }
                    let scratch = store
                        .scratch(&target.id)
                        .map_err(ApiError::storage)?
                        .filter(|rev| {
                            rev.state == RevisionState::Scratch && !rev.is_compiled_program()
                        })
                        .ok_or_else(|| {
                            conflict("token loss replay needs a private scratch target with a GGUF")
                        })?;
                    if scratch.gguf.is_none() {
                        return Err(conflict("convert the replay target to private GGUF first"));
                    }
                }
                Spec::Crosscoder { .. } => {
                    return Err(conflict(
                        "crosscoder reports are imported; the controller cannot replay their offline training",
                    ));
                }
            }
        }
        let slot = SLOT.try_acquire().map_err(|_| {
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "Another finding replay is active".into(),
            )
        })?;
        row.summary.replay_of = Some(id);
        row.summary.model = target;
        row.summary.state = Status::Running;
        row.summary.verdict = inconclusive("Replay is running.");
        row.experiments.clear();
        row.comparison.clear();
        store.add_finding(&row, &key).map_err(ApiError::storage)?;
        (original, row, slot)
    };
    let mut row = draft.clone();
    tokio::spawn(async move {
        let _slot = slot;
        match run(&app, &original, &mut row).await {
            Ok(()) => row.summary.state = Status::Ready,
            Err(err) => {
                row.summary.state = Status::Failed;
                row.summary.error = Some(err.1);
                row.summary.verdict = inconclusive("Replay did not complete.");
                row.comparison.clear();
            }
        }
        if let Err(err) = app
            .store
            .lock()
            .map_err(|e| e.to_string())
            .and_then(|store| store.finish_finding(&row).map_err(|e| e.to_string()))
        {
            eprintln!(
                "Could not finalize finding replay {}: {err}",
                row.summary.id
            );
        }
    });
    Ok((StatusCode::ACCEPTED, Json(draft)))
}

async fn run(app: &Arc<App>, source: &Finding, row: &mut Finding) -> Result<()> {
    let model = &row.summary.model;
    let mut pin_difference = false;
    for (i, experiment) in source.experiments.iter().enumerate() {
        let key = format!("finding-{}-{i}", row.summary.id);
        let paths: Vec<_> = experiment
            .metrics
            .iter()
            .map(|metric| metric.path.clone())
            .collect();
        let replayed = match &experiment.spec {
            Spec::Job { request } => run_job(app, model, request, &key, &paths).await?,
            Spec::TokenLoss {
                arguments,
                preparation,
                ..
            } => run_loss(app, model, arguments, preparation, &key, &paths).await?,
            Spec::Crosscoder { .. } => unreachable!("checked before replay"),
        };
        pin_difference |= [
            "executable_sha256",
            "manifest_sha256",
            "backend",
            "lanternfish_commit",
            "heldout_sha256",
            "fixture_sha256",
            "runner_sha256",
            "converter_version",
            "corpus_sha256",
        ]
        .iter()
        .any(|key| experiment.artifact.pins[*key] != replayed.artifact.pins[*key]);
        row.comparison
            .extend(
                experiment
                    .metrics
                    .iter()
                    .zip(&replayed.metrics)
                    .map(|(original, replay)| Comparison {
                        experiment: i + 1,
                        path: original.path.clone(),
                        original: original.value.clone(),
                        replay: replay.value.clone(),
                    }),
            );
        row.experiments.push(replayed);
    }
    row.summary.verdict = inconclusive(if pin_difference {
        "Runtime or fixture pins differ. Check provenance before comparing these numbers."
    } else {
        "Replay measured the same paths. Review the comparison before giving a verdict."
    });
    Ok(())
}

async fn run_job(
    app: &Arc<App>,
    model: &Model,
    request: &Value,
    key: &str,
    paths: &[String],
) -> Result<Experiment> {
    let mut spec: NewJob = serde_json::from_value(request.clone()).map_err(ApiError::storage)?;
    let sha = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        input_hash(&store, model)?
    };
    spec.input_ids = vec![model.id.clone()];
    spec.host = None; // This job must run on the replacement model's own worker.
    if matches!(spec.kind.as_str(), "circuit" | "sensitivity") {
        spec.options.insert("model_sha256".into(), json!(sha));
    }
    let mut headers = HeaderMap::new();
    headers.insert("Idempotency-Key", key.parse().map_err(ApiError::storage)?);
    headers.insert("Brewery-Allow-Duplicate", "1".parse().unwrap());
    let job =
        reply(crate::web::add_job(State(app.clone()), headers, Ok(Json(spec))).await?).await?;
    let job_id = identifier(&job)?;
    wait(app, job_id).await?;
    job_evidence(
        &*app.store.lock().map_err(ApiError::storage)?,
        model,
        job_id,
        paths,
    )
}

async fn run_loss(
    app: &Arc<App>,
    model: &Model,
    arguments: &Value,
    preparation: &TokenPrep,
    key: &str,
    paths: &[String],
) -> Result<Experiment> {
    let mut edit = preparation.arguments.clone();
    if preparation.edit == "prune" {
        edit["sha256"] = json!(model.sha256);
    }
    let api = crate::scratch::router().with_state(app.clone());
    let child = post(
        &api,
        &format!("/scratch/{}/{}", model.id, preparation.edit),
        key,
        edit,
    )
    .await?;
    let id = identifier(&child)?;
    let convert = post(
        &api,
        &format!("/scratch/{id}/gguf"),
        &format!("{key}-gguf"),
        json!({"sha256":child["sha256"], "assets_id":preparation.assets_id}),
    )
    .await?;
    wait(app, identifier(&convert)?).await?;
    let check = post(
        &api,
        &format!("/scratch/{id}/checks"),
        &format!("{key}-checks"),
        json!({"sha256":child["sha256"]}),
    )
    .await?;
    wait(app, identifier(&check)?).await?;
    let mut query: crate::diff::LossSlice =
        serde_json::from_value(arguments.clone()).map_err(ApiError::storage)?;
    query.a.clone_from(&model.id);
    query.b = id.into();
    token_loss(app, model, query, paths).await
}

async fn post(api: &Router, path: &str, key: &str, body: Value) -> Result<Value> {
    let request = Request::post(path)
        .header("Content-Type", "application/json")
        .header("Idempotency-Key", key)
        .body(Body::from(
            serde_json::to_vec(&body).map_err(ApiError::storage)?,
        ))
        .map_err(ApiError::storage)?;
    reply(
        api.clone()
            .oneshot(request)
            .await
            .expect("infallible router"),
    )
    .await
}

async fn reply(response: Response) -> Result<Value> {
    let code = response.status();
    let bytes = to_bytes(response.into_body(), 256 * 1024)
        .await
        .map_err(ApiError::storage)?;
    let body: Value = serde_json::from_slice(&bytes).map_err(ApiError::storage)?;
    if !code.is_success() {
        return Err(ApiError(
            code,
            body["error"]
                .as_str()
                .unwrap_or("Replay operation failed")
                .into(),
        ));
    }
    Ok(body)
}

fn identifier(reply: &Value) -> Result<&str> {
    reply["id"]
        .as_str()
        .ok_or_else(|| ApiError::storage("replay response has no ID"))
}

async fn wait(app: &Arc<App>, id: &str) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(1200);
    loop {
        let job = app
            .store
            .lock()
            .map_err(ApiError::storage)?
            .job(id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("replay job"))?;
        match job.state {
            JobState::Succeeded => return Ok(()),
            JobState::Failed | JobState::Cancelled => {
                return Err(conflict(format!(
                    "Replay job {id} did not succeed: {}",
                    job.error.as_deref().unwrap_or(&job.progress)
                )));
            }
            _ if Instant::now() >= deadline => {
                return Err(conflict(format!("Replay job {id} did not finish in time")));
            }
            _ => tokio::time::sleep(Duration::from_millis(250)).await,
        }
    }
}
