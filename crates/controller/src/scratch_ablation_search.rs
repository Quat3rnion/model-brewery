//! Submit an HGN forward screen or coordinate the GGUF screen and scratch checks.
use super::*;
use crate::{
    domain::{Job, State as JobState},
    web::{read_submission, replay},
};
use axum::{
    body::to_bytes,
    response::{IntoResponse, Response},
};
use brewery_worker::{
    ablation_search::{self as search, ScreenReport, ScreenRequest},
    behavior_direction::{self as directions, Behavior},
    scratch_eval::{
        BehaviorConfig,
        paired::{PairedOutcome, discordant, holm_adjust, mcnemar_exact_p, wilson95},
    },
};
use serde_json::{Value, json};
use std::time::Duration;

const SCOPE: &str = "scratch-search-ablation";
const MIN_BUDGET: u64 = 64 << 20;
const MAX_BUDGET: u64 = 64 << 30;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    sha256: String,
    direction_id: String,
    assets_id: String,
    seed: u32,
    trial_budget: usize,
    max_children: usize,
    disk_budget_bytes: u64,
    behavior_direction: BehaviorConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    alphas: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_k_direction_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HgnRequest {
    sha256: String,
    direction_id: String,
    source_layer: usize,
    seed: u32,
    first_layer: Option<usize>,
    last_layer: Option<usize>,
    targets: Vec<brewery_worker::trace::hgn_projection::Target>,
    #[serde(skip_serializing_if = "Option::is_none")]
    alphas: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    second_source_layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_k_direction_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum SubmissionRequest {
    Gguf(Request),
    Hgn(HgnRequest),
}

pub(super) async fn submit(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<SubmissionRequest>,
) -> Result<Response> {
    let req = match read(payload)? {
        SubmissionRequest::Gguf(req) => req,
        SubmissionRequest::Hgn(req) => return submit_hgn(&app, id, headers, req),
    };
    let (_, key) = read_submission(&headers, SCOPE, Ok(Json((&id, &req))))?;
    req.behavior_direction
        .validate()
        .map_err(|e| ApiError::bad(e.to_string()))?;
    if !brewery_worker::is_digest(&req.sha256)
        || !(search::MIN_TRIALS..=search::MAX_TRIALS).contains(&req.trial_budget)
        || !(1..=search::MAX_CHILDREN.min(req.trial_budget)).contains(&req.max_children)
        || !(MIN_BUDGET..=MAX_BUDGET).contains(&req.disk_budget_bytes)
    {
        return Err(ApiError::bad(
            "invalid parent hash, trial limit, child limit or disk budget",
        ));
    }
    let store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, key.as_ref())? {
        return Ok(response);
    }
    let row = visible(&store, &id)?;
    let (gguf, _) = jobs::trace_input(&store, &id)?;
    if row.state != RevisionState::Scratch || row.sha256 != req.sha256 {
        return Err(conflict(
            "search the current hash of an unrequested scratch revision",
        ));
    }
    admit(&app, &row.provenance.worker)?;
    let assets = store
        .model(&req.assets_id)
        .map_err(ApiError::storage)?
        .filter(|m| m.kind == "assets" && m.host == gguf.host)
        .ok_or_else(|| {
            ApiError::bad("use Library conversion assets on the local scratch worker")
        })?;
    let gguf_record = row.gguf.as_ref().unwrap();
    let conversion = store
        .envelope(&gguf_record.job_id)
        .map_err(ApiError::storage)?;
    if !conversion
        .inputs
        .get(1)
        .is_some_and(|m| m.id == assets.id && m.path == assets.path)
    {
        return Err(ApiError::bad(
            "use the same Library assets as the parent GGUF conversion",
        ));
    }
    screen_request(&store, &req, &row)?;
    budget(
        &app,
        &store,
        &row,
        req.max_children,
        req.disk_budget_bytes,
        0,
    )?;
    let (job, envelope) = jobs::private_job(search::JOB, vec![gguf, assets], json!(req));
    let tx = store.submission_transaction().map_err(ApiError::storage)?;
    Store::enqueue(&tx, &job, &envelope, key.as_ref()).map_err(ApiError::storage)?;
    tx.commit().map_err(ApiError::storage)?;
    Ok((StatusCode::ACCEPTED, Json(job)).into_response())
}

fn submit_hgn(app: &App, id: String, headers: HeaderMap, req: HgnRequest) -> Result<Response> {
    let (_, key) = read_submission(&headers, "hgn-ablation-screen", Ok(Json((&id, &req))))?;
    if !brewery_worker::is_digest(&req.sha256) {
        return Err(ApiError::bad("invalid HGN checkpoint SHA-256"));
    }
    let store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, key.as_ref())? {
        return Ok(response);
    }
    let model = store
        .model(&id)
        .map_err(ApiError::storage)?
        .filter(|model| brewery_worker::trace::is_hgn(&model.kind))
        .ok_or_else(|| ApiError::not_found("Library HGN checkpoint"))?;
    let dir = crate::directions::hgn_layer(&store, &model, &req.direction_id, req.source_layer)?;
    if dir.sha256 != req.sha256 {
        return Err(ApiError::bad(
            "HGN direction does not match the requested checkpoint hash",
        ));
    }
    if req.second_source_layer == Some(req.source_layer) {
        return Err(ApiError::bad(
            "second_source_layer must differ from source_layer",
        ));
    }
    let second = |id: &str| -> Result<Option<Vec<f32>>> {
        Ok(req
            .second_source_layer
            .map(|layer| crate::directions::hgn_layer(&store, &model, id, layer))
            .transpose()?
            .map(|d| d.vector))
    };
    let last_k_variant = if let Some(id) = &req.last_k_direction_id {
        let variant = crate::directions::hgn_layer(&store, &model, id, req.source_layer)?;
        if variant.sha256 != dir.sha256
            || variant.assets != dir.assets
            || variant.behavior != dir.behavior
            || variant.held_out_pairs != dir.held_out_pairs
        {
            return Err(ApiError::bad(
                "last-K fit must match the checkpoint, assets, and reviewed contrast",
            ));
        }
        Some(search::hgn::Variant {
            direction_id: id.clone(),
            last_k: variant.last_k.unwrap_or(0),
            vector: variant.vector,
            second_vector: second(id)?,
        })
    } else {
        None
    };
    let options = json!(search::hgn::Request {
        model_sha256: dir.sha256,
        direction_id: req.direction_id.clone(),
        source_layer: req.source_layer,
        first_layer: req.first_layer,
        last_layer: req.last_layer,
        targets: req.targets.clone(),
        behavior: dir.behavior,
        seed: req.seed,
        vector: dir.vector,
        last_k: dir.last_k,
        alphas: Some(req.alphas.clone().unwrap_or_else(|| vec![1.0, 2.0])),
        second_source_layer: req.second_source_layer,
        second_vector: second(&req.direction_id)?,
        last_k_variant,
        assets_sha256: dir.assets,
        held_out_pairs: dir.held_out_pairs,
    });
    search::hgn::Request::parse(&options).map_err(|e| ApiError::bad(e.to_string()))?;
    let (mut job, envelope) = jobs::private_job(search::hgn::JOB, vec![model], options);
    job.options = json!(req).as_object().unwrap().clone();
    let hosts = app.effective_hosts().map_err(ApiError::storage)?;
    if crate::domain::candidates(&job, &envelope, &hosts).is_empty() {
        return Err(conflict(
            "HGN screening needs trace and behavior_direction on the non-Hy4 Library worker",
        ));
    }
    let tx = store.submission_transaction().map_err(ApiError::storage)?;
    Store::enqueue(&tx, &job, &envelope, key.as_ref()).map_err(ApiError::storage)?;
    tx.commit().map_err(ApiError::storage)?;
    Ok((StatusCode::ACCEPTED, Json(job)).into_response())
}

fn screen_request(store: &Store, req: &Request, row: &Revision) -> Result<ScreenRequest> {
    let artifact = crate::directions::load(store, &req.direction_id, directions::VERSION)?;
    let gguf = row
        .gguf
        .as_ref()
        .ok_or_else(|| conflict("convert the scratch parent to private GGUF"))?;
    if artifact["model_sha256"] != gguf.sha256 || gguf.revision_sha256 != row.sha256 {
        return Err(ApiError::bad(
            "fit the direction on the same GGUF bytes as the private parent",
        ));
    }
    if !artifact["position"].is_null()
        || artifact["last_k"].as_u64().unwrap_or(1) != 1
        || artifact.get("lanternfish_commit").is_none()
    {
        return Err(ApiError::bad("fit the directions at the last user token"));
    }
    let behavior = serde_json::from_value::<Behavior>(artifact["behavior"].clone())
        .map_err(ApiError::storage)?;
    let name = match behavior {
        Behavior::Slop => "slop",
        Behavior::Sycophancy => "sycophancy",
        _ => return Err(ApiError::bad("search supports slop and sycophancy only")),
    };
    let last_k_variant = if let Some(id) = &req.last_k_direction_id {
        let variant = crate::directions::load(store, id, directions::VERSION)?;
        if variant.get("lanternfish_commit").is_none()
            || variant["model_sha256"] != artifact["model_sha256"]
            || variant["behavior"] != artifact["behavior"]
            || !variant["position"].is_null()
            || variant["last_k"].as_u64().unwrap_or(1) < 2
        {
            return Err(ApiError::bad(
                "last-K fit must use the same GGUF bytes and reviewed contrast",
            ));
        }
        Some(search::FitVariant {
            direction_id: id.clone(),
            last_k: variant["last_k"].as_u64().expect("verified last-K fit") as usize,
            fit: serde_json::from_value(variant["fit"].clone()).map_err(ApiError::storage)?,
        })
    } else {
        None
    };
    let screen = ScreenRequest {
        model_sha256: gguf.sha256.clone(),
        direction_id: req.direction_id.clone(),
        fit: serde_json::from_value(artifact["fit"].clone()).map_err(ApiError::storage)?,
        behavior,
        seed: req.seed,
        trial_budget: req.trial_budget,
        max_children: req.max_children,
        confirmation_sha256: req.behavior_direction.prompt_sets[name].clone(),
        alphas: req.alphas.clone(),
        last_k_variant,
    };
    screen.check().map_err(|e| ApiError::bad(e.to_string()))?;
    Ok(screen)
}

fn child_disk(store: &Store, row: &Revision) -> Result<(u64, u64)> {
    let root = store.scratch_root();
    let bytes = open_regular(&root.join(&row.id).join(scratch::MODEL))
        .and_then(|f| Ok(f.metadata()?.len()))
        .map_err(ApiError::storage)?;
    let gguf = row
        .gguf
        .as_ref()
        .ok_or_else(|| conflict("private GGUF required"))?
        .bytes;
    let per_child = bytes
        .saturating_add(gguf)
        .saturating_add(128 << 20)
        .saturating_add(scratch::ablation::MAX_TOTAL_BYTES + scratch::MERGE_DELTA_OVERHEAD)
        .saturating_add(16 << 20);
    Ok((bytes, per_child))
}

fn budget(
    app: &App,
    store: &Store,
    row: &Revision,
    children: usize,
    limit: u64,
    used: u64,
) -> Result<u64> {
    let (bytes, per_child) = child_disk(store, row)?;
    // Existing BF16 exporter: allow its four-times input staging estimate,
    // plus one output's slack, a transient trace and bounded JSON receipts.
    let staging = bytes
        .saturating_mul(4)
        .saturating_add(search::RESERVE + (128 << 20));
    let extra = per_child
        .saturating_mul(children as u64)
        .saturating_add(staging);
    let reserve = admit(app, &row.provenance.worker)?.scratch_disk_reserve();
    if limit < used.saturating_add(extra)
        || brewery_worker::available(store.scratch_root()).map_err(ApiError::storage)?
            < extra.saturating_add(reserve)
    {
        return Err(conflict(
            "ablation search disk budget or available scratch space is too small",
        ));
    }
    Ok(per_child)
}

pub(crate) fn allocate(app: &App) -> std::result::Result<Option<Job>, crate::Error> {
    let mut store = app.store.lock().expect("store lock");
    if let Some(job) = store
        .by_state(JobState::Running)?
        .into_iter()
        .find(|j| j.kind == search::JOB)
    {
        return Ok(Some(job));
    }
    let Some(mut job) = store
        .by_state(JobState::Queued)?
        .into_iter()
        .find(|j| j.kind == search::JOB)
    else {
        return Ok(None);
    };
    job.state = JobState::Running;
    job.progress = "Preparing bounded CPU ablation screen".into();
    store.update(&mut job, None)?;
    Ok(Some(job))
}

pub(crate) async fn drive(app: Arc<App>, job: Job) -> std::result::Result<(), crate::Error> {
    let result = execute(&app, &job).await;
    let mut store = app.store.lock().expect("store lock");
    let mut current = store.job(&job.id)?.ok_or("search job disappeared")?;
    let receipt = match result {
        Ok(receipt) => {
            // A late cancel cannot hide the completed receipt and its
            // retained children after execute's final cancellation check.
            current.state = JobState::Succeeded;
            current.progress = "Ablation search complete; not release-qualified".into();
            current.has_result = true;
            Some(receipt)
        }
        Err(_) if current.cancel_requested => {
            current.state = JobState::Cancelled;
            current.progress = "Ablation search cancelled".into();
            None
        }
        Err(error) => {
            current.state = JobState::Failed;
            current.progress = "Ablation search did not complete".into();
            current.error = Some(error.to_string());
            None
        }
    };
    store.update(&mut current, receipt)
}

async fn response_id(response: Response) -> anyhow::Result<String> {
    let data: Value = serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await?)?;
    Ok(data["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("scratch endpoint did not return a job"))?
        .into())
}

fn api(error: ApiError) -> anyhow::Error {
    anyhow::anyhow!(error.1)
}

fn active(store: &Store, job: &Job) -> anyhow::Result<()> {
    let row = store
        .job(&job.id)
        .map_err(box_error)?
        .ok_or_else(|| anyhow::anyhow!("search job disappeared"))?;
    anyhow::ensure!(!row.cancel_requested, "search cancelled");
    Ok(())
}

fn box_error(error: crate::Error) -> anyhow::Error {
    anyhow::Error::msg(error.to_string())
}

async fn wait(app: &App, parent: &Job, child: &str) -> anyhow::Result<Value> {
    loop {
        {
            let mut store = app.store.lock().expect("store lock");
            let job = store
                .job(child)
                .map_err(box_error)?
                .ok_or_else(|| anyhow::anyhow!("search subjob disappeared"))?;
            if active(&store, parent).is_err()
                && !(job.kind == search::SCREEN && job.state == JobState::Succeeded)
            {
                if !job.state.terminal() {
                    store.cancel(child).map_err(box_error)?;
                } else {
                    anyhow::bail!("search cancelled");
                }
            } else if job.state == JobState::Succeeded {
                return store
                    .job_result(child)
                    .map_err(box_error)?
                    .ok_or_else(|| anyhow::anyhow!("search subjob has no receipt"));
            } else if job.state.terminal() {
                anyhow::bail!("{child}: {}", job.error.as_deref().unwrap_or(&job.progress));
            } else {
                let progress = format!("{}: {}", job.kind, job.progress);
                store
                    .sample_progress(&parent.id, Some(&progress), job.progress_detail)
                    .map_err(box_error)?;
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn prepare(app: &App, parent: &Job) -> anyhow::Result<(Request, ScreenRequest, String, u64)> {
    let mut store = app.store.lock().expect("store lock");
    let envelope = store.envelope(&parent.id).map_err(box_error)?;
    let req: Request = serde_json::from_value(Value::Object(envelope.options))?;
    let row = visible(&store, &envelope.inputs[0].id).map_err(api)?;
    let screen = screen_request(&store, &req, &row).map_err(api)?;
    let options = serde_json::to_value(&screen)?;
    let key_string = format!("{}-screen", parent.id);
    let key = Submission::new("ablation-search-screen", &key_string, &screen)?;
    let id = match store.replay(&key).map_err(box_error)? {
        Replay::Matched(id) => id,
        Replay::Conflict => anyhow::bail!("search was resumed with a different direction"),
        Replay::New => {
            active(&store, parent)?;
            let (gguf, sha) = jobs::trace_input(&store, &row.id).map_err(api)?;
            let path = gguf.path.clone();
            let (mut job, mut envelope) = jobs::private_job(search::SCREEN, vec![gguf], options);
            envelope.scratch_pins.insert(path, sha);
            job.options.clear();
            let id = job.id.clone();
            jobs::enqueue(
                app,
                &mut store,
                job,
                envelope,
                Some(&key),
                StatusCode::ACCEPTED,
            )
            .map_err(api)?;
            id
        }
    };
    Ok((req, screen, id, row.gguf.as_ref().unwrap().bytes))
}

async fn execute(app: &Arc<App>, parent: &Job) -> anyhow::Result<Value> {
    let (req, spec, screen_job, parent_gguf_bytes) = prepare(app, parent)?;
    let report: ScreenReport = serde_json::from_value(wait(app, parent, &screen_job).await?)?;
    report.verify(&spec)?;
    let mut children = Vec::new();
    let mut checked = Vec::new();
    let mut used = 0_u64;
    let mut failure = None;
    for &trial in &report.selected {
        let next = run_child(app, parent, &req, &report, trial, used, &mut children).await;
        match next {
            Ok((row, bytes)) => {
                checked.push(row);
                used = used.saturating_add(bytes);
            }
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    let cancelled = active(&app.store.lock().expect("store lock"), parent).is_err();
    if failure.is_some() || cancelled {
        // Locate children from a previous controller process too. An edit ID
        // is written atomically with its search-only idempotency key.
        for &trial in &report.selected {
            let edit = edit_request(&req, &report, trial);
            let name = format!("{}-{trial}-edit", parent.id);
            let key = Submission::new(
                "ablation-search-edit",
                &name,
                &(&parent.input_ids[0], &edit),
            )?;
            let store = app.store.lock().expect("store lock");
            if let Replay::Matched(id) = store.replay(&key).map_err(box_error)?
                && !children.contains(&id)
            {
                children.push(id);
            }
        }
        for child in &children {
            discard(app, child).await?;
        }
        anyhow::bail!(
            "{}",
            failure.map_or_else(|| "search cancelled".into(), |e| e.to_string())
        );
    }
    let p_values = checked
        .iter()
        .map(|r| r["exact_mcnemar_p"].as_f64().unwrap())
        .collect::<Vec<_>>();
    for (row, adjusted) in checked.iter_mut().zip(holm_adjust(&p_values)) {
        row["holm_adjusted_p"] = json!(adjusted);
        row["confirmed_improvement"] = json!(
            adjusted < 0.05
                && row["improved"].as_u64() > row["worsened"].as_u64()
                && row["w4_passed"] == true
                && row["parent_behavior_observed"] == true
        );
    }
    let moved = report
        .trials
        .iter()
        .any(|t| t.moves_beyond_control(report.prompts));
    let conclusion = if report.parent_hits == 0 {
        "stock_screen_marker_not_observed"
    } else if !moved {
        "no_screen_setting_outperformed_matched_random_control"
    } else if checked.is_empty() {
        "screen_effect_not_on_pareto_front"
    } else if checked.iter().all(|r| r["confirmed_improvement"] != true) {
        "screen_effect_not_confirmed_by_weight_edit"
    } else {
        "screen_effect_confirmed_by_weight_edit"
    };
    Ok(json!({
        "schema_version":1, "kind":search::JOB, "release_qualified":false,
        "parent_safetensors_sha256":req.sha256, "parent_gguf_bytes":parent_gguf_bytes,
        "disk_budget_bytes":req.disk_budget_bytes, "charged_retained_bytes":used,
        "screen_job_id":screen_job,"backend":"cpu","screen":report,"children":checked,
        "finding":conclusion,
    }))
}

fn edit_request(request: &Request, report: &ScreenReport, trial: usize) -> ablation::Request {
    let candidate = report.trials[trial].candidate;
    let settings = candidate.settings();
    ablation::Request {
        sha256: request.sha256.clone(),
        direction: ablation::Source {
            id: if candidate.last_k.is_some() {
                request
                    .last_k_direction_id
                    .as_ref()
                    .expect("validated last-K trial")
                    .clone()
            } else {
                request.direction_id.clone()
            },
            version: directions::VERSION,
            layer: candidate.source_layer,
        },
        first_layer: settings.first_layer,
        last_layer: settings.last_layer,
        kernel: settings.kernel,
        variant: settings.variant,
        alpha: settings.alpha,
        include_embeddings: false,
    }
}

async fn run_child(
    app: &Arc<App>,
    parent: &Job,
    request: &Request,
    report: &ScreenReport,
    trial: usize,
    used: u64,
    children: &mut Vec<String>,
) -> anyhow::Result<(Value, u64)> {
    let id = parent.input_ids[0].clone();
    let req = edit_request(request, report, trial);
    let parent_job_id = parent.id.clone();
    let layer_key = format!("{}-{trial}-edit", parent.id);
    let app_copy = app.clone();
    let budget_limit = request.disk_budget_bytes;
    let slot = app.scratch_slots.clone().acquire_owned().await?;
    let (child, bytes) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let _slot = slot;
        let key = Submission::new("ablation-search-edit", &layer_key, &(&id, &req))?;
        let bytes = {
            let store = app_copy.store.lock().expect("store lock");
            let root = visible(&store, &id).map_err(api)?;
            match store.replay(&key).map_err(box_error)? {
                Replay::Matched(existing) => {
                    let row = visible(&store, &existing).map_err(api)?;
                    // A resumed search already charged and reserved this child.
                    return Ok((row, child_disk(&store, &root).map_err(api)?.1));
                }
                Replay::Conflict => {
                    anyhow::bail!("search was resumed with different edit parameters")
                }
                Replay::New => {
                    let parent = store
                        .job(&parent_job_id)
                        .map_err(box_error)?
                        .ok_or_else(|| anyhow::anyhow!("search job disappeared"))?;
                    anyhow::ensure!(!parent.cancel_requested, "search cancelled");
                    budget(&app_copy, &store, &root, 1, budget_limit, used).map_err(api)?
                }
            }
        };
        let (_, Json(child)) = ablation::apply(&app_copy, id, req, Some(&key)).map_err(api)?;
        Ok((child, bytes))
    })
    .await??;
    children.push(child.id.clone());
    let conv = jobs::Convert {
        sha256: child.sha256.clone(),
        assets_id: request.assets_id.clone(),
    };
    let c_key = format!("{}-{trial}-convert", parent.id);
    let key = Submission::new("ablation-search-convert", &c_key, &(&child.id, &conv))?;
    let converted = jobs::submit_convert(app, &child.id, &conv, Some(&key)).map_err(api)?;
    let convert_job = response_id(converted).await?;
    let converted = wait(app, parent, &convert_job).await?;
    {
        let store = app.store.lock().expect("store lock");
        let root = visible(&store, &parent.input_ids[0]).map_err(api)?;
        let gguf = root
            .gguf
            .ok_or_else(|| anyhow::anyhow!("parent GGUF missing"))?;
        anyhow::ensure!(
            converted["gguf"]["bytes"]
                .as_u64()
                .is_some_and(|n| n <= gguf.bytes + (128 << 20)),
            "child GGUF exceeded the accounted disk allowance"
        );
    }
    let options = checks::Run {
        sha256: child.sha256.clone(),
        behavior_direction: Some(serde_json::from_value(serde_json::to_value(
            &request.behavior_direction,
        )?)?),
        registered_marker: None,
    };
    let c_key = format!("{}-{trial}-checks", parent.id);
    let app_copy = app.clone();
    let child_id = child.id.clone();
    let slot = app.scratch_slots.clone().acquire_owned().await?;
    let check = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let _slot = slot;
        let key = Submission::new("ablation-search-check", &c_key, &(&child_id, &options))?;
        checks::check(&app_copy, child_id, options, Some(&key)).map_err(api)
    })
    .await??;
    let checks_job = response_id(check).await?;
    wait(app, parent, &checks_job).await?;
    let bundle = {
        let store = app.store.lock().expect("store lock");
        let row = visible(&store, &child.id).map_err(api)?;
        serde_json::to_value(
            row.check_bundle
                .ok_or_else(|| anyhow::anyhow!("missing W4 check bundle"))?,
        )?
    };
    let behavior = match report.behavior {
        Behavior::Slop => "slop",
        _ => "sycophancy",
    };
    let measured = &bundle["behavior_direction"];
    let pairs: Vec<PairedOutcome> =
        serde_json::from_value(measured["paired_outcomes"][behavior].clone())?;
    anyhow::ensure!(pairs.len() == 32, "A2 report needs 32 paired outcomes");
    let p = pairs.iter().filter(|x| x.parent).count();
    let c = pairs.iter().filter(|x| x.child).count();
    let (improved, worsened) = discordant(&pairs);
    Ok((
        json!({
            "trial":trial, "id":child.id,"sha256":child.sha256,
            "gguf_sha256":converted["gguf"]["sha256"],"checks_job_id":checks_job,
            "n":pairs.len(),"parent_positives":p,"child_positives":c,
            "parent_wilson95":wilson95(p, pairs.len()),"child_wilson95":wilson95(c, pairs.len()),
            "improved":improved,"worsened":worsened,"exact_mcnemar_p":mcnemar_exact_p(improved,worsened),
            "w4_passed":(["integrity","smoke","behavior"].into_iter().all(|name| bundle[name]["status"] == "passed")),
            "parent_behavior_observed": p > 0,
            "metrics":bundle["measured_report"]["metrics"],
            "behavior_metrics":measured["metrics"],
        }),
        bytes,
    ))
}

async fn discard(app: &Arc<App>, child: &str) -> anyhow::Result<()> {
    let app_copy = app.clone();
    let child = child.to_string();
    let slot = app.scratch_slots.clone().acquire_owned().await?;
    tokio::task::spawn_blocking(move || {
        let _slot = slot;
        let result = change(
            &app_copy,
            Some(child),
            RevisionState::Discarded,
            None,
            None,
            None,
        );
        match result {
            Ok(_) | Err(ApiError(StatusCode::NOT_FOUND, _)) => {}
            Err(error) => return Err(api(error)),
        }
        Ok(())
    })
    .await?
}
