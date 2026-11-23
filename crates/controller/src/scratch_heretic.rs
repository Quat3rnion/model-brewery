use super::*;
use crate::{
    domain::{Job, Model},
    web::{read_submission, replay},
};
use axum::response::Response;
use brewery_worker::heretic::{self, Invocation, Options};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Run {
    parent_id: String,
    base_sha256: String,
    profile: String,
    trials: u8,
    trial_index: u8,
    seed: u32,
    assets_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Reproduce {
    parent_id: String,
    base_sha256: String,
    manifest: Value,
    assets_id: Option<String>,
}

pub(super) async fn run(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    payload: Payload<Run>,
) -> Result<Response> {
    let body = read(payload)?;
    let invocation = Invocation {
        profile: body.profile.clone(),
        trials: body.trials,
        trial_index: body.trial_index,
        seed: body.seed,
    };
    invocation
        .validate()
        .map_err(|e| ApiError::bad(e.to_string()))?;
    let (_, submission) = read_submission(&headers, "scratch-heretic", Ok(Json(&body)))?;
    submit(
        &app,
        &body.parent_id,
        &body.base_sha256,
        body.assets_id.as_deref(),
        json!(invocation),
        None,
        submission.as_ref(),
    )
}

pub(super) async fn reproduce(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    payload: Payload<Reproduce>,
) -> Result<Response> {
    let body = read(payload)?;
    let (_, submission) = read_submission(&headers, "scratch-heretic-reproduce", Ok(Json(&body)))?;
    let manifest = &body.manifest;
    submit(
        &app,
        &body.parent_id,
        &body.base_sha256,
        body.assets_id.as_deref(),
        Value::Null,
        Some(manifest.clone()),
        submission.as_ref(),
    )
}

fn hf_commit(store: &Store, parent: &Model, base: &str, manifest: &Value) -> Result<String> {
    let Some(job_id) = parent.job_id.as_deref() else {
        return Err(conflict(
            "community reproduction requires a verified Hugging Face base download",
        ));
    };
    let receipt = store
        .job_result(job_id)
        .map_err(ApiError::storage)?
        .unwrap_or_default();
    let artifact = &receipt["artifact"];
    let commit = receipt["commit"]
        .as_str()
        .ok_or_else(|| conflict("missing verified Hugging Face base commit"))?;
    if receipt["kind"] != "hf_download"
        || receipt["verified"] != true
        || receipt["library_model_id"] != parent.id
        || receipt["repo_id"] != manifest["settings"]["model"]
        || manifest["settings"]["model_commit"] != commit
        || artifact["path"] != parent.path
        || artifact["kind"] != "safetensors"
        || artifact["sha256"] != base
        || parent.sha256.as_deref() != Some(base)
    {
        return Err(conflict(
            "community manifest and Library base commit or SafeTensors SHA-256 do not match",
        ));
    }
    Ok(commit.to_owned())
}

fn submit(
    app: &App,
    parent_id: &str,
    base_sha256: &str,
    assets_id: Option<&str>,
    invocation: Value,
    manifest: Option<Value>,
    submission: Option<&Submission<'_>>,
) -> Result<Response> {
    if !brewery_worker::is_digest(base_sha256) {
        return Err(ApiError::bad(
            "expected a lowercase base SafeTensors SHA-256",
        ));
    }
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, submission)? {
        return Ok(response);
    }
    let parent = store
        .model(parent_id)
        .map_err(ApiError::storage)?
        .filter(|m| m.kind == "safetensors")
        .ok_or_else(|| ApiError::bad("select one standalone Library SafeTensors revision"))?;
    let lane = admit(app, &parent.host)?;
    let runs = lane
        .heretic_runs()
        .filter(|_| lane.scratch_gpu_allowed())
        .ok_or_else(|| {
            conflict(
                "configure the pinned Heretic runtime and a GPU queue wrapper on the local worker",
            )
        })?;
    let path = FsPath::new(&parent.path);
    require_source(app, &parent.host, path)?;
    if path.canonicalize().ok().as_deref() != Some(path)
        || path.extension().is_none_or(|ext| ext != "safetensors")
        || parent
            .sha256
            .as_deref()
            .is_some_and(|sha| sha != base_sha256)
    {
        return Err(conflict(
            "use the current SHA-256 of a canonical standalone Library SafeTensors file",
        ));
    }
    if store.scratch_count().map_err(ApiError::storage)? >= LIMIT {
        return Err(conflict("scratch revision limit reached"));
    }
    let bytes = open_regular(path)
        .and_then(|f| Ok(f.metadata()?.len()))
        .map_err(ApiError::storage)?;
    let scratch_root = store.scratch_root();
    private_root(scratch_root)?;
    let reserve = lane
        .scratch_disk_reserve()
        .saturating_add(scratch::MERGE_DELTA_OVERHEAD);
    if bytes > scratch::MAX_FILE
        || brewery_worker::available(scratch_root).map_err(ApiError::storage)?
            < bytes.saturating_mul(2).saturating_add(reserve)
    {
        return Err(conflict(
            "not enough disk space for the Heretic scratch child and delta",
        ));
    }
    let commit = manifest
        .as_ref()
        .map(|manifest| hf_commit(&store, &parent, base_sha256, manifest))
        .transpose()?;
    let kind = if manifest.is_some() {
        heretic::REPRODUCE
    } else {
        heretic::RUN
    };
    let options = json!(Options {
        kind: kind.to_owned(),
        base_sha256: base_sha256.into(),
        base_hf_commit: commit,
        invocation,
        manifest
    });
    Options::parse(kind, &options).map_err(|e| ApiError::bad(e.to_string()))?;
    let pins = BTreeMap::from([(parent.path.clone(), base_sha256.to_owned())]);
    let mut inputs = vec![parent];
    if let Some(id) = assets_id {
        let assets = store
            .model(id)
            .map_err(ApiError::storage)?
            .filter(|assets| assets.host == inputs[0].host && assets.kind == "assets")
            .filter(|assets| {
                let path = FsPath::new(&assets.path);
                path.canonicalize().is_ok_and(|p| p == path && p.is_dir())
                    && lane.configured.iter().any(|root| path.starts_with(root))
            })
            .ok_or_else(|| {
                ApiError::bad("use a Library assets directory on the same local worker")
            })?;
        inputs.push(assets);
    }
    let (job, mut envelope) = jobs::private_job(kind, inputs, options);
    envelope.scratch_pins = pins;
    envelope.heretic_runs = Some(runs);
    jobs::enqueue(
        app,
        &mut store,
        job,
        envelope,
        submission,
        StatusCode::ACCEPTED,
    )
}

pub(crate) fn finish(
    store: &Store,
    job: &Job,
    result: &Value,
) -> std::result::Result<(Revision, Value, Uncommitted), crate::Error> {
    let envelope = store.envelope(&job.id)?;
    let options = Options::parse(&job.kind, &Value::Object(envelope.options))?;
    let base = envelope
        .inputs
        .first()
        .ok_or("missing Heretic Library base")?;
    let current = store.model(&base.id)?.filter(|m| {
        m.kind == base.kind
            && m.host == base.host
            && m.path == base.path
            && m.sha256
                .as_deref()
                .is_none_or(|sha| sha == options.base_sha256)
    });
    if envelope.scratch_pins.get(&base.path) != Some(&options.base_sha256) || job.host != base.host
    {
        return Err("Heretic Library base changed during the job".into());
    }
    let current = current.ok_or("Heretic Library base changed during the job")?;
    if let Some(manifest) = &options.manifest {
        hf_commit(store, &current, &options.base_sha256, manifest).map_err(|e| e.1)?;
    }
    let runs = envelope
        .heretic_runs
        .as_deref()
        .ok_or("missing private Heretic runs root")?;
    let output = options.verify(result, Some(runs))?;
    let output_meta = fs::symlink_metadata(&output)?;
    if !output_meta.is_dir()
        || output_meta.uid() != unsafe { libc::geteuid() }
        || output_meta.mode() & 0o7777 != 0o700
        || output.canonicalize()? != output
    {
        return Err("Heretic run output is not a canonical private directory".into());
    }
    if store.scratch_count()? >= LIMIT || store.scratch(&job.id)?.is_some() {
        return Err("scratch revision limit reached or Heretic job already imported".into());
    }
    let root = store.scratch_root();
    remove(root, &job.id).map_err(|e| e.1)?;
    let guard = Uncommitted::new(root.join(&job.id)).map_err(|e| e.1)?;
    let built = scratch::import(
        FsPath::new(&base.path),
        &options.base_sha256,
        &output.join(scratch::MODEL),
        result["export_sha256"].as_str().unwrap(),
        &guard.path,
    )?;
    if built.delta.changed_tensors == 0 {
        return Err("Heretic export has no changed weights".into());
    }
    let journal = &result["journal"];
    let provenance = Provenance {
        ablation: None,
        schema: "brewery.scratch-provenance.v1".into(),
        parent_id: base.id.clone(),
        parent_sha256: options.base_sha256,
        revision_id: job.id.clone(),
        revision_sha256: built.sha256.clone(),
        worker: job.host.clone(),
        method: job.kind.clone(),
        operations: Vec::new(),
        merge: None,
        pruning: None,
        pruning_trace: None,
        tensors: built.tensors,
        tool: "Heretic (separate AGPL-3.0-or-later program)".into(),
        tool_version: format!("2.1.0+{}", heretic::COMMIT),
        runtime_version: journal["runtime"]["environment_sha256"]
            .as_str()
            .unwrap()
            .into(),
        heretic: Some(journal.clone()),
        locate_edit: None,
        created_at_ms: now(),
        compiled_program: None,
    };
    brewery_worker::atomic(&guard.path.join(PROVENANCE), &provenance)?;
    File::open(root)?.sync_all()?;
    let result = json!({"scratch_id":job.id,"sha256":built.sha256,"method":job.kind,
                        "qualification":"not release-qualified"});
    let child = Revision {
        id: job.id.clone(),
        kind: "scratch".into(),
        state: RevisionState::Scratch,
        sha256: built.sha256,
        qualification: "not release-qualified".into(),
        library_model_id: None,
        provenance,
        delta: built.delta,
        check_bundle: None,
        gguf: None,
    };
    Ok((child, result, guard))
}
