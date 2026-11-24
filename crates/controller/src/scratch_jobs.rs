use super::*;
use crate::{
    domain::{Envelope, Job, Model, NewJob},
    web::{read_submission, replay},
};
use axum::response::{IntoResponse, Response};
use brewery_worker::locate_edit;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) use brewery_worker::scratch_eval::{CHECKS as CHECK, GGUF as CONVERT};
pub(super) const GGUF: &str = "model.gguf";
const OUTPUT: &str = "converted.gguf";

#[derive(Deserialize, Serialize)]
pub(crate) struct PrivateGguf {
    pub(crate) revision_sha256: String,
    pub(crate) sha256: String,
    pub(crate) converter_version: String,
    pub(super) bytes: u64,
    pub(crate) job_id: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Convert {
    pub(super) sha256: String,
    pub(super) assets_id: String,
}

pub(super) async fn convert(
    axum::extract::State(app): axum::extract::State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<Convert>,
) -> Result<Response> {
    let request = read(payload)?;
    let (_, submission) = read_submission(&headers, "scratch-gguf", Ok(Json((&id, &request))))?;
    submit_convert(&app, &id, &request, submission.as_ref())
}

pub(super) fn submit_convert(
    app: &App,
    id: &str,
    request: &Convert,
    submission: Option<&Submission<'_>>,
) -> Result<Response> {
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, submission)? {
        return Ok(response);
    }
    let row = visible(&store, id)?;
    if row.is_compiled_program() {
        return Err(conflict(
            "compiled program architectures do not support GGUF conversion",
        ));
    }
    if row.state != RevisionState::Scratch || row.sha256 != request.sha256 {
        return Err(conflict(
            "convert the current hash of an unrequested scratch revision",
        ));
    }
    if row.gguf.is_some()
        || store
            .scratch_has_jobs(id, Some(CONVERT))
            .map_err(ApiError::storage)?
    {
        return Err(conflict(
            "this revision already has a private GGUF or conversion",
        ));
    }
    let lane = admit(&app, &row.provenance.worker)?;
    let assets = store
        .model(&request.assets_id)
        .map_err(ApiError::storage)?
        .filter(|m| m.kind == "assets" && m.host == row.provenance.worker)
        .filter(|m| {
            let path = FsPath::new(&m.path);
            path.canonicalize().is_ok_and(|p| p == path && p.is_dir())
                && lane.configured.iter().any(|root| path.starts_with(root))
        })
        .ok_or_else(|| ApiError::bad("use a Library assets directory on the local worker"))?;
    let source = input(&store, &row, scratch::MODEL, "safetensors");
    let pins = BTreeMap::from([(source.path.clone(), row.sha256)]);
    let (job, mut envelope) = private_job(
        CONVERT,
        vec![source, assets],
        json!({
            "output_format":"gguf", "gguf_type":"BF16"
        }),
    );
    envelope.scratch_pins = pins;
    enqueue(
        app,
        &mut store,
        job,
        envelope,
        submission,
        StatusCode::ACCEPTED,
    )
}

pub(crate) fn input(store: &Store, row: &Revision, name: &str, kind: &str) -> Model {
    Model {
        id: row.id.clone(),
        name: format!("Scratch revision ({})", row.id),
        host: row.provenance.worker.clone(),
        path: store
            .scratch_root()
            .join(&row.id)
            .join(name)
            .to_string_lossy()
            .into_owned(),
        kind: kind.into(),
        sha256: None,
        bytes: None,
        available: None,
        reason: None,
        job_id: None,
    }
}

pub(crate) fn trace_input(store: &Store, id: &str) -> Result<(Model, String)> {
    let row = visible(store, id)?;
    if row.is_hgn() {
        return Err(conflict("native HGN scratch cannot be converted to GGUF"));
    }
    let gguf = row
        .gguf
        .as_ref()
        .filter(|gguf| gguf.revision_sha256 == row.sha256)
        .ok_or_else(|| conflict("convert this revision to private GGUF first"))?;
    Ok((input(store, &row, GGUF, "gguf"), gguf.sha256.clone()))
}

pub(super) fn private_job(kind: &str, inputs: Vec<Model>, options: Value) -> (Job, Envelope) {
    let host = inputs[0].host.clone();
    NewJob {
        kind: kind.into(),
        host: None,
        input_ids: inputs.iter().map(|m| m.id.clone()).collect(),
        options: options.as_object().unwrap().clone(),
        prompt_id: None,
        prompt: None,
    }
    .records(host, inputs, None, crate::gguf::Format::Gguf)
}

pub(crate) fn enqueue(
    app: &App,
    store: &mut Store,
    job: Job,
    envelope: Envelope,
    submission: Option<&Submission<'_>>,
    code: StatusCode,
) -> Result<Response> {
    let job = prepare_enqueue(app, job, &envelope)?;
    let tx = store.submission_transaction().map_err(ApiError::storage)?;
    Store::enqueue(&tx, &job, &envelope, submission).map_err(ApiError::storage)?;
    tx.commit().map_err(ApiError::storage)?;
    Ok((code, Json(job)).into_response())
}

pub(crate) fn prepare_enqueue(app: &App, mut job: Job, envelope: &Envelope) -> Result<Job> {
    let lane = admit(app, &job.host)?;
    job.local_edit = true;
    job.scratch_job = true;
    let hosts = app.effective_hosts().map_err(ApiError::storage)?;
    if crate::domain::candidates(&job, envelope, &hosts).is_empty() {
        return Err(conflict("no eligible local worker for this scratch job"));
    }
    let backend = if job.kind == locate_edit::JOB {
        locate_edit::Request::parse(&Value::Object(envelope.options.clone()))
            .map_err(ApiError::storage)?
            .edit
            .backend
    } else if job.kind == brewery_worker::sensitivity::JOB {
        brewery_worker::sensitivity::Request::resolved(&Value::Object(envelope.options.clone()))
            .map_err(ApiError::storage)?
            .0
            .backend
    } else if brewery_worker::heretic::job(&job.kind) {
        locate_edit::Backend::Cuda
    } else {
        locate_edit::Backend::Cpu
    };
    lane.snapshot_scratch(&job.host, &job.id, &envelope.scratch_pins, backend)
        .map_err(ApiError::storage)?;
    Ok(job)
}

pub(crate) fn finish(
    store: &Store,
    job: &Job,
    result: &Value,
) -> std::result::Result<Option<(Revision, Value)>, crate::Error> {
    let envelope = store.envelope(&job.id)?;
    let id = &envelope.inputs[if job.kind == CHECK { 3 } else { 0 }].id;
    let mut row = store
        .scratch(id)?
        .filter(|r| r.state != RevisionState::Discarded)
        .ok_or("private revision is no longer available")?;
    if job.kind == brewery_worker::ablation_search::SCREEN {
        let req = brewery_worker::ablation_search::ScreenRequest::parse(&Value::Object(
            envelope.options.clone(),
        ))?;
        if envelope.scratch_pins.get(&envelope.inputs[0].path) != Some(&req.model_sha256) {
            return Err("screen result does not match the pinned model".into());
        }
        serde_json::from_value::<brewery_worker::ablation_search::ScreenReport>(result.clone())?
            .verify(&req)?;
        return Ok(None);
    }
    if job.kind == brewery_worker::trace::JOB {
        let model = &envelope.inputs[0];
        if result["model_id"] != model.id
            || result["model_sha256"].as_str()
                != envelope.scratch_pins.get(&model.path).map(String::as_str)
        {
            return Err("scratch trace result does not match the pinned model".into());
        }
        return Ok(None);
    }
    if job.kind == brewery_worker::sensitivity::JOB {
        if result["revision_sha256"].as_str() != Some(&row.sha256) {
            return Err("sensitivity artifact does not match the current scratch revision".into());
        }
        return Ok(None);
    }
    if job.kind == brewery_worker::circuit::JOB {
        let model = &envelope.inputs[0];
        if result["model_sha256"].as_str()
            != envelope.scratch_pins.get(&model.path).map(String::as_str)
        {
            return Err("scratch circuit result does not match the pinned GGUF".into());
        }
        return Ok(None);
    }
    if job.kind == CHECK && row.state == RevisionState::Scratch {
        return checks::finish(row, &envelope, job, result, &store.scratch_root().join(id))
            .map(Some);
    }
    if job.kind != CONVERT || row.gguf.is_some() || row.state != RevisionState::Scratch {
        return Err("invalid scratch job result or revision state".into());
    }
    let source = &envelope.inputs[0];
    let artifact = &result["artifact"];
    let tool = &result["backend_receipt"];
    let sha = artifact["sha256"]
        .as_str()
        .filter(|sha| brewery_worker::is_digest(sha))
        .ok_or("private GGUF result has no output SHA-256")?;
    let bytes = artifact["bytes"]
        .as_u64()
        .filter(|n| *n >= 4)
        .ok_or("private GGUF result has no output length")?;
    let quantizer = tool["executable_sha256"]
        .as_str()
        .filter(|sha| brewery_worker::is_digest(sha))
        .ok_or("private GGUF result has no converter identity")?;
    let exporter = tool["hf_exporter_sha256"]
        .as_str()
        .filter(|sha| brewery_worker::is_digest(sha))
        .ok_or("private GGUF result has no exporter identity")?;
    let staged = output_path(store, job);
    if artifact["path"].as_str() != staged.to_str()
        || result["kind"] != "convert"
        || result["gguf_type"] != "BF16"
        || artifact["kind"] != "gguf"
        || result["file_verification"]["status"] != "passed"
        || result["inputs"][0]["id"] != row.id
        || result["inputs"][0]["sha256"] != row.sha256
        || envelope.scratch_pins.get(&source.path) != Some(&row.sha256)
    {
        return Err("private GGUF result does not match the pinned revision".into());
    }
    let directory = store.scratch_root().join(id);
    let destination = directory.join(GGUF);
    // A retry after a controller exit may find the verified rename already done.
    let moved = !staged.exists();
    let mut file = open_regular(if moved { &destination } else { &staged })?;
    let mut magic = [0; 4];
    use std::io::Read as _;
    file.read_exact(&mut magic)?;
    if &magic != b"GGUF"
        || file.metadata()?.len() != bytes
        || scratch::file_sha256(&mut file)? != sha
    {
        return Err("private GGUF file does not match its conversion result".into());
    }
    if !moved {
        file.sync_all()?;
        fs::rename(&staged, &destination)?;
        File::open(directory)?.sync_all()?;
        File::open(staged.parent().unwrap())?.sync_all()?;
    }
    row.gguf = Some(PrivateGguf {
        revision_sha256: row.sha256.clone(),
        sha256: sha.into(),
        converter_version: format!(
            "llama-quantize sha256:{quantizer}; HF exporter sha256:{exporter}"
        ),
        bytes,
        job_id: job.id.clone(),
    });
    let safe_result =
        json!({"scratch_id":id, "gguf":row.gguf, "qualification":"not release-qualified"});
    Ok(Some((row, safe_result)))
}

fn output_path(store: &Store, job: &Job) -> PathBuf {
    store
        .scratch_root()
        .with_file_name("local-edit")
        .join(&job.host)
        .join("worker/jobs")
        .join(&job.id)
        .join("work")
        .join(OUTPUT)
}

pub(crate) fn clean_failed_conversion(store: &Store, job: &Job) {
    if job.kind == CONVERT
        && let Err(error) = fs::remove_file(output_path(store, job))
        && error.kind() != std::io::ErrorKind::NotFound
    {
        eprintln!(
            "Could not remove private conversion output for {}: {error}",
            job.id
        );
    }
}
