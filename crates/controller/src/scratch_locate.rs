//! Submit a pinned private edit; materialize its W3 child on verified success.
use super::*;
use crate::{
    domain::Job,
    web::{read_submission, replay},
};
use axum::response::Response;
use brewery_worker::locate_edit::{self, Edit, Factors, Method, Request};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Deserialize, Serialize)]
pub(super) struct EditBody {
    sha256: String,
    #[serde(flatten)]
    edit: Edit,
}

macro_rules! route {
    ($handler:ident, $method:ident) => {
        pub(super) async fn $handler(
            State(app): State<Arc<App>>,
            Path(parent): Path<String>,
            headers: HeaderMap,
            payload: Payload<EditBody>,
        ) -> Result<Response> {
            run(&app, &parent, &headers, read(payload)?, Method::$method)
        }
    };
}
route!(rome, Rome);
route!(memit, Memit);
route!(alphaedit, Alphaedit);

fn run(
    app: &App,
    parent: &str,
    headers: &HeaderMap,
    body: EditBody,
    method: Method,
) -> Result<Response> {
    let request = Request {
        method,
        edit: body.edit,
    };
    request
        .validate()
        .map_err(|e| ApiError::bad(e.to_string()))?;
    if !brewery_worker::is_digest(&body.sha256) {
        return Err(ApiError::bad("expected a lowercase parent SHA-256"));
    }
    let (_, submission) = read_submission(
        headers,
        "scratch-locate-edit",
        Ok(Json((parent, &body.sha256, &request))),
    )?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, submission.as_ref())? {
        return Ok(response);
    }
    let row = visible(&store, parent)?;
    if row.sha256 != body.sha256 || row.state != RevisionState::Scratch {
        return Err(conflict(
            "use the current hash of an unrequested scratch parent",
        ));
    }
    let (gguf, gguf_sha) = jobs::trace_input(&store, parent)?;
    let safe = jobs::input(&store, &row, scratch::MODEL, "safetensors");
    let lane = admit(app, &row.provenance.worker)?;
    let backend = request.edit.backend;
    if !lane.locate_edit_configured(backend) {
        return Err(conflict(&format!(
            "configure the pinned Lanternfish {} editor on this local worker",
            backend.label()
        )));
    }
    if backend.is_gpu() && !lane.scratch_gpu_allowed() {
        return Err(conflict(&format!(
            "{} edit requires GPU load permission and a GPU admission wrapper on this worker",
            backend.label()
        )));
    }
    if store.scratch_count().map_err(ApiError::storage)? >= LIMIT {
        return Err(conflict("scratch revision limit reached"));
    }
    let parent_len = open_regular(FsPath::new(&safe.path))
        .and_then(|f| Ok(f.metadata()?.len()))
        .map_err(ApiError::storage)?;
    // Private F32 expansion uses at most twice the BF16 model size and
    // bounded cache factors; only the final W3 child persists.
    let budget = parent_len
        .saturating_mul(3)
        .saturating_add(512 << 20)
        .saturating_add(backend.runtime_stage_reserve())
        .saturating_add(lane.scratch_disk_reserve());
    if brewery_worker::available(store.scratch_root()).map_err(ApiError::storage)? < budget {
        return Err(conflict("not enough disk space for locate-then-edit"));
    }
    let pins = BTreeMap::from([
        (gguf.path.clone(), gguf_sha),
        (safe.path.clone(), row.sha256),
    ]);
    let (job, mut envelope) = jobs::private_job(
        locate_edit::JOB,
        vec![gguf, safe],
        serde_json::to_value(&request).map_err(ApiError::storage)?,
    );
    envelope.scratch_pins = pins;
    jobs::enqueue(
        app,
        &mut store,
        job,
        envelope,
        submission.as_ref(),
        StatusCode::ACCEPTED,
    )
}

pub(crate) fn finish(
    store: &Store,
    job: &Job,
    result: &Value,
) -> std::result::Result<(Revision, Value, Uncommitted), crate::Error> {
    let envelope = store.envelope(&job.id)?;
    let request = Request::parse(&Value::Object(envelope.options))?;
    let backend = request.edit.backend;
    let pin = backend.pin();
    let [gguf, safe] = envelope.inputs.as_slice() else {
        return Err("missing private editor inputs".into());
    };
    let row = store
        .scratch(&safe.id)?
        .ok_or("missing private editor parent")?;
    let gguf_pin = envelope
        .scratch_pins
        .get(&gguf.path)
        .ok_or("missing edit model SHA-256")?;
    if !(row.state == RevisionState::Scratch
        && safe.id == gguf.id
        && row.provenance.worker == job.host
        && row.provenance.worker == safe.host
        && gguf.path == jobs::input(store, &row, jobs::GGUF, "gguf").path
        && safe.path == jobs::input(store, &row, scratch::MODEL, "safetensors").path
        && row
            .gguf
            .as_ref()
            .is_some_and(|g| &g.sha256 == gguf_pin && g.revision_sha256 == row.sha256)
        && envelope.scratch_pins.get(&safe.path) == Some(&row.sha256)
        && result["parent_sha256"] == row.sha256
        && result["kind"] == locate_edit::JOB
        && result["executable_sha256"] == pin.sha256
        && result["lanternfish_commit"] == pin.commit)
    {
        return Err("private edit receipt does not match the pinned parent".into());
    }
    let factors: Factors = serde_json::from_value(result["factors"].clone())?;
    factors.verify(&request, gguf_pin)?;
    if store.scratch_count()? >= LIMIT {
        return Err("scratch revision limit reached".into());
    }
    let root = store.scratch_root();
    // The job ID is unique and private. Recover only its own uncommitted
    // directory if the controller exited before the SQLite commit.
    if store.scratch(&job.id)?.is_some() {
        return Err("scratch edit already committed".into());
    }
    remove(root, &job.id).map_err(|e| e.1)?;
    let guard = Uncommitted::new(root.join(&job.id)).map_err(|e| e.1)?;
    let operations = factors.operations();
    let output = scratch::build(
        FsPath::new(&safe.path),
        &row.sha256,
        &guard.path,
        &operations,
    )?;
    let mut locate = json!({
        "request":request,"parent_gguf_sha256":gguf_pin,"covariance_sha256":factors.covariance_sha256,
        "gguf_weight_dtype":factors.source_dtype,"gradient_dtype":factors.gradient_dtype,
        "optimization_nll":factors.losses,"runtime_libraries":pin.libraries,
    });
    if let Some(audit) = &factors.provenance {
        locate["provenance"] = serde_json::to_value(audit)?;
    }
    let provenance = Provenance {
        schema: "brewery.scratch-provenance.v1".into(),
        parent_id: row.id,
        parent_sha256: row.sha256,
        revision_id: job.id.clone(),
        revision_sha256: output.sha256.clone(),
        worker: job.host.clone(),
        method: request.method.key().into(),
        operations,
        merge: None,
        pruning: None,
        pruning_trace: None,
        ablation: None,
        tensors: output.tensors,
        tool: "Lanternfish locate-edit and Brewery SafeTensors scratch editor".into(),
        tool_version: pin.commit.into(),
        runtime_version: format!("{}-ggml-backward-f32; sha256:{}", backend.key(), pin.sha256),
        locate_edit: Some(locate),
        created_at_ms: now(),
        compiled_program: None,
    };
    brewery_worker::atomic(&guard.path.join(PROVENANCE), &provenance)?;
    File::open(root)?.sync_all()?;
    let result = json!({
        "scratch_id":job.id, "sha256":output.sha256, "method":provenance.method,
        "optimization_nll":factors.losses, "qualification":"not release-qualified"
    });
    let child = Revision {
        id: job.id.clone(),
        kind: "scratch".into(),
        state: RevisionState::Scratch,
        sha256: output.sha256,
        qualification: "not release-qualified".into(),
        library_model_id: None,
        provenance,
        delta: output.delta,
        check_bundle: None,
        gguf: None,
    };
    Ok((child, result, guard))
}
