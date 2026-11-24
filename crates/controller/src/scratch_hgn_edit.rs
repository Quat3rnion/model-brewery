//! Queue native HGN edits on the source worker, without publishing a model path.
use super::*;
use crate::{
    domain::{Envelope, Job, Model, candidates},
    web::{read_submission, replay},
};
use axum::response::{IntoResponse, Response};
use brewery_worker::{behavior_direction, hgn_edit as native};
use serde_json::{Value, json};

pub(super) const KIND: &str = "scratch_hgn";

#[derive(Deserialize, Serialize)]
pub(super) struct Evidence {
    pub(super) job_id: String,
    root_model: String,
    root_sha256: String,
    report: native::Report,
}

#[derive(Deserialize, Serialize)]
struct Bound {
    parent: String,
    root_model: String,
    root_sha256: String,
    response: ablation::Request,
    native: native::Request,
}

pub(super) fn is_parent(app: &App, id: &str) -> Result<bool> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(store
        .scratch(id)
        .map_err(ApiError::storage)?
        .is_some_and(|r| r.is_hgn())
        || store
            .model(id)
            .map_err(ApiError::storage)?
            .is_some_and(|m| native::is_hgn(&m.kind)))
}

fn key() -> anyhow::Result<[u8; 32]> {
    let path = std::env::var_os(native::KEY_FILE)
        .ok_or_else(|| anyhow::anyhow!("private HGN edit signer is not configured"))?;
    native::read_key(&PathBuf::from(path))
}

fn source(store: &Store, parent: &str, expected: &str) -> Result<(Model, Option<String>, String)> {
    let (root, producer, root_sha, worker) =
        if let Some(row) = store.scratch(parent).map_err(ApiError::storage)? {
            if !row.is_hgn() || row.state != RevisionState::Scratch || row.sha256 != expected {
                return Err(conflict(
                    "use the pinned hash of a private HGN scratch parent",
                ));
            }
            let evidence = row
                .provenance
                .ablation
                .and_then(|a| a.hgn)
                .ok_or_else(|| ApiError::storage("HGN scratch producer is missing"))?;
            (
                evidence.root_model,
                Some(evidence.job_id),
                evidence.root_sha256,
                Some(row.provenance.worker),
            )
        } else {
            (parent.into(), None, expected.into(), None)
        };
    let model = store
        .model(&root)
        .map_err(ApiError::storage)?
        .filter(|m| {
            native::is_hgn(&m.kind)
                && worker.as_ref().is_none_or(|host| host == &m.host)
                && m.sha256.as_deref().is_none_or(|sha| sha == root_sha)
                && FsPath::new(&m.path).is_absolute()
                && FsPath::new(&m.path).extension().is_some_and(|e| e == "hgn")
                && !native::private_path(FsPath::new(&m.path))
        })
        .ok_or_else(|| ApiError::bad("select an approved standalone HGN Library parent"))?;
    Ok((model, producer, root_sha))
}

pub(super) fn submit(
    app: &App,
    id: String,
    req: ablation::Request,
    headers: &HeaderMap,
) -> Result<Response> {
    use scratch::ablation::{Shape, Variant};
    let ((id, req), submission) =
        read_submission(headers, "scratch-hgn-ablate", Ok(Json((id, req))))?;
    if req.include_embeddings
        || req.variant != Variant::Plain
        || req.alpha != 1.0
        || !matches!(req.kernel.shape, Shape::Flat)
        || req.kernel.peak >= 4096
        || !(0.0..=4096.0).contains(&req.kernel.width)
        || req.kernel.width == 0.0
        || !(0.0..=1.0).contains(&req.kernel.min_weight)
        || req.direction.version != behavior_direction::VERSION
    {
        return Err(ApiError::bad(
            "HGN edits require plain, flat, alpha 1, no embeddings and a version 2 direction",
        ));
    }
    let store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, submission.as_ref())? {
        return Ok(response);
    }
    key().map_err(|_| conflict("private HGN edit signer is not configured"))?;
    let (model, producer, root_sha256) = source(&store, &id, &req.sha256)?;
    let result = crate::directions::load(&store, &req.direction.id, req.direction.version)?;
    if result["model_id"] != model.id
        || result["model_sha256"] != root_sha256
        || result["runtime"] != brewery_worker::trace::HGN_RUNTIME
    {
        return Err(ApiError::bad(
            "fit the direction on the same pinned HGN Library root",
        ));
    }
    let fit: behavior_direction::Fit =
        serde_json::from_value(result["fit"].clone()).map_err(ApiError::storage)?;
    let direction = fit
        .layers
        .into_iter()
        .find(|row| row.layer == req.direction.layer)
        .and_then(|row| row.unit_vector)
        .ok_or_else(|| ApiError::bad("the source layer has no nonzero fitted direction"))?;
    let native = native::Request {
        parent_sha256: req.sha256.clone(),
        producer,
        first_layer: req.first_layer,
        last_layer: req.last_layer,
        direction,
    };
    native.check().map_err(|e| ApiError::bad(e.to_string()))?;
    let bound = Bound {
        parent: id,
        root_model: model.id.clone(),
        root_sha256,
        response: req,
        native,
    };
    if store.scratch_count().map_err(ApiError::storage)? >= LIMIT {
        return Err(conflict("scratch revision limit reached"));
    }
    let (mut job, envelope) = jobs::private_job(native::JOB, vec![model], json!(bound));
    job.scratch_job = true;
    job.options.clear();
    if candidates(
        &job,
        &envelope,
        &app.effective_hosts().map_err(ApiError::storage)?,
    )
    .is_empty()
    {
        return Err(conflict("enable native HGN edits on the source worker"));
    }
    let tx = store.submission_transaction().map_err(ApiError::storage)?;
    Store::enqueue(&tx, &job, &envelope, submission.as_ref()).map_err(ApiError::storage)?;
    tx.commit().map_err(ApiError::storage)?;
    Ok((StatusCode::ACCEPTED, Json(job)).into_response())
}

pub(crate) fn wire(job: &Job, envelope: &Envelope) -> std::result::Result<Vec<u8>, crate::Error> {
    let bound: Bound = serde_json::from_value(json!(envelope.options))?;
    let inputs = json!(envelope.inputs);
    let input: brewery_worker::Input = serde_json::from_value(inputs[0].clone())?;
    let grant = native::grant(&job.id, &input, bound.native, &key()?)?;
    Ok(serde_json::to_vec(
        &json!({"kind":native::JOB,"inputs":inputs,"options":{(native::GRANT):grant}}),
    )?)
}

pub(crate) fn finish(
    store: &Store,
    job: &Job,
    result: &Value,
) -> std::result::Result<(Revision, Value), crate::Error> {
    let envelope = store.envelope(&job.id)?;
    let bound: Bound = serde_json::from_value(json!(envelope.options))?;
    let (root, producer, sha) = source(store, &bound.parent, &bound.native.parent_sha256)
        .map_err(|_| "private HGN parent is no longer available")?;
    if root.id != bound.root_model
        || root.host != job.host
        || envelope
            .inputs
            .first()
            .is_none_or(|input| input.path != root.path || input.kind != root.kind)
        || producer != bound.native.producer
        || sha != bound.root_sha256
        || store.scratch_count()? >= LIMIT
        || result["kind"] != native::JOB
        || result.as_object().is_none_or(|o| o.len() != 2)
    {
        return Err("HGN parent or worker changed before receipt commit".into());
    }
    let report: native::Report = serde_json::from_value(result["report"].clone())?;
    bound.native.verify(&report)?;
    let parent_sha256 = report.parent_sha256.clone();
    let revision_sha256 = report.revision_sha256.clone();
    let tensors = report
        .tensors
        .iter()
        .map(|(name, t)| t.info(name.clone()))
        .collect();
    let changed_tensors = report
        .tensors
        .values()
        .filter(|r| r.rel_error > 0.0)
        .count();
    let changed_bytes = report
        .tensors
        .values()
        .filter(|r| r.rel_error > 0.0)
        .map(|r| r.bytes)
        .sum();
    let runtime_version = report.encoder_sha256.clone();
    let evidence = bound.response.evidence(Evidence {
        job_id: job.id.clone(),
        root_model: root.id,
        root_sha256: bound.root_sha256,
        report,
    });
    let row = Revision {
        id: job.id.clone(),
        kind: KIND.into(),
        state: RevisionState::Scratch,
        sha256: revision_sha256.clone(),
        qualification: "not release-qualified".into(),
        library_model_id: None,
        provenance: Provenance {
            schema: "brewery.scratch-provenance.v1".into(),
            parent_id: bound.parent,
            parent_sha256: parent_sha256.clone(),
            revision_id: job.id.clone(),
            revision_sha256: revision_sha256.clone(),
            worker: job.host.clone(),
            method: "ablate_direction".into(),
            operations: vec![],
            merge: None,
            pruning: None,
            pruning_trace: None,
            ablation: Some(evidence),
            tensors,
            tool: "brewery native HGN editor".into(),
            tool_version: env!("CARGO_PKG_VERSION").into(),
            runtime_version,
            locate_edit: None,
            heretic: None,
            created_at_ms: now(),
            compiled_program: None,
        },
        delta: Summary {
            schema: "brewery.hgn-edit-summary.v1".into(),
            parent_sha256,
            revision_sha256: revision_sha256.clone(),
            changed_tensors,
            changed_bytes,
            delta_bytes: 0,
            delta_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
        },
        check_bundle: None,
        gguf: None,
    };
    Ok((
        row,
        json!({"kind":native::JOB,"scratch_id":job.id,"sha256":revision_sha256,
        "release_qualified":false}),
    ))
}
