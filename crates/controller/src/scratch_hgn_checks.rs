//! Queue and bind checks for a remote, private HGN scratch revision.
use super::*;
use crate::domain::{Envelope, Job, Model, candidates};
use axum::response::{IntoResponse, Response};
use brewery_worker::hgn_checks as native;
use serde_json::{Value, json};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bound {
    revision: String,
    request: native::Request,
}

fn plan<'a>(store: &Store, row: &'a Revision) -> Result<(Model, Bound, &'a str)> {
    let p = &row.provenance;
    let (ablation, evidence) = p
        .ablation
        .as_ref()
        .and_then(|a| Some((a, a.hgn.as_ref()?)))
        .ok_or_else(|| conflict("missing native HGN edit provenance"))?;
    let (root, parent_producer, root_sha256) =
        hgn_edit::source(store, &p.parent_id, &p.parent_sha256)?;
    if root.id != evidence.root_model
        || root_sha256 != evidence.root_sha256
        || root.host != p.worker
        || evidence.job_id != row.id
        || p.revision_id != row.id
        || p.revision_sha256 != row.sha256
        || row.delta.parent_sha256 != p.parent_sha256
        || row.delta.revision_sha256 != row.sha256
        || evidence.report.parent_sha256 != p.parent_sha256
        || evidence.report.revision_sha256 != row.sha256
    {
        return Err(conflict("HGN check pins do not match provenance"));
    }
    let request = native::Request {
        parent_sha256: p.parent_sha256.clone(),
        revision_sha256: row.sha256.clone(),
        root_sha256,
        parent_producer,
        child_producer: evidence.job_id.clone(),
        first_layer: ablation.settings.first_layer,
        last_layer: ablation.settings.last_layer,
    };
    request.verify().map_err(ApiError::storage)?;
    Ok((
        root,
        Bound {
            revision: row.id.clone(),
            request,
        },
        &p.parent_id,
    ))
}

pub(super) fn submit(
    app: &App,
    store: &Store,
    row: &Revision,
    key: Option<&Submission<'_>>,
) -> Result<Response> {
    hgn_edit::key().map_err(|_| conflict("private HGN check signer is not configured"))?;
    if store
        .scratch_has_jobs(&row.id, Some(native::JOB))
        .map_err(ApiError::storage)?
    {
        return Err(conflict("checks are already running for this revision"));
    }
    let (root, bound, parent_id) = plan(store, row)?;
    let (mut job, envelope) = jobs::private_job(native::JOB, vec![root], json!(bound));
    job.scratch_job = true;
    job.input_ids = vec![parent_id.into(), row.id.clone()];
    job.options.clear();
    if candidates(
        &job,
        &envelope,
        &app.effective_hosts().map_err(ApiError::storage)?,
    )
    .is_empty()
    {
        return Err(conflict(
            "enable native HGN edits and CPU trace on the source worker",
        ));
    }
    let tx = store.submission_transaction().map_err(ApiError::storage)?;
    Store::enqueue(&tx, &job, &envelope, key).map_err(ApiError::storage)?;
    tx.commit().map_err(ApiError::storage)?;
    Ok((StatusCode::ACCEPTED, Json(job)).into_response())
}

pub(crate) fn wire(job: &Job, envelope: &Envelope) -> std::result::Result<Vec<u8>, crate::Error> {
    let bound: Bound = serde_json::from_value(json!(envelope.options))?;
    let inputs = json!(envelope.inputs);
    let input = serde_json::from_value(inputs[0].clone())?;
    let grant = native::grant(&job.id, &input, bound.request, &hgn_edit::key()?)?;
    Ok(serde_json::to_vec(&json!({
        "kind": native::JOB, "inputs": inputs, "options": {(native::GRANT): grant}
    }))?)
}

pub(crate) fn finish(
    store: &Store,
    job: &Job,
    result: &Value,
) -> std::result::Result<(Revision, Value), crate::Error> {
    let envelope = store.envelope(&job.id)?;
    let bound: Bound = serde_json::from_value(json!(envelope.options))?;
    let mut row = store
        .scratch(&bound.revision)?
        .filter(|r| r.is_hgn() && r.state == RevisionState::Scratch)
        .ok_or("HGN scratch revision is not available")?;
    let (root, current, _) = plan(store, &row).map_err(|_| "HGN check pins changed")?;
    if root.host != job.host
        || json!(current) != json!(bound)
        || !matches!(envelope.inputs.as_slice(), [input]
            if input.id == root.id && input.host == root.host
                && input.path == root.path && input.kind == root.kind)
    {
        return Err("HGN check source changed".into());
    }
    let feedback = &row
        .provenance
        .ablation
        .as_ref()
        .unwrap()
        .hgn
        .as_ref()
        .unwrap()
        .report;
    let report: native::Report = serde_json::from_value(result.clone())?;
    report.verify(&bound.request, feedback)?;
    row.check_bundle = Some(checks::hgn_bundle(&row, job.id.clone(), report));
    let result = json!({"scratch_id": row.id, "check_bundle": row.check_bundle});
    Ok((row, result))
}

pub(crate) fn failed(
    store: &Store,
    job: &Job,
) -> std::result::Result<Option<Revision>, crate::Error> {
    let envelope = store.envelope(&job.id)?;
    let bound: Bound = serde_json::from_value(json!(envelope.options))?;
    let Some(mut row) = store.scratch(&bound.revision)?.filter(|r| {
        r.is_hgn()
            && r.state == RevisionState::Scratch
            && r.sha256 == bound.request.revision_sha256
            && r.provenance.worker == job.host
    }) else {
        return Ok(None);
    };
    row.check_bundle = Some(checks::hgn_failed_bundle(&row, job.id.clone()));
    Ok(Some(row))
}
