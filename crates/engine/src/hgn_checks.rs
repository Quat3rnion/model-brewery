use crate::{bounded, integrity::Stamp, trace::hgn::Runtime, Job};
use anyhow::{ensure, Context, Result};
use brewery_worker::{hgn_checks as checks, hgn_edit};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(super) fn run(job: &Job, work: &Path) -> Result<Value> {
    let [parent, child] = job.inputs.as_slice() else {
        anyhow::bail!("stored HGN checks need two resolved checkpoints");
    };
    ensure!(
        job.prompt.is_none()
            && hgn_edit::is_hgn(&parent.kind)
            && parent.kind == child.kind
            && parent.host == child.host
            && hgn_edit::private_path(&child.path),
        "invalid stored HGN check inputs"
    );
    let options: checks::EngineOptions = serde_json::from_value(job.options.clone())?;
    let req = &options.request;
    req.verify()?;
    let feedback_path = child.path.with_file_name(hgn_edit::REPORT);
    let bytes = bounded(
        brewery_worker::weight_source::open_regular(&feedback_path)?,
        brewery_worker::LIMIT,
    )?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes)) == options.feedback_sha256,
        "stored HGN feedback SHA-256 mismatch"
    );
    let feedback: hgn_edit::Report = serde_json::from_slice(&bytes)?;
    let a = Stamp::regular(&parent.path)?;
    let b = Stamp::regular(&child.path)?;
    let rows = checks::audit(
        &parent.path,
        &child.path,
        &req.parent_sha256,
        &req.revision_sha256,
        req.first_layer..=req.last_layer,
        &feedback,
    )
    .context("decoded HGN store audit failed")?;
    let runtime = Runtime::stage(work, &req.root_sha256)?;
    let scores = runtime.checks(work, &parent.path, &child.path)?;
    a.check(&parent.path)?;
    b.check(&child.path)?;
    let mut result = runtime.receipt();
    result["schema"] = json!(checks::SCHEMA);
    result["parent_sha256"] = json!(req.parent_sha256);
    result["revision_sha256"] = json!(req.revision_sha256);
    result["root_sha256"] = json!(req.root_sha256);
    result["seed"] = json!(checks::SEED);
    result["tensors"] = serde_json::to_value(rows)?;
    result["prompts"] = serde_json::to_value(scores)?;
    let report: checks::Report = serde_json::from_value(result.clone())?;
    report.verify(req, &feedback)?;
    Ok(result)
}
