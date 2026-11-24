use crate::Job;
use anyhow::{Context, Result, ensure};
use brewery_worker::{hgn_edit as edit, weight_source::scratch::hgn_ablation};
use serde_json::{Value, json};
use std::{
    env,
    fs::DirBuilder,
    os::unix::fs::DirBuilderExt,
    path::{Path, PathBuf},
};

pub(super) fn run(job: &Job, cwd: &Path) -> Result<Value> {
    let [input] = job.inputs.as_slice() else {
        anyhow::bail!("expected one private HGN source");
    };
    let req: edit::Request = serde_json::from_value(job.options.clone())?;
    req.check()?;
    ensure!(
        edit::is_hgn(&input.kind) && job.prompt.is_none(),
        "invalid private HGN input"
    );
    let cache =
        PathBuf::from(env::var_os(edit::CACHE).context("HGN edit cache is not configured")?);
    let reserve = env::var(edit::RESERVE)?.parse()?;
    ensure!(cache.is_absolute(), "expected absolute HGN cache");
    let private = cwd.join(edit::PRIVATE);
    DirBuilder::new().mode(0o700).create(&private)?;
    let report = hgn_ablation::build(
        &input.path,
        &req.parent_sha256,
        &private.join(edit::MODEL),
        &cache,
        req.first_layer..=req.last_layer,
        &req.direction,
        reserve,
    )?;
    req.verify(&report)?;
    brewery_worker::atomic(&private.join(edit::REPORT), &report)?;
    Ok(json!({"kind":edit::JOB,"report":report}))
}
