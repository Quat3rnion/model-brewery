use crate::{integrity, progress::Progress, trace::hgn::Runtime, Job};
use anyhow::{ensure, Context, Result};
use brewery_worker::{
    ablation_search::hgn::{self, Report, Request},
    behavior_direction::{self as directions, Behavior, Split},
    trace,
};
use serde_json::{json, Value};
use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let req = Request::parse(&job.options)?;
    let [input] = job.inputs.as_slice() else {
        anyhow::bail!("HGN screen requires one Library checkpoint");
    };
    ensure!(
        trace::is_hgn(&input.kind)
            && job.prompt.is_none()
            && input.sha256.as_ref() == Some(&req.model_sha256),
        "invalid HGN screen input"
    );
    let cache = PathBuf::from(env::var_os(directions::CACHE).context("missing contrast cache")?);
    ensure!(cache.is_absolute(), "contrast cache must be absolute");
    let contrast_sha256 = req.behavior.contrast_sha256();
    let pairs = directions::load_pairs(&cache, req.behavior, contrast_sha256)?;
    let prompts = |split| {
        pairs.iter().filter(|p| p.split == split).map(|p| json!({
        "positive":p.positive.hgn_chat_prefix(), "negative":p.negative.hgn_chat_prefix(),
        "agree_a":p.expected_agreement.as_deref() == Some("(A)"),
    })).collect::<Vec<_>>()
    };
    let held_out = prompts(Split::HeldOut);
    ensure!(
        held_out.len() == req.held_out_pairs,
        "HGN screen holdout does not match fit"
    );
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let model = &ids[0];
    ensure!(
        model.files.len() == 1 && model.sha256 == req.model_sha256,
        "HGN screen hash mismatch"
    );
    let private = tempfile::Builder::new()
        .prefix(".screen-hgn-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(cwd)?;
    let runtime = Runtime::stage(private.path(), &model.sha256)?;
    let pins = runtime.receipt();
    ensure!(
        pins["assets_sha256"] == serde_json::to_value(&req.assets_sha256)?,
        "HGN direction companion pins differ from this checkpoint's runtime assets"
    );
    let spec = json!({
        "behavior":req.behavior, "held_out":held_out,
        "train": if req.behavior == Behavior::Slop { prompts(Split::Train) } else { vec![] },
        "trials":req.trial_specs()?,
    });
    let mut phase = progress.start("screen_hgn_marker", "items", Some(1));
    model.check()?;
    let scores = runtime.screen(private.path(), &input.path, &spec)?;
    model.check()?;
    scores.verify(&req)?;
    let (pareto, selected) = scores.frontier();
    let report = Report {
        kind: hgn::JOB.into(),
        schema_version: 1,
        release_qualified: false,
        model_sha256: req.model_sha256.clone(),
        direction_id: req.direction_id.clone(),
        source_layer: req.source_layer,
        first_layer: req.first_layer,
        last_layer: req.last_layer,
        targets: req.targets.clone(),
        second_source_layer: req.second_source_layer,
        configurations: req.configurations()?,
        behavior: req.behavior,
        seed: req.seed,
        runtime: trace::HGN_RUNTIME.into(),
        runtime_sha256: trace::HGN_RUNTIME_SHA256.trim_end().into(),
        executable_sha256: pins["executable_sha256"]
            .as_str()
            .context("missing HGN Python pin")?
            .into(),
        assets_sha256: req.assets_sha256.clone(),
        contrast_sha256: contrast_sha256.into(),
        scores,
        pareto,
        selected,
    };
    report.verify(&req, None)?;
    phase.advance(1);
    phase.finish();
    Ok(serde_json::to_value(report)?)
}
