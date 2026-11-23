use crate::{Job, integrity, progress::Progress, trace::Runtime};
use anyhow::{Context, Result, ensure};
use brewery_worker::{behavior_direction as direction, trace};
use serde_json::{Value, json};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let request = direction::Request::parse(&job.options)?;
    ensure!(
        job.prompt.is_none() && job.inputs.len() == 1 && job.inputs[0].kind == "gguf",
        "direction needs one Library GGUF and the pinned prompt set"
    );
    let input = &job.inputs[0];
    let model_id = input.id.as_deref().context("missing Library model ID")?;
    ensure!(
        input.sha256.as_deref() == Some(request.model_sha256.as_str()),
        "direction needs the resolved GGUF revision"
    );
    let cache =
        PathBuf::from(env::var_os(direction::CACHE).context("direction cache is not configured")?);
    ensure!(cache.is_absolute(), "direction cache path must be absolute");
    let pairs = request.load_pairs(&cache)?;
    let splits: Vec<_> = pairs.iter().map(|p| p.split).collect();
    let [model] = integrity::check_inputs(&job.inputs, progress)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("missing verified model"))?;
    ensure!(
        model.files.len() == 1 && input.path.is_file() && model.sha256 == request.model_sha256,
        "direction needs a single-file GGUF with matching SHA-256"
    );
    let runtime = Runtime::stage(cwd)?;
    let out = runtime.path().join("direction-trace");
    let mut phase = progress.start("trace_prompts", "items", Some((pairs.len() * 2) as u64));
    let fit = direction::compute(&request.layers, request.role, &splits, |row, positive| {
        let pair = &pairs[row];
        let prompt = if positive {
            &pair.positive
        } else {
            &pair.negative
        }
        .qwen_chat_prefix();
        let capture = |spec: &trace::Capture| {
            model.check()?;
            let verified = runtime.capture(&input.path, &prompt, spec, &out, None)?;
            model.check()?;
            let captured = trace::direction_rows(&out, spec, &verified)?;
            fs::remove_dir_all(&out)?;
            Ok::<_, anyhow::Error>(captured)
        };
        let probe = if request.position.is_none() {
            Some(capture(&trace::direction_probe())?.0)
        } else {
            None
        };
        let position = request
            .position
            .unwrap_or_else(|| probe.as_ref().unwrap().len() - 1);
        let (tokens, rows) = capture(&trace::direction_capture(
            &request.layers,
            request.role,
            position,
        ))?;
        ensure!(
            probe.is_none_or(|expected| expected == tokens),
            "direction trace token IDs changed"
        );
        phase.advance(1);
        Ok(rows)
    })?;
    phase.finish();
    let result = json!({
        "kind":direction::JOB, "schema_version":1, "release_qualified":false,
        "model_id":model_id, "model_sha256":model.sha256, "position":request.position,
        "behavior":request.behavior, "contrast_sha256":request.behavior.contrast_sha256(),
        "lanternfish_commit": trace::COMMIT, "executable_sha256":trace::SHA256,
        "fit":fit,
    });
    request.verify_result(&result, model_id, &model.sha256)?;
    Ok(result)
}
