use crate::{
    child_budget, child_threads, gguf_conversion::Tool, integrity, native_json_admitted,
    progress::Progress, Job,
};
use anyhow::{ensure, Context, Result};
use brewery_worker::{is_digest, scratch_eval};
use serde_json::Value;
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn configured(key: &str) -> Result<PathBuf> {
    let path = env::var_os(key)
        .map(PathBuf::from)
        .with_context(|| format!("{key} is required"))?;
    integrity::canonical(&path)?;
    Ok(path)
}

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    ensure!(
        job.options.as_object().is_some_and(|options| options.is_empty())
            && job.prompt.is_none()
            && job.inputs.iter().map(|i| i.kind.as_str()).eq(scratch_eval::CHECK_INPUTS)
            && job.inputs.iter().all(|i| i.sha256.as_deref().is_some_and(is_digest) && i.path.is_file()),
        "scratch checks need four pinned files in parent/child GGUF, parent/child SafeTensors order"
    );
    let script = configured(scratch_eval::SCRIPT)?;
    let cache = configured(scratch_eval::CACHE)?;
    ensure!(
        script.is_file() && cache.is_dir(),
        "invalid scratch checks script or cache"
    );
    let private = tempfile::Builder::new()
        .prefix(".scratch-checks-")
        .tempdir_in(cwd)?;
    let runner = Tool::pinned(
        private.path(),
        scratch_eval::RUNNER,
        scratch_eval::RUNNER_SHA,
        "scratch-check-logits",
        "scratch checks runner",
    )?;
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    ensure!(
        ids.iter().all(|id| id.files.len() == 1),
        "scratch checks need single-file models"
    );
    let threads = child_threads(None)?.min(16);
    let mut cmd = Command::new(Path::new("/usr/bin/python3").canonicalize()?);
    cmd.env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", private.path())
        .env("TMPDIR", private.path())
        .env("LC_ALL", "C")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("HIP_VISIBLE_DEVICES", "")
        .env("ROCR_VISIBLE_DEVICES", "")
        .env("GGML_VK_VISIBLE_DEVICES", "")
        .args(["-I", "-B"])
        .arg(script)
        .arg("--runner")
        .arg(runner.path())
        .arg("--cache")
        .arg(cache)
        .arg("--threads")
        .arg(threads.to_string());
    child_budget(&mut cmd, threads);
    for (flag, input) in [
        "--parent-gguf",
        "--child-gguf",
        "--parent-safetensors",
        "--child-safetensors",
    ]
    .into_iter()
    .zip(&job.inputs)
    {
        cmd.arg(flag).arg(&input.path);
    }
    ids.iter().try_for_each(integrity::Identity::check)?;
    let (report, _) = native_json_admitted(
        &mut cmd,
        &cwd.join("scratch-checks.log"),
        None,
        Some((progress, scratch_eval::CHECKS)),
    )?;
    ids.iter().try_for_each(integrity::Identity::check)?;
    scratch_eval::check_report(
        &report,
        std::array::from_fn(|i| ids[i].sha256.as_str()),
        &runner.sha,
    )?;
    Ok(report)
}
