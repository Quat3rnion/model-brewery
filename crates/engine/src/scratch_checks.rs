use crate::{
    child_budget, child_threads, gguf_conversion::Tool, integrity, native_json_admitted,
    progress::Progress, w0_libraries, Job,
};
use anyhow::{ensure, Context, Result};
use brewery_worker::{is_digest, scratch_eval};
use serde_json::Value;
use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
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

#[cfg(test)]
mod tests;

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    run_with_libraries(job, cwd, progress, w0_libraries::PINS)
}

fn run_with_libraries(
    job: &Job,
    cwd: &Path,
    progress: &Progress,
    libraries: &[(&str, &str)],
) -> Result<Value> {
    let behavior = scratch_eval::check_options(&job.options)?;
    ensure!(
        job.prompt.is_none()
            && job
                .inputs
                .iter()
                .map(|i| i.kind.as_str())
                .eq(scratch_eval::CHECK_INPUTS)
            && job
                .inputs
                .iter()
                .all(|i| i.sha256.as_deref().is_some_and(is_digest) && i.path.is_file()),
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
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(cwd)?;
    let runner = Tool::pinned(
        private.path(),
        scratch_eval::RUNNER,
        scratch_eval::RUNNER_SHA,
        "scratch-check-logits",
        "scratch checks runner",
    )?;
    let libs = w0_libraries::stage(
        private.path(),
        scratch_eval::RUNNER_LIB_DIR,
        "scratch checks",
        libraries,
    )?;
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    ensure!(
        ids.iter().all(|id| id.files.len() == 1),
        "scratch checks need single-file models"
    );
    let threads = child_threads(None)?.min(16);
    let mut cmd = Command::new(Path::new("/usr/bin/python3").canonicalize()?);
    cmd.env_clear()
        // Python passes this search path to each native runner invocation.
        .env("LD_LIBRARY_PATH", &libs)
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
    if let Some(config) = &behavior {
        let path = configured(scratch_eval::CONTRAST_CACHE)?;
        ensure!(path.is_dir(), "invalid contrast cache directory");
        cmd.arg("--contrast-cache")
            .arg(path)
            .arg("--behavior-config")
            .arg(serde_json::to_string(config)?);
    }
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
        behavior.as_ref(),
    )?;
    Ok(report)
}
