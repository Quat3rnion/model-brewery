use crate::{
    child_threads, gguf_conversion::Tool, integrity, progress::Progress, w0_libraries, write_json,
    Job,
};
use anyhow::{ensure, Context, Result};
use brewery_worker::{is_digest, locate_edit, weight_source::open_regular};
use serde_json::{json, Value};
use std::{env, fs, io::BufReader, os::unix::fs::PermissionsExt, path::Path};

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let request = locate_edit::Request::parse(&job.options)?;
    let backend = request.edit.backend;
    let pin = backend.pin();
    if backend.is_gpu() {
        ensure!(
            env::var(locate_edit::GPU_GATE).as_deref() == Ok("1"),
            "{} edit requires GPU load permission",
            backend.label()
        );
    }
    ensure!(
        job.prompt.is_none()
            && job
                .inputs
                .iter()
                .map(|i| i.kind.as_str())
                .eq(locate_edit::INPUTS)
            && job
                .inputs
                .iter()
                .all(|i| i.sha256.as_deref().is_some_and(is_digest)),
        "locate-then-edit needs pinned private GGUF and SafeTensors inputs"
    );
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    ensure!(
        ids.iter().all(|id| id.files.len() == 1),
        "expected single-file inputs"
    );
    let model = &job.inputs[0].path;
    let work = tempfile::Builder::new()
        .prefix(".locate-edit-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(cwd)?;
    let tool = Tool::pinned_digest(
        work.path(),
        pin.executable,
        pin.sha256,
        "lanternfish",
        "Lanternfish",
    )?;
    let libs = w0_libraries::stage(work.path(), pin.lib_dir, "Lanternfish edit", pin.libraries)?;
    let cache = cwd
        .ancestors()
        .nth(3)
        .context("missing private worker root")?
        .join("edit-moments");
    fs::create_dir_all(&cache)?;
    ensure!(
        integrity::canonical(&cache)?.is_dir(),
        "invalid moment cache"
    );
    let request_file = work.path().join("request.json");
    let mut native_request = serde_json::to_value(&request)?;
    // Lanternfish selects the backend by flag, not in its strict JSON request.
    native_request.as_object_mut().unwrap().remove("backend");
    write_json(&request_file, &native_request)?;
    let out = work.path().join("factors.json");
    let threads = child_threads(None)?.min(16);
    let mut command = tool.command(work.path(), threads);
    command
        .env("LD_LIBRARY_PATH", &libs)
        .arg("locate-edit")
        .arg("--model")
        .arg(model)
        .arg("--request")
        .arg(request_file)
        .arg("--out")
        .arg(&out)
        .arg("--cache")
        .arg(cache)
        .arg("--threads")
        .arg(threads.to_string());
    if backend.is_gpu() {
        command.args(["--backend", backend.key()]);
        for &key in backend.visibility_vars() {
            match env::var_os(key) {
                Some(devices) => command.env(key, devices),
                None => command.env_remove(key),
            };
        }
    }
    if backend == locate_edit::Backend::Cuda {
        command
            .env("NVIDIA_TF32_OVERRIDE", "0")
            .env("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32");
    } else if backend == locate_edit::Backend::Rocm {
        // HIP also recognizes this CUDA alias. An empty value can hide all GPUs.
        command.env_remove(locate_edit::CUDA_DEVICES);
    }
    ids.iter().try_for_each(integrity::Identity::check)?;
    let mut phase = progress.start("locate_edit", "items", Some(1));
    tool.capture(&mut command, work.path())
        .context("Lanternfish locate-then-edit failed")?;
    ids.iter().try_for_each(integrity::Identity::check)?;
    let file = open_regular(&out)?;
    ensure!(
        file.metadata()?.len() <= brewery_worker::LIMIT,
        "edit factor export exceeds byte limit"
    );
    let factors: locate_edit::Factors = serde_json::from_reader(BufReader::new(file))?;
    factors.verify(&request, &ids[0].sha256)?;
    phase.advance(1);
    phase.finish();
    Ok(json!({
        "kind":locate_edit::JOB,"parent_sha256":ids[1].sha256,
        "executable_sha256":pin.sha256,"lanternfish_commit":pin.commit,
        "factors":factors
    }))
}
