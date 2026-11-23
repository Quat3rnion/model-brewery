use crate::{
    Job, child_threads, gguf_conversion::Tool, integrity, progress::Progress, w0_libraries,
    write_json,
};
use anyhow::{Context, Result, ensure};
use brewery_worker::trace::{self, Request};
use serde_json::{Value, json};
use std::{fs, os::unix::fs::PermissionsExt, path::Path};

const ROUTING_LIBS: &[(&str, &str)] = &[
    (
        "libllama.so.0",
        "dc933d3ad896057d3f824dd0867242456a87d62d29693076cc48448630348cd4",
    ),
    (
        "libggml.so.0",
        "5dabaac15aaaeafbfe8d2b6d48d17c7591c99e7c45f0ba1f37551c84326900a1",
    ),
    (
        "libggml-base.so.0",
        "3f8e5b01616fdaf8006495e8f600b365cd9336bbea86f95b4f2bcc2b82a2933f",
    ),
    (
        "libggml-cpu.so.0",
        "80a3aa5789309cb75a28bb441190410fada3407b93176aadb1ed37a4feec25dc",
    ),
];

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let request = Request::worker(&job.options, job.prompt.as_deref())?;
    ensure!(
        job.inputs.len() == 1 && job.inputs[0].kind == "gguf",
        "trace needs a GGUF input"
    );
    let input = &job.inputs[0];
    let model_id = input
        .id
        .as_deref()
        .filter(|s| !s.is_empty())
        .context("trace needs a Library model ID")?;
    ensure!(
        input
            .sha256
            .as_deref()
            .is_some_and(brewery_worker::is_digest),
        "trace needs a resolved GGUF digest"
    );
    let root = cwd.join(trace::JOB);
    ensure!(!root.exists(), "trace output already exists");
    let private = tempfile::Builder::new()
        .prefix(".trace-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(cwd)?;
    let (executable, sha, commit, lib_dir, pins) = if request.routing() {
        (
            trace::ROUTING_EXECUTABLE,
            trace::ROUTING_SHA256,
            trace::ROUTING_COMMIT,
            trace::ROUTING_LIB_DIR,
            ROUTING_LIBS,
        )
    } else {
        (
            trace::EXECUTABLE,
            trace::SHA256,
            trace::COMMIT,
            trace::LIB_DIR,
            w0_libraries::PINS,
        )
    };
    let tool = Tool::pinned_digest(
        private.path(),
        executable,
        sha,
        "lanternfish",
        "Lanternfish",
    )?;
    let libs = w0_libraries::stage(private.path(), lib_dir, "Lanternfish trace", pins)?;
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let id = &ids[0];
    trace::check_model_revision(&job.options, &id.sha256)?;
    ensure!(
        id.files.len() == 1 && input.path.is_file(),
        "trace needs a single-file GGUF"
    );
    let threads = child_threads(None)?.min(64);
    let staging = private.path().join("export");
    fs::create_dir(&staging)?;
    let intervention = private.path().join("interventions.json");
    let has_intervention = !request.interventions.is_empty();
    if has_intervention {
        write_json(
            &intervention,
            &serde_json::to_value(&request.interventions)?,
        )?;
    }
    let mut capture = request.capture;
    let mut receipts = Vec::new();
    let mut phase = progress.start("trace_prompts", "items", Some(request.prompts.len() as u64));
    for (i, prompt) in request.prompts.iter().enumerate() {
        ensure!(capture.byte_cap > 0, "trace byte cap exhausted");
        let spec = private.path().join(format!("spec-{i}.json"));
        write_json(&spec, &serde_json::to_value(&capture)?)?;
        let out = staging.join(i.to_string());
        let mut command = tool.command(private.path(), threads);
        command
            .env("LD_LIBRARY_PATH", &libs)
            .arg("trace")
            .arg("--model")
            .arg(&input.path)
            .arg(format!("--prompt={prompt}"))
            .arg("--spec")
            .arg(&spec)
            .arg("--out")
            .arg(&out)
            .arg("--threads")
            .arg(threads.to_string());
        if has_intervention {
            command.arg("--intervention").arg(&intervention);
        }
        id.check()?;
        tool.capture(&mut command, private.path())
            .context("Lanternfish trace failed")?;
        id.check()?;
        let verified = trace::verify(&out, &capture)?;
        trace::verify_interventions(&out, &verified, &request.interventions)?;
        capture.byte_cap -= verified.bytes;
        receipts.push(verified);
        phase.advance(1);
    }
    fs::rename(&staging, &root)?;
    phase.finish();
    Ok(
        json!({"kind":trace::JOB,"schema_version":1,"model_id":model_id,
         "model_sha256":id.sha256,"lanternfish_commit":commit,"executable_sha256":sha,
        "bytes":receipts.iter().map(|p| p.bytes).sum::<u64>(),"prompts":receipts}),
    )
}
