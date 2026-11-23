use crate::{child_threads, gguf_conversion::Tool, integrity, progress::Progress, write_json, Job};
use anyhow::{ensure, Context, Result};
use brewery_worker::trace::{self, Request};
use serde_json::{json, Value};
use std::{fs, path::Path};

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
    let private = tempfile::Builder::new().prefix(".trace-").tempdir_in(cwd)?;
    let tool = Tool::pinned_digest(
        private.path(),
        trace::EXECUTABLE,
        trace::SHA256,
        "lanternfish",
        "Lanternfish",
    )?;
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
        write_json(&intervention, &serde_json::to_value(&request.interventions)?)?;
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
        capture.byte_cap -= verified.bytes;
        receipts.push(verified);
        phase.advance(1);
    }
    fs::rename(&staging, &root)?;
    phase.finish();
    Ok(
        json!({"kind":trace::JOB,"schema_version":1,"model_id":model_id,
        "model_sha256":id.sha256,"lanternfish_commit":trace::COMMIT,"executable_sha256":trace::SHA256,
        "bytes":receipts.iter().map(|p| p.bytes).sum::<u64>(),"prompts":receipts}),
    )
}
