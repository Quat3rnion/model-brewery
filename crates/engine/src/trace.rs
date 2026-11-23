use crate::{
    Job, child_threads, gguf_conversion::Tool, integrity, progress::Progress, w0_libraries,
    write_json,
};
use anyhow::{Context, Result, ensure};
use brewery_worker::trace::{self, Request};
use serde_json::{Value, json};
use std::{fs, os::unix::fs::PermissionsExt, path::Path};

mod hgn;

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let request = Request::worker(&job.options, job.prompt.as_deref())?;
    ensure!(
        job.inputs.len() == 1 && matches!(job.inputs[0].kind.as_str(), "gguf" | trace::HGN_KIND),
        "trace needs a single GGUF or HGN input"
    );
    let input = &job.inputs[0];
    let is_hgn = input.kind == trace::HGN_KIND;
    ensure!(
        !is_hgn || request.interventions.is_empty(),
        "HGN traces do not support interventions"
    );
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
        "trace needs a resolved model digest"
    );
    let root = cwd.join(trace::JOB);
    ensure!(!root.exists(), "trace output already exists");
    let private = tempfile::Builder::new()
        .prefix(".trace-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(cwd)?;
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let id = &ids[0];
    trace::check_model_revision(&job.options, &id.sha256)?;
    ensure!(
        id.files.len() == 1 && input.path.is_file(),
        "trace needs a single-file GGUF or HGN checkpoint"
    );
    let hgn = is_hgn
        .then(|| hgn::Runtime::stage(private.path(), &id.sha256))
        .transpose()?;
    let lanternfish = if is_hgn {
        None
    } else {
        Some((
            Tool::pinned_digest(
                private.path(),
                trace::EXECUTABLE,
                trace::SHA256,
                "lanternfish",
                "Lanternfish",
            )?,
            w0_libraries::stage(
                private.path(),
                trace::LIB_DIR,
                "Lanternfish trace",
                w0_libraries::PINS,
            )?,
        ))
    };
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
        let mut command = if let Some((tool, libs)) = &lanternfish {
            let mut command = tool.command(private.path(), threads);
            command.env("LD_LIBRARY_PATH", libs).arg("trace");
            command
        } else {
            hgn.as_ref().unwrap().command(private.path(), threads)
        };
        command
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
        if let Some(runtime) = &hgn {
            runtime.capture(&mut command, private.path())?;
        } else if let Some((tool, _)) = &lanternfish {
            tool.capture(&mut command, private.path())
                .context("Lanternfish trace failed")?;
        }
        id.check()?;
        let verified = trace::verify(&out, &capture)?;
        capture.byte_cap -= verified.bytes;
        receipts.push(verified);
        phase.advance(1);
    }
    fs::rename(&staging, &root)?;
    phase.finish();
    let mut receipt = hgn.map_or_else(
        || json!({"lanternfish_commit":trace::COMMIT,"executable_sha256":trace::SHA256}),
        |runtime| runtime.receipt(),
    );
    receipt["kind"] = json!(trace::JOB);
    receipt["schema_version"] = json!(1);
    receipt["model_id"] = json!(model_id);
    receipt["model_sha256"] = json!(id.sha256);
    receipt["bytes"] = json!(receipts.iter().map(|p| p.bytes).sum::<u64>());
    receipt["prompts"] = json!(receipts);
    Ok(receipt)
}
