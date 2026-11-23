use crate::{
    Job, child_threads, gguf_conversion::Tool, integrity, progress::Progress, w0_libraries,
    write_json,
};
use anyhow::{Context, Result, ensure};
use brewery_worker::trace::{self, Request};
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

const TRACE_LIBRARIES: &[(&str, &str)] = &[
    (
        "libllama.so.0",
        "f4a59a07c8ebea5d02779f82e16cf23d5815a288fede9cdbf851f73b1854c005",
    ),
    (
        "libggml.so.0",
        "c51f659bc7c64ab3b675fb56cdddf1beab54c75818ace8325c7e49ff1a60e3e4",
    ),
    (
        "libggml-base.so.0",
        "c445afaddb3d37bbb8ef6d979786f66f4cae71b5597fee0feb398f9b961289b8",
    ),
    (
        "libggml-cpu.so.0",
        "fd7b911f677b93bef0b2fd31877da45bc5e78bd7197fe33df5ed4b76f75b08cd",
    ),
];

pub(crate) struct Runtime {
    private: tempfile::TempDir,
    tool: Tool,
    libs: PathBuf,
    threads: usize,
}

impl Runtime {
    pub(crate) fn stage(cwd: &Path) -> Result<Self> {
        let private = tempfile::Builder::new()
            .prefix(".trace-")
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir_in(cwd)?;
        let tool = Tool::pinned_digest(
            private.path(),
            trace::EXECUTABLE,
            trace::SHA256,
            "lanternfish",
            "Lanternfish",
        )?;
        let libs = w0_libraries::stage(
            private.path(),
            trace::LIB_DIR,
            "Lanternfish trace",
            TRACE_LIBRARIES,
        )?;
        Ok(Self {
            private,
            tool,
            libs,
            threads: child_threads(None)?.min(64),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        self.private.path()
    }

    pub(crate) fn capture(
        &self,
        model: &Path,
        prompt: &str,
        capture: &trace::Capture,
        out: &Path,
        intervention: Option<&Path>,
    ) -> Result<trace::Export> {
        let spec = self.path().join("current-spec.json");
        write_json(&spec, &serde_json::to_value(capture)?)?;
        let mut command = self.tool.command(self.path(), self.threads);
        command
            .env("LD_LIBRARY_PATH", &self.libs)
            .arg("trace")
            .arg("--model")
            .arg(model)
            .arg(format!("--prompt={prompt}"))
            .arg("--spec")
            .arg(&spec)
            .arg("--out")
            .arg(out)
            .arg("--threads")
            .arg(self.threads.to_string());
        if let Some(path) = intervention {
            command.arg("--intervention").arg(path);
        }
        self.tool
            .capture(&mut command, self.path())
            .context("Lanternfish trace failed")?;
        fs::remove_file(&spec)?;
        trace::verify(out, capture)
    }
}

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
    let runtime = Runtime::stage(cwd)?;
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let id = &ids[0];
    trace::check_model_revision(&job.options, &id.sha256)?;
    ensure!(
        id.files.len() == 1 && input.path.is_file(),
        "trace needs a single-file GGUF"
    );
    let staging = runtime.path().join("export");
    fs::create_dir(&staging)?;
    let intervention = runtime.path().join("interventions.json");
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
        let out = staging.join(i.to_string());
        id.check()?;
        let verified = runtime.capture(
            &input.path,
            prompt,
            &capture,
            &out,
            has_intervention.then_some(intervention.as_path()),
        )?;
        id.check()?;
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
