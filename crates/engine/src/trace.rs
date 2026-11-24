use crate::{
    child_threads, gguf_conversion::Tool, integrity, progress::Progress, w0_libraries, write_json,
    Job,
};
use anyhow::{ensure, Context, Result};
use brewery_worker::trace::{self, LanternfishRuntime, Request};
use serde_json::{json, Value};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

const PROJECTION_LIBRARIES: &[(&str, &str)] = &[
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
        Self::stage_trace(cwd, LanternfishRuntime::Base)
    }

    pub(crate) fn stage_screen(cwd: &Path) -> Result<Self> {
        use brewery_worker::ablation_search as search;
        Self::stage_for(
            cwd,
            search::EXECUTABLE,
            search::SHA256,
            trace::PROJECTION_LIB_DIR,
            PROJECTION_LIBRARIES,
        )
    }

    fn stage_trace(cwd: &Path, runtime: LanternfishRuntime) -> Result<Self> {
        let (executable, lib_dir, sha, _) = runtime.pins();
        let libs = match runtime {
            LanternfishRuntime::Base => w0_libraries::PINS,
            LanternfishRuntime::Projection => PROJECTION_LIBRARIES,
            LanternfishRuntime::Routing => ROUTING_LIBS,
        };
        Self::stage_for(cwd, executable, sha, lib_dir, libs)
    }

    fn stage_for(
        cwd: &Path,
        variable: &str,
        sha256: &str,
        lib_dir: &str,
        pins: &[(&str, &str)],
    ) -> Result<Self> {
        let private = tempfile::Builder::new()
            .prefix(".trace-")
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir_in(cwd)?;
        let tool = Tool::pinned_digest(
            private.path(),
            variable,
            sha256,
            "lanternfish",
            "Lanternfish",
        )?;
        let libs = w0_libraries::stage(private.path(), lib_dir, "Lanternfish trace", pins)?;
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

pub(crate) mod hgn;

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let request = Request::worker(&job.options, job.prompt.as_deref())?;
    ensure!(
        job.inputs.len() == 1
            && (job.inputs[0].kind == "gguf" || trace::is_hgn(&job.inputs[0].kind)),
        "trace needs a single GGUF or HGN input"
    );
    let input = &job.inputs[0];
    let is_hgn = trace::is_hgn(&input.kind);
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
    let selection = if is_hgn {
        None
    } else {
        Some(request.lanternfish_runtime()?)
    };
    let runtime = selection
        .map(|r| Runtime::stage_trace(cwd, r))
        .transpose()?;
    let hgn_dir = if is_hgn {
        Some(
            tempfile::Builder::new()
                .prefix(".trace-hgn-")
                .permissions(fs::Permissions::from_mode(0o700))
                .tempdir_in(cwd)?,
        )
    } else {
        None
    };
    let work = if let Some(hgn_dir) = &hgn_dir {
        hgn_dir.path()
    } else {
        runtime.as_ref().context("missing trace runtime")?.path()
    };
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let id = &ids[0];
    trace::check_model_revision(&job.options, &id.sha256)?;
    ensure!(
        id.files.len() == 1 && input.path.is_file(),
        "trace needs a single-file GGUF or HGN checkpoint"
    );
    let hgn = is_hgn
        .then(|| hgn::Runtime::stage(work, &id.sha256))
        .transpose()?;
    let staging = work.join("export");
    fs::create_dir(&staging)?;
    let intervention = work.join("interventions.json");
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
        let verified = if let Some(runtime) = &hgn {
            runtime.capture_projected(
                work,
                &input.path,
                prompt,
                &capture,
                &out,
                request.hgn_projection.as_ref(),
            )?
        } else {
            runtime
                .as_ref()
                .context("missing Lanternfish runtime")?
                .capture(
                    &input.path,
                    prompt,
                    &capture,
                    &out,
                    has_intervention.then_some(intervention.as_path()),
                )?
        };
        id.check()?;
        trace::verify_interventions(&out, &verified, &request.interventions)?;
        if is_hgn {
            trace::verify_hgn_projection(&out, &verified, request.hgn_projection.as_ref())?;
        }
        capture.byte_cap -= verified.bytes;
        receipts.push(verified);
        phase.advance(1);
    }
    fs::rename(&staging, &root)?;
    phase.finish();
    let mut receipt = if let Some(runtime) = hgn {
        runtime.receipt()
    } else {
        let (_, _, sha, commit) = selection.context("missing Lanternfish pin")?.pins();
        json!({"lanternfish_commit":commit,"executable_sha256":sha})
    };
    receipt["kind"] = json!(trace::JOB);
    receipt["schema_version"] = json!(1);
    receipt["model_id"] = json!(model_id);
    receipt["model_sha256"] = json!(id.sha256);
    receipt["bytes"] = json!(receipts.iter().map(|p| p.bytes).sum::<u64>());
    receipt["prompts"] = json!(receipts);
    Ok(receipt)
}
