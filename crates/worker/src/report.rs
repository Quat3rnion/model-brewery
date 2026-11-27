use super::*;
use std::{
    io::{Read, Seek, SeekFrom},
    process::ExitStatus,
    time::{SystemTime, UNIX_EPOCH},
};

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// Read only complete lines from a bounded tail. The engine keeps writing to a
// private regular file; status never adds a pipe or holds the state lock.
fn tail(base: &Path) -> Option<String> {
    let mut f = File::open(base.join("engine-stderr.log")).ok()?;
    let start = f.metadata().ok()?.len().saturating_sub(16 * 1024);
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut b = Vec::new();
    f.take(16 * 1024).read_to_end(&mut b).ok()?;
    let b = if start > 0 {
        b.get(b.iter().position(|c| *c == b'\n')? + 1..)?
    } else {
        &b
    };
    Some(String::from_utf8_lossy(b).into_owned())
}

pub fn live(base: &Path, kind: &str, admission: bool, started: u64) -> String {
    let log = tail(base).unwrap_or_default();
    let known = log.split_inclusive('\n').rev().find_map(|line| {
        let line = line.strip_suffix('\n')?.trim_end_matches('\r');
        Some(match line {
            "brewery-engine: tracing left" | "brewery-engine: tracing reference" => {
                "Tracing reference model"
            }
            "brewery-engine: tracing right" => "Tracing second model",
            "gpu-thermal-run: waiting for shared GPU batch lock" => "Waiting for GPU batch lock",
            "gpu-thermal-run: admission-only; running command is NOT thermally paused"
            | "gpu-thermal-run: thermal gate reopened" => stage(kind),
            line if line.starts_with("gpu-thermal-run: waiting for thermal gate: ") => {
                "Waiting for thermal admission"
            }
            line if line.starts_with("resource-queue: ticket ") && line.contains(": waiting,") => {
                "Waiting for GPU reservation"
            }
            line if line.starts_with("resource-queue: ticket ")
                && line.ends_with(": granted, running") =>
            {
                stage(kind)
            }
            _ => return None,
        })
    });
    let stage = known.unwrap_or_else(|| {
        if admission && log.is_empty() {
            "Starting admission wrapper"
        } else {
            stage(kind)
        }
    });
    format!("{stage} ({}s elapsed)", now().saturating_sub(started))
}

fn stage(kind: &str) -> &'static str {
    match kind {
        "compare" => "Comparing models",
        brewery_worker::trace::JOB => "Preparing CPU trace",
        brewery_worker::behavior_direction::JOB => "Fitting CPU behavior directions",
        brewery_worker::ablation_search::SCREEN | brewery_worker::ablation_search::hgn::JOB => {
            "Preparing CPU ablation screen"
        }
        brewery_worker::scratch_eval::GGUF => "Converting private BF16 GGUF",
        brewery_worker::scratch_eval::CHECKS => "Preparing CPU scratch checks",
        brewery_worker::hgn_checks::JOB => "Checking stored HGN writers and CPU forward",
        brewery_worker::locate_edit::JOB => "Computing locate-then-edit factors",
        "inspect" => "Inspecting model",
        "plan" => "Planning conversion",
        "convert" => "Converting model",
        "verify" => "Verifying model",
        "generate" => "Generating response",
        "evaluate" => "Evaluating model",
        "baseline" => "Running baseline",
        "hf_download" => "Preparing Hugging Face download",
        brewery_worker::qsa::JOB => "Training QSA indexers",
        _ => "Running engine",
    }
}

pub fn engine_exit(base: &Path, status: ExitStatus) -> String {
    let log = tail(base).unwrap_or_default().to_ascii_lowercase();
    let action = if log.lines().any(|s| {
        (s.starts_with("cuda error:")
            || s.starts_with("ggml_cuda")
            || s.starts_with("cuda runtime error"))
            && (s.contains("out of memory") || s.contains("cudaerrormemoryallocation"))
    }) {
        "CUDA out of memory; check free GPU memory or use a smaller job"
    } else if log.contains("cuda backend unavailable:") {
        "CUDA backend unavailable; check that the worker has a visible CUDA GPU"
    } else if log.contains("rocm backend unavailable:") {
        "ROCm backend unavailable; check that the worker has a visible ROCm GPU"
    } else if log.lines().any(|s| {
        s.starts_with("gpu-thermal-run:")
            && (s.contains("permission denied") || s.contains("scope") || s.contains("lock"))
    }) || (log.contains("resource-queue:")
        && (status.code() == Some(75) || !log.contains(": granted, running")))
    {
        "GPU admission failed; check the worker admission lock and scope"
    } else if log.contains("no space left on device") || log.contains("disk quota exceeded") {
        "Worker disk space exhausted; check free space and cache"
    } else if log.contains("error while loading shared libraries:") {
        "Native tool could not load a shared library; check worker runtime dependencies"
    } else if log.lines().any(|s| s.starts_with("bf16-to-hgn: config ")) {
        "Converter rejected model configuration; check the input model and converter support"
    } else {
        "Native engine failed; inspect the private worker log"
    };
    format!("{action} ({status}; engine-stderr.log)")
}
