use anyhow::{ensure, Result};
use serde::Deserialize;
use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt, path::Path};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    phase: String,
    unit: String,
    completed: u64,
    total: u64,
    updated_at_ms: u64,
}

pub(crate) struct Counter {
    pub(crate) phase: &'static str,
    pub(crate) completed: u64,
    pub(crate) total: u64,
    stage: u8,
}

pub(crate) fn read_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let meta = file.metadata()?;
    ensure!(
        meta.is_file() && meta.len() <= limit,
        "invalid native status file"
    );
    crate::bounded(file, limit)
}

pub(crate) fn read(path: &Path, command: &str, started_ms: u64) -> Option<Counter> {
    let wire: Wire = serde_json::from_slice(&read_file(path, 4096).ok()?).ok()?;
    let (stage, phase) = match (command, wire.phase.as_str()) {
        ("convert", "prepare_rotations_values") => (1, "prepare_rotations_values"),
        ("convert", "convert_passes_values") => (2, "convert_passes_values"),
        ("verify", "verify_payloads_values") => (1, "verify_payloads_values"),
        _ => return None,
    };
    (wire.unit == "items"
        && (started_ms..=crate::progress::now_ms()).contains(&wire.updated_at_ms)
        && wire.total > 0
        && wire.completed <= wire.total)
        .then_some(Counter {
            phase,
            completed: wire.completed,
            total: wire.total,
            stage,
        })
}

pub(crate) fn follows(next: &Counter, last: Option<&Counter>) -> bool {
    last.is_none_or(|last| {
        next.stage > last.stage
            || (next.stage == last.stage
                && next.total == last.total
                && next.completed > last.completed)
    })
}
