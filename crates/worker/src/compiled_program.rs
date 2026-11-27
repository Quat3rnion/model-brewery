//! Local CPU program compiler and reference runner. The executable paths are
//! operator settings, never request parameters.
use crate::{Result, decode_limited};
use anyhow::{Context, ensure};
use serde::de::DeserializeOwned;
use std::{
    ffi::OsStr,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

pub const SCRIPT: &str = "BREWERY_COMPILED_PROGRAM_SCRIPT";
pub const PYTHON: &str = "BREWERY_COMPILED_PROGRAM_PYTHON";

pub fn run<T: DeserializeOwned>(python: &Path, script: &Path, args: &[&OsStr]) -> Result<T> {
    ensure!(
        python.is_absolute() && script.is_absolute(),
        "compiler paths must be absolute"
    );
    let mut child = Command::new(python)
        .args(["-B", "-s"])
        .arg(script)
        .args(args)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("JAX_PLATFORMS", "cpu")
        .env("PYTHONHASHSEED", "0")
        .env("TMPDIR", "/arboretum/tmp")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("launch compiled program runtime")?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                ensure!(
                    status.success(),
                    "compiled program runtime failed: {status}"
                );
                return decode_limited(child.stdout.take().unwrap(), 4096);
            }
            Ok(None) if start.elapsed() < Duration::from_secs(30) => {
                thread::sleep(Duration::from_millis(10));
            }
            result => {
                let _ = child.kill();
                let _ = child.wait();
                result?;
                anyhow::bail!("compiled program runtime timed out");
            }
        }
    }
}
