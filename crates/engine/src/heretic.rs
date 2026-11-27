use crate::{Job, bounded};
use anyhow::{Context, Result, ensure};
use brewery_worker::heretic;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::{self, File},
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

pub fn run(job: &Job, cwd: &Path) -> Result<Value> {
    let options = heretic::Options::parse(&job.kind, &job.options)?;
    let [source, others @ ..] = job.inputs.as_slice() else {
        anyhow::bail!("Heretic needs one pinned Library SafeTensors input");
    };
    if let [assets] = others {
        ensure!(
            assets.kind == "assets" && assets.host == source.host,
            "Heretic assets must be on the base worker"
        );
        crate::integrity::check_inputs(
            others,
            &crate::progress::Progress::new(cwd, [&assets.path]),
        )?;
    } else {
        ensure!(
            others.is_empty(),
            "Heretic accepts at most one local assets directory"
        );
    }
    ensure!(
        job.prompt.is_none()
            && source.kind == "safetensors"
            && source.sha256.as_deref() == Some(&options.base_sha256),
        "Heretic Library base SHA-256 mismatch"
    );
    let runtime = PathBuf::from(env::var_os(heretic::RUNTIME).context("Missing Heretic runtime")?)
        .canonicalize()?;
    let meta = runtime.metadata()?;
    ensure!(
        meta.is_dir() && meta.mode() & 0o7777 == 0o700 && meta.uid() == unsafe { libc::geteuid() },
        "Heretic runtime must be a private worker-owned directory"
    );
    let runs = runtime.join("runs");
    ensure!(
        runs.canonicalize()? == runs && runs.is_dir(),
        "invalid Heretic run directory"
    );
    for (name, pinned) in [
        (
            "run.py",
            include_bytes!("../../../tools/heretic/run.py").as_slice(),
        ),
        (
            "uv.lock",
            include_bytes!("../../../tools/heretic/uv.lock").as_slice(),
        ),
    ] {
        let local = brewery_worker::weight_source::open_regular(&runtime.join(name))?;
        ensure!(
            Sha256::digest(bounded(local, 2 << 20)?) == Sha256::digest(pinned),
            "Heretic {name} does not match the pinned release"
        );
    }
    let bytes = source.path.metadata()?.len();
    ensure!(
        bytes <= brewery_worker::weight_source::scratch::MAX_FILE
            && brewery_worker::available(&runs)?
                >= bytes.saturating_mul(2).saturating_add(512 << 20),
        "insufficient Heretic scratch space"
    );
    let run = tempfile::Builder::new()
        .prefix("brewery-")
        .rand_bytes(6)
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(&runs)?;
    let request = cwd.join("heretic-request.json");
    brewery_worker::atomic(&request, &options)?;
    let mut command = Command::new(runtime.join("venv/bin/python"));
    command
        .env_clear()
        .args(["-IB"])
        .arg(runtime.join("run.py"))
        .arg("run")
        .arg(&runtime)
        .arg(run.path())
        .arg(&source.path)
        .arg(request)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let [assets] = others {
        command.arg(&assets.path);
    }
    if let Some(devices) = env::var_os(brewery_worker::locate_edit::CUDA_DEVICES) {
        command.env(brewery_worker::locate_edit::CUDA_DEVICES, devices);
    }
    let status = command.status()?;
    if !status.success() {
        // Do not send generated samples or commands to the Controller.
        if let Ok(log) = File::open(run.path().join("heretic.log")) {
            let _ = std::io::copy(
                &mut std::io::Read::take(log, 256 << 10),
                &mut File::create(cwd.join("heretic.log"))?,
            );
        }
        anyhow::bail!("Pinned Heretic did not complete local export");
    }
    let journal: Value = serde_json::from_slice(&bounded(
        File::open(run.path().join("journal.json"))?,
        384 << 10,
    )?)?;
    let result = json!({
        "kind":job.kind,
        "run_directory":run.path(),
        "export_sha256":journal["export_sha256"],
        "journal":journal
    });
    options.verify(&result, Some(&runs))?;
    let _ = run.keep();
    Ok(result)
}
