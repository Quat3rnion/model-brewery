use super::*;
use std::{
    os::fd::{AsRawFd, FromRawFd},
    thread,
    time::{Duration, Instant},
};

pub fn detach(config: &Path, path: &Path, id: &str) -> Result<()> {
    let stderr = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.join("supervisor.log"))?;
    let mut c = Command::new(env::current_exe()?);
    c.args(["supervise", id])
        .env("BREWERY_WORKER_CONFIG", config)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr);
    unsafe {
        c.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            libc::signal(libc::SIGHUP, libc::SIG_IGN);
            Ok(())
        });
    }
    c.spawn()?;
    Ok(())
}

pub fn supervise(w: &Worker, id: &str) -> Result<()> {
    // Orphan grandchildren (including a setsid() child) reparent here, not to init.
    ensure!(
        unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } == 0,
        "could not become child subreaper"
    );
    let base = w.directory(id)?;
    let job = {
        let _guard = w.state_lock()?;
        let mut r = w.require(id)?;
        if r.reply.state != "queued" {
            return Ok(());
        }
        ensure!(r.action == "start", "invalid detached action");
        let mut job = r.job.clone().context("job disappeared")?;
        ensure!(
            r.inputs.len() == job.inputs.len(),
            "unresolved worker inputs"
        );
        job.inputs = r.inputs.values().cloned().collect();
        r.boot = Some(boot_id()?);
        r.started = Some(report::now());
        r.started_ms = Some(progress::now_ms());
        r.reply = Reply::new("running", "executing on worker");
        w.put(id, &r)?;
        job
    };
    let download = job.kind == "hf_download";
    let held = if download {
        download_lock(w, id).map(Some)
    } else {
        Ok(None)
    };
    let result = match &held {
        Ok(_) => execute(w, id, &base, &job),
        Err(e) => Err(anyhow::anyhow!("{e}")),
    };
    let _guard = w.state_lock()?;
    let mut r = w.require(id)?;
    if (r.cancel || result.is_err())
        && weight_source::job(&job.kind)
        && job
            .inputs
            .first()
            .is_some_and(|i| weight_source::supports(&i.kind))
    {
        let work = base.join("work");
        // The engine may have been killed before its scratch guard ran.
        if let Ok(entries) = fs::read_dir(&work) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if name
                    .to_str()
                    .is_some_and(|s| s.starts_with(".weight-safetensors-") || s == "edited.hf")
                {
                    let _ = fs::remove_dir_all(entry.path());
                } else if matches!(
                    name.to_str(),
                    Some("edited.safetensors" | "weight-proposal.json")
                ) {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
    r.reply = if r.cancel {
        Reply::new("cancelled", "process tree stopped")
    } else {
        let result = result.and_then(|value| {
            if download {
                publish_download(w, &base)?;
            }
            Ok(value)
        });
        match result {
            Ok(value) => Reply::success(value),
            Err(e) => Reply::error(e.to_string()),
        }
    };
    w.put(id, &r)
}

fn download_lock(w: &Worker, id: &str) -> Result<File> {
    // One bounded private resume stage per dedicated host. Waiting jobs
    // still observe cancellation without needing the transfer to finish.
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(w.cfg.root.join("hf-download.lock"))?;
    loop {
        ensure!(!w.require(id)?.cancel, "cancel requested");
        match fs2::FileExt::try_lock_exclusive(&lock) {
            Ok(()) => return Ok(lock),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(300))
            }
            Err(e) => return Err(e.into()),
        }
    }
}

fn download_stage(w: &Worker) -> PathBuf {
    w.cfg.root.join("hf-download-stage")
}

fn publish_download(w: &Worker, base: &Path) -> Result<()> {
    // Both the state and download lock are held after worker verification.
    let stage = download_stage(w);
    let work = base.join("work");
    let src = stage.join("data");
    let dest = work.join("download");
    let stage_dir = File::open(&stage)?;
    let work_dir = File::open(&work)?;
    fs::rename(&src, &dest)?;
    if let Err(e) = work_dir.sync_all().and_then(|()| stage_dir.sync_all()) {
        let _ = fs::rename(&dest, &src);
        return Err(e.into());
    }
    Ok(())
}

fn execute(w: &Worker, id: &str, base: &Path, job: &Job) -> Result<Value> {
    let download = w.cfg.hf_download.as_ref();
    ensure!(
        (job.kind == "hf_download") == download.is_some(),
        "Worker is not configured for this operation"
    );
    let work = base.join("work");
    fs::create_dir(&work)?;
    let jobfile = base.join("engine-input.json");
    atomic(&jobfile, job)?;
    let output = base.join("engine-output.json");
    let stderr = base.join("engine-stderr.log");
    {
        let r = w.require(id)?;
        ensure!(!r.cancel, "cancel requested");
    }
    let mut command = if let Some(wrapper) = &w.cfg.admission {
        let mut c = Command::new(wrapper);
        c.args(["--wait", "--"]).arg(&w.cfg.engine);
        c
    } else {
        Command::new(&w.cfg.engine)
    };
    command
        .current_dir(&work)
        .envs(&w.cfg.engine_env)
        .stdin(File::open(jobfile)?)
        .stdout(File::create(&output)?)
        .stderr(File::create(stderr)?);
    if let Some(download) = download {
        let policy = base.join("hf-download-policy.json");
        atomic(
            &policy,
            &(download, download_stage(w), w.cfg.disk_reserve_bytes),
        )?;
        command
            .env_clear()
            .env("BREWERY_HF_DOWNLOAD_CONFIG", policy);
    }
    if weight_source::job(&job.kind)
        && job
            .inputs
            .first()
            .is_some_and(|i| weight_source::supports(&i.kind))
    {
        for key in [weight_source::PUBLICATION, weight_source::VERIFY_BYTES] {
            if !w.cfg.engine_env.contains_key(key) {
                command.env_remove(key);
            }
        }
        command.env(
            "BREWERY_WEIGHT_SAFETENSORS_DISK_RESERVE_BYTES",
            w.cfg.disk_reserve_bytes.to_string(),
        );
    }
    if let Some(policy) = &w.cfg.managed_settings
        && matches!(
            job.kind.as_str(),
            "benchmark" | "baseline" | "evaluate" | "generate"
        )
    {
        ensure!(
            w.cfg.pinned_publication.is_none()
                && !w
                    .cfg
                    .engine_env
                    .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
                && env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_none(),
            "source-pinned workers cannot apply operational settings"
        );
        let catalog = policy
            .load_catalog()
            .map_err(|_| anyhow::anyhow!("managed catalog unavailable"))?;
        let settings = operational::Settings::load(&w.cfg.root)
            .map_err(|_| anyhow::anyhow!("managed settings unavailable"))?;
        settings.validate(
            policy,
            &catalog,
            operational::eval_ready(policy, &w.cfg.engine_env),
        )?;
        if matches!(job.kind.as_str(), "evaluate" | "generate") {
            ensure!(
                settings.evaluation_enabled,
                "evaluation not enabled on worker"
            );
        }
        if job.kind == "baseline" {
            ensure!(
                policy.baseline_allowed,
                "baseline disabled by worker policy"
            );
        }
        if job.kind == "benchmark" {
            let config = base.join("managed-benchmark.json");
            atomic(&config, &settings.benchmark_config(&catalog, &job.options)?)?;
            command.env("BREWERY_BENCHMARK_CONFIG", config);
        } else if matches!(job.kind.as_str(), "baseline" | "evaluate") {
            let config = base.join("managed-baseline.json");
            atomic(&config, &settings.baseline_config(&job.options)?)?;
            command.env("BREWERY_BASELINE_CONFIG", config);
        }
    }
    // Redirected regular files avoid pipe backpressure and keep remote login FDs closed.
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!("Allowlisted engine or wrapper launch failed: {e:#}");
            bail!(
                "Cannot launch the worker engine or admission wrapper; check configured executables and their permissions"
            );
        }
    };
    let mut stopping = None;
    let mut child_status = None;
    let mut monitor_error = None;
    loop {
        if child_status.is_none() {
            match child.try_wait() {
                Ok(status) => child_status = status,
                Err(e) => {
                    eprintln!("Engine wait failed: {e}");
                    monitor_error = Some(
                        "Worker could not read engine exit status; check private supervisor.log",
                    );
                    stopping.get_or_insert(Instant::now());
                }
            }
        }
        if child_status.is_some() {
            reap_adopted();
        }
        match w.require(id) {
            Ok(r) if r.cancel => {
                stopping.get_or_insert(Instant::now());
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Engine state monitor failed: {e:#}");
                monitor_error =
                    Some("Worker could not read job state; check private supervisor.log");
                stopping.get_or_insert(Instant::now());
            }
        }
        let descendants = descendants(std::process::id());
        if child_status.is_some() && descendants.is_empty() {
            break;
        }
        if child_status.is_some() && stopping.is_none() {
            monitor_error = Some(
                "Engine exited with child processes still active; check private supervisor.log",
            );
            stopping = Some(Instant::now());
        }
        if let Some(start) = stopping {
            let signal = if start.elapsed() < Duration::from_secs(2) {
                libc::SIGTERM
            } else {
                libc::SIGKILL
            };
            for pid in descendants.into_iter().rev() {
                signal_descendant(pid, signal);
            }
        }
        thread::sleep(Duration::from_millis(100));
    }
    if let Some(e) = monitor_error {
        bail!("{e}");
    }
    let status = child_status.context("engine status lost")?;
    if !status.success() {
        bail!("{}", report::engine_exit(base, status));
    }
    let value: Value = match load(&output) {
        Ok(value) => value,
        Err(_) => bail!(
            "Engine output is not a single bounded JSON result; check private engine-output.json"
        ),
    };
    ensure!(
        value.is_object(),
        "Engine output is not a JSON object; check private engine-output.json"
    );
    if let Some(download) = download {
        let spec = hf_download::Spec::parse(&job.options)?;
        let mut path = work.join("download");
        if let Some(filename) = &spec.filename {
            path.push(filename);
        }
        ensure!(
            value["kind"] == "hf_download"
                && value["verified"] == true
                && value["repo_id"] == spec.repo_id
                && value["revision"] == spec.revision
                && value.get("filename").map(Value::as_str) == spec.filename.as_deref().map(Some)
                && value["commit"].as_str().is_some_and(|commit| {
                    hf_download::is_commit(commit)
                        && (!hf_download::is_commit(&spec.revision) || commit == spec.revision)
                })
                && value["artifact"]["path"].as_str().map(Path::new) == Some(path.as_path())
                && value["artifact"]["kind"] == spec.artifact_kind()
                && value["artifact"]["bytes"]
                    .as_u64()
                    .is_some_and(|n| n > 0 && n <= download.max_bytes),
            "Download engine did not return a verified managed artifact"
        );
        let data = download_stage(w).join("data");
        let mut staging = value.clone();
        let mut path = data.clone();
        if let Some(name) = spec.filename.as_deref() {
            path.push(name);
            // A selected artifact is exactly one file; do not expand split GGUF names.
            staging["artifact"]["kind"] = json!("safetensors");
        }
        staging["artifact"]["path"] = json!(path);
        let mut emit = w.progress_callback(id)?;
        if let Err(e) = verify_artifact(
            &data,
            &staging,
            &mut progress::Reporter::new(&mut emit, None),
        ) {
            eprintln!("Download staging verification failed: {e:#}");
            bail!("Download artifact failed worker validation");
        }
        return Ok(value);
    }
    if job.kind == "weight_apply" {
        let source = job.inputs.first().context("weight apply has no source")?;
        let filename = match source.kind.as_str() {
            "gguf" => "edited.gguf",
            "hgn" => {
                ensure!(
                    value["source_format"] == "hgn",
                    "weight apply did not attest HGN format"
                );
                "edited.hgn"
            }
            "hf" | "safetensors" => {
                ensure!(
                    value["source_format"] == source.kind
                        && value["proposal_sha256"].is_string()
                        && value["proposal_sha256"] == job.options["proposal_sha256"],
                    "weight apply receipt did not bind SafeTensors format and approval"
                );
                if source.kind == "hf" {
                    "edited.hf"
                } else {
                    "edited.safetensors"
                }
            }
            _ => bail!("invalid weight apply source format"),
        };
        ensure!(
            value["kind"] == "weight_apply"
                && value["verified"] == true
                && value["report"]["verified"] == true
                && value["report"]["source_sha256"] == job.options["source_sha256"]
                && value["artifact"]["kind"] == source.kind
                && value["artifact"]["path"].as_str().map(Path::new)
                    == Some(work.join(filename).as_path())
                && value["artifact"]["sha256"] == value["report"]["output_sha256"],
            "weight apply did not return a verified edited artifact"
        );
        if job.options["packed"] == true {
            ensure!(
                value["packed"] == true
                    && value["proposal_sha256"].is_string()
                    && value["proposal_sha256"] == job.options["proposal_sha256"],
                "weight apply receipt did not bind packed approval"
            );
        } else {
            ensure!(
                value.get("packed").is_none(),
                "unexpected packed apply receipt"
            );
        }
    }
    let mut emit = w.progress_callback(id)?;
    if let Err(e) = verify_artifact(&work, &value, &mut progress::Reporter::new(&mut emit, None)) {
        eprintln!("Engine artifact verification failed: {e:#}");
        bail!(
            "Engine artifact failed worker validation; check the private supervisor.log for path, size or checksum details"
        );
    }
    Ok(value)
}

fn verify_artifact(work: &Path, result: &Value, reporter: &mut progress::Notify<'_>) -> Result<()> {
    let Some(artifact) = result.get("artifact") else {
        return Ok(());
    };
    let path = PathBuf::from(artifact["path"].as_str().context("artifact path missing")?);
    let p = cache::authorized(&path, &[work.to_owned()])?;
    let kind = artifact["kind"].as_str().context("artifact kind missing")?;
    if result["kind"] == "weight_apply" && weight_source::supports(kind) {
        weight_source::inventory(&p, kind)?;
    }
    let d = cache::describe(&p, kind, work, "verify_output", reporter)?;
    ensure!(
        d.root.starts_with(work),
        "artifact bundle outside private workdir"
    );
    let sha = if d.bundle.entry == "." {
        d.bundle.digest.clone()
    } else {
        d.bundle
            .files
            .iter()
            .find(|f| f.path == d.bundle.entry)
            .context("artifact entry missing")?
            .sha256
            .clone()
    };
    ensure!(
        artifact["sha256"].as_str() == Some(&sha),
        "artifact SHA-256 does not match"
    );
    ensure!(
        artifact["bytes"].as_u64() == Some(d.bundle.bytes()?),
        "artifact size does not match"
    );
    if let Some(digest) = artifact.get("bundle_sha256") {
        ensure!(
            digest.as_str() == Some(&d.bundle.digest),
            "artifact bundle SHA-256 does not match"
        );
    }
    if result["kind"] == "weight_apply" && kind == "hf" {
        ensure!(
            artifact["bundle_sha256"] == d.bundle.digest
                && artifact["files"] == to_value(&d.bundle.files)?,
            "edited HF artifact manifest does not match"
        );
    }
    Ok(())
}

fn children(pid: u32) -> Vec<u32> {
    let Ok(tasks) = fs::read_dir(format!("/proc/{pid}/task")) else {
        return Vec::new();
    };
    let mut pids = Vec::new();
    for task in tasks.flatten() {
        let list = fs::read_to_string(task.path().join("children")).unwrap_or_default();
        pids.extend(
            list.split_ascii_whitespace()
                .filter_map(|s| s.parse::<u32>().ok()),
        );
    }
    pids
}

fn descendants(root: u32) -> Vec<u32> {
    let mut seen = BTreeSet::new();
    let mut remaining = children(root);
    let mut ordered = Vec::new();
    while let Some(pid) = remaining.pop() {
        if seen.insert(pid) {
            ordered.push(pid);
            remaining.extend(children(pid));
        }
    }
    ordered
}

fn ppid(pid: u32) -> Option<u32> {
    let s = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    s.rsplit_once(')')?
        .1
        .split_ascii_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

fn signal_descendant(pid: u32, sig: i32) {
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
    if fd < 0 {
        return;
    }
    let f = unsafe { File::from_raw_fd(fd as i32) };
    let mut p = pid;
    for _ in 0..256 {
        let Some(parent) = ppid(p) else {
            return;
        };
        if parent == std::process::id() {
            unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    f.as_raw_fd(),
                    sig,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                );
            }
            return;
        }
        if parent <= 1 {
            break;
        }
        p = parent;
    }
}

fn reap_adopted() {
    loop {
        if unsafe { libc::waitpid(-1, std::ptr::null_mut(), libc::WNOHANG) } <= 0 {
            break;
        }
    }
}
