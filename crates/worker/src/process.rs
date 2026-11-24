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
        if !w.conversion_slot(&job)? {
            return Ok(());
        }
        if w.resource_enabled() && resources::engine_job(&job.kind) {
            let bounds = w.cfg.resource_bounds.unwrap_or_default();
            let ram = job
                .resources
                .min_free_ram_bytes
                .unwrap_or(0)
                .max(bounds.min_free_ram_bytes.map_or(0, |b| b.min));
            // Preserve legacy launch behavior unless a reserve policy applies.
            let disk = if job.resources.disk_reserve_bytes.is_some()
                || bounds.disk_reserve_bytes.is_some()
            {
                w.resource_floor(&r)
            } else {
                0
            };
            if let Err(error) =
                resources::ram_check(ram).and_then(|_| resources::disk_check(&w.cfg.root, disk))
            {
                r.reply = Reply::error(error);
                return w.put(id, &r);
            }
        }
        job.inputs = r.inputs.values().cloned().collect();
        r.boot = Some(boot_id()?);
        r.started = Some(report::now());
        r.started_ms = Some(progress::now_ms());
        r.reply = Reply::new("running", "executing on worker");
        w.put(id, &r)?;
        job
    };
    let download = job.kind == "hf_download";
    let hy4_encode = w.cfg.hy4_plan.as_ref().is_some_and(|p| p.encode.is_some());
    let held = if hy4_encode {
        hy4_encode_slot(w).map(Some)
    } else if download {
        download_lock(w, id).map(Some)
    } else {
        Ok(None)
    };
    let mut created_work = false;
    let result = match &held {
        Ok(_) => execute(w, id, &base, &job, &mut created_work),
        Err(e) => Err(anyhow::anyhow!("{e}")),
    };
    let mut _trace_lock = None;
    let result = result.and_then(|value| {
        if job.kind == trace::JOB {
            _trace_lock = Some(lock(&w.cfg.root.join("trace.lock"))?);
        }
        Ok(value)
    });
    let _guard = w.state_lock()?;
    let mut r = w.require(id)?;
    // The state lock orders cancellation against the durable terminal reply.
    // A failed create_dir never gives this run ownership of existing output.
    if job.kind == brewery_worker::exl3_layer::JOB
        && created_work
        && (r.cancel || result.is_err())
        && let Err(error) =
            fs::remove_dir_all(base.join("work")).and_then(|()| File::open(&base)?.sync_all())
    {
        eprintln!("EXL3 work directory cleanup failed: {error}");
        r.reply = Reply::error("EXL3 work directory cleanup failed");
        return w.put(id, &r);
    }
    if hy4_encode && held.is_ok() && (r.cancel || result.is_err()) {
        for name in [hy4::FILE, hy4::RECEIPT, ".gate.partial"] {
            if let Err(e) = fs::remove_file(base.join("work").join(name))
                && e.kind() != std::io::ErrorKind::NotFound
            {
                eprintln!("Hy4 scratch cleanup failed: {e}");
            }
        }
    }
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
    let result = result.and_then(|value| {
        if !r.cancel {
            if download {
                publish_download(w, &base)?;
            } else if job.kind == trace::JOB {
                let request = trace::Request::worker(&job.options, job.prompt.as_deref())?;
                publish_trace(w, id, &base, request.retain_jobs)?;
            }
        }
        Ok(value)
    });
    if (matches!(
        job.kind.as_str(),
        trace::JOB | behavior_direction::JOB | sensitivity::JOB | circuit::JOB | damage::JOB
    ) || scratch_eval::private_output(&job.kind))
        && created_work
        && (r.cancel || result.is_err())
        && let Err(error) =
            fs::remove_dir_all(base.join("work")).and_then(|()| File::open(&base)?.sync_all())
    {
        eprintln!("Private work directory cleanup failed: {error}");
        r.reply = Reply::error("Private work directory cleanup failed");
        return w.put(id, &r);
    }
    r.reply = if r.cancel {
        Reply::new("cancelled", "process tree stopped")
    } else {
        match result {
            Ok(value) => Reply::success(value),
            Err(e) => Reply::error(e.to_string()),
        }
    };
    w.put(id, &r)
}

fn publish_trace(w: &Worker, id: &str, base: &Path, retain: usize) -> Result<()> {
    let root = w.cfg.root.join("trace-results");
    fs::create_dir_all(&root)?;
    ensure!(fs::symlink_metadata(&root)?.is_dir(), "invalid trace store");
    let dest = root.join(id);
    ensure!(!dest.exists(), "trace result already exists");
    let source = base.join("work/trace");
    let mut older = Vec::new();
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && entry.file_name().to_str().is_some_and(|s| token(s).is_ok())
        {
            older.push((entry.metadata()?.modified()?, entry.path()));
        }
    }
    older.sort_unstable();
    for prompt in fs::read_dir(&source)? {
        let dir = prompt?.path();
        for file in fs::read_dir(&dir)? {
            File::open(file?.path())?.sync_all()?;
        }
        File::open(&dir)?.sync_all()?;
    }
    File::open(&source)?.set_modified(std::time::SystemTime::now())?;
    File::open(&source)?.sync_all()?;
    fs::rename(&source, &dest)?;
    let prune = (|| -> Result<()> {
        let remove = (older.len() + 1).saturating_sub(retain);
        for (_, old) in older.into_iter().take(remove) {
            fs::remove_dir_all(old)?;
        }
        File::open(&root)?.sync_all()?;
        File::open(&w.cfg.root)?.sync_all()?;
        Ok(())
    })();
    if prune.is_err() {
        let _ = fs::rename(&dest, &source);
    }
    prune
}

fn hy4_encode_slot(w: &Worker) -> Result<File> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(w.cfg.root.join("hy4-encode.lock"))?;
    fs2::FileExt::try_lock_exclusive(&file).context("Another Hy4 encode is active")?;
    for entry in fs::read_dir(&w.jobs)? {
        let work = entry?.path().join("work");
        ensure!(
            [hy4::FILE, hy4::RECEIPT, ".gate.partial"]
                .iter()
                .all(|name| !work.join(name).exists()),
            "An existing Hy4 result or partial must be retired by the operator"
        );
    }
    Ok(file)
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

fn execute(w: &Worker, id: &str, base: &Path, job: &Job, created_work: &mut bool) -> Result<Value> {
    let download = w.cfg.hf_download.as_ref();
    ensure!(
        (job.kind == "hf_download") == download.is_some(),
        "Worker is not configured for this operation"
    );
    let scratch_job =
        scratch_eval::private_output(&job.kind) || job.inputs.iter().any(scratch_eval::granted);
    ensure!(
        scratch_job || w.cfg.scratch_access.is_none(),
        "scratch worker config is scoped to one job"
    );
    if scratch_job {
        let access = w
            .cfg
            .scratch_access
            .as_ref()
            .filter(|a| a.job_id == id && a.permits("start", job))
            .context("Scratch access does not match the job")?;
        for (index, input) in job.inputs.iter().enumerate() {
            if let Some(pin) = access.paths.get(&input.path) {
                ensure!(
                    scratch_eval::granted(input)
                        && input.extra.get("sha256").and_then(Value::as_str) == Some(pin),
                    "Scratch input was not resolved with its current pin"
                );
                let mut emit = w.progress_callback(id)?;
                let bundle = w.scratch_bundle(
                    &input.path,
                    pin,
                    &mut progress::Reporter::new(&mut emit, Some((index, job.inputs.len()))),
                )?;
                ensure!(
                    input.extra.get("bundle_sha256").and_then(Value::as_str)
                        == Some(&bundle.digest),
                    "Scratch input manifest changed"
                );
            }
        }
    }
    // Resolve and supervise are different processes. Do not hand off a newly
    // replaced grant root just because the earlier resolution succeeded.
    let _source_guard = if !w.cfg.source_grants.is_empty()
        && let Some(input) = job.inputs.first()
        && w.source(&input.path).is_err()
    {
        w.editor_grant("start", job, 0, input)?
    } else {
        None
    };
    let work = base.join("work");
    if job.kind == trace::JOB {
        let request = trace::Request::worker(&job.options, job.prompt.as_deref())?;
        resources::disk_check(
            &w.cfg.root,
            w.disk_floor(job.resources.disk_reserve_bytes)
                .checked_add(request.capture.byte_cap + (4 << 20))
                .context("trace disk reserve overflow")?,
        )?;
    }
    let gradient_backend = if matches!(job.kind.as_str(), sensitivity::JOB | circuit::JOB) {
        let backend = if job.kind == circuit::JOB {
            locate_edit::Backend::Cpu
        } else {
            sensitivity::Request::resolved(&job.options)?.0.backend
        };
        ensure!(
            sensitivity::configured(&w.cfg.engine_env, backend, w.cfg.admission.as_deref()),
            "pinned Lanternfish gradient backend is unavailable"
        );
        // The native runtime may expand BF16 weights to a private F32 GGUF.
        let model_bytes = weight_source::open_regular(&job.inputs[0].path)?
            .metadata()?
            .len();
        resources::disk_check(
            &w.cfg.root,
            w.disk_floor(job.resources.disk_reserve_bytes)
                .saturating_add(sensitivity::RUNTIME_RESERVE + sensitivity::MAX_GRADIENT_BYTES)
                .saturating_add(model_bytes.saturating_mul(2)),
        )?;
        Some(backend)
    } else if job.kind == damage::JOB {
        let req = damage::Request::parse(&job.options)?;
        ensure!(
            damage::configured(&w.cfg.engine_env, &req, w.cfg.admission.as_deref()),
            "pinned damage map backend is unavailable"
        );
        let [a, b] = job.inputs.as_slice() else {
            bail!("missing damage map sources")
        };
        let size = |input: &Input| -> Result<u64> {
            Ok(weight_source::open_regular(&input.path)?.metadata()?.len())
        };
        let (a, b) = (size(a)?, size(b)?);
        ensure!(
            [a, b]
                .into_iter()
                .all(|size| size <= damage::MAX_MODEL_BYTES),
            "damage map input exceeds disk budget"
        );
        // Each F32 model or corpus bundle can have one extra untied head.
        let expanded = a * 2 + damage::MAX_GRADIENT_BYTES;
        let scratch = if req.objective == damage::Objective::CorpusNll {
            expanded * 2
        } else {
            (expanded + damage::MAX_GRADIENT_BYTES).max(b + damage::MAX_GRADIENT_BYTES / 2)
        } + (128 << 20)
            + req.backend.runtime_stage_reserve();
        resources::disk_check(
            &w.cfg.root,
            w.disk_floor(job.resources.disk_reserve_bytes)
                .saturating_add(scratch),
        )?;
        Some(req.backend)
    } else {
        None
    };
    let hgn_direction = job.kind == behavior_direction::JOB && trace::is_hgn(&job.inputs[0].kind);
    if job.kind == behavior_direction::JOB {
        resources::disk_check(
            &w.cfg.root,
            w.disk_floor(job.resources.disk_reserve_bytes)
                .checked_add(trace::HARD_BYTE_CAP + (16 << 20))
                .context("direction disk reserve overflow")?,
        )?;
    }
    if job.kind == ablation_search::SCREEN {
        resources::disk_check(
            &w.cfg.root,
            w.disk_floor(job.resources.disk_reserve_bytes)
                .saturating_add(ablation_search::RESERVE),
        )?;
    }
    let edit_backend = if job.kind == locate_edit::JOB {
        let backend = locate_edit::Request::parse(&job.options)?.edit.backend;
        let bytes = weight_source::open_regular(&job.inputs[0].path)?
            .metadata()?
            .len();
        resources::disk_check(
            &w.cfg.root,
            w.disk_floor(job.resources.disk_reserve_bytes)
                .saturating_add(bytes.saturating_mul(2))
                .saturating_add(512 << 20)
                .saturating_add(backend.runtime_stage_reserve()),
        )?;
        backend
    } else {
        locate_edit::Backend::Cpu
    };
    let gpu_edit = edit_backend.is_gpu() || heretic::job(&job.kind);
    if gpu_edit {
        ensure!(
            scratch_job
                && w.cfg.admission.is_some()
                && w.cfg
                    .engine_env
                    .get(locate_edit::GPU_GATE)
                    .is_some_and(|v| v == "1"),
            "This edit requires GPU load permission and a GPU admission wrapper"
        );
    }
    fs::create_dir(&work)?;
    *created_work = true;
    let jobfile = base.join("engine-input.json");
    atomic(&jobfile, job)?;
    let output = base.join("engine-output.json");
    let stderr = base.join("engine-stderr.log");
    {
        let r = w.require(id)?;
        ensure!(!r.cancel, "cancel requested");
    }
    let mut command = if job.kind == exl3_layer::JOB {
        let spec = exl3_layer::Spec::parse(&job.options)?;
        if spec.backend == exl3_layer::Backend::Cpu {
            Command::new(&w.cfg.engine)
        } else {
            let queue = w
                .cfg
                .exl3_gpu_queue
                .as_ref()
                .context("EXL3 GPU admission is not configured on this worker")?;
            // The queue starts the engine only after acquiring the shared GPU.
            // Do not add the generic admission wrapper inside or outside this run.
            let mut c = Command::new(queue);
            c.args(["run", "--resource", "gpu", "--exclusive", "--flock", "gpu"])
                .arg("--holder")
                .arg(format!("brewery-exl3-{}-{id}", std::process::id()))
                .arg("--purpose")
                .arg(format!("EXL3 {} layer verification", spec.backend.name()))
                .args(["--wait", "30", "--max-hold", "120", "--"])
                .arg(&w.cfg.engine);
            c
        }
    } else if let Some(plan) = &w.cfg.hy4_plan {
        ensure!(
            plan.validate("start", job),
            "Invalid pinned Hy4 plan request"
        );
        let mut c = Command::new("/usr/bin/python3");
        c.env_clear()
            .args(["-I", "-B"])
            .arg(&w.cfg.engine)
            .arg(&plan.host)
            .arg(&plan.source)
            .arg(&plan.halo_tools);
        if let Some(encode) = &plan.encode {
            c.args([&encode.control, &encode.preflight, &encode.quantizer])
                .env("BREWERY_HY4_PROGRESS", base.join(hy4::PROGRESS_FILE))
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1");
        }
        c
    } else if let Some(wrapper) = w.cfg.admission.as_ref().filter(|_| {
        gradient_backend.map_or(
            gpu_edit
                || (!matches!(job.kind.as_str(), trace::JOB | behavior_direction::JOB)
                    && !scratch_job),
            |b| b.is_gpu(),
        )
    }) {
        let mut c = Command::new(wrapper);
        c.args(["--wait", "--"]).arg(&w.cfg.engine);
        c
    } else {
        Command::new(&w.cfg.engine)
    };
    let hgn_trace = job.kind == trace::JOB && trace::is_hgn(&job.inputs[0].kind);
    if !hgn_trace
        && (scratch_job
            || gradient_backend.is_some()
            || matches!(job.kind.as_str(), trace::JOB | behavior_direction::JOB))
    {
        let trace_keys = if job.kind == trace::JOB {
            let (executable, libraries, _, _) =
                trace::Request::worker(&job.options, job.prompt.as_deref())?
                    .lanternfish_runtime()?
                    .pins();
            [executable, libraries]
        } else {
            [trace::EXECUTABLE, trace::LIB_DIR]
        };
        let keys: &[&str] = match job.kind.as_str() {
            scratch_eval::GGUF => &[
                "BREWERY_GGUF_SAFETENSORS",
                "BREWERY_GGUF_QUANTIZER",
                "BREWERY_GGUF_HF_EXPORTER",
                "BREWERY_GGUF_PYTHON",
                "BREWERY_LLAMA_CPP",
                "BREWERY_CONVERTER_DISK_RESERVE_BYTES",
            ],
            scratch_eval::CHECKS => &[
                scratch_eval::SCRIPT,
                scratch_eval::RUNNER,
                scratch_eval::RUNNER_SHA,
                scratch_eval::RUNNER_LIB_DIR,
                scratch_eval::CACHE,
                scratch_eval::CONTRAST_CACHE,
            ],
            locate_edit::JOB => &[],
            sensitivity::JOB | damage::JOB => &[
                sensitivity::PIN,
                sensitivity::MANIFEST_SHA,
                locate_edit::GPU_GATE,
            ],
            circuit::JOB => &[sensitivity::PIN, sensitivity::MANIFEST_SHA],
            kind if heretic::job(kind) => &[heretic::RUNTIME],
            behavior_direction::JOB if hgn_direction => &[
                trace::HGN_PYTHON,
                trace::HGN_PYTHON_SHA256,
                trace::HGN_ASSETS,
                behavior_direction::CACHE,
            ],
            behavior_direction::JOB => {
                &[trace::EXECUTABLE, trace::LIB_DIR, behavior_direction::CACHE]
            }
            ablation_search::SCREEN => &[
                ablation_search::EXECUTABLE,
                trace::PROJECTION_LIB_DIR,
                behavior_direction::CACHE,
            ],
            _ => &trace_keys,
        };
        command.env_clear().envs(
            keys.iter()
                .copied()
                .chain(
                    (job.kind == locate_edit::JOB)
                        .then(|| edit_backend.env_keys())
                        .into_iter()
                        .flatten(),
                )
                .chain(["BREWERY_CPU_THREADS"])
                .chain(
                    gradient_backend
                        .into_iter()
                        .flat_map(|b| b.visibility_vars().iter().copied()),
                )
                .chain(heretic::job(&job.kind).then_some(locate_edit::CUDA_DEVICES))
                .filter_map(|key| w.cfg.engine_env.get(key).map(|value| (key, value))),
        );
        if gpu_edit || gradient_backend.is_some_and(locate_edit::Backend::is_gpu) {
            command.env("PATH", "/usr/bin:/bin");
        }
    } else if hgn_trace {
        command.env_clear().envs(
            [
                trace::HGN_PYTHON,
                trace::HGN_PYTHON_SHA256,
                trace::HGN_ASSETS,
                "BREWERY_CPU_THREADS",
            ]
            .into_iter()
            .filter_map(|key| w.cfg.engine_env.get(key).map(|value| (key, value))),
        );
    } else {
        command.envs(&w.cfg.engine_env);
    }
    command
        .current_dir(&work)
        .stdin(File::open(jobfile)?)
        .stdout(File::create(&output)?)
        .stderr(File::create(stderr)?);
    if w.resource_enabled() && resources::engine_job(&job.kind) {
        let bounds = w.cfg.resource_bounds.unwrap_or_default();
        if let Some(requested) = job
            .resources
            .cpu_threads
            .or(bounds.cpu_threads.map(|range| range.max))
        {
            let configured = w
                .cfg
                .engine_env
                .get("BREWERY_CPU_THREADS")
                .cloned()
                .or_else(|| {
                    if scratch_job {
                        None
                    } else {
                        env::var("BREWERY_CPU_THREADS").ok()
                    }
                });
            let ceiling = bounds.cpu_threads.map_or(64, |range| range.max);
            resources::cap_cpu(&mut command, requested.min(ceiling), configured.as_deref())?;
        }
        if matches!(
            job.kind.as_str(),
            "inspect" | "plan" | "convert" | "verify" | scratch_eval::GGUF
        ) {
            let reserve = if bounds.disk_reserve_bytes.is_some()
                || job.resources.disk_reserve_bytes.is_some()
            {
                Some(w.disk_floor(job.resources.disk_reserve_bytes))
            } else {
                None
            };
            if let Some(reserve) = reserve {
                command.env("BREWERY_CONVERTER_DISK_RESERVE_BYTES", reserve.to_string());
            }
        }
    }
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
        for key in [
            weight_source::PUBLICATION,
            weight_source::VERIFY_BYTES,
            weight_source::CHUNK_INDEX,
            weight_source::CHUNK_INDEX_SHA256,
        ] {
            if !w.cfg.engine_env.contains_key(key) {
                command.env_remove(key);
            }
        }
        command.env(
            "BREWERY_WEIGHT_SAFETENSORS_DISK_RESERVE_BYTES",
            w.disk_floor(job.resources.disk_reserve_bytes).to_string(),
        );
    }
    if job.kind == brewery_worker::exl3_layer::JOB {
        resources::disk_check(
            &work,
            w.disk_floor(job.resources.disk_reserve_bytes)
                .checked_add(256 << 10)
                .context("EXL3 disk reserve overflow")?,
        )?;
        command.env(
            brewery_worker::exl3_layer::HELPER,
            w.cfg
                .engine_env
                .get(brewery_worker::exl3_layer::HELPER)
                .context("EXL3 helper is not configured")?,
        );
        for key in [weight_source::GATE, weight_source::VERIFY_BYTES] {
            command.env(
                key,
                w.cfg
                    .engine_env
                    .get(key)
                    .context("EXL3 source verification is not configured")?,
            );
        }
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
    if job.kind == behavior_direction::JOB {
        if hgn_direction {
            let configured_pin = w
                .cfg
                .engine_env
                .get(trace::HGN_PYTHON_SHA256)
                .map(String::as_str);
            ensure!(
                configured_pin.is_some() && value["executable_sha256"].as_str() == configured_pin,
                "HGN direction Python pin does not match the worker"
            );
        }
        behavior_direction::Request::parse(&job.options)?.verify_result(
            &value,
            job.inputs[0].extra["id"].as_str().unwrap_or_default(),
            job.inputs[0].extra["sha256"].as_str().unwrap_or_default(),
            &job.inputs[0].kind,
        )?;
        return Ok(value);
    }
    if job.kind == ablation_search::SCREEN {
        let req = ablation_search::ScreenRequest::parse(&job.options)?;
        ensure!(
            job.inputs[0].extra["sha256"] == req.model_sha256,
            "screen input hash mismatch"
        );
        serde_json::from_value::<ablation_search::ScreenReport>(value.clone())?.verify(&req)?;
        return Ok(value);
    }
    if job.kind == scratch_eval::CHECKS {
        let hashes = std::array::from_fn(|i| {
            job.inputs[i]
                .extra
                .get("sha256")
                .and_then(Value::as_str)
                .unwrap_or_default()
        });
        let runner = w
            .cfg
            .engine_env
            .get(scratch_eval::RUNNER_SHA)
            .context("scratch checks runner pin is missing")?;
        let behavior = scratch_eval::check_options(&job.options)?;
        scratch_eval::check_report(&value, hashes, runner, behavior.as_ref())?;
        return Ok(value);
    }
    if job.kind == locate_edit::JOB {
        let request = locate_edit::Request::parse(&job.options)?;
        let pin = request.edit.backend.pin();
        let factors: locate_edit::Factors = serde_json::from_value(value["factors"].clone())?;
        factors.verify(
            &request,
            job.inputs[0].extra["sha256"].as_str().unwrap_or_default(),
        )?;
        ensure!(
            value["kind"] == locate_edit::JOB
                && value["parent_sha256"] == job.inputs[1].extra["sha256"]
                && value["executable_sha256"] == pin.sha256
                && value["lanternfish_commit"] == pin.commit,
            "private edit receipt does not match input or executable pin"
        );
        return Ok(value);
    }
    if heretic::job(&job.kind) {
        let spec = heretic::Options::parse(&job.kind, &job.options)?;
        let runtime = Path::new(
            w.cfg
                .engine_env
                .get(heretic::RUNTIME)
                .context("Missing Heretic runtime")?,
        )
        .canonicalize()?;
        spec.verify(&value, Some(&runtime.join("runs")))?;
        ensure!(
            job.inputs[0].extra["sha256"] == spec.base_sha256,
            "Heretic base input changed"
        );
        return Ok(value);
    }
    if job.kind == scratch_eval::GGUF {
        ensure!(
            value["kind"] == "convert"
                && value["output_format"] == "gguf"
                && value["gguf_type"] == "BF16"
                && value["file_verification"]["status"] == "passed"
                && value["inputs"][0]["sha256"] == job.inputs[0].extra["sha256"]
                && value["artifact"]["path"].as_str().map(Path::new)
                    == Some(work.join("converted.gguf").as_path())
                && value["artifact"]["kind"] == "gguf",
            "Scratch GGUF did not return a verified conversion receipt"
        );
    }
    if job.kind == trace::JOB {
        let request = trace::Request::worker(&job.options, job.prompt.as_deref())?;
        let runtime_matches = if trace::is_hgn(&job.inputs[0].kind) {
            let python_pin = w
                .cfg
                .engine_env
                .get(trace::HGN_PYTHON_SHA256)
                .map(String::as_str);
            value["runtime"] == trace::HGN_RUNTIME
                && value["runtime_sha256"] == trace::HGN_RUNTIME_SHA256.trim_end()
                && python_pin.is_some()
                && value["executable_sha256"].as_str() == python_pin
                && ["sidecar", "config", "tokenizer"]
                    .into_iter()
                    .all(|name| value["assets_sha256"][name].as_str().is_some_and(is_digest))
        } else {
            let (_, _, sha, commit) = request.lanternfish_runtime()?.pins();
            value["lanternfish_commit"] == commit && value["executable_sha256"] == sha
        };
        ensure!(
            value["kind"] == trace::JOB
                && value["schema_version"] == 1
                && value["model_id"].is_string()
                && value["model_id"] == job.inputs[0].extra["id"]
                && value["model_sha256"].as_str().is_some_and(is_digest)
                && value["model_sha256"] == job.inputs[0].extra["sha256"]
                && runtime_matches,
            "trace receipt does not match resolved input and runtime"
        );
        let exports: Vec<trace::Export> = serde_json::from_value(value["prompts"].clone())?;
        let root = work.join(trace::JOB);
        ensure!(
            fs::symlink_metadata(&root)?.is_dir()
                && fs::read_dir(&root)?.count() == request.prompts.len()
                && exports.len() == request.prompts.len(),
            "incomplete trace results"
        );
        let mut spec = request.capture;
        let cap = spec.byte_cap;
        for (i, export) in exports.into_iter().enumerate() {
            let dir = root.join(i.to_string());
            ensure!(
                trace::verify(&dir, &spec)? == export,
                "trace receipt does not match export"
            );
            trace::verify_interventions(&dir, &export, &request.interventions)?;
            spec.byte_cap -= export.bytes;
        }
        ensure!(
            value["bytes"].as_u64() == Some(cap - spec.byte_cap),
            "invalid trace receipt byte count"
        );
        return Ok(value);
    }
    if job.kind == sensitivity::JOB {
        let (request, revision) = sensitivity::Request::resolved(&job.options)?;
        let input = &job.inputs[0];
        request.check_model(input.extra["sha256"].as_str().unwrap_or_default())?;
        let model_id = input.extra["id"]
            .as_str()
            .context("missing sensitivity source ID")?;
        let artifact: sensitivity::Artifact = serde_json::from_value(value)?;
        artifact.check(&request, model_id, &revision)?;
        ensure!(
            w.cfg.engine_env.get(sensitivity::MANIFEST_SHA) == Some(&artifact.manifest_sha256),
            "sensitivity artifact runtime SHA-256 does not match worker pin"
        );
        return Ok(serde_json::to_value(artifact)?);
    }
    if job.kind == circuit::JOB {
        let request = circuit::Request::parse(&job.options)?;
        let input = &job.inputs[0];
        request.check_model(input.extra["sha256"].as_str().unwrap_or_default())?;
        let model_id = input.extra["id"]
            .as_str()
            .context("missing circuit source ID")?;
        let artifact: circuit::Artifact = serde_json::from_value(value)?;
        artifact.check(&request, model_id)?;
        ensure!(
            w.cfg.engine_env.get(sensitivity::MANIFEST_SHA) == Some(&artifact.manifest_sha256),
            "circuit artifact runtime SHA-256 does not match worker pin"
        );
        return Ok(serde_json::to_value(artifact)?);
    }
    if job.kind == damage::JOB {
        let request = damage::Request::parse(&job.options)?;
        let [a, b] = job.inputs.as_slice() else {
            bail!("missing damage map sources")
        };
        let ids = [
            a.extra["id"].as_str().context("missing reference ID")?,
            b.extra["id"].as_str().context("missing quantized ID")?,
        ];
        damage::check_result(&request, &value, ids)?;
        ensure!(
            w.cfg
                .engine_env
                .get(sensitivity::MANIFEST_SHA)
                .map(String::as_str)
                == value["manifest_sha256"].as_str(),
            "damage artifact runtime SHA-256 does not match worker pin"
        );
        return Ok(value);
    }
    if w.cfg.hy4_plan.as_ref().is_some_and(|p| p.encode.is_some()) {
        let artifact = &value["artifact"];
        let file = work.join(hy4::FILE);
        let receipt = work.join(hy4::RECEIPT);
        let file_metadata = fs::symlink_metadata(&file)?;
        let receipt_metadata = fs::symlink_metadata(&receipt)?;
        ensure!(
            value["kind"] == "hy4_gate_encode"
                && value["verified"] == true
                && artifact["kind"] == "gguf"
                && artifact["bytes"] == file_metadata.len()
                && file_metadata.is_file()
                && file_metadata.nlink() == 1
                && receipt_metadata.is_file()
                && receipt_metadata.nlink() == 1
                && receipt_metadata.len() <= 1 << 20
                && artifact["sha256"].as_str().is_some_and(is_digest)
                && artifact["path"].as_str().map(Path::new) == Some(file.as_path())
                && value["receipt_path"].as_str().map(Path::new) == Some(receipt.as_path())
                && load::<Value>(&receipt)? == value
                && fs::read_dir(&work)?.count() == 2,
            "Hy4 encode did not return a verified fixed-path receipt"
        );
        return Ok(value);
    }
    if w.cfg.hy4_plan.is_some() {
        ensure!(
            value["kind"] == "hy4_plan"
                && value["payload_verified"] == false
                && value["runtime_ready"] == false
                && value.get("artifact").is_none()
                && fs::read_dir(&work)?.next().is_none(),
            "Hy4 planner did not return an artifact-free metadata receipt"
        );
        return Ok(value);
    }
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
    if job.kind == brewery_worker::exl3_layer::JOB {
        use brewery_worker::exl3_layer::{FILE, MAX_OUTPUT, Spec};
        let source = &job.inputs[0];
        Spec::parse(&job.options)?.check_receipt(&source.host, &value)?;
        let path = work.join(FILE);
        ensure!(
            value["artifact"]["path"].as_str().map(Path::new) == Some(&path)
                && fs::symlink_metadata(&path).is_ok_and(|m| m.is_file() && m.len() <= MAX_OUTPUT)
                && value["inputs"][0]["sha256"] == source.extra["sha256"]
                && value["inputs"][0]["bundle_sha256"] == source.extra["bundle_sha256"],
            "EXL3 artifact is not the verified layer from the resolved source"
        );
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
