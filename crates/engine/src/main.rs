mod benchmark;
mod compare;
mod converter;
mod exl3_layer;
mod gguf;
mod gguf_conversion;
mod hf_download;
mod hgn_recipe;
mod integrity;
mod native_progress;
mod progress;
mod recipe;
mod scratch_checks;
mod scratch_locate;
mod trace;
mod weight;

use anyhow::{Context, Result, bail, ensure};
use integrity::{check_inputs, file_digest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const JOB_LIMIT: u64 = 16 << 20;
const NATIVE_JSON_LIMIT: u64 = 64 << 20;

#[derive(Deserialize)]
struct Job {
    kind: String,
    inputs: Vec<Input>,
    #[serde(default = "empty_options")]
    options: Value,
    prompt: Option<String>,
}

#[derive(Deserialize)]
struct Input {
    path: PathBuf,
    host: String,
    kind: String,
    id: Option<String>,
    name: Option<String>,
    source_host: Option<String>,
    sha256: Option<String>,
    bundle_sha256: Option<String>,
    files: Option<Vec<integrity::Member>>,
    bytes: Option<u64>,
    // Set only by the worker on the resolved input. This labels the
    // verification phase; it is not a substitute for checking payloads.
    integrity_basis: Option<IntegrityBasis>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum IntegrityBasis {
    PinnedPublication,
}

fn empty_options() -> Value {
    json!({})
}

fn bounded(mut reader: impl Read, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    (&mut reader).take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "JSON exceeds {limit} bytes");
    Ok(bytes)
}

fn read_json(path: &Path) -> Result<Value> {
    let data = bounded(File::open(path)?, NATIVE_JSON_LIMIT)?;
    serde_json::from_slice(&data).with_context(|| format!("invalid JSON at {}", path.display()))
}

fn create_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("cannot create fresh output {}", path.display()))
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut file = create_file(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn tool(var: &str, default: &str) -> Result<PathBuf> {
    let config = env::var_os(var).unwrap_or_else(|| default.into());
    let path = PathBuf::from(&config);
    if path.is_absolute() {
        return path
            .canonicalize()
            .with_context(|| format!("invalid {var} executable"));
    }
    ensure!(
        path.components().count() == 1 && path.file_name().is_some(),
        "{var} must be an absolute executable or a command name"
    );
    env::split_paths(&env::var_os("PATH").unwrap_or_default())
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join(&path))
        .find_map(|candidate| {
            use std::os::unix::fs::PermissionsExt;
            let meta = fs::metadata(&candidate).ok()?;
            if !meta.is_file() || meta.permissions().mode() & 0o111 == 0 {
                return None;
            }
            candidate.canonicalize().ok()
        })
        .with_context(|| format!("{var} executable {:?} is not installed", path))
}

fn native_json(cmd: &mut Command, log: &Path) -> Result<(Value, Value)> {
    native_json_admitted(cmd, log, None, None)
}

fn child_threads(requested: Option<u64>) -> Result<usize> {
    let default = brewery_hash_pool::cpu_threads()?;
    let cap = if env::var_os("BREWERY_CPU_THREADS").is_some() {
        default
    } else {
        std::thread::available_parallelism().map_or(1, usize::from)
    };
    match requested {
        None => Ok(default),
        Some(n) => {
            ensure!((1..=64).contains(&n), "threads must be 1..64");
            Ok((n as usize).min(cap))
        }
    }
}

fn child_budget(cmd: &mut Command, threads: usize) {
    for &key in brewery_hash_pool::CPU_ENV {
        cmd.env(key, threads.to_string());
    }
}

fn native_json_admitted(
    cmd: &mut Command,
    log: &Path,
    admission: Option<&Path>,
    tracking: Option<(&progress::Progress, &'static str)>,
) -> Result<(Value, Value)> {
    let Some((progress, name)) = tracking else {
        return native_capture(cmd, log, admission);
    };
    // All converter arguments are absolute. Use a dedicated cwd for native
    // counters so they cannot replace the engine snapshot or enter an artifact.
    let cwd = env::current_dir()?.canonicalize()?;
    let scratch = matches!(
        name,
        "convert" | "verify" | brewery_worker::scratch_eval::CHECKS
    )
    .then(|| {
        tempfile::Builder::new()
            .prefix(".native-progress-")
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir_in(cwd)
            .ok()
    })
    .flatten();
    let native_path = scratch.as_ref().map(|dir| {
        let path = dir.path().join("progress.json");
        cmd.current_dir(dir.path())
            .env("BREWERY_NATIVE_PROGRESS", &path);
        path
    });
    let admitted_pid = admission.map(|_| log.with_file_name("admitted-pid"));
    let started_ms = progress::now_ms();
    std::thread::scope(|scope| {
        let (stop, watch) = std::sync::mpsc::channel();
        let mut phase = progress.start(
            if admission.is_some() {
                "admission"
            } else {
                name
            },
            "items",
            None,
        );
        let monitor = scope.spawn(move || {
            let mut admitted = admitted_pid.is_none();
            let mut last = None;
            let mut exited = None;
            loop {
                if !admitted
                    && admitted_pid
                        .as_deref()
                        .is_some_and(|path| admitted_process(path).is_ok())
                {
                    phase.advance(1);
                    phase.finish();
                    phase = progress.start(name, "items", None);
                    admitted = true;
                }
                if admitted {
                    if let Some(next) = native_path
                        .as_deref()
                        .and_then(|p| native_progress::read(p, name, started_ms))
                    {
                        if native_progress::follows(&next, last.as_ref()) {
                            if last.as_ref().is_none_or(|prev: &native_progress::Counter| {
                                prev.phase != next.phase
                            }) {
                                phase = progress.start_at(
                                    next.phase,
                                    "items",
                                    Some(next.total),
                                    next.completed,
                                );
                            } else {
                                phase.set_completed(next.completed);
                                if next.completed == next.total {
                                    phase.checkpoint();
                                }
                            }
                            last = Some(next);
                        }
                    }
                }
                if exited.is_some() {
                    if exited == Some(true) && last.is_none() {
                        phase.advance(1);
                    }
                    phase.finish();
                    break;
                }
                match watch.recv_timeout(std::time::Duration::from_millis(250)) {
                    Ok(success) => exited = Some(success),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
        });
        let result = native_capture(cmd, log, admission);
        let _ = stop.send(result.is_ok());
        monitor.join().ok();
        result
    })
}

fn admitted_process(path: &Path) -> Result<u32> {
    let raw = native_progress::read_file(path, 32)?;
    let observed = std::str::from_utf8(&raw)?.trim().parse::<u32>()?;
    ensure!(observed > 0, "invalid admitted converter PID");
    Ok(observed)
}

fn native_capture(
    cmd: &mut Command,
    log: &Path,
    admission: Option<&Path>,
) -> Result<(Value, Value)> {
    let executable = PathBuf::from(cmd.get_program());
    let (binary, stamp) = file_digest(&executable, "binary".into())?;
    let mut log_file = create_file(log)?;
    let pid_file = log.with_file_name("admitted-pid");
    let mut wrapped;
    let cmd = if let Some(helper) = admission {
        ensure!(
            helper.is_absolute(),
            "native admission helper must be absolute"
        );
        ensure!(
            !pid_file.exists(),
            "native admission PID receipt already exists"
        );
        wrapped = Command::new(helper);
        wrapped.arg(&pid_file).arg(&executable).args(cmd.get_args());
        if let Some(cwd) = cmd.get_current_dir() {
            wrapped.current_dir(cwd);
        }
        for (key, val) in cmd.get_envs() {
            if let Some(val) = val {
                wrapped.env(key, val);
            } else {
                wrapped.env_remove(key);
            }
        }
        &mut wrapped
    } else {
        cmd
    };
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("cannot start {}", executable.display()))?;
    let pid = child.id();
    let mut log_pipe = child.stderr.take().expect("child stderr is piped");
    let relay = std::thread::spawn(move || -> io::Result<()> {
        let mut chunk = [0; 8192];
        // Relay progress immediately so worker cancellation/status can see it.
        // Bound both outputs without causing the native subprocess to SIGPIPE.
        let mut remaining = 16 << 20;
        loop {
            let n = log_pipe.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            let emit = n.min(remaining);
            if emit > 0 {
                log_file.write_all(&chunk[..emit])?;
                let _ = io::stderr().write_all(&chunk[..emit]);
                remaining -= emit;
                if remaining == 0 {
                    log_file.write_all(b"\n[brewery-engine: log limit reached]\n")?;
                }
            }
        }
        Ok(())
    });
    let stdout = bounded(
        child.stdout.take().expect("child stdout is piped"),
        NATIVE_JSON_LIMIT,
    );
    if stdout.is_err() {
        let _ = child.kill();
    }
    let exit = child.wait()?;
    relay
        .join()
        .map_err(|_| anyhow::anyhow!("native log relay panicked"))??;
    let stdout = stdout?;
    ensure!(
        exit.success(),
        "{} exited with {exit} (log: {})",
        executable.display(),
        log.display()
    );
    stamp.check(&executable)?;
    let pid = if admission.is_some() {
        admitted_process(&pid_file)?
    } else {
        pid
    };
    let data = serde_json::from_slice(&stdout).with_context(|| {
        format!(
            "{} did not return a single JSON value",
            executable.display()
        )
    })?;
    let (stderr, _) = file_digest(log, "stderr".into())?;
    Ok((
        data,
        json!({"executable":executable,"executable_sha256":binary.sha256,"pid":pid,
               "exit_code":exit.code(),"stderr_log":log,"stderr_sha256":stderr.sha256}),
    ))
}

fn delegate(bytes: &[u8]) -> Result<i32> {
    let mut command = Command::new(tool("BREWERY_EVAL", "brewery-eval")?);
    child_budget(&mut command, child_threads(None)?);
    let mut child = command
        .stdin(Stdio::piped())
        .spawn()
        .context("cannot start brewery-eval")?;
    let written = child
        .stdin
        .take()
        .expect("child stdin is piped")
        .write_all(bytes);
    let status = child.wait()?;
    if status.success() {
        written?;
    }
    Ok(status.code().unwrap_or(1))
}

fn run() -> Result<i32> {
    let bytes = bounded(io::stdin().lock(), JOB_LIMIT)?;
    // Leave the eval envelope intact, including fields only brewery-eval knows.
    let value: Value = serde_json::from_slice(&bytes).context("expected one job JSON object")?;
    if matches!(
        value["kind"].as_str(),
        Some("generate" | "evaluate" | "baseline")
    ) {
        return delegate(&bytes);
    }
    let job: Job = serde_json::from_value(value).context("invalid resolved job")?;
    ensure!(job.options.is_object(), "options must be an object");
    let cwd = env::current_dir()?.canonicalize()?;
    let progress = progress::Progress::new(&cwd, job.inputs.iter().map(|input| &input.path));
    let result = match job.kind.as_str() {
        "benchmark" => benchmark::run(&job, &cwd, &progress)?,
        "compare" => compare::run(&job, &cwd, &progress)?,
        brewery_worker::trace::JOB => trace::run(&job, &cwd, &progress)?,
        brewery_worker::scratch_eval::CHECKS => scratch_checks::run(&job, &cwd, &progress)?,
        brewery_worker::locate_edit::JOB => scratch_locate::run(&job, &cwd, &progress)?,
        brewery_worker::scratch_eval::GGUF => {
            ensure!(
                brewery_worker::scratch_eval::gguf_options(&job.options)
                    && job
                        .inputs
                        .iter()
                        .map(|i| i.kind.as_str())
                        .eq(["safetensors", "assets"])
                    && job.inputs[0]
                        .sha256
                        .as_deref()
                        .is_some_and(brewery_worker::is_digest),
                "scratch GGUF needs pinned standalone SafeTensors, assets, and BF16 options"
            );
            gguf_conversion::run(&job, &cwd, &progress)?
        }
        "hf_download" => hf_download::run(&job, &cwd, &progress)?,
        brewery_worker::exl3_layer::JOB => exl3_layer::run(&job, &cwd, &progress)?,
        "weight_inspect" | "weight_propose" | "weight_tune" | "weight_apply" => {
            weight::run(&job, &cwd, &progress)?
        }
        "inspect" | "plan" | "convert" | "verify" if job.options["output_format"] == "gguf" => {
            gguf_conversion::run(&job, &cwd, &progress)?
        }
        "inspect" | "plan" | "convert" | "verify"
            if job.options["output_format"] == brewery_worker::hgn_recipe::FORMAT =>
        {
            hgn_recipe::run(&job, &cwd, &progress)?
        }
        "inspect" | "plan" | "convert" | "verify" => converter::run(&job, &cwd, &progress)?,
        _ => bail!("unsupported job kind {:?}", job.kind),
    };
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, &result)?;
    out.write_all(b"\n")?;
    Ok(0)
}

fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("brewery-engine: {e:#}");
            2
        }
    };
    std::process::exit(code);
}
