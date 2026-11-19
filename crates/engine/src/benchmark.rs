use crate::{bounded, integrity, progress::Progress, Job};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    env,
    fs::{File, OpenOptions},
    io::{self, Seek, Write},
    net::{Ipv4Addr, Ipv6Addr},
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
        process::CommandExt,
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const CONFIG_LIMIT: u64 = 128 << 10;
const OUTPUT_LIMIT: u64 = 64 << 10;
const RAW_LIMIT: u64 = 4 << 20;
const RUN_LIMIT: Duration = Duration::from_secs(300);
const POLL: Duration = Duration::from_millis(20);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    benchmark_target: String,
    benchmarks: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    targets: BTreeMap<String, Target>,
    adapters: BTreeMap<String, Adapter>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    endpoint: String,
    model: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Adapter {
    executable: PathBuf,
    version: String,
    protocol: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Output {
    metrics: Vec<Metric>,
    samples: Option<u64>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Metric {
    name: String,
    value: f64,
    unit: String,
    direction: Direction,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum Direction {
    Higher,
    Lower,
    Neutral,
}

// If the engine cannot establish that the process tree stopped, do not start
// another adapter; the worker supervisor will finish cleaning up the job.
#[derive(Debug)]
struct CleanupError;

impl std::fmt::Display for CleanupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cannot stop benchmark adapter process tree")
    }
}

impl std::error::Error for CleanupError {}

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    ensure!(job.inputs.is_empty(), "benchmark jobs do not accept inputs");
    ensure!(
        job.prompt.is_none(),
        "benchmark jobs do not accept a prompt"
    );
    let options: Options = serde_json::from_value(job.options.clone())
        .ok()
        .context("benchmark options must contain only benchmark_target and benchmarks")?;
    ensure!(
        valid_id(&options.benchmark_target),
        "invalid benchmark target ID"
    );
    ensure!(
        (1..=8).contains(&options.benchmarks.len())
            && options.benchmarks.iter().all(|id| valid_id(id))
            && options.benchmarks.iter().collect::<HashSet<_>>().len() == options.benchmarks.len(),
        "benchmarks must contain 1 to 8 distinct IDs"
    );
    let config = load_config()?;
    let target = config
        .targets
        .get(&options.benchmark_target)
        .context("benchmark target is not configured on this worker")?;
    ensure!(
        options
            .benchmarks
            .iter()
            .all(|id| config.adapters.contains_key(id)),
        "benchmark adapter is not configured on this worker"
    );
    let cpu_threads = brewery_hash_pool::cpu_threads()?.to_string();
    // Reap children adopted when a short-lived adapter leaves a subprocess behind.
    ensure!(
        unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } == 0,
        "cannot initialize benchmark process cleanup"
    );
    let mut runs = Vec::with_capacity(options.benchmarks.len());
    let mut suites = progress.start("benchmark", "suites", Some(options.benchmarks.len() as u64));
    for id in &options.benchmarks {
        runs.push(run_one(
            id,
            &options.benchmark_target,
            target,
            &config.adapters[id],
            cwd,
            &cpu_threads,
        )?);
        // A recorded failure counts as a finished suite, not a successful run.
        suites.advance(1);
        suites.checkpoint();
    }
    let mut publish = progress.start("publish_output", "items", Some(1));
    let succeeded = runs.iter().filter(|r| r["status"] == "succeeded").count();
    let status = match succeeded {
        0 => "failed",
        n if n == runs.len() => "complete",
        _ => "partial",
    };
    let mut report = json!({
        "kind":"benchmark", "schema":"brewery-benchmark-report-v1",
        "target":options.benchmark_target, "model":target.model,
        "status":status, "runs":runs, "release_qualified":false
    });
    let path = cwd.join("benchmark-report.json");
    let mut bytes = serde_json::to_vec(&report)?;
    bytes.push(b'\n');
    save_private(&path, &bytes)?;
    publish.advance(1);
    publish.finish();
    report["artifact"] = integrity::identify_output(&path, progress)?.artifact(&path, "report");
    Ok(report)
}

fn valid_id(id: &str) -> bool {
    let start = |c: u8| c.is_ascii_lowercase() || c.is_ascii_digit();
    id.len() <= 64
        && id.bytes().next().is_some_and(start)
        && id.bytes().all(|c| start(c) || matches!(c, b'-' | b'_'))
}

fn label(text: &str, limit: usize) -> bool {
    !text.trim().is_empty() && text.len() <= limit && !text.chars().any(char::is_control)
}

fn loopback(endpoint: &str) -> bool {
    if endpoint.len() > 2048
        || !endpoint
            .bytes()
            .all(|c| c.is_ascii_graphic() && !matches!(c, b'\\' | b'@' | b'?' | b'#'))
    {
        return false;
    }
    let Some(url) = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
    else {
        return false;
    };
    // Parse only numeric addresses; no DNS or URL parser normalization.
    let (authority, path) = url.split_once('/').unwrap_or((url, ""));
    let mut path = path.bytes();
    while let Some(c) = path.next() {
        match c {
            // RFC 3986 path characters; raw '@' is rejected above.
            b if b.is_ascii_alphanumeric() || b"/-._~!$&'()*+,;=:".contains(&b) => {}
            b'%' if (0..2).all(|_| path.next().is_some_and(|b| b.is_ascii_hexdigit())) => {}
            _ => return false,
        }
    }
    let valid_port = |p: &str| {
        !p.is_empty()
            && p.bytes().all(|b| b.is_ascii_digit())
            && p.parse::<u16>().is_ok_and(|port| port > 0)
    };
    if let Some(ipv6) = authority.strip_prefix('[') {
        return ipv6.split_once(']').is_some_and(|(ip, suffix)| {
            ip.parse::<Ipv6Addr>().is_ok_and(|ip| ip.is_loopback())
                && (suffix.is_empty() || suffix.strip_prefix(':').is_some_and(valid_port))
        });
    }
    let (host, port) = authority
        .split_once(':')
        .map_or((authority, None), |(h, p)| (h, Some(p)));
    host.parse::<Ipv4Addr>().is_ok_and(|ip| ip.is_loopback()) && port.is_none_or(valid_port)
}

fn load_config() -> Result<Config> {
    let path = env::var_os("BREWERY_BENCHMARK_CONFIG")
        .context("benchmarking is not configured on this worker")?;
    let config: Option<Config> = (|| {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
            .ok()?;
        if !file.metadata().ok()?.is_file() {
            return None;
        }
        serde_json::from_slice(&bounded(file, CONFIG_LIMIT).ok()?).ok()
    })();
    // Do not echo file paths, contents, serde messages or invalid endpoints.
    let config = config.context("invalid worker benchmark configuration")?;
    ensure!(
        config.targets.iter().all(|(id, t)| valid_id(id) && label(&t.model, 256) && loopback(&t.endpoint)),
        "configured benchmark targets require IDs, model labels, and numeric loopback HTTP(S) endpoints without credentials, queries, or fragments"
    );
    ensure!(
        config.adapters.iter().all(|(id, a)| valid_id(id)
            && a.executable.is_absolute() && label(&a.version, 128) && label(&a.protocol, 128)),
        "configured benchmark adapters require IDs, absolute executables, and version/protocol labels"
    );
    Ok(config)
}

fn run_one(
    id: &str,
    target_id: &str,
    target: &Target,
    adapter: &Adapter,
    cwd: &Path,
    cpu_threads: &str,
) -> Result<Value> {
    let started = Instant::now();
    let mut run = json!({
        "benchmark":id, "version":adapter.version, "protocol":adapter.protocol,
        "status":"failed", "metrics":[]
    });
    let request = json!({
        "benchmark":id, "target":target_id, "endpoint":target.endpoint,
        "model":target.model, "version":adapter.version, "protocol":adapter.protocol
    });
    let dir = tempfile::Builder::new()
        .prefix(".benchmark-")
        .tempdir_in(cwd)
        .context("cannot create benchmark work directory")?;
    let result = (|| -> Result<Output> {
        let path = adapter
            .executable
            .canonicalize()
            .ok()
            .context("adapter executable is not available")?;
        ensure!(
            path.metadata()
                .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0),
            "adapter executable is not a regular executable file"
        );
        let (digest, stamp) = integrity::file_digest(&path, "adapter".into())
            .ok()
            .context("cannot hash adapter executable")?;
        run["executable_sha256"] = json!(digest.sha256);

        let result = execute(&path, dir.path(), &request, cpu_threads)?;
        stamp
            .check(&path)
            .ok()
            .context("adapter executable changed during the run")?;
        let output: Output = serde_json::from_slice(&result)
            .ok()
            .context("adapter did not return valid metric JSON")?;
        let mut names = HashSet::new();
        ensure!(
            (1..=64).contains(&output.metrics.len())
                && output.samples.is_none_or(|n| n <= 1_000_000_000)
                && output.metrics.iter().all(|m| {
                    label(&m.name, 96)
                        && label(&m.unit, 32)
                        && names.insert(&m.name)
                        && m.value.is_finite()
                }),
            "adapter metrics exceed limits or are invalid"
        );
        Ok(output)
    })();
    match result {
        Ok(output) => {
            run["status"] = json!("succeeded");
            run["metrics"] = json!(output.metrics);
            if let Some(samples) = output.samples {
                run["samples"] = json!(samples);
            }
        }
        Err(error) if error.is::<CleanupError>() => return Err(error),
        Err(error) => run["error"] = json!(error.to_string()),
    }
    match preserve_raw(dir.path(), cwd, id) {
        Ok(Some(raw)) => run["raw_artifact"] = raw,
        Ok(None) => {}
        Err(_) => {
            run["raw_error"] =
                json!("adapter raw.json is unsafe, oversized, invalid, or could not be saved")
        }
    }
    dir.close()
        .ok()
        .context("cannot clean benchmark work directory")?;
    run["duration_ms"] = json!(started.elapsed().as_millis() as u64);
    Ok(run)
}

fn execute(executable: &Path, dir: &Path, request: &Value, cpu_threads: &str) -> Result<Vec<u8>> {
    // Anonymous files avoid pipe stalls (including subprocesses holding pipe ends open).
    let mut input = tempfile::tempfile_in(dir)
        .ok()
        .context("cannot prepare adapter stdin")?;
    serde_json::to_writer(&mut input, request)
        .ok()
        .context("cannot write adapter stdin")?;
    input.rewind().ok().context("cannot rewind adapter stdin")?;
    let mut stdout = tempfile::tempfile_in(dir)
        .ok()
        .context("cannot prepare adapter stdout")?;
    let output = stdout
        .try_clone()
        .ok()
        .context("cannot prepare adapter stdout")?;
    let mut command = Command::new(executable);
    command
        .current_dir(dir)
        .process_group(0)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", dir)
        .env("TMPDIR", dir)
        .envs(
            brewery_hash_pool::CPU_ENV
                .iter()
                .map(|&key| (key, cpu_threads)),
        )
        .stdin(input)
        .stdout(output)
        .stderr(Stdio::null());
    // Limit each file the adapter creates (including stdout and optional raw.json).
    unsafe {
        command.pre_exec(|| {
            let limit = libc::rlimit {
                rlim_cur: RAW_LIMIT,
                rlim_max: RAW_LIMIT,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .ok()
        .context("cannot start benchmark adapter")?;
    let pid = child.id() as libc::pid_t;
    let started = Instant::now();
    let monitored = loop {
        if started.elapsed() >= RUN_LIMIT {
            break Err("adapter timed out");
        }
        match stdout.metadata() {
            Ok(m) if m.len() > OUTPUT_LIMIT => break Err("adapter stdout exceeds limit"),
            Err(_) => break Err("cannot inspect adapter stdout"),
            _ => {}
        }
        // Leave the leader unreaped until the group is signaled so its PID
        // cannot be reused by an unrelated process group.
        let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                pid as _,
                info.as_mut_ptr(),
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if rc == -1 {
            if io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                break Err("cannot monitor benchmark adapter");
            }
        } else if unsafe { info.assume_init().si_pid() } == pid {
            break Ok(());
        }
        thread::sleep(POLL);
    };

    unsafe { libc::kill(-pid, libc::SIGKILL) };
    let deadline = Instant::now() + Duration::from_secs(2);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(POLL),
            _ => return Err(CleanupError.into()),
        }
    };
    stop_adopted(deadline).map_err(|_| CleanupError)?;
    if let Err(error) = monitored {
        anyhow::bail!("{error}");
    }
    ensure!(status.success(), "adapter exited unsuccessfully");
    stdout.rewind().ok().context("cannot read adapter stdout")?;
    bounded(stdout, OUTPUT_LIMIT)
        .ok()
        .context("adapter stdout exceeds limit or could not be read")
}

fn stop_adopted(deadline: Instant) -> Result<()> {
    let children = format!("/proc/self/task/{}/children", std::process::id());
    loop {
        // The engine is single-threaded and has one adapter at a time. All
        // remaining direct children are from that adapter, even across sessions.
        let result = unsafe { libc::waitpid(-1, std::ptr::null_mut(), libc::WNOHANG) };
        if result > 0 {
            continue;
        }
        if result == -1 && io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD) {
            break;
        }
        ensure!(
            Instant::now() < deadline,
            "cannot stop benchmark adapter subprocesses"
        );
        let adopted = std::fs::read_to_string(&children)
            .ok()
            .context("cannot inspect benchmark adapter subprocesses")?;
        for pid in adopted
            .split_ascii_whitespace()
            .filter_map(|s| s.parse::<libc::pid_t>().ok())
        {
            // No wait/reap between reading a child PID and signaling it.
            if pid > 0 {
                unsafe { libc::kill(pid, libc::SIGKILL) };
            }
        }
        thread::sleep(POLL);
    }
    Ok(())
}

fn preserve_raw(dir: &Path, cwd: &Path, id: &str) -> Result<Option<Value>> {
    ensure!(dir.canonicalize()? == dir, "adapter work directory changed");
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(dir.join("raw.json"))
    {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let meta = file.metadata()?;
    ensure!(
        meta.is_file() && meta.nlink() == 1 && meta.len() <= RAW_LIMIT,
        "invalid raw file"
    );
    let bytes = bounded(file, RAW_LIMIT)?;
    serde_json::from_slice::<serde::de::IgnoredAny>(&bytes)?;
    let path = cwd.join(format!("benchmark-raw-{id}.json"));
    save_private(&path, &bytes)?;
    Ok(Some(
        integrity::identify(&path, false)?.artifact(&path, "report"),
    ))
}

fn save_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file: File = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error.into());
    }
    Ok(())
}
