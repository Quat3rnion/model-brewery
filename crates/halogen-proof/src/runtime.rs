use super::{is_sha256, GenerationBundle, Qualification, Result, Role};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::FileExt,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

pub(crate) struct Completion<'a> {
    pub(crate) bundle: &'a GenerationBundle,
    pub(crate) prompts: &'a [String],
    pub(crate) max_tokens: u32,
    pub(crate) observe: &'a mut dyn FnMut(u64),
}

const DOCKER: &str = "/usr/bin/docker";
const OBSERVER: &str = include_str!("observe.py");
const COMPLETER: &str = include_str!("complete.py");
const LEASE: &str = include_str!("lease.py");
const CHECKPOINT: &str = "/brewery-checkpoint.hgn";
const LOOKUP: &str = "/brewery-lookup.hgn";
const VISION: &str = "/brewery-vision.hgn";
const PORT: &str = "28730";
const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Serialize)]
pub struct Proof<'a> {
    schema: &'static str,
    image_id: String,
    container_id: String,
    qualification: &'a Qualification,
    observation: Observation,
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Observation {
    engine_pid: u32,
    start_ticks: u64,
    listener_inode: u64,
    mapped: Vec<Mapping>,
}

#[derive(Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Mapping {
    path: String,
    major: u32,
    minor: u32,
    inode: u64,
    bytes: u64,
}

/// Start the supported image's engine mode and verify a fresh, isolated load.
///
/// Call only while holding the GPU gate on a host with sufficient spare RAM.
/// The image ID must refer to an installed, trusted Halogen image. This
/// operation can allocate tens of GiB and take minutes.
pub fn prove<'a>(
    bundle: &'a Qualification,
    image_id: &str,
    timeout_seconds: u64,
) -> Result<Proof<'a>> {
    Ok(run(bundle, image_id, timeout_seconds, None)?.0)
}

pub(crate) fn run<'a>(
    bundle: &'a Qualification,
    image_id: &str,
    timeout_seconds: u64,
    completion: Option<Completion<'_>>,
) -> Result<(Proof<'a>, Vec<String>)> {
    if !image_id.strip_prefix("sha256:").is_some_and(is_sha256) {
        return Err("provide a pinned, local Docker image ID (sha256:<64 lowercase hex>)".into());
    }
    if !(1..=1800).contains(&timeout_seconds) {
        return Err("engine timeout must be 1 to 1800 seconds".into());
    }
    let duration = Duration::from_secs(timeout_seconds);
    let control = Duration::from_secs(15);
    let cpu_threads = completion
        .is_some()
        .then(brewery_hash_pool::cpu_threads)
        .transpose()
        .map_err(|error| format!("{error:#}"))?;
    let mut create = container_create(cpu_threads);
    create.args([
        "--pull=never",
        "--network=none",
        "--init",
        "--ipc=private",
        "--restart=no",
        "--oom-score-adj=500",
        "--pids-limit=512",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--entrypoint=python3",
    ]);
    for var in [
        "HALOGEN_CHECKPOINT=/brewery-checkpoint.hgn",
        "HALOGEN_PORT=28730",
        "HALOGEN_BIND=127.0.0.1",
        "HALOGEN_KV_SLOTS=1",
        "HALOGEN_CTX=4096",
        "HALOGEN_KV_POOL_POSITIONS=8192",
        "HALOGEN_PROMPT_CACHE=0",
        "HALOGEN_CACHE_DIR=",
        "HALOGEN_WEIGHTS_LOCK=0",
        "HALOGEN_DOWNLOAD=",
        "HALOGEN_NPU_MODELS=",
        "HALOGEN_CK_OVERLAY=none",
        "HALOGEN_ENGINE_WATCHDOG_S=0",
    ] {
        create.args(["--env", var]);
    }
    let mut members: Vec<_> = bundle.files.iter().collect();
    members.sort_by_key(|file| file.role);
    let mut expected = Vec::new();
    for file in &members {
        file.check_unchanged()?;
        let source = file
            .path
            .to_str()
            .ok_or("Docker bind source must be UTF-8")?;
        if source.contains([',', '"', '\r', '\n']) {
            return Err("Docker bind source cannot contain comma, quote, CR, or LF".into());
        }
        let target = match file.role {
            Role::Checkpoint => CHECKPOINT,
            Role::Lookup => {
                create.args(["--env", "HALOGEN_NGRAM_TABLE=/brewery-lookup.hgn"]);
                LOOKUP
            }
            Role::Vision => {
                create.args(["--env", "HALOGEN_VISION_TOWER=/brewery-vision.hgn"]);
                VISION
            }
            Role::Draft => return Err("external draft HGN is unsupported".into()),
        };
        create.args([
            "--mount",
            &format!("type=bind,src={source},dst={target},readonly"),
        ]);
        expected.push(Mapping {
            path: target.to_owned(),
            major: file.identity.dev_major,
            minor: file.identity.dev_minor,
            inode: file.identity.inode,
            bytes: file.identity.bytes,
        });
    }
    let available = memory_headroom(&expected)?;
    create.args(["--memory", &(available - 8 * GIB).to_string()]);
    create.args(["--memory-swap", &(available - 8 * GIB).to_string()]);
    if let Some(Completion {
        bundle: generation,
        max_tokens,
        ..
    }) = completion.as_ref()
    {
        let path = generation.stage.path().canonicalize()?;
        let path = path
            .to_str()
            .filter(|p| !p.contains([',', '"', '\r', '\n']))
            .ok_or("invalid tokenizer staging path")?;
        for var in [
            "HALOGEN_TOKENIZER=/brewery-tokenizer",
            "HALOGEN_API_PORT=28731",
            "HALOGEN_MODEL_ID=brewery-hgn",
            "HALOGEN_ENABLE_THINKING=0",
            "HALOGEN_FORK=0",
        ] {
            create.args(["--env", var]);
        }
        create.args(["--env", &format!("HALOGEN_MAX_TOK={max_tokens}")]);
        create.args(["--env", &format!("HALOGEN_MAX_TOKENS_CAP={max_tokens}")]);
        create.args(["--env", &format!("HALOGEN_MAX_TOKENS_DEFAULT={max_tokens}")]);
        create.args([
            "--mount",
            &format!("type=bind,src={path},dst=/brewery-tokenizer,readonly"),
        ]);
    } else {
        create.args(["--env", "HALOGEN_MAX_TOK=256"]);
    }
    let lease = tempfile::tempdir()?;
    let lease_path = lease.path().canonicalize()?;
    let lease_source = lease_path
        .to_str()
        .filter(|p| !p.contains([',', '"', '\r', '\n']))
        .ok_or("invalid lease path")?;
    create.args([
        "--mount",
        &format!("type=bind,src={lease_source},dst=/brewery-lease,readonly"),
    ]);
    create.args([
        image_id,
        "-I",
        "-c",
        LEASE,
        if completion.is_some() {
            "all"
        } else {
            "engine"
        },
    ]);

    let actual = docker_checked(
        docker().args(["image", "inspect", "--format", "{{.Id}}", image_id]),
        control,
    )?;
    if actual.trim() != image_id {
        return Err("Docker image did not resolve to the requested pinned image ID".into());
    }
    let created = docker_checked(&mut create, control)?;
    let id = created.trim();
    if !is_sha256(id) {
        return Err("Docker did not return an unambiguous container ID".into());
    }
    let mut container = Container { id, remove: true };
    eprintln!("halogen-proof: isolated container {id}");
    for file in &members {
        file.check_unchanged()?;
    }
    memory_headroom(&expected)?;
    renew(&lease_path)?;
    let _heartbeat = Heartbeat::start(lease_path)?;
    docker_checked(docker().args(["start", id]), control)?;

    let observe = |seconds: u64, deadline| {
        docker_checked(
            docker().args([
                "exec",
                id,
                "python3",
                "-I",
                "-c",
                OBSERVER,
                "/usr/local/bin/flash_serve",
                "1",
                PORT,
                &seconds.to_string(),
                &serde_json::to_string(&expected)?,
            ]),
            deadline,
        )
    };
    let observed = observe(timeout_seconds, duration + control);
    let observed = match observed {
        Ok(s) => s,
        Err(e) => {
            if let Ok(log) = docker_checked(docker().args(["logs", "--tail", "12", id]), control) {
                return Err(format!("{e}\nRecent isolated runtime log:\n{log}").into());
            }
            return Err(e);
        }
    };
    let observation: Observation = serde_json::from_str(&observed)?;
    if observation.engine_pid <= 1
        || observation.start_ticks == 0
        || observation.listener_inode == 0
        || observation.mapped != expected
    {
        return Err("runtime observation differs from the qualified bundle".into());
    }
    let texts = if let Some(Completion {
        bundle: generation,
        prompts,
        max_tokens,
        observe: count,
    }) = completion
    {
        count(0);
        let mut input = tempfile::tempfile()?;
        serde_json::to_writer(
            &mut input,
            &serde_json::json!({
                "prompts":prompts,"max_tokens":max_tokens,"tokenizer":generation.assets
            }),
        )?;
        input.flush()?;
        input.rewind()?;
        let raw = docker_capture_observed(
            docker().args(["exec", "-i", id, "python3", "-I", "-c", COMPLETER]),
            Duration::from_secs(120 + (prompts.len() as u64 * 180)),
            Some(input),
            90 * 1024 * 1024,
            Some((prompts.len() as u64, count)),
        )?;
        let after: Observation = serde_json::from_str(&observe(5, control)?)?;
        if observation != after {
            return Err("Halogen engine identity changed during generation".into());
        }
        let texts: Vec<String> = serde_json::from_str(&raw)?;
        if texts.len() != prompts.len() || texts.iter().any(|t| t.len() > 65536) {
            return Err("invalid isolated generation output".into());
        }
        texts
    } else {
        Vec::new()
    };
    for file in members {
        file.check_unchanged()?;
    }
    container.cleanup()?;
    Ok((
        Proof {
            schema: "halogen-linux-engine-load-v1",
            image_id: image_id.to_owned(),
            container_id: id.to_owned(),
            qualification: bundle,
            observation,
        },
        texts,
    ))
}

fn memory_headroom(files: &[Mapping]) -> Result<u64> {
    let info = fs::read_to_string("/proc/meminfo")?;
    let available = info
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|line| line.split_whitespace().next())
        .ok_or("cannot determine host MemAvailable")?
        .parse::<u64>()?
        .checked_mul(1024)
        .ok_or("invalid host MemAvailable")?;
    let required = files
        .iter()
        .try_fold(12 * GIB, |sum, f| sum.checked_add(f.bytes))
        .ok_or("bundle size overflow")?;
    if available < required {
        return Err(format!(
            "insufficient host memory for an isolated load: {available} bytes available, need at least {required}"
        )
        .into());
    }
    Ok(available)
}

fn docker() -> Command {
    let mut command = Command::new(DOCKER);
    // Never inherit a remote Docker host/context, proxy, or Halogen config.
    command
        .env_clear()
        .args(["--host", "unix:///var/run/docker.sock"]);
    command
}

fn container_create(cpu_threads: Option<usize>) -> Command {
    let mut command = docker();
    command.arg("create");
    // Docker's sanitized client environment does not cross this boundary.
    // Pass explicit values to the existing isolated generation container.
    if let Some(budget) = cpu_threads {
        for key in brewery_hash_pool::CPU_ENV {
            command.args(["--env", &format!("{key}={budget}")]);
        }
    }
    command
}

fn docker_checked(command: &mut Command, timeout: Duration) -> Result<String> {
    docker_capture(command, timeout, None, 65536)
}

fn docker_capture(
    command: &mut Command,
    timeout: Duration,
    stdin: Option<fs::File>,
    limit: u64,
) -> Result<String> {
    docker_capture_observed(command, timeout, stdin, limit, None)
}

fn docker_capture_observed(
    command: &mut Command,
    timeout: Duration,
    stdin: Option<fs::File>,
    limit: u64,
    mut counts: Option<(u64, &mut dyn FnMut(u64))>,
) -> Result<String> {
    let verb = command
        .get_args()
        .nth(2)
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let mut out = tempfile::tempfile()?;
    let mut err = tempfile::tempfile()?;
    let mut child = command
        .stdin(stdin.map_or_else(Stdio::null, Stdio::from))
        .stdout(out.try_clone()?)
        .stderr(err.try_clone()?)
        .spawn()?;
    let deadline = Instant::now() + timeout;
    let (mut offset, mut previous) = (0, 0);
    let status = loop {
        let status = child.try_wait()?;
        if let Some((total, observe)) = counts.as_mut() {
            if let Ok(Some(n)) = consume_counts(&err, &mut offset, &mut previous, *total) {
                observe(n);
            }
        }
        if let Some(status) = status {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("Docker {verb} timed out").into());
        }
        thread::sleep(Duration::from_millis(25));
    };
    fn read_capture(file: &mut fs::File, limit: u64) -> Result<String> {
        let mut bytes = Vec::new();
        file.seek(SeekFrom::Start(0))?;
        file.take(limit + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > limit {
            return Err("Docker response exceeds size limit".into());
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    if !status.success() {
        return Err(format!(
            "Docker {verb} failed ({status}): {}",
            read_capture(&mut err, 65536)?.trim()
        )
        .into());
    }
    let mut stdout = read_capture(&mut out, limit)?;
    if verb == "logs" {
        stdout.push_str(&read_capture(&mut err, 65536)?);
    }
    Ok(stdout)
}

fn consume_counts(
    file: &fs::File,
    offset: &mut u64,
    previous: &mut u64,
    total: u64,
) -> std::io::Result<Option<u64>> {
    let before = *previous;
    let mut bytes = [0u8; 4096];
    // pread does not change the child writer's offset. Counters are strict
    // sequential ordinals, and only the bundled helper emits them.
    while *offset < 65536 {
        let cap = bytes.len().min((65536 - *offset) as usize);
        let n = file.read_at(&mut bytes[..cap], *offset)?;
        let Some(end) = bytes[..n].iter().rposition(|&b| b == b'\n') else {
            break;
        };
        *offset += end as u64 + 1;
        for line in bytes[..end].split(|&b| b == b'\n') {
            if let Some(n) = line
                .strip_prefix(b"brewery-completed:")
                .and_then(|line| std::str::from_utf8(line).ok())
                .filter(|line| !line.is_empty() && line.bytes().all(|c| c.is_ascii_digit()))
                .and_then(|line| line.parse::<u64>().ok())
            {
                if n == *previous + 1 && n <= total {
                    *previous = n;
                }
            }
        }
        if n < cap {
            break;
        }
    }
    Ok((*previous > before).then_some(*previous))
}

struct Container<'a> {
    id: &'a str,
    remove: bool,
}

impl Container<'_> {
    fn cleanup(&mut self) -> Result<()> {
        // Give the pinned engine time to release GPU allocations cleanly first.
        let stopped = docker_checked(
            docker().args(["stop", "--time", "90", self.id]),
            Duration::from_secs(105),
        );
        let removed = docker_checked(
            docker().args(["rm", "--volumes", self.id]),
            Duration::from_secs(15),
        );
        if removed.is_ok() {
            self.remove = false;
        }
        stopped?;
        removed?;
        Ok(())
    }
}

fn renew(root: &Path) -> Result<()> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } == -1 {
        return Err(std::io::Error::last_os_error().into());
    }
    let timestamp = i128::from(time.tv_sec) * 1_000_000_000 + i128::from(time.tv_nsec);
    let mut lease = tempfile::NamedTempFile::new_in(root)?;
    write!(lease, "{timestamp}")?;
    lease.persist(root.join("current"))?;
    Ok(())
}

struct Heartbeat {
    stop: std::sync::mpsc::Sender<()>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Heartbeat {
    fn start(path: std::path::PathBuf) -> Result<Self> {
        let (stop, rx) = std::sync::mpsc::channel();
        let thread = thread::Builder::new()
            .name("halogen-lease".to_owned())
            .spawn(move || {
                while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                    rx.recv_timeout(Duration::from_secs(2))
                {
                    if let Err(e) = renew(&path) {
                        eprintln!("halogen-proof: could not renew runtime lease: {e}");
                        break;
                    }
                }
            })?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Container<'_> {
    fn drop(&mut self) {
        if self.remove {
            if let Err(e) = self.cleanup() {
                eprintln!("could not clean up isolated container {}: {e}", self.id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::container_create;

    #[test]
    fn forwards_budget_to_the_container_not_the_docker_client() {
        let command = container_create(Some(2));
        let mut args = command.get_args();
        assert_eq!(args.next().unwrap(), "--host");
        assert_eq!(args.next().unwrap(), "unix:///var/run/docker.sock");
        assert_eq!(args.next().unwrap(), "create");
        for key in brewery_hash_pool::CPU_ENV {
            assert_eq!(args.next().unwrap(), "--env");
            assert_eq!(args.next().unwrap(), format!("{key}=2").as_str());
        }
        assert!(args.next().is_none());
        assert_eq!(command.get_envs().count(), 0);
        assert_eq!(container_create(None).get_args().count(), 3);
    }
}
