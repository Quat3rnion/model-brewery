use anyhow::{Context, Result, bail, ensure};
use brewery_hash_pool::ProgressEvent;
use brewery_worker::*;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    io::{BufRead, BufReader, Read},
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Deserialize)]
struct Config {
    hosts: BTreeMap<String, Host>,
}

#[derive(Deserialize)]
struct Host {
    ssh: Option<String>,
    ssh_config: Option<PathBuf>,
    local: Option<bool>,
    worker: Option<PathBuf>,
    config: Option<PathBuf>,
    remote_worker: Option<String>,
    remote_config: Option<String>,
}

impl Host {
    fn command(&self) -> Result<Command> {
        match (
            &self.ssh,
            &self.ssh_config,
            self.local,
            &self.worker,
            &self.config,
            &self.remote_worker,
            &self.remote_config,
        ) {
            (Some(alias), config, None, None, None, worker, worker_config)
                if (worker.is_none() && worker_config.is_none())
                    || (config.is_some() && worker.is_some() && worker_config.is_some()) =>
            {
                token(alias)?;
                let mut c = Command::new("/usr/bin/ssh");
                if let Some(config) = config {
                    absolute(config)?;
                    c.arg("-F").arg(config);
                }
                c.args([
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "ServerAliveInterval=15",
                    "-o",
                    "ServerAliveCountMax=2",
                    "-o",
                    "ClearAllForwardings=yes",
                    "-o",
                    "ForwardAgent=no",
                    "-o",
                    "ForwardX11=no",
                    "-o",
                    "PermitLocalCommand=no",
                    "-o",
                    "ControlMaster=no",
                    "-o",
                    "ControlPath=none",
                    "--",
                    alias,
                ]);
                if let (Some(worker), Some(config)) = (worker, worker_config) {
                    ensure!(
                        managed_remote_path(worker)
                            && managed_remote_path(config)
                            && worker.strip_suffix("/brewery-worker")
                                == config.strip_suffix("/worker.json")
                            && worker.ends_with("/brewery-worker"),
                        "invalid managed SSH executable or config path"
                    );
                    c.args([
                        format!("BREWERY_WORKER_CONFIG={config}"),
                        worker.clone(),
                        "rpc".into(),
                    ]);
                } else {
                    c.args(["brewery-worker", "rpc"]);
                }
                Ok(c)
            }
            (None, None, Some(true), Some(worker), Some(config), None, None) => {
                absolute(worker)?;
                absolute(config)?;
                let mut c = Command::new(worker);
                c.arg("rpc").env("BREWERY_WORKER_CONFIG", config);
                Ok(c)
            }
            _ => bail!("host must select exactly one SSH or explicit local worker"),
        }
    }
}

fn managed_remote_path(path: &str) -> bool {
    path.len() <= 1050
        && path.strip_prefix('/').is_some_and(|path| {
            path.split('/').all(|part| {
                !matches!(part, "" | "." | "..")
                    && part
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
            })
        })
}

#[derive(Debug)]
enum Error {
    Transport(anyhow::Error),
    Remote(anyhow::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(e) | Self::Remote(e) => write!(f, "{e:#}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self {
        Self::Remote(e)
    }
}
type Wire<T> = std::result::Result<T, Error>;

impl Config {
    fn describe(
        &self,
        source: &str,
        dest: &str,
        id: &str,
        input: Input,
        index: usize,
        count: usize,
    ) -> Wire<Description> {
        let mut c = self
            .hosts
            .get(source)
            .context("host not allowlisted")?
            .command()?;
        c.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = c
            .spawn()
            .context("launch source worker transport")
            .map_err(Error::Transport)?;
        let result = (|| {
            serde_json::to_writer(
                child.stdin.take().unwrap(),
                &Rpc::DescribeProgress {
                    input,
                    index,
                    count,
                },
            )
            .context("write source request")
            .map_err(Error::Transport)?;
            let mut stdout = BufReader::new(child.stdout.take().unwrap());
            let mut frame = Vec::new();
            let mut last_forward: Option<Instant> = None;
            let mut had_threads = false;
            loop {
                frame.clear();
                (&mut stdout)
                    .take(LIMIT + 1)
                    .read_until(b'\n', &mut frame)
                    .context("read source response")
                    .map_err(Error::Transport)?;
                if frame.len() as u64 > LIMIT || frame.last() != Some(&b'\n') {
                    return Err(Error::Transport(anyhow::anyhow!(
                        "incomplete or oversized source response"
                    )));
                }
                let reply: Value = serde_json::from_slice(&frame)
                    .context("invalid source response")
                    .map_err(Error::Transport)?;
                if let Some(detail) = reply.get("progress") {
                    if frame.len() as u64 > PROGRESS_LIMIT {
                        return Err(Error::Transport(anyhow::anyhow!(
                            "oversized source progress"
                        )));
                    }
                    let detail: ProgressDetail = serde_json::from_value(detail.clone())
                        .context("invalid source progress")
                        .map_err(Error::Transport)?;
                    if !detail.valid()
                        || detail.input_index != Some(index as u64 + 1)
                        || detail.input_count != Some(count as u64)
                        || !matches!(detail.phase.as_str(), "inventory_input" | "hash_input")
                    {
                        return Err(Error::Transport(anyhow::anyhow!("invalid source progress")));
                    }
                    if (had_threads && detail.threads.is_empty())
                        || last_forward.is_none_or(|at| at.elapsed() >= Duration::from_millis(250))
                    {
                        had_threads = !detail.threads.is_empty();
                        self.rpc::<Value>(
                            dest,
                            Rpc::StageProgress {
                                id: id.into(),
                                detail,
                            },
                        )?;
                        last_forward = Some(Instant::now());
                    }
                } else if let Some(error) = reply["error"].as_str() {
                    return Err(Error::Remote(anyhow::anyhow!("{source}: {error}")));
                } else {
                    return serde_json::from_value(reply["ok"].clone())
                        .context("invalid source description")
                        .map_err(Error::Transport);
                }
            }
        })();
        if result.is_err() {
            let _ = child.kill();
        }
        let status = child
            .wait()
            .context("wait for source transport")
            .map_err(Error::Transport)?;
        if result.is_ok() && !status.success() {
            return Err(Error::Transport(anyhow::anyhow!(
                "worker transport to {source} exited {status}"
            )));
        }
        result
    }

    fn rpc<T: DeserializeOwned>(&self, alias: &str, request: Rpc) -> Wire<T> {
        let mut c = self
            .hosts
            .get(alias)
            .context("host not allowlisted")?
            .command()?;
        c.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = c
            .spawn()
            .context("launch worker transport")
            .map_err(Error::Transport)?;
        let mut stdin = child.stdin.take().unwrap();
        if let Err(e) = serde_json::to_writer(&mut stdin, &request) {
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
            return Err(Error::Transport(e.into()));
        }
        drop(stdin);
        let limit = match request {
            Rpc::SuggestPaths(_) | Rpc::Stat(_) => SUGGEST_JSON_LIMIT,
            Rpc::TraceRead { .. } => trace::RESPONSE_CAP as u64 + 128,
            _ => LIMIT,
        };
        let value = decode_limited::<Value>(child.stdout.take().unwrap(), limit);
        if value.is_err() {
            let _ = child.kill();
        }
        let status = child
            .wait()
            .context("wait for worker transport")
            .map_err(Error::Transport)?;
        if !status.success() {
            return Err(Error::Transport(anyhow::anyhow!(
                "worker transport to {alias} exited {status}"
            )));
        }
        let response = value.map_err(Error::Transport)?;
        if let Some(error) = response["error"].as_str() {
            return Err(Error::Remote(anyhow::anyhow!("{alias}: {error}")));
        }
        let value = response
            .get("ok")
            .context("invalid worker response")
            .map_err(Error::Transport)?;
        serde_json::from_value(value.clone())
            .context("invalid worker reply")
            .map_err(Error::Transport)
    }
}

#[derive(Default)]
struct InitialRelaySlots(BTreeSet<usize>);

impl InitialRelaySlots {
    fn should_publish(&mut self, slot: usize, event: ProgressEvent) -> bool {
        // Pool slots can start in any order. Publish each slot's first
        // assignment promptly even if all other slots are blocked on reads.
        event == ProgressEvent::Started && self.0.insert(slot)
    }
}

fn stage(cfg: &Config, action: &str, dest: &str, id: &str, job: Job) -> Wire<Reply> {
    if action == "cleanup" {
        return cfg.rpc(dest, Rpc::Cleanup { id: id.into() });
    }
    let mut total = 0u64;
    let mut missing_bytes = 0u64;
    let mut free = u64::MAX;
    let mut seen = BTreeSet::new();
    let preview = action == "preview_transfer";
    let input_count = job.inputs.len();
    for (index, input) in job.inputs.into_iter().enumerate() {
        ensure_wire(
            index != 0
                || !(weight_source::job(&job.kind) || job.kind == brewery_worker::exl3_layer::JOB)
                || !weight_source::supports(&input.kind)
                || input.host == dest,
            "SafeTensors weight jobs require a direct local source",
        )?;
        if action == "start" && input.host == dest {
            cfg.rpc::<Input>(
                dest,
                Rpc::Resolve {
                    id: id.into(),
                    index,
                    input,
                    bundle: None,
                },
            )?;
            continue;
        }
        let src = input.host.clone();
        let d = cfg.describe(&src, dest, id, input.clone(), index, input_count)?;
        d.bundle.validate()?;
        let bytes = d.bundle.bytes()?;
        total = total.checked_add(bytes).context("transfer size overflow")?;
        let plan: Plan = cfg.rpc(
            dest,
            Rpc::Plan {
                id: id.into(),
                bundle: d.bundle.clone(),
                preview,
            },
        )?;
        if plan.offsets.len() != d.bundle.files.len() {
            return Err(Error::Transport(anyhow::anyhow!(
                "target returned invalid offsets"
            )));
        }
        if d.bundle
            .files
            .iter()
            .zip(&plan.offsets)
            .any(|(f, offset)| *offset > f.bytes)
        {
            return Err(Error::Transport(anyhow::anyhow!(
                "target returned an offset beyond the source file"
            )));
        }
        free = free.min(plan.available_bytes);
        if seen.insert(d.bundle.digest.clone()) {
            let missing =
                d.bundle
                    .files
                    .iter()
                    .zip(&plan.offsets)
                    .try_fold(0u64, |n, (f, o)| {
                        n.checked_add(
                            f.bytes
                                .checked_sub(*o)
                                .context("offset exceeds source length")?,
                        )
                        .context("size overflow")
                    })?;
            missing_bytes = missing_bytes
                .checked_add(missing)
                .context("size overflow")?;
        }
        if !preview && !plan.cached {
            let mut copied = plan.offsets.iter().sum::<u64>();
            let mut files_done = d
                .bundle
                .files
                .iter()
                .zip(&plan.offsets)
                .filter(|(entry, offset)| entry.bytes == **offset)
                .count() as u64;
            let pending: Vec<_> = d
                .bundle
                .files
                .iter()
                .zip(plan.offsets)
                .filter(|(entry, offset)| *offset < entry.bytes)
                .collect();
            // Bound live transport processes and buffered chunks independently
            // of the CPU budget. Each file keeps its existing ordered offsets.
            const RELAYS: usize = 4;
            let mut slots = PoolSlots::default();
            let mut last = None::<Instant>;
            let mut initial_slots = InitialRelaySlots::default();
            let send = |copied, files_done, slots: &PoolSlots| {
                let detail = ProgressDetail {
                    phase: "copy_input".into(),
                    unit: ProgressUnit::Bytes,
                    completed: copied,
                    total: Some(bytes),
                    current_file: None,
                    files_completed: Some(files_done),
                    files_total: Some(d.bundle.files.len() as u64),
                    input_index: Some(index as u64 + 1),
                    input_count: Some(input_count as u64),
                    updated_at_ms: 1, // Destination replaces this with its timestamp.
                    threads: slots.rows(),
                };
                cfg.rpc::<Value>(
                    dest,
                    Rpc::StageProgress {
                        id: id.into(),
                        detail: bound_progress(detail, 192),
                    },
                )
                .map(|_| ())
            };
            let result = pending.chunks(RELAYS).try_for_each(|batch| -> Wire<()> {
                brewery_hash_pool::map_cpu_observed(
                    batch,
                    |_, &(entry, start), checkpoint| {
                        let mut offset = start;
                        while offset < entry.bytes {
                            checkpoint(0)?;
                            let data: String = cfg.rpc(
                                &src,
                                Rpc::Read {
                                    root: d.root.clone(),
                                    entry: entry.clone(),
                                    offset,
                                },
                            )?;
                            checkpoint(0)?;
                            let next: u64 = cfg.rpc(
                                dest,
                                Rpc::Write {
                                    id: id.into(),
                                    digest: d.bundle.digest.clone(),
                                    entry: entry.clone(),
                                    offset,
                                    data,
                                },
                            )?;
                            if next <= offset || next > entry.bytes {
                                return Err(Error::Transport(anyhow::anyhow!(
                                    "target returned invalid resume offset"
                                ))
                                .into());
                            }
                            checkpoint(next - offset)?;
                            offset = next;
                        }
                        Ok(())
                    },
                    |slot, file, event| {
                        slots.update(
                            slot,
                            event,
                            batch[file].0.path.rsplit('/').next().unwrap_or(""),
                        );
                        match event {
                            ProgressEvent::Advanced(n) => copied += n,
                            ProgressEvent::Finished => files_done += 1,
                            _ => {}
                        }
                        let show_initial = initial_slots.should_publish(slot, event);
                        if event != ProgressEvent::Stopped
                            && (show_initial
                                || last.is_none_or(|at| at.elapsed() >= Duration::from_millis(300)))
                        {
                            send(copied, files_done, &slots)?;
                            last = Some(Instant::now());
                        }
                        Ok(())
                    },
                )
                .map(|_| ())
                .map_err(|e| e.downcast::<Error>().unwrap_or_else(Error::Remote))
            });
            if !pending.is_empty() {
                let cleared = send(copied, files_done, &PoolSlots::default());
                result.and(cleared)?;
            }
            cfg.rpc::<Value>(
                dest,
                Rpc::Seal {
                    id: id.into(),
                    digest: d.bundle.digest.clone(),
                },
            )?;
        }
        let resolved = Input {
            path: plan.path,
            host: dest.into(),
            ..input
        };
        cfg.rpc::<Input>(
            dest,
            Rpc::Resolve {
                id: id.into(),
                index,
                input: resolved,
                bundle: Some(d.bundle),
            },
        )?;
    }
    if action == "start" {
        cfg.rpc(dest, Rpc::Start { id: id.into() })
    } else {
        cfg.rpc(
            dest,
            Rpc::Finish {
                id: id.into(),
                failure: None,
                result: Some(
                    json!({"bytes":total,"missing_bytes":missing_bytes,"available_bytes":free}),
                ),
            },
        )
    }
}

fn config(dest: &str) -> Wire<Config> {
    token(dest)?;
    let path = env::var_os("BREWERY_DISPATCH_CONFIG")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME")
                .map(|h| PathBuf::from(h).join(".config/model-brewery/dispatch.json"))
        })
        .context("set BREWERY_DISPATCH_CONFIG")?;
    let cfg: Config = load(&path)?;
    ensure_wire(cfg.hosts.contains_key(dest), "host not allowlisted")?;
    Ok(cfg)
}

fn run() -> Wire<Value> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.as_slice() {
        [action, dest] if action == "suggest-paths" => {
            let cfg = config(dest)?;
            let request: SuggestRequest =
                decode_limited(std::io::stdin().lock(), SUGGEST_JSON_LIMIT)?;
            validate_suggest_prefix(&request.prefix)?;
            let reply: PathSuggestions = cfg.rpc(dest, Rpc::SuggestPaths(request))?;
            Ok(serde_json::to_value(reply).context("serialize path suggestions")?)
        }
        [action, dest, id] if action == "trace-read" => {
            token(id)?;
            let cfg = config(dest)?;
            let slice: trace::Slice = decode_limited(std::io::stdin().lock(), 2048)?;
            slice.check()?;
            cfg.rpc(dest, Rpc::TraceRead { id: id.into(), slice })
        }
        [action, dest] if action == "stat" => {
            let cfg = config(dest)?;
            let request: StatRequest = decode_limited(std::io::stdin().lock(), SUGGEST_JSON_LIMIT)?;
            let reply: PathState = cfg.rpc(dest, Rpc::Stat(request))?;
            Ok(serde_json::to_value(reply).context("serialize path state")?)
        }
        [action, dest, id] if !matches!(action.as_str(), "suggest-paths" | "stat") => {
            token(id)?;
            let cfg = config(dest)?;
            let reply = run_job(&cfg, action, dest, id)?;
            Ok(serde_json::to_value(reply).context("serialize reply")?)
        }
        _ => Err(anyhow::anyhow!(
            "usage: brewery-dispatch ACTION HOST JOB_ID | brewery-dispatch (suggest-paths|stat) HOST"
        )
        .into()),
    }
}

fn run_job(cfg: &Config, action: &str, dest: &str, id: &str) -> Wire<Reply> {
    match action {
        "status" => cfg.rpc(dest, Rpc::Status { id: id.into() }),
        "cancel" => cfg.rpc(dest, Rpc::Cancel { id: id.into() }),
        "start" | "transfer" | "preview_transfer" | "cleanup" => {
            let stdin = std::io::stdin();
            let job: Job = if action == "cleanup" && !stdin.is_terminal() {
                // Cleanup also accepts EOF for older callers.
                use std::io::BufRead;
                let mut lock = stdin.lock();
                if lock.fill_buf().context("read cleanup stdin")?.is_empty() {
                    cleanup_job()
                } else {
                    decode(lock)?
                }
            } else if action == "cleanup" {
                cleanup_job()
            } else {
                decode(stdin.lock())?
            };
            let resource_request = !job.resources.is_empty();
            ensure_wire(
                action != "cleanup" || job.inputs.is_empty(),
                "cleanup must not contain inputs",
            )?;
            ensure_wire(
                !matches!(action, "preview_transfer" | "transfer") || !job.inputs.is_empty(),
                "transfer needs at least one input",
            )?;
            let reply: Reply = cfg.rpc(
                dest,
                Rpc::Prepare {
                    id: id.into(),
                    action: action.into(),
                    job: job.clone(),
                },
            )?;
            if reply.state != "queued" {
                return Ok(reply);
            }
            if resource_request && !reply.resources_enforced {
                return cfg.rpc(
                    dest,
                    Rpc::Finish {
                        id: id.into(),
                        failure: Some("Worker does not support requested resource controls".into()),
                        result: None,
                    },
                );
            }
            match stage(cfg, action, dest, id, job) {
                Err(Error::Remote(e)) => cfg.rpc(
                    dest,
                    Rpc::Finish {
                        id: id.into(),
                        failure: Some(format!("{e:#}")),
                        result: None,
                    },
                ),
                result => result,
            }
        }
        _ => Err(anyhow::anyhow!("invalid dispatcher action").into()),
    }
}

use std::io::IsTerminal;

fn ensure_wire(condition: bool, message: &str) -> Wire<()> {
    if condition {
        Ok(())
    } else {
        Err(anyhow::anyhow!("{message}").into())
    }
}

fn cleanup_job() -> Job {
    Job {
        kind: "cache-cleanup".into(),
        inputs: vec![],
        options: object(),
        prompt: None,
        resources: resources::Limits::default(),
    }
}

fn main() {
    let (reply, status) = match run() {
        Ok(r) => (r, 0),
        Err(Error::Transport(e)) => (json!(Reply::error(format!("{e:#}"))), 1),
        Err(Error::Remote(e)) => (json!(Reply::error(format!("{e:#}"))), 2),
    };
    println!("{}", serde_json::to_string(&reply).unwrap());
    std::process::exit(status);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_publishes_both_initial_assignments_when_the_higher_slot_starts_first() {
        let mut initial = InitialRelaySlots::default();
        assert!(initial.should_publish(2, ProgressEvent::Started));
        assert!(!initial.should_publish(2, ProgressEvent::Advanced(0)));
        assert!(initial.should_publish(1, ProgressEvent::Started));
        assert!(!initial.should_publish(2, ProgressEvent::Finished));
        assert!(!initial.should_publish(2, ProgressEvent::Started));
    }

    #[test]
    fn ssh_alias_is_allowlisted_and_command_is_constant() {
        let mut h: Host = serde_json::from_value(json!({
            "ssh": "workhorse", "ssh_config": "/arboretum/ssh config",
            "name":"Workhorse", "capabilities":{"compare":true}, "baseline_sources":[]
        }))
        .unwrap();
        let c = h.command().unwrap();
        let argv: Vec<_> = c.get_args().map(|s| s.to_str().unwrap()).collect();
        assert_eq!(c.get_program(), "/usr/bin/ssh");
        assert_eq!(&argv[..2], ["-F", "/arboretum/ssh config"]);
        assert_eq!(
            &argv[argv.len() - 4..],
            ["--", "workhorse", "brewery-worker", "rpc"]
        );
        assert!(
            argv.contains(&"StrictHostKeyChecking=yes")
                && argv.contains(&"BatchMode=yes")
                && argv.contains(&"-T")
        );
        for invalid in [
            "-oProxyCommand=x",
            "host;touch /x",
            "user@host",
            "host\nother",
        ] {
            h.ssh = Some(invalid.into());
            assert!(h.command().is_err());
        }
        h.ssh = Some("workhorse".into());
        h.ssh_config = Some("relative".into());
        assert!(h.command().is_err());
    }
}
