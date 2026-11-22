use crate::{
    backend_hf,
    dispatch::Dispatcher,
    domain::{Host, Job, Model},
    infra_hosts, local_edit, remote_edit,
    suggestions::read_only_dispatch,
};
use brewery_worker::{PathState, StatRequest};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{process::Command, sync::Semaphore, time::timeout};

const CACHE_TTL: Duration = Duration::from_secs(30);
const IN_FLIGHT: usize = 4;
const WORKER_START_INTERVAL: Duration = Duration::from_millis(250);
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

pub struct ModelAvailability {
    dispatch: PathBuf,
    backend: Option<(PathBuf, PathBuf, Option<PathBuf>)>,
    ssh: Arc<infra_hosts::Registry>,
    local_edit: Arc<local_edit::Lanes>,
    remote_edit: Arc<remote_edit::Installer>,
    cache: Mutex<Cache>,
    permits: Arc<Semaphore>,
    config_permits: Arc<Semaphore>,
}

#[derive(Default)]
struct Cache {
    paths: HashMap<Probe, Entry>,
    workers: HashMap<String, WorkerProbe>,
}

struct WorkerProbe {
    started: Instant,
    in_flight: bool,
    config_gate: Arc<Semaphore>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct Probe {
    executor: Executor,
    // Static lanes are fixed for this controller instance; managed SSH
    // workers have an installation binding. Remote editor IDs are generations.
    generation: Option<String>,
    host: String,
    path: PathBuf,
}

enum Entry {
    Waiting(Instant),
    Pending,
    Checked(Instant, Check),
}

#[derive(Clone, Copy)]
enum Check {
    State(PathState),
    Unsupported,
    Unknown,
    EditorUnverified,
}

impl Check {
    fn apply(self, model: &mut Model) {
        let (available, reason) = match self {
            Self::State(PathState::File { bytes }) => {
                // Do not pair a receipt hash with a size from a different file revision.
                if model.sha256.is_none() {
                    model.bytes = Some(bytes);
                }
                (Some(true), None)
            }
            Self::State(PathState::Directory) => (Some(true), None),
            Self::State(PathState::Missing) => (
                Some(false),
                Some("The model path does not exist on the worker."),
            ),
            Self::State(PathState::OutsideRoots) => (
                Some(false),
                Some("The worker does not allow access to this path."),
            ),
            Self::Unsupported => (
                None,
                Some("Worker does not report file state. Update the worker."),
            ),
            Self::Unknown => (
                None,
                Some("The controller could not check this path on the worker."),
            ),
            Self::EditorUnverified => (
                None,
                Some(
                    "The standard worker could not find or access this path. Editor access is not checked.",
                ),
            ),
        };
        model.available = available;
        model.reason = reason.map(str::to_owned);
    }
}

// Dispatch uses a separate config for these executors. A successful model
// publication may not be visible in the owning host's default worker state.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum Executor {
    Default { editor: Option<Editor> },
    Backend { editor: bool },
    LocalEdit,
    RemoteEdit(uuid::Uuid),
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum Editor {
    Local,
    Remote(uuid::Uuid),
}

impl Executor {
    fn for_model(host: &Host, model: &Model, producer: Option<&Job>) -> Self {
        if model.host == backend_hf::ID {
            return Self::Backend {
                editor: model.kind == "weight-calibration"
                    || producer.is_some_and(|job| job.kind == "weight_apply"),
            };
        }
        if let Some(job) = producer {
            if job.local_edit {
                return Self::LocalEdit;
            }
            if let Some(editor) = job.remote_editor {
                return Self::RemoteEdit(editor);
            }
        }
        let weight_input = matches!(
            model.kind.as_str(),
            "gguf" | "hf" | "safetensors" | "hgn" | "weight-calibration"
        );
        let editor = if !weight_input {
            None
        } else if host.local_edit {
            Some(Editor::Local)
        } else {
            host.remote_editor.map(Editor::Remote)
        };
        Self::Default { editor }
    }
}

impl ModelAvailability {
    pub fn new(dispatcher: &Dispatcher) -> Self {
        Self {
            dispatch: dispatcher.command.clone(),
            backend: dispatcher.backend.as_ref().map(|lane| {
                (
                    lane.dispatch.clone(),
                    lane.dispatch_config.clone(),
                    lane.edit_config.clone(),
                )
            }),
            ssh: dispatcher.ssh_hosts.clone(),
            local_edit: dispatcher.local_edit.clone(),
            remote_edit: dispatcher.remote_edit.clone(),
            cache: Mutex::new(Cache::default()),
            permits: Arc::new(Semaphore::new(IN_FLIGHT)),
            config_permits: Arc::new(Semaphore::new(IN_FLIGHT)),
        }
    }

    pub fn refresh(self: &Arc<Self>, hosts: &[Host], jobs: &[Job], models: &mut [Model]) {
        let hosts: HashMap<_, _> = hosts.iter().map(|host| (host.id.as_str(), host)).collect();
        let jobs: HashMap<_, _> = jobs.iter().map(|job| (job.id.as_str(), job)).collect();
        let now = Instant::now();
        let mut cache = self.cache.lock().expect("model state cache");
        let mut registered = HashSet::new();
        let mut due = HashMap::<_, (Instant, Probe)>::new();
        for model in models {
            let Some(host) = hosts.get(model.host.as_str()) else {
                model.available = Some(false);
                model.reason = Some("The owning worker is not configured or is disabled.".into());
                continue;
            };
            let producer = model.job_id.as_deref().and_then(|id| jobs.get(id).copied());
            let probe = Probe {
                executor: Executor::for_model(host, model, producer),
                generation: host.provision_binding.clone(),
                host: model.host.clone(),
                // Normalize lexically; path lookup and symlink checks belong to the worker.
                path: Path::new(&model.path).components().collect(),
            };
            registered.insert(probe.clone());
            let due_at = match *cache
                .paths
                .entry(probe.clone())
                .or_insert(Entry::Waiting(now))
            {
                Entry::Pending => continue,
                Entry::Checked(at, check) if now.saturating_duration_since(at) < CACHE_TTL => {
                    check.apply(model);
                    continue;
                }
                Entry::Checked(at, _) => at + CACHE_TTL,
                Entry::Waiting(at) => at,
            };
            if cache.workers.get(&host.id).is_some_and(|worker| {
                worker.in_flight
                    || now.saturating_duration_since(worker.started) < WORKER_START_INTERVAL
            }) {
                continue;
            }
            let next = due
                .entry(host.id.as_str())
                .or_insert_with(|| (due_at, probe.clone()));
            if due_at < next.0 {
                *next = (due_at, probe);
            }
        }
        cache
            .paths
            .retain(|probe, entry| matches!(entry, Entry::Pending) || registered.contains(probe));
        cache.workers.retain(|host, worker| {
            worker.in_flight
                || worker.config_gate.available_permits() == 0
                || hosts.contains_key(host.as_str())
        });
        // Give each worker one slot at a time, rotating by least recent start.
        // Within a worker, a path already due stays ahead of later registrations.
        let mut due: Vec<_> = due.into_iter().collect();
        due.sort_by_key(|(host, _)| (cache.workers.get(*host).map(|worker| worker.started), *host));
        for (_, (_, probe)) in due {
            let Ok(permit) = self.permits.clone().try_acquire_owned() else {
                break;
            };
            let worker = cache
                .workers
                .entry(probe.host.clone())
                .or_insert_with(|| WorkerProbe {
                    started: now,
                    in_flight: true,
                    config_gate: Arc::new(Semaphore::new(1)),
                });
            worker.started = now;
            worker.in_flight = true;
            let config_gate = worker.config_gate.clone();
            cache.paths.insert(probe.clone(), Entry::Pending);
            let this = self.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let check = timeout(PROBE_TIMEOUT, this.check(&probe, config_gate))
                    .await
                    .unwrap_or(Check::Unknown);
                let mut cache = this.cache.lock().expect("model state cache");
                cache
                    .workers
                    .get_mut(&probe.host)
                    .expect("pending worker probe")
                    .in_flight = false;
                cache
                    .paths
                    .insert(probe, Entry::Checked(Instant::now(), check));
            });
        }
    }

    async fn command(&self, probe: &Probe, config_gate: Arc<Semaphore>) -> Option<Command> {
        let host = &probe.host;
        let mut command = Command::new(&self.dispatch);
        let config = match probe.executor {
            Executor::Default { .. } if probe.generation.is_some() => {
                // A timeout cannot cancel a running filesystem syscall. Keep
                // both permits in the blocking task. One worker with a stuck
                // config check must not use all slots on subsequent probes.
                let worker_permit = config_gate.try_acquire_owned().ok()?;
                let permit = self.config_permits.clone().try_acquire_owned().ok()?;
                let ssh = self.ssh.clone();
                let host = host.clone();
                tokio::task::spawn_blocking(move || {
                    let _permits = (permit, worker_permit);
                    ssh.config_for(&[&host])
                })
                .await
                .ok()?
                .ok()?
            }
            Executor::Default { .. } => None,
            Executor::Backend { editor } => {
                let (dispatch, primary, edit) = self.backend.as_ref()?;
                command = Command::new(dispatch);
                command.env_clear();
                Some(if editor { edit.as_ref()? } else { primary }.clone())
            }
            Executor::LocalEdit => {
                let lane = self.local_edit.get(host)?;
                command = Command::new(&self.local_edit.dispatch);
                command.env_clear().env("PATH", "/usr/bin:/bin");
                Some(lane.config.clone())
            }
            Executor::RemoteEdit(id) => Some(self.remote_edit.config_for(id)),
        };
        if let Some(config) = config {
            command.env("BREWERY_DISPATCH_CONFIG", config);
        }
        Some(command)
    }

    async fn check(&self, probe: &Probe, config_gate: Arc<Semaphore>) -> Check {
        let Some(mut command) = self.command(probe, config_gate).await else {
            return Check::Unknown;
        };
        let Ok(body) = serde_json::to_vec(&StatRequest {
            path: probe.path.to_string_lossy().into_owned(),
        }) else {
            return Check::Unknown;
        };
        command.args(["stat", &probe.host]);
        let Ok((output, status)) = read_only_dispatch(command, body).await else {
            return Check::Unknown;
        };
        if status.success() {
            return match serde_json::from_slice(&output) {
                Ok(PathState::Missing | PathState::OutsideRoots)
                    if matches!(probe.executor, Executor::Default { editor: Some(_) }) =>
                {
                    // Read-only Stat on the ordinary lane cannot attest
                    // approval on a remote editor or under a per-job local grant.
                    Check::EditorUnverified
                }
                Ok(state) => Check::State(state),
                Err(_) => Check::Unknown,
            };
        }
        #[derive(serde::Deserialize)]
        struct ErrorReply {
            error: String,
        }
        let unsupported = serde_json::from_slice::<ErrorReply>(&output).is_ok_and(|reply| {
            reply.error.contains("unknown variant `stat`, expected")
                || reply.error == "usage: brewery-dispatch ACTION HOST JOB_ID | brewery-dispatch suggest-paths HOST"
        });
        if unsupported {
            Check::Unsupported
        } else {
            Check::Unknown
        }
    }
}
