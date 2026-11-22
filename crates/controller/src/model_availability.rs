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
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{process::Command, sync::Semaphore};

const CACHE_TTL: Duration = Duration::from_secs(30);
const IN_FLIGHT: usize = 4;

pub struct ModelAvailability {
    dispatch: PathBuf,
    backend: Option<(PathBuf, PathBuf, Option<PathBuf>)>,
    ssh: Arc<infra_hosts::Registry>,
    local_edit: Arc<local_edit::Lanes>,
    remote_edit: Arc<remote_edit::Installer>,
    cache: Mutex<HashMap<String, Entry>>,
    permits: Arc<Semaphore>,
}

enum Entry {
    Pending,
    Checked(Instant, Check),
}

#[derive(Clone, Copy)]
enum Check {
    State(PathState),
    Unsupported,
    Unknown,
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
        };
        model.available = available;
        model.reason = reason.map(str::to_owned);
    }
}

// Dispatch uses a separate config for these executors. A successful model
// publication may not be visible in the owning host's default worker state.
enum Executor {
    Default,
    Backend { editor: bool },
    LocalEdit,
    RemoteEdit(uuid::Uuid),
}

impl Executor {
    fn for_model(model: &Model, producer: Option<&Job>) -> Self {
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
        Self::Default
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
            cache: Mutex::new(HashMap::new()),
            permits: Arc::new(Semaphore::new(IN_FLIGHT)),
        }
    }

    pub fn refresh(self: &Arc<Self>, hosts: &[Host], jobs: &[Job], models: &mut [Model]) {
        let worker_ids: HashSet<_> = hosts.iter().map(|host| host.id.as_str()).collect();
        let mut cache = self.cache.lock().expect("model state cache");
        let registered: HashSet<_> = models
            .iter()
            .filter(|model| worker_ids.contains(model.host.as_str()))
            .map(|model| model.id.as_str())
            .collect();
        cache.retain(|id, entry| {
            matches!(entry, Entry::Pending) || registered.contains(id.as_str())
        });
        let mut due = Vec::new();
        for model in models {
            if !worker_ids.contains(model.host.as_str()) {
                model.available = Some(false);
                model.reason = Some("The owning worker is not configured or is disabled.".into());
                continue;
            }
            let last_check = match cache.get(&model.id) {
                Some(Entry::Pending) => continue,
                Some(Entry::Checked(at, check)) if at.elapsed() < CACHE_TTL => {
                    check.apply(model);
                    continue;
                }
                Some(Entry::Checked(at, _)) => Some(*at),
                None => None,
            };
            due.push((last_check, model));
        }
        // New and least recently checked records go first, so a large
        // library or several slow paths cannot starve the last records.
        due.sort_by_key(|(last_check, _)| *last_check);
        let jobs: HashMap<_, _> = jobs.iter().map(|job| (job.id.as_str(), job)).collect();
        for (_, model) in due {
            let Ok(permit) = self.permits.clone().try_acquire_owned() else {
                break;
            };
            let producer = model.job_id.as_deref().and_then(|id| jobs.get(id).copied());
            let executor = Executor::for_model(model, producer);
            let (id, host, path) = (model.id.clone(), model.host.clone(), model.path.clone());
            cache.insert(id.clone(), Entry::Pending);
            let this = self.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let check = this.check(executor, &host, path).await;
                this.cache
                    .lock()
                    .expect("model state cache")
                    .insert(id, Entry::Checked(Instant::now(), check));
            });
        }
    }

    fn command(&self, executor: Executor, host: &str) -> Option<Command> {
        let mut command = Command::new(&self.dispatch);
        let config = match executor {
            Executor::Default => self.ssh.config_for(&[host]).ok()?,
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

    async fn check(&self, executor: Executor, host: &str, path: String) -> Check {
        let Some(mut command) = self.command(executor, host) else {
            return Check::Unknown;
        };
        let Ok(body) = serde_json::to_vec(&StatRequest { path }) else {
            return Check::Unknown;
        };
        command.args(["stat", host]);
        let Ok((output, status)) = read_only_dispatch(command, body).await else {
            return Check::Unknown;
        };
        if status.success() {
            return serde_json::from_slice(&output)
                .map(Check::State)
                .unwrap_or(Check::Unknown);
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
