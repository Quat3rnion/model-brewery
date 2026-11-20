use crate::{
    App, Error, backend_hf,
    domain::{Envelope, Job, ProgressDetail, State, candidates},
};
use serde::Deserialize;
use serde_json::{Map, Value, value::RawValue};
use std::{path::PathBuf, process::Stdio, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncRead, AsyncReadExt as _, AsyncWriteExt as _},
    process::Command,
    task::JoinSet,
    time::{Instant, sleep, timeout},
};

const RESULT_LIMIT: usize = 32 * 1024 * 1024;
const SHORT_RPC: Duration = Duration::from_secs(30);

pub struct Dispatcher {
    pub command: PathBuf,
    pub backend: Option<backend_hf::Lane>,
    pub local_edit: Arc<crate::local_edit::Lanes>,
    pub ssh_hosts: Arc<crate::infra_hosts::Registry>,
    pub remote_edit: Arc<crate::remote_edit::Installer>,
    pub launch_timeout: Duration,
    pub poll: Duration,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum WorkerState {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Missing,
}

#[derive(Deserialize)]
struct WorkerReply {
    state: WorkerState,
    progress: String,
    progress_detail: Option<Box<RawValue>>,
    result: Option<Map<String, Value>>,
    error: Option<String>,
}

impl Dispatcher {
    async fn call(
        &self,
        action: &str,
        job: &Job,
        envelope: Option<&Envelope>,
    ) -> Result<WorkerReply, Error> {
        let duration = if envelope.is_some() {
            self.launch_timeout
        } else {
            SHORT_RPC
        };
        let payload = envelope
            .map(|envelope| {
                if job.dispatch_resources.is_empty() {
                    serde_json::to_vec(envelope)
                } else {
                    let mut wire = serde_json::to_value(envelope)?;
                    wire["resources"] = serde_json::to_value(job.dispatch_resources)?;
                    serde_json::to_vec(&wire)
                }
            })
            .transpose()?;
        let backend = if job.host == backend_hf::ID {
            Some(
                self.backend
                    .as_ref()
                    .ok_or("Backend downloader is unavailable")?,
            )
        } else {
            None
        };
        let local = if job.local_edit {
            Some(
                self.local_edit
                    .get(&job.host)
                    .ok_or("Local CPU editor is unavailable")?,
            )
        } else {
            None
        };
        let mut command = Command::new(if local.is_some() {
            &self.local_edit.dispatch
        } else {
            backend.map_or(&self.command, |lane| &lane.dispatch)
        });
        if let Some(lane) = local {
            let config = if job.directory_revision.is_some() {
                lane.job_config(&job.id)
            } else {
                lane.config.clone()
            };
            command
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("BREWERY_DISPATCH_CONFIG", config);
        } else if let Some(id) = job.remote_editor {
            command.env("BREWERY_DISPATCH_CONFIG", self.remote_edit.config_for(id));
        } else if let Some(lane) = backend {
            command.env_clear().env(
                "BREWERY_DISPATCH_CONFIG",
                lane.config_for(&job.kind)
                    .ok_or("Backend executor unavailable for this operation")?,
            );
        } else {
            let mut ids = vec![job.host.as_str()];
            if let Some(envelope) = envelope {
                ids.extend(envelope.inputs.iter().map(|model| model.host.as_str()));
            }
            if let Some(config) = self.ssh_hosts.config_for(&ids)? {
                command.env("BREWERY_DISPATCH_CONFIG", config);
            }
        }
        let mut child = command
            .args([action, &job.host, &job.id])
            .stdin(if payload.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;
        let mut stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let io = async {
            let write = async {
                if let (Some(pipe), Some(data)) = (stdin.as_mut(), payload) {
                    // Some dispatch rejections close stdin without reading. Preserve
                    // the exit status and stdout rather than hiding it behind EPIPE.
                    if let Err(err) = pipe.write_all(&data).await
                        && err.kind() != std::io::ErrorKind::BrokenPipe
                    {
                        return Err(err);
                    }
                }
                drop(stdin);
                Ok(())
            };
            tokio::try_join!(
                write,
                read_limited(stdout, RESULT_LIMIT),
                read_limited(stderr, 1024 * 1024),
                child.wait()
            )
        };
        let (_, stdout, stderr, status) = timeout(duration, io)
            .await
            .map_err(|_| format!("{action} timed out"))??;
        if !status.success() {
            if backend.is_some() || local.is_some() {
                return Err(format!("Independent executor {action} failed ({status})").into());
            }
            let text = serde_json::from_slice::<Value>(&stdout)
                .ok()
                .and_then(|value| {
                    value
                        .get("error")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| String::from_utf8_lossy(&stderr).trim().to_owned());
            return Err(format!("{action} exited {status}: {}", bounded(&text, 2048)).into());
        }
        Ok(serde_json::from_slice(&stdout)?)
    }
}

pub(crate) async fn read_limited(
    pipe: impl AsyncRead + Unpin,
    max: usize,
) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    pipe.take(max as u64 + 1).read_to_end(&mut bytes).await?;
    if bytes.len() > max {
        return Err(std::io::Error::other(
            "dispatcher output exceeded size limit",
        ));
    }
    Ok(bytes)
}

pub async fn run(app: Arc<App>, dispatch: Dispatcher) -> Result<(), Error> {
    let dispatch = Arc::new(dispatch);
    let mut jobs = JoinSet::new();
    let recovered = {
        let mut store = app.store.lock().expect("store lock");
        let mut running = store.by_state(State::Running)?;
        for job in &mut running {
            if (job.host == backend_hf::ID
                && dispatch
                    .backend
                    .as_ref()
                    .and_then(|lane| lane.config_for(&job.kind))
                    .is_none())
                || (job.local_edit && dispatch.local_edit.get(&job.host).is_none())
            {
                job.state = if job.cancel_requested {
                    State::Cancelled
                } else {
                    State::Failed
                };
                job.progress = "Independent executor is unavailable after restart".into();
                job.progress_detail = None;
                job.error = None;
                store.update(job, None)?;
                continue;
            }
            job.progress = "Controller was interrupted; reconciling with worker".into();
            job.progress_detail = None;
            job.error = None;
            store.update(job, None)?;
        }
        running.retain(|job| job.state == State::Running);
        running
    };
    for job in recovered {
        let (app, dispatch) = (app.clone(), dispatch.clone());
        jobs.spawn(async move { drive(&app, &dispatch, job, true).await });
    }
    loop {
        for job in allocate(&app)? {
            let (app, dispatch) = (app.clone(), dispatch.clone());
            jobs.spawn(async move { drive(&app, &dispatch, job, false).await });
        }
        tokio::select! {
            Some(completed) = jobs.join_next() => completed??,
            () = sleep(dispatch.poll) => {},
        }
    }
}

fn allocate(app: &App) -> Result<Vec<Job>, Error> {
    let mut store = app.store.lock().expect("store lock");
    let hosts = app.effective_hosts()?;
    store.advance_conversions()?;
    let active = store.by_state(State::Running)?;
    let mut busy = std::collections::BTreeMap::<String, std::collections::BTreeSet<u64>>::new();
    for job in &active {
        busy.entry(job.host.clone())
            .or_default()
            .insert(job.host_slot.unwrap_or(0));
    }
    // Recovered work can have an unchanged alias but a disabled pool flag.
    // Treat it as exclusive until it finishes.
    if active.iter().any(|job| {
        !job.pool_slot
            || !hosts
                .iter()
                .any(|host| host.id == job.host && host.quantization_pool)
    }) {
        return Ok(Vec::new());
    }
    let mut launch = Vec::new();
    for mut job in store.by_state(State::Queued)? {
        let envelope = store.envelope(&job.id)?;
        let eligible = candidates(&job, &envelope, &hosts);
        if eligible.is_empty()
            || !envelope
                .inputs
                .iter()
                .all(|input| hosts.iter().any(|host| host.id == input.host))
        {
            job.state = State::Failed;
            job.progress = "Worker configuration changed before dispatch".into();
            job.error = Some("No eligible destination or input host remains".into());
            store.update(&mut job, None)?;
            continue;
        }
        let pool_slot = job.kind == "convert" && eligible.iter().all(|host| host.quantization_pool);
        // A global task is a barrier: never skip it to start work behind it.
        if !pool_slot && !busy.is_empty() {
            break;
        }
        let Some((host, slot)) = eligible.into_iter().find_map(|host| {
            let cap = if pool_slot {
                host.effective_resources().concurrency.unwrap_or(1)
            } else {
                1
            };
            let occupied = busy.get(&host.id);
            if occupied.is_some_and(|slots| slots.len() as u64 >= cap) {
                return None;
            }
            (0..cap)
                .find(|slot| occupied.is_none_or(|slots| !slots.contains(slot)))
                .map(|slot| (host, slot))
        }) else {
            // A pinned, busy pool worker need not block another idle worker.
            continue;
        };
        job.host = host.id.clone();
        if job.local_edit {
            let Some(access) = app
                .editor_grants(&job, &envelope)?
                .filter(|access| access.revision == job.directory_revision)
            else {
                job.state = State::Failed;
                job.progress = "Model directory access changed before dispatch".into();
                job.error =
                    Some("Review the current model directory access and submit a new job".into());
                store.update(&mut job, None)?;
                continue;
            };
            if access.revision.is_some() {
                let lane = app
                    .local_edit
                    .get(&job.host)
                    .ok_or("local editor unavailable")?;
                if lane
                    .snapshot(
                        &job.host,
                        &job.id,
                        &access.grants,
                        &app.local_edit.protected,
                    )
                    .is_err()
                {
                    job.state = State::Failed;
                    job.progress = "Cannot pin model directory access for this job".into();
                    store.update(&mut job, None)?;
                    continue;
                }
            }
        }
        job.pool_slot = pool_slot;
        job.host_slot = pool_slot.then_some(slot);
        job.dispatch_resources = if job.local_edit {
            Default::default()
        } else {
            host.resources.for_job(&job.kind)
        };
        if !pool_slot {
            job.dispatch_resources.concurrency = None;
        }
        job.state = State::Running;
        job.progress = "Dispatching to worker".into();
        store.update(&mut job, None)?;
        busy.entry(job.host.clone()).or_default().insert(slot);
        launch.push(job);
        if !pool_slot {
            break;
        }
    }
    Ok(launch)
}

async fn drive(app: &App, dispatch: &Dispatcher, job: Job, recovered: bool) -> Result<(), Error> {
    let envelope = app.store.lock().expect("store lock").envelope(&job.id)?;
    let action = match job.kind.as_str() {
        "transfer" => "transfer",
        "transfer_preview" => "preview_transfer",
        "cache_cleanup" => "cleanup",
        _ => "start",
    };
    let mut launch = !recovered;
    let mut established = false;
    let mut interrupt = false;
    let cadence = dispatch.poll.max(Duration::from_millis(250));
    let mut next_sample = Instant::now();
    loop {
        let current = app
            .store
            .lock()
            .expect("store lock")
            .job(&job.id)?
            .ok_or("active job disappeared")?;
        if launch && !current.cancel_requested && !interrupt {
            let mut hosts = app.effective_hosts_for_allocated()?;
            if job.local_edit
                && let Some((host, lane)) = hosts
                    .iter_mut()
                    .find(|h| h.id == job.host)
                    .zip(dispatch.local_edit.get(&job.host))
            {
                lane.advertise(host);
            }
            if let Some(generation) = job.remote_editor
                && dispatch.remote_edit.config_for(generation).is_file()
                && let Some(host) = hosts.iter_mut().find(|h| h.id == job.host)
            {
                dispatch.remote_edit.advertise(host, generation);
            }
            let allowed = candidates(&job, &envelope, &hosts)
                .iter()
                .any(|h| h.id == job.host)
                && envelope
                    .inputs
                    .iter()
                    .all(|m| hosts.iter().any(|h| h.id == m.host));
            if !allowed {
                interrupt = true;
            }
        }
        let cancelling = current.cancel_requested || interrupt;
        let (op, reply) = if cancelling {
            ("cancel", dispatch.call("cancel", &job, None).await)
        } else if launch {
            // Transfer staging can be long. Cancel intent is durable and the
            // worker's tombstone prevents a racing or abandoned launch. Both
            // dispatcher futures are scoped to this select and kill on drop.
            tokio::select! {
                reply = dispatch.call(action, &job, Some(&envelope)) => (action, reply),
                result = wait_for_cancel(app, &job.id, dispatch.poll) => {
                    result?;
                    launch = false;
                    continue;
                },
                result = sample_launch_progress(app, dispatch, &job), if action != "cleanup" => return result,
            }
        } else {
            ("status", dispatch.call("status", &job, None).await)
        };
        launch = false;
        match reply {
            Err(err) => {
                update(
                    app,
                    &job,
                    State::Running,
                    "Worker unavailable; keeping the compute slot",
                    Some(err.to_string()),
                    None,
                )?;
            }
            Ok(reply) => {
                use WorkerState::*;
                match reply.state {
                    Succeeded => {
                        if interrupt && !current.cancel_requested && job.kind == "hf_download" {
                            update(
                                app,
                                &job,
                                State::Failed,
                                "Interrupted Hugging Face download was not registered",
                                Some("The controller interrupted this job; not restarted automatically".into()),
                                None,
                            )?;
                            return Ok(());
                        }
                        update(
                            app,
                            &job,
                            State::Succeeded,
                            &reply.progress,
                            None,
                            reply.result.map(Value::Object),
                        )?;
                        return Ok(());
                    }
                    Failed | Cancelled => {
                        let was_interrupted = interrupt && !current.cancel_requested;
                        let state = if matches!(reply.state, Cancelled) && !was_interrupted {
                            State::Cancelled
                        } else {
                            State::Failed
                        };
                        let error = if was_interrupted {
                            Some("Interrupted job was not running on its worker; not restarted automatically".into())
                        } else {
                            reply.error
                        };
                        // Running/failed/cancelled replies cannot publish partials.
                        update(app, &job, state, &reply.progress, error, None)?;
                        return Ok(());
                    }
                    Running => {
                        established = true;
                        if !cancelling && Instant::now() >= next_sample {
                            record_sample(app, &job, Some(&reply.progress), reply.progress_detail)?;
                            next_sample = Instant::now() + cadence;
                        }
                    }
                    Queued | Missing => {
                        if cancelling {
                            update(
                                app,
                                &job,
                                State::Running,
                                "Waiting for cancellation confirmation",
                                None,
                                None,
                            )?;
                        } else if matches!(reply.state, Missing) && (established || recovered)
                            || matches!(reply.state, Queued) && recovered && !established
                        {
                            // Missing is not evidence that a previous dispatcher
                            // request cannot arrive later. Require a tombstone.
                            interrupt = true;
                            update(
                                app,
                                &job,
                                State::Running,
                                "Interrupted; securing a worker cancellation before releasing the slot",
                                None,
                                None,
                            )?;
                        } else {
                            // Live reconnect only. Status=queued means resumable
                            // staging until the worker has reported running.
                            launch = op == "status" && !established;
                            if Instant::now() >= next_sample {
                                let detail = if matches!(reply.state, Queued) {
                                    reply.progress_detail
                                } else {
                                    None
                                };
                                record_sample(app, &job, Some(&reply.progress), detail)?;
                                next_sample = Instant::now() + cadence;
                            }
                        }
                    }
                }
            }
        }
        sleep(dispatch.poll).await;
    }
}

async fn sample_launch_progress(app: &App, dispatch: &Dispatcher, job: &Job) -> Result<(), Error> {
    let cadence = dispatch
        .poll
        .saturating_mul(5)
        .max(Duration::from_millis(250));
    loop {
        // At most one sample in flight per job. Slow RPCs don't produce bursts.
        sleep(cadence).await;
        // A sample cannot publish results, release a slot, or trigger a
        // retry. Missing, terminal, and offline replies only clear stale detail.
        match dispatch.call("status", job, None).await {
            Ok(reply) if matches!(reply.state, WorkerState::Queued | WorkerState::Running) => {
                record_sample(app, job, Some(&reply.progress), reply.progress_detail)?;
            }
            _ => record_sample(app, job, None, None)?,
        }
    }
}

fn record_sample(
    app: &App,
    job: &Job,
    progress: Option<&str>,
    raw: Option<Box<RawValue>>,
) -> Result<(), Error> {
    let detail = raw.as_deref().and_then(ProgressDetail::from_worker);
    let progress = progress.map(|text| bounded(text, 2048));
    app.store
        .lock()
        .expect("store lock")
        .sample_progress(&job.id, progress.as_deref(), detail)
}

async fn wait_for_cancel(app: &App, id: &str, poll: Duration) -> Result<(), Error> {
    loop {
        if app
            .store
            .lock()
            .expect("store lock")
            .job(id)?
            .is_some_and(|job| job.cancel_requested)
        {
            return Ok(());
        }
        sleep(poll).await;
    }
}

fn update(
    app: &App,
    job: &Job,
    state: State,
    progress: &str,
    error: Option<String>,
    result: Option<Value>,
) -> Result<(), Error> {
    let mut store = app.store.lock().expect("store lock");
    let mut current = store.job(&job.id)?.ok_or("active job disappeared")?;
    let progress = if state == State::Running && current.cancel_requested {
        "Cancellation requested; waiting for the worker"
    } else {
        progress
    };
    let progress = bounded(progress, 2048);
    let error = error.map(|s| bounded(&s, 4096));
    if current.state != state
        || current.progress != progress
        || current.error != error
        || current.progress_detail.is_some()
    {
        current.state = state;
        current.progress = progress;
        current.error = error;
        current.progress_detail = None;
        current.has_result = result.is_some();
        store.update(&mut current, result)?;
    }
    Ok(())
}

fn bounded(s: &str, max: usize) -> String {
    s[..s.floor_char_boundary(max)].to_owned()
}
