//! Authority is local configuration, never a role asserted by a client.
use crate::{
    App, Error,
    dispatch::read_limited,
    domain::{Host, State as JobState},
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use brewery_worker::resources::Limits;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{io::AsyncWriteExt, process::Command, sync::Semaphore, time::timeout};

#[path = "infra_policy.rs"]
mod infra_policy;
#[path = "model_directories.rs"]
mod model_directories;

const INSTALL_UNAVAILABLE: &str =
    "Installation requires an operator-configured helper and a pinned, pre-staged runtime";
const POLICY_UNAVAILABLE: &str = "Only the controller's existing quantization pool eligibility is adjustable; hardware and reservations remain operator-managed";
const PROVISION_UNAVAILABLE: &str =
    "SSH aliases, pinned host keys, worker provisioning, and admission remain operator-managed";

pub struct Admin {
    catalog: Catalog,
    db: Mutex<Connection>,
    changes: Semaphore,
    command: PathBuf,
    runtime_helper: Option<PathBuf>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Catalog {
    workers: BTreeMap<String, Worker>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Worker {
    ssh_alias: Option<String>,
    #[serde(default)]
    local: bool,
    #[serde(default)]
    initially_active: bool,
    #[serde(default)]
    runtimes: BTreeMap<String, Runtime>,
    #[serde(skip)]
    binding: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Runtime {
    label: String,
    capabilities: Vec<String>,
    sha256: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Lifecycle {
    Unregistered,
    Pending,
    Active,
    Inactive,
}

#[derive(Serialize, Deserialize)]
struct Policy {
    status: Lifecycle,
    allow_pool: bool,
    #[serde(default)]
    enabled: BTreeMap<String, String>,
    #[serde(default)]
    installed: BTreeMap<String, String>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum Action {
    Enroll {
        worker: String,
    },
    Approve {
        worker: String,
    },
    Deactivate {
        worker: String,
    },
    EnableRuntime {
        worker: String,
        runtime: String,
    },
    DisableRuntime {
        worker: String,
        runtime: String,
    },
    InstallRuntime {
        worker: String,
        runtime: String,
    },
    SetResourcePolicy {
        worker: String,
        allow_pool: bool,
    },
    SetHostPolicy {
        worker: String,
        revision: u64,
        paused: bool,
        disabled_jobs: Vec<String>,
    },
    SetHostResources {
        worker: String,
        revision: u64,
        resources: Limits,
    },
    SetWeightEditor {
        worker: String,
        revision: u64,
        enabled: bool,
    },
    SetupWeightEditor {
        worker: String,
        revision: u64,
        model_directory: String,
        credential: String,
    },
}

impl Action {
    fn worker(&self) -> &str {
        match self {
            Self::Enroll { worker }
            | Self::Approve { worker }
            | Self::Deactivate { worker }
            | Self::EnableRuntime { worker, .. }
            | Self::DisableRuntime { worker, .. }
            | Self::InstallRuntime { worker, .. }
            | Self::SetResourcePolicy { worker, .. }
            | Self::SetHostPolicy { worker, .. }
            | Self::SetHostResources { worker, .. }
            | Self::SetWeightEditor { worker, .. }
            | Self::SetupWeightEditor { worker, .. } => worker,
        }
    }
}

#[derive(Clone, Copy)]
struct ApiError(StatusCode, &'static str);
impl ApiError {
    fn storage(_: impl std::fmt::Display) -> Self {
        Self(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Administration storage unavailable",
        )
    }
    fn conflict(reason: &'static str) -> Self {
        Self(StatusCode::CONFLICT, reason)
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error":self.1}))).into_response()
    }
}
type ApiResult<T> = Result<T, ApiError>;

fn token(s: &str) -> bool {
    (1..=128).contains(&s.len())
        && s.as_bytes()[0].is_ascii_alphanumeric()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}
fn hex(s: &[u8]) -> bool {
    s.len() == 64
        && s.iter()
            .all(|&b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn private(path: &Path, limit: usize) -> Result<Vec<u8>, Error> {
    if !path.is_absolute() {
        return Err("admin configuration must use absolute paths".into());
    }
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let m = f.metadata()?;
    if !m.is_file()
        || m.mode() & 0o077 != 0
        || m.uid() != unsafe { libc::geteuid() }
        || m.nlink() != 1
    {
        return Err("admin configuration must be a private controller-owned regular file".into());
    }
    let mut bytes = Vec::new();
    f.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("admin configuration exceeds size limit".into());
    }
    Ok(bytes)
}

fn trusted_file(path: &Path, executable: bool) -> bool {
    path.is_absolute()
        && path.canonicalize().is_ok_and(|p| p == path)
        && fs::symlink_metadata(path).is_ok_and(|m| {
            m.is_file()
                && m.mode() & 0o022 == 0
                && (!executable || m.mode() & 0o111 != 0)
                && (m.uid() == 0 || m.uid() == unsafe { libc::geteuid() })
        })
}

fn catalog_binding(entry: &Worker, profile: &Value) -> Result<String, Error> {
    if entry.local {
        let paths = profile["worker"].as_str().zip(profile["config"].as_str());
        if entry.ssh_alias.is_some()
            || profile["local"].as_bool() != Some(true)
            || !profile["ssh"].is_null()
            || !profile["ssh_config"].is_null()
            || !paths.is_some_and(|(worker, config)| {
                trusted_file(Path::new(worker), true) && trusted_file(Path::new(config), false)
            })
        {
            return Err(
                "local admin catalog entry requires a trusted local worker and config".into(),
            );
        }
        let (worker, config) = paths.unwrap();
        return Ok(format!(
            "local:{:x}",
            Sha256::digest(format!("{worker}\0{config}"))
        ));
    }
    let Some(alias) = &entry.ssh_alias else {
        return Err("admin catalog must specify a strict SSH alias or local worker".into());
    };
    if !token(alias)
        || profile["ssh"].as_str() != Some(alias)
        || ["local", "worker", "config"]
            .iter()
            .any(|k| !profile[k].is_null())
        || profile["ssh_config"]
            .as_str()
            .is_some_and(|p| !Path::new(p).is_absolute())
        || (!profile["ssh_config"].is_null() && !profile["ssh_config"].is_string())
    {
        return Err("admin catalog must match a strict SSH dispatch alias".into());
    }
    Ok(alias.clone())
}

impl Admin {
    pub fn from_env(
        data: &Path,
        hosts: &[Host],
        dispatch: Option<&Path>,
        command: &Path,
    ) -> Result<Option<Self>, Error> {
        let mut catalog = env::var_os("BREWERY_ADMIN_CATALOG")
            .map(|p| -> Result<Catalog, Error> {
                Ok(serde_json::from_slice(&private(
                    Path::new(&p),
                    128 * 1024,
                )?)?)
            })
            .transpose()?
            .unwrap_or_default();
        let runtime_helper = env::var_os("BREWERY_ADMIN_RUNTIME_HELPER").map(PathBuf::from);
        if runtime_helper
            .as_deref()
            .is_some_and(|path| !trusted_file(path, true))
        {
            return Err("runtime helper must be a trusted canonical executable".into());
        }
        if !catalog.workers.is_empty() {
            let config: Value = serde_json::from_reader(fs::File::open(
                dispatch.ok_or("admin workers require dispatch configuration")?,
            )?)?;
            if catalog.workers.len() > 64 {
                return Err("too many admin catalog workers".into());
            }
            for (id, entry) in &mut catalog.workers {
                let profile = &config["hosts"][id];
                let host = hosts
                    .iter()
                    .find(|h| h.id == *id)
                    .ok_or("admin catalog worker is not in dispatch configuration")?;
                if !token(id) {
                    return Err("invalid admin worker ID".into());
                }
                entry.binding = catalog_binding(entry, profile)?;
                let mut capabilities = BTreeSet::new();
                if entry.runtimes.len() > 32 {
                    return Err("too many catalog runtimes".into());
                }
                for (runtime, definition) in &entry.runtimes {
                    if !token(runtime)
                        || definition.label.trim().is_empty()
                        || definition.label.len() > 96
                        || definition.label.chars().any(char::is_control)
                        || definition.capabilities.is_empty()
                        || definition
                            .sha256
                            .as_ref()
                            .is_some_and(|s| !hex(s.as_bytes()))
                        || definition.capabilities.iter().any(|kind| {
                            !matches!(
                                kind.as_str(),
                                "inspect"
                                    | "plan"
                                    | "convert"
                                    | "verify"
                                    | "generate"
                                    | "evaluate"
                                    | "baseline"
                                    | "benchmark"
                                    | "compare"
                                    | "convert_components"
                            ) || host.capabilities.get(kind) != Some(&true)
                                || !capabilities.insert(kind)
                        })
                    {
                        return Err(
                            "invalid admin runtime catalog entry or capability ceiling".into()
                        );
                    }
                }
            }
        }
        let path = data.join("admin.db");
        OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&path)?;
        let db = Connection::open(path)?;
        db.execute_batch("CREATE TABLE IF NOT EXISTS admin_policy(worker TEXT PRIMARY KEY, alias TEXT NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS admin_audit(id INTEGER PRIMARY KEY AUTOINCREMENT, at_ms INTEGER NOT NULL, body TEXT NOT NULL, outcome TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS infra_host_policy(worker TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS infra_directory_plan(worker TEXT PRIMARY KEY, body TEXT NOT NULL);")?;
        Ok(Some(Self {
            catalog,
            db: Mutex::new(db),
            changes: Semaphore::new(1),
            command: command.to_owned(),
            runtime_helper,
        }))
    }

    fn legacy_host(&self, db: &Connection, mut host: Host) -> Result<Option<Host>, Error> {
        if let Some(entry) = self.catalog.workers.get(&host.id) {
            let policy = policy(db, &host, entry)?;
            if policy.status != Lifecycle::Active {
                return Ok(None);
            }
            host.quantization_pool &= policy.allow_pool;
            for (id, runtime) in &entry.runtimes {
                if !enabled(&policy, id, runtime) {
                    for kind in &runtime.capabilities {
                        host.capabilities.insert(kind.clone(), false);
                        host.reasons.insert(
                            kind.clone(),
                            "Disabled by infrastructure administration".into(),
                        );
                    }
                }
            }
        }
        Ok(Some(host))
    }

    pub fn effective_hosts(&self, hosts: &[Host], app: &App) -> Result<Vec<Host>, Error> {
        self.hosts_with_policy(hosts, Some(app))
    }

    fn hosts_with_policy(
        &self,
        hosts: &[Host],
        app: Option<&App>,
    ) -> Result<Vec<Host>, Error> {
        let db = self.db.lock().map_err(|_| "admin storage unavailable")?;
        let mut effective = Vec::new();
        for host in hosts {
            let mut host = host.clone();
            let policy = app
                .map(|app| {
                    let policy = infra_policy::Policy::load(&db, &host.id)?;
                    apply_editor(&mut host, &policy, &app.local_edit);
                    if policy.weight_editor == Some(true) {
                        advertise_remote(app, &mut host, &policy);
                    }
                    Ok::<_, Error>(policy)
                })
                .transpose()?;
            if let Some(mut host) = self.legacy_host(&db, host)? {
                if let Some(policy) = policy {
                    policy.restrict(&mut host)?;
                }
                effective.push(host);
            }
        }
        Ok(effective)
    }

    fn view(&self, app: &App) -> Result<Value, Error> {
        let store = app
            .store
            .lock()
            .map_err(|_| "controller storage unavailable")?;
        let hosts = app.defined_hosts()?;
        let mut busy = BTreeSet::new();
        for job in store.by_state(JobState::Running)? {
            busy.insert(job.host);
            busy.extend(
                store
                    .envelope(&job.id)?
                    .inputs
                    .into_iter()
                    .map(|input| input.host),
            );
        }
        let operational: BTreeMap<_, _> = app
            .settings
            .effective(&hosts)
            .into_iter()
            .map(|host| (host.id.clone(), host))
            .collect();
        let db = self.db.lock().map_err(|_| "admin storage unavailable")?;
        let mut host_rows = Vec::new();
        let mut workers = Vec::new();
        for host in &hosts {
            let host_policy = infra_policy::Policy::load(&db, &host.id)?;
            let mut installed = host.clone();
            if let Some(lane) = app.local_edit.get(&host.id) {
                lane.advertise(&mut installed);
            }
            let remote_available = advertise_remote(app, &mut installed, &host_policy);
            let mut configured = operational[&host.id].clone();
            apply_editor(&mut configured, &host_policy, &app.local_edit);
            if host_policy.weight_editor == Some(true) && remote_available {
                advertise_remote(app, &mut configured, &host_policy);
            }
            let mut effective = self.legacy_host(&db, configured.clone())?;
            if let Some(effective) = &mut effective {
                host_policy.restrict(effective)?;
            }
            let mut editor = editor_view(host, &configured, effective.as_ref(), &app.local_edit, &host_policy);
            if app.ssh_hosts.editor_transport(&host.id).is_some() {
                editor["available"] = json!(remote_available);
                if remote_available {
                    editor.as_object_mut().expect("editor object").remove("reason");
                } else {
                    editor["reason"] = json!("Set up and verify a remote editor on this SSH worker.");
                }
                let credentials = app.remote_edit.choices(&host.id);
                editor["setup"] = json!({
                    "status": if remote_available { "verified" } else { "needs_setup" },
                    "model_directory": host_policy.remote_editor.as_ref().map(|r| &r.directory),
                    "credentials": credentials,
                    "reason": credentials.is_empty().then_some("No provision credential is configured for this SSH worker.")
                });
            }
            host_rows.push(json!({"id":host.id,"name":host.name,"revision":host_policy.revision,
                "paused":host_policy.paused,"draining":host_policy.paused && busy.contains(&host.id),
                "disabled_jobs":host_policy.disabled_jobs,"installed_jobs":infra_policy::job_kinds(&installed),
                "weight_editor":editor,
                "model_directories":model_directories::view(&app.local_edit, &host.id, &host_policy),
                "resources":host_policy.resources,"resource_bounds":host.resource_bounds,
                "effective_resources":effective.as_ref().unwrap_or(host).effective_resources(),
                "allowed_jobs":effective.as_ref().map_or_else(Vec::new, infra_policy::job_kinds)}));
            let Some(entry) = self.catalog.workers.get(&host.id) else {
                continue;
            };
            let policy = policy(&db, host, entry)?;
            let mut runtimes = Vec::new();
            for (id, runtime) in &entry.runtimes {
                let available = self.runtime_helper.is_some() && runtime.sha256.is_some();
                let mut row = json!({"id":id,"label":runtime.label,"enabled":enabled(&policy,id,runtime),"available":available,
                    "installed":runtime.sha256.as_ref().is_some_and(|sha| policy.installed.get(id)==Some(sha)),
                    "can_install":available && matches!(policy.status,Lifecycle::Inactive|Lifecycle::Pending)});
                if !available {
                    row["reason"] = json!(INSTALL_UNAVAILABLE);
                }
                runtimes.push(row);
            }
            workers.push(json!({"id":host.id,"name":host.name,"status":policy.status,
                "can_enroll":policy.status==Lifecycle::Unregistered,
                "can_approve":matches!(policy.status,Lifecycle::Pending|Lifecycle::Inactive),
                "can_deactivate":policy.status==Lifecycle::Active,
                "allow_pool":policy.allow_pool && host.quantization_pool,"pool_ceiling":host.quantization_pool,
                "runtimes":runtimes}));
        }
        let mut query =
            db.prepare("SELECT id,at_ms,body,outcome FROM admin_audit ORDER BY id DESC LIMIT 50")?;
        let mut cursor = query.query([])?;
        let mut audit = Vec::new();
        while let Some(row) = cursor.next()? {
            let mut v: Value = serde_json::from_str(&row.get::<_, String>(2)?)?;
            v["id"] = json!(row.get::<_, i64>(0)?);
            v["at_ms"] = json!(row.get::<_, i64>(1)?);
            v["outcome"] = json!(row.get::<_, String>(3)?);
            audit.push(v);
        }
        Ok(
            json!({"hosts":host_rows,"workers":workers,"audit":audit,"unavailable":{
                "install_runtime": self.runtime_helper.is_none().then_some(INSTALL_UNAVAILABLE),
                "host_provisioning":PROVISION_UNAVAILABLE,"resource_controls":POLICY_UNAVAILABLE
            }}),
        )
    }
}

impl App {
    pub(crate) fn editor_grants(
        &self,
        job: &crate::domain::Job,
        envelope: &crate::domain::Envelope,
    ) -> Result<Option<model_directories::Access>, Error> {
        let lane = self
            .local_edit
            .get(&job.host)
            .ok_or("local editor unavailable")?;
        let admin = self.admin.as_ref().ok_or("administration unavailable")?;
        let db = admin.db.lock().map_err(|_| "admin storage unavailable")?;
        let policy = infra_policy::Policy::load(&db, &job.host)?;
        Ok(model_directories::access(
            &self.local_edit,
            lane,
            job,
            envelope,
            &policy,
        ))
    }

    // A durable allocation may reconnect after its host has started draining.
    // Keep immutable and legacy admission checks without applying new job policy.
    pub(crate) fn effective_hosts_for_allocated(&self) -> Result<Vec<Host>, Error> {
        let hosts = self.settings.effective(&self.defined_hosts()?);
        match &self.admin {
            Some(admin) => admin.hosts_with_policy(&hosts, None),
            None => Ok(hosts),
        }
    }
}

fn editor_available(
    host: &Host,
    lanes: &crate::local_edit::Lanes,
    policy: &infra_policy::Policy,
) -> bool {
    lanes
        .get(&host.id)
        .is_some_and(|lane| !lane.configured.is_empty() || !policy.model_directories.is_empty())
        || (host.id == crate::backend_hf::ID && host.capabilities["weight_inspect"])
}

fn advertise_remote(app: &App, host: &mut Host, policy: &infra_policy::Policy) -> bool {
    let binding = app.ssh_hosts.editor_transport(&host.id);
    if !app.remote_edit.available(policy.remote_editor.as_ref(), binding.as_ref()) {
        return false;
    }
    app.remote_edit.advertise(
        host,
        policy.remote_editor.as_ref().expect("verified receipt").generation,
    );
    true
}

fn apply_editor(host: &mut Host, policy: &infra_policy::Policy, lanes: &crate::local_edit::Lanes) {
    match policy.weight_editor {
        Some(true) if editor_available(host, lanes, policy) && lanes.get(&host.id).is_some() => {
            lanes.get(&host.id).expect("prepared lane").advertise(host);
        }
        Some(true) if editor_available(host, lanes, policy) => {}
        Some(_) => crate::local_edit::disable(host),
        None => {}
    }
}

fn editor_view(
    host: &Host,
    configured: &Host,
    effective: Option<&Host>,
    lanes: &crate::local_edit::Lanes,
    policy: &infra_policy::Policy,
) -> Value {
    let available = editor_available(host, lanes, policy);
    let enabled = configured.capabilities["weight_inspect"]
        || configured.capabilities["weight_propose"]
        || configured.capabilities["weight_tune"]
        || configured.capabilities["weight_apply"];
    let capabilities: BTreeMap<_, _> = crate::local_edit::CAPS
        .iter()
        .map(|&cap| {
            (
                cap,
                effective.is_some_and(|host| !host.scheduling_paused && host.capabilities[cap]),
            )
        })
        .collect();
    let mut row = json!({"available":available,"enabled":enabled,"capabilities":capabilities});
    if !available {
        row["reason"] = json!(editor_reason(lanes, &host.id));
    }
    row
}

fn editor_reason(lanes: &crate::local_edit::Lanes, id: &str) -> &'static str {
    if lanes.get(id).is_some() {
        "The editor is installed, but this worker has no approved model directory."
    } else {
        lanes.unavailable_reason(id)
    }
}

fn enabled(policy: &Policy, id: &str, runtime: &Runtime) -> bool {
    runtime
        .sha256
        .as_ref()
        .is_some_and(|sha| policy.enabled.get(id) == Some(sha))
}

fn policy(db: &Connection, host: &Host, entry: &Worker) -> Result<Policy, Error> {
    let mut query = db.prepare("SELECT alias,body FROM admin_policy WHERE worker=?")?;
    let mut rows = query.query([&host.id])?;
    if let Some(row) = rows.next()? {
        let alias: String = row.get(0)?;
        if alias != entry.binding {
            return Err("admin catalog alias changed; operator reconciliation required".into());
        }
        return Ok(serde_json::from_str(&row.get::<_, String>(1)?)?);
    }
    Ok(Policy {
        status: if entry.initially_active {
            Lifecycle::Active
        } else {
            Lifecycle::Unregistered
        },
        allow_pool: host.quantization_pool,
        installed: BTreeMap::new(),
        enabled: BTreeMap::new(),
    })
}

pub fn router() -> Router<Arc<App>> {
    Router::new()
        .merge(model_directories::router())
        .route("/admin/state", get(state))
        .route(
            "/admin/actions",
            post(change).layer(DefaultBodyLimit::max(4096)),
        )
}

fn configured(app: &App) -> ApiResult<&Admin> {
    app.admin.as_ref().ok_or(ApiError(
        StatusCode::SERVICE_UNAVAILABLE,
        "Infrastructure administration is not configured",
    ))
}

async fn state(State(app): State<Arc<App>>) -> ApiResult<Json<Value>> {
    configured(&app)?
        .view(&app)
        .map(Json)
        .map_err(ApiError::storage)
}

async fn change(
    State(app): State<Arc<App>>,
    action: Result<Json<Action>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    configured(&app)?;
    let Json(action) =
        action.map_err(|_| ApiError(StatusCode::BAD_REQUEST, "Invalid administration action"))?;
    // Keep an accepted management action scoped to the controller, not to the
    // browser connection. A disconnect cannot cause two helpers to overlap.
    tokio::spawn(apply(app, action))
        .await
        .map_err(ApiError::storage)?
}

async fn apply(app: Arc<App>, action: Action) -> ApiResult<Json<Value>> {
    let admin = configured(&app)?;
    let _permit = admin.changes.try_acquire().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "Another administration action is in progress",
        )
    })?;
    let id = action.worker();
    if let Action::SetupWeightEditor {
        revision,
        model_directory,
        credential,
        ..
    } = &action
    {
        if !crate::remote_edit::model_directory(model_directory) {
            return Err(ApiError(StatusCode::BAD_REQUEST, "Use an absolute canonical model directory on the SSH worker"));
        }
        let transport = {
            let _store = app.store.lock().map_err(ApiError::storage)?;
            let hosts = app.defined_hosts().map_err(ApiError::storage)?;
            let host = hosts.iter().find(|h| h.id == id)
                .ok_or(ApiError(StatusCode::NOT_FOUND, "Unknown defined worker"))?;
            let transport = app.ssh_hosts.editor_transport(id)
                .ok_or(ApiError::conflict("Select an existing SSH worker"))?;
            let db = admin.db.lock().map_err(ApiError::storage)?;
            if infra_policy::Policy::load(&db, id).map_err(ApiError::storage)?.revision != *revision {
                return Err(ApiError::conflict("Host policy revision is stale"));
            }
            if admin.legacy_host(&db, host.clone()).map_err(ApiError::storage)?.is_none() {
                return Err(ApiError::conflict("The existing SSH worker is not admitted"));
            }
            if !app.remote_edit.accepts(id, credential) {
                return Err(ApiError::conflict("Select a configured provision credential for this worker"));
            }
            record_audit(&db, &action, "started")?;
            transport
        };
        let receipt = app
            .remote_edit
            .setup(&admin.command, id, &transport, model_directory, credential)
            .await;
        {
            let _store = app.store.lock().map_err(ApiError::storage)?;
            let mut db = admin.db.lock().map_err(ApiError::storage)?;
            let tx = db.transaction().map_err(ApiError::storage)?;
            let mut current = infra_policy::Policy::load(&tx, id).map_err(ApiError::storage)?;
            let receipt = match receipt {
                Ok(receipt) if current.revision == *revision
                    && app.ssh_hosts.editor_transport(id).as_ref() == Some(&transport) => receipt,
                _ => {
                    record_audit(&tx, &action, "failed")?;
                    tx.commit().map_err(ApiError::storage)?;
                    return Err(ApiError::conflict("Could not verify remote editor setup. Check the pinned host, provision key, remote directory, and packaged runtime."));
                }
            };
            current.revision = current.revision.checked_add(1)
                .ok_or(ApiError::conflict("Host policy revision limit reached"))?;
            current.remote_editor = Some(receipt);
            current.weight_editor = Some(true);
            current.save(&tx, id).map_err(ApiError::storage)?;
            record_audit(&tx, &action, "applied")?;
            tx.commit().map_err(ApiError::storage)?;
        }
        return admin.view(&app).map(Json).map_err(ApiError::storage);
    }
    if let Action::SetHostPolicy { revision, .. }
    | Action::SetHostResources { revision, .. }
    | Action::SetWeightEditor { revision, .. } = &action
    {
        {
            // Serialize against queue allocation and admission.
            let _store = app.store.lock().map_err(ApiError::storage)?;
            let hosts = app.defined_hosts().map_err(ApiError::storage)?;
            let host = hosts
                .iter()
                .find(|host| host.id == id)
                .ok_or(ApiError(StatusCode::NOT_FOUND, "Unknown defined worker"))?;
            let mut db = admin.db.lock().map_err(ApiError::storage)?;
            let tx = db.transaction().map_err(ApiError::storage)?;
            let mut current = infra_policy::Policy::load(&tx, id).map_err(ApiError::storage)?;
            if current.revision != *revision {
                return Err(ApiError::conflict("Host policy revision is stale"));
            }
            match &action {
                Action::SetHostPolicy {
                    paused,
                    disabled_jobs,
                    ..
                } => {
                    let mut installed = host.clone();
                    if let Some(lane) = app.local_edit.get(id) {
                        lane.advertise(&mut installed);
                    }
                    advertise_remote(&app, &mut installed, &current);
                    current.disabled_jobs = infra_policy::validate_jobs(&installed, disabled_jobs)
                        .ok_or(ApiError(StatusCode::BAD_REQUEST, "Disabled jobs must be distinct and within the installed worker capabilities"))?;
                    current.paused = *paused;
                }
                Action::SetHostResources { resources, .. } => {
                    host.resource_bounds
                        .unwrap_or_default()
                        .validate(resources)
                        .map_err(|_| {
                            ApiError(
                                StatusCode::BAD_REQUEST,
                                "Resources must be within the installed worker bounds",
                            )
                        })?;
                    current.resources = *resources;
                }
                Action::SetWeightEditor { enabled, .. } => {
                    if !editor_available(host, &app.local_edit, &current)
                        && !advertise_remote(&app, &mut host.clone(), &current) {
                        return Err(ApiError::conflict(editor_reason(&app.local_edit, id)));
                    }
                    current.weight_editor = Some(*enabled);
                }
                _ => unreachable!(),
            }
            current.revision = current
                .revision
                .checked_add(1)
                .ok_or(ApiError::conflict("Host policy revision limit reached"))?;
            current.save(&tx, id).map_err(ApiError::storage)?;
            record_audit(&tx, &action, "applied")?;
            tx.commit().map_err(ApiError::storage)?;
        }
        return admin.view(&app).map(Json).map_err(ApiError::storage);
    }
    let entry = admin
        .catalog
        .workers
        .get(id)
        .ok_or(ApiError(StatusCode::NOT_FOUND, "Unknown catalog worker"))?;
    let host = app
        .hosts
        .iter()
        .find(|h| h.id == id)
        .ok_or(ApiError(StatusCode::NOT_FOUND, "Unknown catalog worker"))?;
    let mut p = {
        let db = admin.db.lock().map_err(ApiError::storage)?;
        policy(&db, host, entry).map_err(ApiError::storage)?
    };
    match &action {
        Action::Enroll { .. } if p.status == Lifecycle::Unregistered => {
            p.status = Lifecycle::Pending
        }
        Action::Approve { .. } if matches!(p.status, Lifecycle::Pending | Lifecycle::Inactive) => {
            let reply = external(
                &admin.command,
                &["suggest-paths", id],
                Some(b"{\"prefix\":\"\"}"),
                Duration::from_secs(8),
            )
            .await;
            if !reply.is_some_and(|v| {
                v["items"].as_array().is_some_and(|a| a.len() <= 32) && v["truncated"].is_boolean()
            }) {
                return Err(ApiError::conflict(
                    "Worker did not pass its configured transport readiness probe",
                ));
            }
            p.status = Lifecycle::Active;
        }
        Action::Deactivate { .. } if p.status == Lifecycle::Active => {
            p.status = Lifecycle::Inactive
        }
        Action::SetResourcePolicy { allow_pool, .. } => {
            if *allow_pool && !host.quantization_pool {
                return Err(ApiError::conflict(
                    "Requested policy exceeds the configured pool ceiling",
                ));
            }
            p.allow_pool = *allow_pool;
        }
        Action::EnableRuntime { runtime, .. }
        | Action::DisableRuntime { runtime, .. }
        | Action::InstallRuntime { runtime, .. } => {
            let definition = entry
                .runtimes
                .get(runtime)
                .ok_or(ApiError(StatusCode::NOT_FOUND, "Unknown catalog runtime"))?;
            match &action {
                Action::DisableRuntime { .. } => {
                    p.enabled.remove(runtime);
                }
                _ => {
                    let (helper, sha) = admin
                        .runtime_helper
                        .as_ref()
                        .zip(definition.sha256.as_deref())
                        .ok_or(ApiError::conflict(INSTALL_UNAVAILABLE))?;
                    if matches!(&action, Action::InstallRuntime { .. }) {
                        if !matches!(p.status, Lifecycle::Inactive | Lifecycle::Pending)
                            || worker_busy(&app, id)?
                        {
                            return Err(ApiError::conflict(
                                "Enroll the worker or deactivate it and wait for its existing jobs to finish before installing",
                            ));
                        }
                        // Record an attempt before giving side effects to the helper.
                        admin.record(&action, None, "started")?;
                        if !helper_digest(helper, "install-pinned", id, runtime, sha).await
                            || !helper_digest(helper, "probe-installed", id, runtime, sha).await
                        {
                            admin.record(&action, None, "failed")?;
                            return Err(ApiError::conflict(
                                "Runtime installation could not be verified",
                            ));
                        }
                        p.installed.insert(runtime.clone(), sha.into());
                    } else {
                        if !helper_digest(helper, "probe-installed", id, runtime, sha).await {
                            return Err(ApiError::conflict(
                                "The pinned runtime is not installed or cannot be verified",
                            ));
                        }
                        p.installed.insert(runtime.clone(), sha.into());
                        p.enabled.insert(runtime.clone(), sha.into());
                    }
                }
            }
        }
        _ => {
            return Err(ApiError::conflict(
                "Worker state does not allow this action",
            ));
        }
    }
    // The queue takes this lock before evaluating effective policy; already
    // allocated work remains tracked for draining.
    {
        let _store = app.store.lock().map_err(ApiError::storage)?;
        admin.record(&action, Some((&entry.binding, &p)), "applied")?;
    }
    admin.view(&app).map(Json).map_err(ApiError::storage)
}

impl Admin {
    fn record(
        &self,
        action: &Action,
        policy: Option<(&str, &Policy)>,
        outcome: &str,
    ) -> ApiResult<()> {
        let mut db = self.db.lock().map_err(ApiError::storage)?;
        let tx = db.transaction().map_err(ApiError::storage)?;
        if let Some((alias, p)) = policy {
            tx.execute("INSERT INTO admin_policy(worker,alias,body) VALUES(?,?,?) ON CONFLICT(worker) DO UPDATE SET alias=excluded.alias,body=excluded.body",
                params![action.worker(),alias,serde_json::to_string(p).map_err(ApiError::storage)?]).map_err(ApiError::storage)?;
        }
        record_audit(&tx, action, outcome)?;
        tx.commit().map_err(ApiError::storage)
    }
}

fn record_audit(tx: &Connection, action: &impl Serialize, outcome: &str) -> ApiResult<()> {
    let at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    tx.execute(
        "INSERT INTO admin_audit(at_ms,body,outcome) VALUES(?,?,?)",
        params![
            at,
            serde_json::to_string(action).map_err(ApiError::storage)?,
            outcome
        ],
    )
    .map_err(ApiError::storage)?;
    Ok(())
}

fn worker_busy(app: &App, id: &str) -> ApiResult<bool> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    for job in store
        .by_state(JobState::Running)
        .map_err(ApiError::storage)?
    {
        if job.host == id
            || store
                .envelope(&job.id)
                .map_err(ApiError::storage)?
                .inputs
                .iter()
                .any(|i| i.host == id)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn helper_digest(helper: &Path, verb: &str, worker: &str, runtime: &str, sha: &str) -> bool {
    external(
        helper,
        &[verb, worker, runtime, sha],
        None,
        Duration::from_secs(30),
    )
    .await
    .is_some_and(|v| {
        v.as_object().is_some_and(|o| o.len() == 1) && v["sha256"].as_str() == Some(sha)
    })
}

async fn external(
    path: &Path,
    args: &[&str],
    payload: Option<&[u8]>,
    limit: Duration,
) -> Option<Value> {
    let mut child = Command::new(path)
        .args(args)
        .stdin(if payload.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .ok()?;
    let output = child.stdout.take()?;
    let stdin = child.stdin.take();
    let (bytes, status) = timeout(limit, async {
        let write = async {
            if let Some((mut stdin, payload)) = stdin.zip(payload) {
                stdin.write_all(payload).await?;
            }
            Ok::<_, std::io::Error>(())
        };
        let (_, bytes, status) = tokio::try_join!(write, read_limited(output, 4096), child.wait())?;
        Ok::<_, std::io::Error>((bytes, status))
    })
    .await
    .ok()?
    .ok()?;
    if !status.success() {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}
