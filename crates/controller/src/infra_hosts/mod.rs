mod api;
pub(crate) mod identity;
pub(crate) mod transport;

use crate::{
    Error,
    domain::{Host, KINDS},
};
pub use api::router;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    net::IpAddr,
    os::unix::fs::{DirBuilderExt as _, MetadataExt as _},
    path::{Path, PathBuf},
    sync::Mutex,
};
use tokio::sync::Semaphore;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    host: String,
    user: String,
    port: u16,
    fingerprint: String,
}

impl Target {
    fn valid(&self) -> bool {
        let bytes = self.user.as_bytes();
        endpoint_valid(&self.host, self.port)
            && (1..=32).contains(&bytes.len())
            && (bytes[0].is_ascii_lowercase() || bytes[0] == b'_')
            && bytes[1..]
                .iter()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-".contains(b))
            && transport::fingerprint_valid(&self.fingerprint)
    }

    fn matches(&self, other: &Self) -> bool {
        self.host.eq_ignore_ascii_case(&other.host)
            && self.user == other.user
            && self.port == other.port
            && self.fingerprint == other.fingerprint
    }
}

fn endpoint_valid(host: &str, port: u16) -> bool {
    port != 0
        && (host.parse::<IpAddr>().is_ok()
            || (host.len() <= 253
                && host.split('.').all(|label| {
                    label.len() <= 63
                        && label
                            .as_bytes()
                            .first()
                            .is_some_and(u8::is_ascii_alphanumeric)
                        && label
                            .as_bytes()
                            .last()
                            .is_some_and(u8::is_ascii_alphanumeric)
                        && label
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
                })))
}

#[derive(Clone, Serialize, Deserialize, Default)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Provision {
    #[default]
    Defined,
    Verified {
        key: String,
    },
    Planned {
        key: String,
        confirmation: String,
        install: transport::Install,
    },
    Installed {
        key: String,
        install: transport::Install,
    },
}

#[derive(Clone, Serialize, Deserialize)]
struct Row {
    id: String,
    name: String,
    target: Target,
    #[serde(default)]
    identity_file: Option<String>,
    // Effective path used to verify or install (including the operator default).
    #[serde(default)]
    checked_identity: Option<String>,
    revision: u64,
    provision: Provision,
}

pub(crate) const SAVED_LOGIN_PREFIX: &str = "saved-ssh:";

#[derive(Serialize)]
pub(crate) struct EditorEndpoint {
    pub host: String,
    pub port: u16,
    pub key: String,
}

pub(crate) struct EditorLogin {
    pub endpoint: EditorEndpoint,
    pub user: String,
    pub identity: PathBuf,
    pub fingerprint: String,
}

impl Row {
    fn view(&self) -> Value {
        let status = match &self.provision {
            Provision::Defined => "defined",
            Provision::Verified { .. } => "verified",
            Provision::Planned { .. } => "planned",
            Provision::Installed { .. } => "installed",
        };
        let mut view = json!({
            "id":self.id,"name":self.name,"host":self.target.host,"user":self.target.user,
            "port":self.target.port,"fingerprint":self.target.fingerprint,
            "revision":self.revision,"status":status,"identity_file":self.identity_file
        });
        if let Provision::Planned { confirmation, .. } = &self.provision {
            view["confirmation"] = json!(confirmation);
        }
        view
    }

    fn dispatch_host(&self) -> Value {
        let capabilities = KINDS
            .iter()
            .map(|&kind| {
                (
                    kind.into(),
                    json!(matches!(
                        kind,
                        "inspect" | "transfer" | "transfer_preview" | "cache_cleanup"
                    )),
                )
            })
            .collect::<Map<String, Value>>();
        json!({"name":self.name,"capabilities":capabilities})
    }
}

#[derive(Clone, Deserialize, Serialize, Default)]
struct State {
    rows: BTreeMap<String, Row>,
    // IDs are durable Library namespaces. Never let a later host inherit
    // paths or historical job identities from a removed definition.
    retired: BTreeSet<String>,
}

pub struct Registry {
    dir: PathBuf,
    base: Map<String, Value>,
    reserved: BTreeSet<String>,
    state: Mutex<State>,
    helper: Option<transport::Helper>,
    default_identity: Option<String>,
    changes: Semaphore,
    discovery: Semaphore,
    #[cfg(test)]
    pause_config_identity_check: Option<Box<dyn Fn() + Send + Sync>>,
}

impl Registry {
    pub fn open(data: &Path, config: Option<&Value>) -> Result<Self, Error> {
        let dir = data.canonicalize()?.join("ssh-hosts");
        if let Err(err) = fs::DirBuilder::new().mode(0o700).create(&dir)
            && err.kind() != io::ErrorKind::AlreadyExists
        {
            return Err(err.into());
        }
        brewery_worker::operational::private_root(&dir)?;
        let state: State = match fs::symlink_metadata(dir.join("registry.json")) {
            Ok(md)
                if md.is_file()
                    && md.uid() == unsafe { libc::geteuid() }
                    && md.mode() & 0o077 == 0
                    && md.nlink() == 1 =>
            {
                serde_json::from_reader(fs::File::open(dir.join("registry.json"))?)?
            }
            Ok(_) => return Err("SSH definition registry is not private".into()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => State::default(),
            Err(err) => return Err(err.into()),
        };
        let base = config
            .and_then(|v| v["hosts"].as_object())
            .cloned()
            .unwrap_or_default();
        let mut reserved: BTreeSet<_> = base.keys().map(|k| k.to_ascii_lowercase()).collect();
        reserved.extend(["auto".into(), crate::backend_hf::ID.into()]);
        let helper = transport::Helper::from_env();
        let out = Self {
            dir,
            base,
            reserved,
            state: Mutex::new(state),
            helper,
            default_identity: identity::configured_default()
                .filter(|path| identity::safe_path(path)),
            changes: Semaphore::new(1),
            discovery: Semaphore::new(1),
            #[cfg(test)]
            pause_config_identity_check: None,
        };
        let state = out.state.lock().expect("SSH registry lock");
        for (id, row) in &state.rows {
            if id != &row.id
                || !id_valid(id)
                || !row.target.valid()
                || row
                    .identity_file
                    .as_deref()
                    .is_some_and(|p| !identity::safe_path(p))
                || row
                    .checked_identity
                    .as_deref()
                    .is_some_and(|p| !identity::safe_path(p))
                || out.reserved.contains(&id.to_ascii_lowercase())
            {
                return Err("invalid saved SSH definition".into());
            }
        }
        if !out.reserved.is_disjoint(&state.retired) {
            return Err("operator host ID conflicts with a retired SSH definition".into());
        }
        out.write_dispatch(&state)?;
        drop(state);
        Ok(out)
    }

    pub fn defined_hosts(&self) -> Result<Vec<Host>, Error> {
        let state = self.state.lock().expect("SSH registry lock");
        state
            .rows
            .values()
            .filter(|r| self.dispatchable(r))
            .map(|row| {
                let mut host = Host::from_config(&row.id, &row.dispatch_host())?;
                if let Provision::Installed { install, .. } = &row.provision {
                    host.provision_binding = Some(crate::settings::binding(&json!([
                        "provisioned",
                        row.id,
                        row.target,
                        install
                    ])));
                }
                Ok(host)
            })
            .collect()
    }

    fn dispatchable(&self, row: &Row) -> bool {
        !self.reserved.contains(&row.id.to_ascii_lowercase())
            && matches!(&row.provision, Provision::Installed { .. })
    }

    fn installed_identity<'a>(&'a self, row: &'a Row) -> Option<&'a str> {
        row.checked_identity
            .as_deref()
            .or(row.identity_file.as_deref())
            .or(self.default_identity.as_deref())
    }

    pub fn config_for(&self, ids: &[&str]) -> Result<Option<PathBuf>, &'static str> {
        let identities: Vec<_> = {
            let state = self.state.lock().expect("SSH registry lock");
            ids.iter()
                .filter_map(|id| state.rows.get(*id))
                .filter(|r| self.dispatchable(r))
                .map(|row| self.installed_identity(row).map(str::to_owned))
                .collect()
        };
        // An installed transport is immutable, and a retired ID cannot be
        // reused. Do not hold the registry lock while checking the key file.
        for identity in &identities {
            #[cfg(test)]
            if let Some(pause) = &self.pause_config_identity_check {
                pause();
            }
            if !identity.as_deref().is_some_and(identity::usable) {
                return Err(
                    "Installed SSH identity is missing or unsafe; restore the configured private key",
                );
            }
        }
        Ok((!identities.is_empty()).then(|| self.dir.join("dispatch.json")))
    }

    /// The editor installer must use the already defined transport identity.
    /// Static aliases are resolved and checked against their existing pin by
    /// the helper; managed rows pass their previously verified public key.
    pub(crate) fn editor_transport(&self, id: &str) -> Option<Value> {
        if let Some(row) = self.base.get(id) {
            let alias = row.get("ssh")?.as_str()?;
            return Some(json!({"alias":alias,"config":row.get("ssh_config")}));
        }
        let state = self.state.lock().expect("SSH registry lock");
        let row = state.rows.get(id)?;
        let Provision::Installed { key, .. } = &row.provision else {
            return None;
        };
        Some(json!({"target":row.target,"key":key}))
    }

    pub(crate) fn lock_editor_logins(&self) -> Option<tokio::sync::SemaphorePermit<'_>> {
        self.changes.try_acquire().ok()
    }

    fn editor_login(&self, row: &Row) -> Option<EditorLogin> {
        let key = match &row.provision {
            Provision::Defined => return None,
            Provision::Verified { key }
            | Provision::Planned { key, .. }
            | Provision::Installed { key, .. } => key,
        };
        let identity = row.checked_identity.as_deref()?;
        let current = if matches!(row.provision, Provision::Installed { .. }) {
            self.installed_identity(row)
        } else {
            row.identity_file
                .as_deref()
                .or(self.default_identity.as_deref())
        };
        if current != Some(identity)
            || !identity::usable(identity)
            || !transport::key_matches(key, &row.target.fingerprint)
        {
            return None;
        }
        Some(EditorLogin {
            endpoint: EditorEndpoint {
                host: row.target.host.clone(),
                port: row.target.port,
                key: key.clone(),
            },
            user: row.target.user.clone(),
            identity: identity.into(),
            fingerprint: row.target.fingerprint.clone(),
        })
    }

    pub(crate) fn editor_login_choices(
        &self,
        host: &str,
        port: u16,
        keys: &[String],
    ) -> Vec<Value> {
        let state = self.state.lock().expect("SSH registry lock");
        state
            .rows
            .values()
            .filter_map(|row| {
                let login = self.editor_login(row)?;
                if !same_hostname(host, &login.endpoint.host)
                    || port != login.endpoint.port
                    || !keys.contains(&login.endpoint.key)
                {
                    return None;
                }
                Some(json!({
                    "id": format!("{SAVED_LOGIN_PREFIX}{}:{}", row.id, row.revision),
                    "label": row.name
                }))
            })
            .collect()
    }

    pub(crate) fn selected_editor_login(&self, credential: &str) -> Option<EditorLogin> {
        let (id, revision) = credential
            .strip_prefix(SAVED_LOGIN_PREFIX)?
            .split_once(':')?;
        let state = self.state.lock().expect("SSH registry lock");
        let row = state.rows.get(id)?;
        if revision != row.revision.to_string() {
            return None;
        }
        self.editor_login(row)
    }

    fn write_dispatch(&self, state: &State) -> Result<(), Error> {
        let mut hosts = self.base.clone();
        for row in state.rows.values().filter(|r| self.dispatchable(r)) {
            let Provision::Installed { key, install } = &row.provision else {
                unreachable!()
            };
            let config =
                transport::ssh_config(&self.dir, row, key, install, self.installed_identity(row))?;
            hosts.insert(row.id.clone(), json!({
                "ssh":row.id,"ssh_config":config,"remote_worker":install.worker,"remote_config":install.config
            }));
        }
        transport::private_write(
            &self.dir.join("dispatch.json"),
            &serde_json::to_vec(&json!({"hosts":hosts}))?,
        )?;
        Ok(())
    }

    fn commit(&self, state: &mut State, next: State) -> Result<(), Error> {
        self.write_dispatch(&next)?;
        if let Err(err) =
            transport::private_write(&self.dir.join("registry.json"), &serde_json::to_vec(&next)?)
        {
            let _ = self.write_dispatch(state);
            return Err(err.into());
        }
        *state = next;
        Ok(())
    }
}

pub(crate) fn same_hostname(left: &str, right: &str) -> bool {
    let left = left.strip_suffix('.').unwrap_or(left);
    let right = right.strip_suffix('.').unwrap_or(right);
    match (left.parse::<IpAddr>(), right.parse::<IpAddr>()) {
        (Ok(left), Ok(right)) => left == right,
        (Err(_), Err(_)) => left.eq_ignore_ascii_case(right),
        _ => false,
    }
}

fn id_valid(id: &str) -> bool {
    id.len() <= 64
        && id.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric)
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}

fn name_valid(name: &str) -> bool {
    !name.trim().is_empty() && name.len() <= 256 && !name.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{App, dispatch::Dispatcher, domain::NewModel, store::Store};
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        time::{sleep, timeout},
    };

    #[test]
    fn state_stays_responsive_while_stat_config_resolution_blocks() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(check_state_stays_responsive());
    }

    async fn check_state_stays_responsive() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("state");
        let mut store = Store::open(&data).unwrap();
        let mut ssh = Registry::open(&data, None).unwrap();
        let mut state = ssh.state.lock().unwrap();
        state.rows.insert(
            "fixture".into(),
            Row {
                id: "fixture".into(),
                name: "Fixture".into(),
                target: Target {
                    host: "127.0.0.1".into(),
                    user: "fixture".into(),
                    port: 22,
                    fingerprint: String::new(),
                },
                identity_file: Some("/fixture-key".into()),
                checked_identity: None,
                revision: 1,
                provision: Provision::Installed {
                    key: String::new(),
                    install: transport::Install {
                        bundle_id: "a".repeat(64),
                        worker: "/fixture-worker".into(),
                        config: "/fixture-config".into(),
                    },
                },
            },
        );
        drop(state);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let gate = Mutex::new((Some(started_tx), release_rx));
        ssh.pause_config_identity_check = Some(Box::new(move || {
            let mut gate = gate.lock().unwrap();
            if let Some(sender) = gate.0.take() {
                let _ = sender.send(());
            }
            // Bound the fixture even if the server does not release the registry.
            let _ = gate.1.recv_timeout(Duration::from_secs(6));
        }));
        let worker = NewModel {
            host: "fixture".into(),
            path: "/models/fixture.gguf".into(),
            kind: "gguf".into(),
            name: None,
        }
        .validate()
        .unwrap();
        store.add_model(&worker, None).unwrap();

        let dispatcher = Dispatcher {
            command: "/usr/bin/false".into(),
            backend: None,
            local_edit: Arc::default(),
            ssh_hosts: Arc::new(ssh),
            remote_edit: Arc::new(crate::remote_edit::Installer::open(&data).unwrap()),
            launch_timeout: Duration::from_secs(1),
            poll: Duration::from_secs(1),
        };
        let app = Arc::new(App {
            store: Mutex::new(store),
            hosts: vec![],
            settings: Default::default(),
            trace_policy: Default::default(),
            hf_tokens: crate::hf_tokens::Store::from_config(None, &data),
            backend_download: false,
            backend_calibration: None,
            backend_edit_formats: Default::default(),
            admin: None,
            local_edit: dispatcher.local_edit.clone(),
            ssh_hosts: dispatcher.ssh_hosts.clone(),
            remote_edit: dispatcher.remote_edit.clone(),
            suggestions: crate::suggestions::PathSuggestions::new(dispatcher.command.clone()),
            model_availability: Arc::new(crate::model_availability::ModelAvailability::new(
                &dispatcher,
            )),
            feed_subscribers: Arc::new(Semaphore::new(crate::feed::SUBSCRIBERS)),
            dispatcher: Arc::new(dispatcher),
            scratch_slots: Arc::new(Semaphore::new(1)),
            diffs: Default::default(),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = crate::web::router(app, addr, data.to_path_buf());
        let server = tokio::spawn(async move { axum::serve(listener, router).await });
        let get_state = async || {
            let mut client = TcpStream::connect(addr).await.unwrap();
            client
                .write_all(format!("GET /api/state HTTP/1.0\r\nHost: {addr}\r\n\r\n").as_bytes())
                .await
                .unwrap();
            let mut out = String::new();
            client.read_to_string(&mut out).await.unwrap();
            assert!(out.starts_with("HTTP/1.0 200 "), "{out}");
            serde_json::from_str::<Value>(out.split_once("\r\n\r\n").unwrap().1).unwrap()
        };
        let initial_started = Instant::now();
        let initial = get_state().await;
        assert!(
            initial_started.elapsed() < Duration::from_secs(1),
            "initial GET /api/state waited for the SSH identity check"
        );
        assert_eq!(
            initial["models"][0]["reason"],
            "The controller has not checked this path on the worker."
        );
        timeout(Duration::from_secs(1), started_rx)
            .await
            .unwrap()
            .unwrap();
        // Time the second poll independently of the async workers. If they
        // are blocked on mutexes, an async timeout alone cannot fire.
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    use std::io::{Read, Write};
                    let mut conn = std::net::TcpStream::connect(addr).unwrap();
                    conn.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
                    write!(conn, "GET /api/state HTTP/1.0\r\nHost: {addr}\r\n\r\n").unwrap();
                    let mut out = String::new();
                    conn.read_to_string(&mut out)
                        .expect("GET /api/state waited for the SSH identity check");
                    assert!(out.starts_with("HTTP/1.0 200 "), "{out}");
                })
                .join()
                .unwrap();
        });
        let checking_since = Instant::now();
        loop {
            let state = get_state().await;
            let model = &state["models"][0];
            assert!(model["available"].is_null(), "{model}");
            if model["reason"] == "The controller could not check this path on the worker." {
                break;
            }
            assert!(
                checking_since.elapsed() < Duration::from_secs(4),
                "the watchdog did not time out the blocked configuration check"
            );
            sleep(Duration::from_millis(20)).await;
        }
        release_tx.send(()).unwrap();
        server.abort();
    }
}
