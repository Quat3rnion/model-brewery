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
        let state = self.state.lock().expect("SSH registry lock");
        let mut managed = false;
        for row in ids
            .iter()
            .filter_map(|id| state.rows.get(*id))
            .filter(|r| self.dispatchable(r))
        {
            if !self.installed_identity(row).is_some_and(identity::usable) {
                return Err(
                    "Installed SSH identity is missing or unsafe; restore the configured private key",
                );
            }
            managed = true;
        }
        Ok(managed.then(|| self.dir.join("dispatch.json")))
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
            row.identity_file.as_deref().or(self.default_identity.as_deref())
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

    pub(crate) fn editor_login_choices(&self, host: &str, port: u16, keys: &[String]) -> Vec<Value> {
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
        let (id, revision) = credential.strip_prefix(SAVED_LOGIN_PREFIX)?.split_once(':')?;
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
    let left = left.trim_end_matches('.');
    let right = right.trim_end_matches('.');
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
