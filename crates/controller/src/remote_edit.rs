//! Provision only a separate, immutable editor lane for a defined SSH host.
use crate::{
    Error,
    domain::Host,
    infra_hosts::{
        EditorLogin, identity, same_hostname,
        transport::{key_matches, private_write, ssh_path},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    env, fs,
    os::unix::fs::{DirBuilderExt, MetadataExt},
    path::{Path, PathBuf},
};
use uuid::Uuid;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Credential {
    label: String,
    worker: String,
    user: String,
    identity: PathBuf,
    fingerprint: String,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Receipt {
    pub generation: Uuid,
    pub directory: String,
    binding: String,
}

pub(crate) struct Installer {
    dir: PathBuf,
    helper: Option<PathBuf>,
    credentials: BTreeMap<String, Credential>,
    inspect: tokio::sync::Semaphore,
}

pub(crate) const NO_LOGIN: &str =
    "Select a current verified SSH login, or configure a provision credential for this worker.";

#[derive(Deserialize)]
struct Installed {
    host: String,
    port: u16,
    key: String,
    root: String,
}

#[derive(Deserialize)]
pub(crate) struct Resolved {
    pub host: String,
    pub port: u16,
    pub keys: Vec<String>,
}

impl Resolved {
    pub fn from_managed(transport: &Value) -> Option<Self> {
        Some(Self {
            host: transport["target"]["host"].as_str()?.into(),
            port: transport["target"]["port"].as_u64()?.try_into().ok()?,
            keys: vec![transport["key"].as_str()?.into()],
        })
    }
}

pub(crate) fn model_directory(value: &str) -> bool {
    value.len() <= 1024
        && value.strip_prefix('/').is_some_and(|p| {
            p.split('/').all(|part| {
                !matches!(part, "" | "." | "..")
                    && !part.chars().any(|c| c.is_control() || c == '\\')
            })
        })
}

fn safe_token(value: &str) -> bool {
    (1..=64).contains(&value.len())
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}

impl Installer {
    pub fn open(data: &Path) -> Result<Self, Error> {
        let dir = data.canonicalize()?.join("remote-edit");
        match fs::DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.into()),
        }
        brewery_worker::operational::private_root(&dir)?;
        let helper = env::var_os("BREWERY_REMOTE_EDITOR_PROVISION")
            .map(PathBuf::from)
            .filter(|p| {
                p.is_absolute()
                    && trusted(p, 0o6022)
                    && p.metadata().is_ok_and(|m| m.mode() & 0o100 != 0)
            });
        let credentials = env::var_os("BREWERY_REMOTE_EDITOR_CREDENTIALS")
            .map(PathBuf::from)
            .map(|p| -> Result<BTreeMap<String, Credential>, Error> {
                if !p.is_absolute() || !trusted(&p, 0o077) || p.metadata()?.len() > 65536 {
                    return Err("Remote editor credentials file must be private".into());
                }
                let catalog: BTreeMap<String, Credential> = serde_json::from_slice(&fs::read(p)?)?;
                for (id, c) in &catalog {
                    if !safe_token(id)
                        || !safe_token(&c.worker)
                        || !safe_token(&c.user)
                        || c.label.trim().is_empty()
                        || c.label.len() > 96
                        || c.label.chars().any(char::is_control)
                        || ssh_path(&c.identity).is_err()
                        || !trusted(&c.identity, 0o077)
                    {
                        return Err("Invalid remote editor credential".into());
                    }
                }
                Ok(catalog)
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            dir,
            helper,
            credentials,
            inspect: tokio::sync::Semaphore::new(1),
        })
    }

    pub fn config_for(&self, id: Uuid) -> PathBuf {
        self.dir.join(id.to_string()).join("dispatch.json")
    }

    pub fn available(&self, receipt: Option<&Receipt>, transport: Option<&Value>) -> bool {
        receipt.zip(transport).is_some_and(|(r, t)| {
            r.binding == crate::settings::binding(t) && self.config_for(r.generation).is_file()
        })
    }

    pub fn choices(&self, id: &str) -> Vec<Value> {
        self.credentials
            .iter()
            .filter(|(_, c)| c.worker == id && trusted(&c.identity, 0o077))
            .map(|(id, c)| json!({"id":id,"label":c.label}))
            .collect()
    }

    pub fn accepts(&self, id: &str, credential: &str) -> bool {
        self.credentials
            .get(credential)
            .is_some_and(|c| c.worker == id && trusted(&c.identity, 0o077))
    }

    fn checked_helper(&self) -> Option<&Path> {
        self.helper.as_deref().filter(|path| {
            trusted(path, 0o6022) && path.metadata().is_ok_and(|md| md.mode() & 0o100 != 0)
        })
    }

    pub async fn inspect_aliases(
        &self,
        aliases: BTreeMap<String, Value>,
    ) -> BTreeMap<String, Resolved> {
        if aliases.is_empty() {
            return BTreeMap::new();
        }
        let Ok(_slot) = self.inspect.try_acquire() else {
            return BTreeMap::new();
        };
        let Some(helper) = self.checked_helper() else {
            return BTreeMap::new();
        };
        let Ok(request) = serde_json::to_vec(&aliases) else {
            return BTreeMap::new();
        };
        if request.len() > 65536 {
            return BTreeMap::new();
        }
        run(helper, &["inspect-aliases"], &[], &request, 2)
            .await
            .ok()
            .and_then(|raw| serde_json::from_slice(&raw).ok())
            .unwrap_or_default()
    }

    pub fn setup_blocker(&self) -> Option<&'static str> {
        if self.checked_helper().is_none() {
            return Some("The trusted remote editor helper is not configured or is unavailable.");
        }
        let package = env::var_os("BREWERY_REMOTE_EDITOR_PACKAGE").map(PathBuf::from);
        let checksum = env::var("BREWERY_REMOTE_EDITOR_PACKAGE_SHA256");
        if !package.is_some_and(|path| {
            path.is_absolute()
                && ssh_path(&path).is_ok()
                && trusted(&path, 0o6022)
                && path
                    .metadata()
                    .is_ok_and(|md| (1..=512 << 20).contains(&md.len()))
        }) || !checksum.is_ok_and(|pin| brewery_worker::is_digest(&pin))
        {
            return Some(
                "The pinned remote editor runtime package is not configured or is unavailable.",
            );
        }
        None
    }

    pub fn advertise(&self, host: &mut Host, generation: Uuid) {
        host.remote_editor = Some(generation);
        for &cap in crate::local_edit::CAPS {
            host.capabilities.insert(cap.into(), true);
            host.reasons.remove(cap);
        }
    }

    pub async fn setup(
        &self,
        dispatcher: &Path,
        id: &str,
        transport: &Value,
        directory: &str,
        credential: &str,
        saved: Option<&EditorLogin>,
    ) -> Result<Receipt, Error> {
        let helper = self.checked_helper().ok_or("No remote editor helper")?;
        let (identity_file, user, fingerprint) = if let Some(saved) = saved {
            if !saved.identity.to_str().is_some_and(identity::usable) {
                return Err(NO_LOGIN.into());
            }
            (&saved.identity, &saved.user, &saved.fingerprint)
        } else {
            let key = self
                .credentials
                .get(credential)
                .filter(|c| c.worker == id && trusted(&c.identity, 0o077))
                .ok_or(NO_LOGIN)?;
            (&key.identity, &key.user, &key.fingerprint)
        };
        if !model_directory(directory) {
            return Err("Invalid model directory".into());
        }
        let generation = Uuid::new_v4();
        let mut request = json!({
            "worker":id,"transport":transport,"model_directory":directory,"generation":generation,
            "identity":identity_file,"user":user,"fingerprint":fingerprint
        });
        if let Some(saved) = saved {
            request["expected_endpoint"] = serde_json::to_value(&saved.endpoint)?;
        }
        let result = run(helper, &[], &[], &serde_json::to_vec(&request)?, 150).await?;
        let installed: Installed = serde_json::from_slice(&result)?;
        if saved.is_some_and(|login| {
            !same_hostname(&installed.host, &login.endpoint.host)
                || installed.port != login.endpoint.port
                || installed.key != login.endpoint.key
        }) || !key_matches(&installed.key, fingerprint)
            || installed.port == 0
            || installed.host.is_empty()
            || !installed
                .host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b".-:".contains(&b))
            || !installed.root.ends_with(&format!(
                "/.local/share/model-brewery/editors/{id}/{generation}"
            ))
            || !model_directory(&installed.root)
            || !installed
                .root
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"/_.-".contains(&b))
        {
            return Err("Invalid editor installer receipt".into());
        }
        let lane = self.dir.join(generation.to_string());
        fs::DirBuilder::new().mode(0o700).create(&lane)?;
        let verify = async {
            if saved.is_some() && !identity_file.to_str().is_some_and(identity::usable) {
                return Err::<(), Error>(NO_LOGIN.into());
            }
            let known = lane.join("known_hosts");
            let ssh = lane.join("ssh");
            let config = self.config_for(generation);
            private_write(&known, format!("{id} {}\n", installed.key).as_bytes())?;
            private_write(&ssh, format!(
                "Host {id}\nHostName {}\nPort {}\nUser {}\nHostKeyAlias {id}\n\
                UserKnownHostsFile \"{}\"\nGlobalKnownHostsFile /dev/null\n\
                StrictHostKeyChecking yes\nHostKeyAlgorithms ssh-ed25519\nUpdateHostKeys no\n\
                IdentityAgent none\nIdentitiesOnly yes\nIdentityFile \"{}\"\n\
                PasswordAuthentication no\nKbdInteractiveAuthentication no\n",
                installed.host, installed.port, user, ssh_path(&known)?, ssh_path(identity_file)?
            ).as_bytes())?;
            private_write(&config, &serde_json::to_vec(&json!({"hosts":{(id):{
                "ssh":id,"ssh_config":ssh,"remote_worker":format!("{}/brewery-worker",installed.root),
                "remote_config":format!("{}/worker.json",installed.root)
            }}}))?)?;
            let status = run(dispatcher, &["status", id, &generation.to_string()], &[("BREWERY_DISPATCH_CONFIG", &config)], b"", 20).await?;
            let status: Value = serde_json::from_slice(&status)?;
            if status["state"] != "missing" {
                return Err::<(), Error>("Editor worker did not open the new state directory".into());
            }
            let payload = serde_json::to_vec(&json!({"prefix":format!("{directory}/")}))?;
            let reply = run(dispatcher, &["suggest-paths", id], &[("BREWERY_DISPATCH_CONFIG", &config)], &payload, 20).await?;
            let reply: Value = serde_json::from_slice(&reply)?;
            if !reply["items"].is_array() || !reply["truncated"].is_boolean() {
                return Err::<(), Error>("Editor worker RPC did not pass readiness".into());
            }
            Ok(())
        }.await;
        if let Err(err) = verify {
            let _ = fs::remove_dir_all(lane);
            return Err(err);
        }
        Ok(Receipt {
            generation,
            directory: directory.into(),
            binding: crate::settings::binding(transport),
        })
    }
}

fn trusted(path: &Path, forbidden: u32) -> bool {
    fs::symlink_metadata(path).is_ok_and(|m| {
        m.is_file()
            && m.nlink() == 1
            && (m.uid() == 0 || m.uid() == unsafe { libc::geteuid() })
            && m.mode() & forbidden == 0
    })
}

async fn run(
    path: &Path,
    args: &[&str],
    vars: &[(&str, &PathBuf)],
    input: &[u8],
    seconds: u64,
) -> Result<Vec<u8>, Error> {
    use std::{process::Stdio, time::Duration};
    use tokio::{io::AsyncWriteExt, process::Command, time::timeout};
    let mut child = Command::new(path)
        .args(args)
        .envs(vars.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0)
        .kill_on_drop(true)
        .spawn()?;
    let pid = child.id().ok_or("Missing installer process ID")? as i32;
    struct Group(i32);
    impl Drop for Group {
        fn drop(&mut self) {
            if self.0 > 0 {
                unsafe {
                    libc::kill(-self.0, libc::SIGKILL);
                }
            }
        }
    }
    let mut group = Group(pid);
    let mut stdin = child.stdin.take().ok_or("Missing installer input")?;
    let stdout = child.stdout.take().ok_or("Missing installer output")?;
    let (output, status) = timeout(Duration::from_secs(seconds), async {
        let ((), output) = tokio::try_join!(
            async move {
                stdin.write_all(input).await?;
                drop(stdin);
                Ok(())
            },
            crate::dispatch::read_limited(stdout, 64 * 1024),
        )?;
        // Keep the group leader unreaped until stdout is closed.
        let status = child.wait().await?;
        group.0 = 0;
        Ok::<_, std::io::Error>((output, status))
    })
    .await??;
    if !status.success() {
        return Err("Remote editor setup or verification failed".into());
    }
    Ok(output)
}
