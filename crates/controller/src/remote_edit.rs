//! Provision only a separate, immutable editor lane for a defined SSH host.
use crate::{
    Error,
    domain::Host,
    infra_hosts::transport::{key_matches, private_write, ssh_path},
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
}

#[derive(Deserialize)]
struct Installed {
    host: String,
    port: u16,
    key: String,
    root: String,
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
        if self.helper.is_none() {
            return Vec::new();
        }
        self.credentials
            .iter()
            .filter(|(_, c)| c.worker == id)
            .map(|(id, c)| json!({"id":id,"label":c.label}))
            .collect()
    }

    pub fn accepts(&self, id: &str, credential: &str) -> bool {
        self.helper.is_some()
            && self
                .credentials
                .get(credential)
                .is_some_and(|c| c.worker == id)
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
    ) -> Result<Receipt, Error> {
        let helper = self.helper.as_ref().ok_or("No remote editor helper")?;
        let key = self
            .credentials
            .get(credential)
            .filter(|c| c.worker == id)
            .ok_or("Unknown worker credential")?;
        if !model_directory(directory) {
            return Err("Invalid model directory".into());
        }
        let generation = Uuid::new_v4();
        let request = serde_json::to_vec(&json!({
            "worker":id,"transport":transport,"model_directory":directory,"generation":generation,
            "identity":key.identity,"user":key.user,"fingerprint":key.fingerprint
        }))?;
        let result = run(helper, &[], &[], &request, 150).await?;
        let installed: Installed = serde_json::from_slice(&result)?;
        if !key_matches(&installed.key, &key.fingerprint)
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
                installed.host, installed.port, key.user, ssh_path(&known)?, ssh_path(&key.identity)?
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
    let ((), output, status) = timeout(Duration::from_secs(seconds), async {
        tokio::try_join!(
            async move { stdin.write_all(input).await },
            crate::dispatch::read_limited(stdout, 64 * 1024),
            child.wait()
        )
    })
    .await??;
    group.0 = 0;
    if !status.success() {
        return Err("Remote editor setup or verification failed".into());
    }
    Ok(output)
}
