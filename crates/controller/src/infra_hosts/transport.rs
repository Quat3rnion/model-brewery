use super::Row;
use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Write as _},
    os::unix::fs::{MetadataExt as _, OpenOptionsExt as _},
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{io::AsyncWriteExt as _, process::Command, time::timeout};

pub(super) fn fingerprint_valid(value: &str) -> bool {
    value
        .strip_prefix("SHA256:")
        .is_some_and(|s| STANDARD_NO_PAD.decode(s).is_ok_and(|b| b.len() == 32))
}

pub(crate) fn key_matches(key: &str, fingerprint: &str) -> bool {
    let Some(blob) = key
        .strip_prefix("ssh-ed25519 ")
        .and_then(|key| STANDARD.decode(key).ok())
    else {
        return false;
    };
    blob.len() == 51
        && blob.starts_with(b"\0\0\0\x0bssh-ed25519\0\0\0\x20")
        && format!("SHA256:{}", STANDARD_NO_PAD.encode(Sha256::digest(blob))) == fingerprint
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(super) struct Install {
    pub(super) bundle_id: String,
    pub(super) worker: String,
    pub(super) config: String,
}

impl Install {
    pub(super) fn valid(&self, id: &str) -> bool {
        if !brewery_worker::is_digest(&self.bundle_id) {
            return false;
        }
        let suffix = format!(
            "/.local/share/model-brewery/workers/{id}/{}",
            self.bundle_id
        );
        let Some(root) = self.worker.strip_suffix("/brewery-worker") else {
            return false;
        };
        root.ends_with(&suffix)
            && self.config == format!("{root}/worker.json")
            && root.len() <= 1024
            && root.strip_prefix('/').is_some_and(|p| {
                p.split('/').all(|c| {
                    !matches!(c, "" | "." | "..")
                        && c.bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
                })
            })
    }
}

#[derive(Deserialize)]
pub(super) struct Reply {
    pub(super) state: String,
    pub(super) host_key: Option<String>,
    pub(super) confirmation: Option<String>,
    #[serde(flatten)]
    pub(super) install: Option<Install>,
}

pub(super) struct Helper(PathBuf);

impl Helper {
    pub(super) fn from_env() -> Option<Self> {
        let path = PathBuf::from(env::var_os("BREWERY_SSH_PROVISION")?);
        if !path.is_absolute() {
            return None;
        }
        let path = path.canonicalize().ok()?;
        let m = path.metadata().ok()?;
        (m.is_file()
            && (m.uid() == 0 || m.uid() == unsafe { libc::geteuid() })
            && m.mode() & 0o6022 == 0
            && m.mode() & 0o100 != 0)
            .then_some(Self(path))
    }

    pub(super) async fn call(
        &self,
        row: &Row,
        op: &str,
        confirmation: Option<&str>,
    ) -> Option<Reply> {
        let mut data = serde_json::json!({
            "op":op,"id":row.id,"host":row.target.host,"user":row.target.user,
            "port":row.target.port,"fingerprint":row.target.fingerprint
        });
        if let Some(confirmation) = confirmation {
            data["confirmation"] = serde_json::json!(confirmation);
        }
        let payload = serde_json::to_vec(&data).ok()?;
        let mut child = Command::new(&self.0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .ok()?;
        let mut group = ProcessGroup(child.id()? as i32);
        let mut stdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;
        let bytes = timeout(Duration::from_secs(180), async {
            let write = async move { stdin.write_all(&payload).await };
            let (_, bytes) =
                tokio::try_join!(write, crate::dispatch::read_limited(stdout, 16 * 1024))?;
            let status = child.wait().await?;
            group.0 = 0;
            if !status.success() {
                return Err(io::Error::other("provision helper failed"));
            }
            Ok(bytes)
        })
        .await
        .ok()?
        .ok()?;
        serde_json::from_slice(&bytes).ok()
    }
}

struct ProcessGroup(i32);
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        if self.0 > 0 {
            unsafe { libc::kill(-self.0, libc::SIGKILL) };
        }
    }
}

pub(crate) fn private_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let scratch = path.with_file_name(format!(".write-{}", uuid::Uuid::new_v4()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&scratch)?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&scratch, path)?;
        File::open(path.parent().expect("private file parent"))?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&scratch);
    }
    result
}

pub(crate) fn ssh_path(path: &Path) -> Result<&str, crate::Error> {
    let text = path.to_str().ok_or("SSH state path is not UTF-8")?;
    if !path.is_absolute() || text.chars().any(|c| c.is_control() || "\"\\%$".contains(c)) {
        return Err("invalid SSH config path".into());
    }
    Ok(text)
}

pub(super) fn ssh_config(
    dir: &Path,
    row: &Row,
    key: &str,
    install: &Install,
) -> Result<PathBuf, crate::Error> {
    if !key_matches(key, &row.target.fingerprint) || !install.valid(&row.id) {
        return Err("invalid saved SSH pin or install target".into());
    }
    let file = dir.join(format!("{}.ssh", row.id));
    let known = dir.join(format!("{}.known-hosts", row.id));
    let mut config = format!(
        "Host {id}\nHostName {host}\nUser {user}\nPort {port}\nHostKeyAlias {id}\n\
        UserKnownHostsFile \"{known}\"\nGlobalKnownHostsFile /dev/null\n\
        StrictHostKeyChecking yes\nHostKeyAlgorithms ssh-ed25519\nUpdateHostKeys no\nCheckHostIP no\n\
        IdentityAgent none\nIdentitiesOnly yes\nPreferredAuthentications publickey\n\
        PasswordAuthentication no\nKbdInteractiveAuthentication no\n",
        id = row.id,
        host = row.target.host,
        user = row.target.user,
        port = row.target.port,
        known = ssh_path(&known)?,
    );
    if let Some(identity) = env::var_os("BREWERY_PROVISION_IDENTITY") {
        config.push_str(&format!(
            "IdentityFile \"{}\"\n",
            ssh_path(Path::new(&identity))?
        ));
    }
    private_write(&known, format!("{} {key}\n", row.id).as_bytes())?;
    private_write(&file, config.as_bytes())?;
    Ok(file)
}
