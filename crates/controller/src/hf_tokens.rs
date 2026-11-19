use brewery_worker::{absolute, hf_download, operational};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Write as _},
    os::{
        fd::AsRawFd as _,
        unix::fs::{MetadataExt as _, OpenOptionsExt as _},
    },
    path::{Path, PathBuf},
};

pub struct Store {
    local: BTreeMap<String, Entry>,
    dist: PathBuf,
}

struct Entry {
    config: PathBuf,
    token: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Worker {
    root: PathBuf,
    hf_download: Option<hf_download::Config>,
    admission: Option<PathBuf>,
    managed_settings: Option<Value>,
    pinned_publication: Option<Value>,
    #[serde(default)]
    source_roots: Vec<PathBuf>,
    #[serde(default)]
    engine_env: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Set {
    pub token: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Remove {}

#[derive(Serialize)]
pub struct Status<'a> {
    pub id: &'a str,
    pub configured: bool,
}

impl Store {
    pub fn from_config(config: Option<&Value>, dist: &Path) -> Self {
        let mut store = Self {
            local: BTreeMap::new(),
            dist: std::path::absolute(dist).unwrap_or_else(|_| dist.to_owned()),
        };
        if let Some(hosts) = config.and_then(|v| v["hosts"].as_object()) {
            for (id, host) in hosts {
                store.add_local(id, host);
            }
        }
        store
    }

    pub fn add_local(&mut self, id: &str, host: &Value) {
        if host["local"] != true || !host["ssh"].is_null() || !host["ssh_config"].is_null() {
            return;
        }
        let (Some(worker), Some(config)) = (host["worker"].as_str(), host["config"].as_str())
        else {
            return;
        };
        if absolute(Path::new(worker)).is_err() || absolute(Path::new(config)).is_err() {
            return;
        }
        let token = if host["capabilities"]["hf_download"] == true {
            host["hf_token_file"].as_str().map(PathBuf::from)
        } else {
            None
        };
        self.local.insert(
            id.into(),
            Entry {
                config: config.into(),
                token,
            },
        );
    }

    // Both operator-owned configs must opt in to the same path. An SSH
    // alias, an ordinary compute worker or a second local alias cannot use it.
    pub fn file(&self, id: &str) -> Option<&Path> {
        let entry = self.local.get(id)?;
        let path = entry.token.as_deref()?;
        let parent = path.parent()?;
        absolute(path).ok()?;
        let own = operational::read::<Worker>(&entry.config).ok()?;
        let config = own.hf_download.as_ref()?;
        config.validate().ok()?;
        let dist = self
            .dist
            .canonicalize()
            .unwrap_or_else(|_| self.dist.clone());
        if config.token_file.as_deref() != Some(path)
            || own.admission.is_some()
            || own.managed_settings.is_some()
            || own.pinned_publication.is_some()
            || !own.source_roots.is_empty()
            || !own.engine_env.is_empty()
            || parent.starts_with(dist)
            || parent.starts_with(&own.root)
        {
            return None;
        }
        for (other_id, other) in &self.local {
            if other_id != id
                && (other.config == entry.config
                    || other.token.as_deref() == Some(path)
                    || operational::read::<Worker>(&other.config).is_ok_and(|other| {
                        other.root == own.root
                            || parent.starts_with(other.root)
                            || other
                                .source_roots
                                .iter()
                                .any(|root| parent.starts_with(root))
                            || other.hf_download.and_then(|c| c.token_file).as_deref() == Some(path)
                    }))
            {
                return None;
            }
        }
        Some(path)
    }
}

fn unsafe_path() -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        "credential storage is not private",
    )
}

pub struct PrivateFile {
    directory: File,
    path: PathBuf,
}

impl PrivateFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let parent = path.parent().ok_or_else(unsafe_path)?;
        let directory = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(parent)?;
        let metadata = directory.metadata()?;
        // Bind every operation to the checked directory inode, not a
        // re-resolved pathname. Requiring its actual path rejects symlinked parents.
        let dir_fd = PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd()));
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.mode() & 0o7777 != 0o700
            || fs::read_link(&dir_fd)?.as_os_str() != parent.as_os_str()
        {
            return Err(unsafe_path());
        }
        let path = dir_fd.join(path.file_name().ok_or_else(unsafe_path)?);
        Ok(Self { directory, path })
    }

    pub fn configured(&self) -> io::Result<bool> {
        match fs::symlink_metadata(&self.path) {
            Ok(metadata)
                if metadata.is_file()
                    && metadata.uid() == unsafe { libc::geteuid() }
                    && metadata.mode() & 0o7777 == 0o600
                    && metadata.nlink() == 1 =>
            {
                Ok(true)
            }
            Ok(_) => Err(unsafe_path()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }

    pub fn change(&self, token: Option<&str>) -> io::Result<bool> {
        let present = self.configured()?;
        if let Some(token) = token {
            let next = self
                .path
                .with_file_name(format!(".hf-token-{}", uuid::Uuid::new_v4()));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&next)?;
            let replaced = (|| {
                file.write_all(token.as_bytes())?;
                file.sync_all()?;
                fs::rename(&next, &self.path)
            })();
            if replaced.is_err() {
                let _ = fs::remove_file(next);
            }
            replaced?;
            self.directory.sync_all()?;
        } else if present {
            fs::remove_file(&self.path)?;
            self.directory.sync_all()?;
        }
        Ok(token.is_some())
    }
}
