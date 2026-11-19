use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};

const SNAPSHOT: &str = "operational-settings.json";
const LIMIT: u64 = 128 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub catalog: PathBuf,
    pub allowed_ports: Vec<u16>,
    pub evaluation_allowed: bool,
    #[serde(default)]
    pub baseline_allowed: bool,
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub revision: u64,
    pub benchmark_targets: Vec<Target>,
    pub enabled_suites: Vec<String>,
    pub baseline_endpoints: Vec<Baseline>,
    pub evaluation_enabled: bool,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub endpoint: String,
    pub model: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Baseline {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub endpoint: String,
    pub model: String,
    pub source_sha256: String,
    pub runtime_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub adapters: BTreeMap<String, Adapter>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Adapter {
    executable: PathBuf,
    version: String,
    protocol: String,
}

pub fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    ensure!(path.is_absolute(), "expected absolute config path");
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(file.metadata()?.is_file(), "expected a config file");
    let mut bytes = Vec::new();
    file.take(LIMIT + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= LIMIT, "config too large");
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn private_root(root: &Path) -> Result<()> {
    crate::absolute(root)?;
    let meta = fs::symlink_metadata(root)?;
    ensure!(
        meta.is_dir()
            && meta.uid() == unsafe { libc::geteuid() }
            && meta.mode() & 0o077 == 0
            && root.canonicalize()? == root,
        "invalid worker settings root"
    );
    Ok(())
}

impl Policy {
    pub fn load_catalog(&self) -> Result<Catalog> {
        ensure!(
            self.allowed_ports.len() <= 64 && !self.allowed_ports.contains(&0),
            "invalid approved ports"
        );
        let catalog: Catalog = read(&self.catalog)?;
        ensure!(
            catalog.adapters.len() <= 32
                && catalog.adapters.iter().all(|(key, a)| id(key)
                    && a.executable.is_absolute()
                    && label(&a.version, 128)
                    && label(&a.protocol, 128)
                    && fs::metadata(&a.executable)
                        .is_ok_and(|m| m.is_file() && m.mode() & 0o111 != 0)),
            "invalid benchmark catalog"
        );
        Ok(catalog)
    }
}

impl Settings {
    pub fn load(root: &Path) -> Result<Self> {
        private_root(root)?;
        match read(&root.join(SNAPSHOT)) {
            Ok(settings) => Ok(settings),
            Err(e)
                if e.downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
            {
                Ok(Self::default())
            }
            Err(e) => Err(e),
        }
    }

    pub fn save(&self, root: &Path) -> Result<()> {
        private_root(root)?;
        let temp = root.join(".operational-settings.tmp");
        // Reserved fixed filename; do not follow links or ever truncate a live file.
        if let Err(err) = fs::remove_file(&temp)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            return Err(err.into());
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)?;
        let result = (|| -> Result<()> {
            file.write_all(&serde_json::to_vec(self)?)?;
            file.sync_all()?;
            fs::rename(&temp, root.join(SNAPSHOT))?;
            File::open(root)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }

    pub fn validate(&self, policy: &Policy, catalog: &Catalog, eval_ready: bool) -> Result<()> {
        ensure!(
            self.benchmark_targets.len() <= 32
                && self.baseline_endpoints.len() <= 32
                && self.enabled_suites.len() <= 32,
            "use no more than 32 targets, suites or baselines"
        );
        ensure!(
            !self.evaluation_enabled || eval_ready,
            "evaluation is not allowed or configured on this worker"
        );
        ensure!(
            policy.baseline_allowed || self.baseline_endpoints.is_empty(),
            "baseline endpoints are disabled by worker policy"
        );
        let mut ids = BTreeSet::new();
        for target in &self.benchmark_targets {
            check_endpoint(&target.endpoint, policy)?;
            ensure!(
                valid_target(&target.id, target.label.as_deref(), &target.model)
                    && ids.insert(&target.id),
                "invalid or duplicate benchmark target"
            );
        }
        ids.clear();
        for baseline in &self.baseline_endpoints {
            check_endpoint(&baseline.endpoint, policy)?;
            ensure!(
                valid_target(&baseline.id, baseline.label.as_deref(), &baseline.model)
                    && ids.insert(&baseline.id)
                    && id(&baseline.runtime_id)
                    && digest(&baseline.source_sha256),
                "invalid or duplicate baseline endpoint"
            );
        }
        ids.clear();
        for suite in &self.enabled_suites {
            ensure!(
                id(suite) && catalog.adapters.contains_key(suite) && ids.insert(suite),
                "invalid, duplicate or unapproved benchmark suite"
            );
        }
        Ok(())
    }

    pub fn benchmark_config(&self, catalog: &Catalog, options: &Value) -> Result<Value> {
        let target = options["benchmark_target"]
            .as_str()
            .context("missing benchmark target ID")?;
        ensure!(
            self.benchmark_targets.iter().any(|item| item.id == target),
            "benchmark target not configured on worker"
        );
        ensure!(
            options["benchmarks"]
                .as_array()
                .is_some_and(|ids| !ids.is_empty()
                    && ids.iter().all(|id| id
                        .as_str()
                        .is_some_and(|id| self.enabled_suites.iter().any(|suite| suite == id)))),
            "benchmark suite not enabled on worker"
        );
        let targets = self
            .benchmark_targets
            .iter()
            .map(|t| {
                (
                    t.id.as_str(),
                    json!({"endpoint":t.endpoint,"model":t.model}),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let adapters = self
            .enabled_suites
            .iter()
            .map(|id| (id, &catalog.adapters[id]))
            .collect::<BTreeMap<_, _>>();
        Ok(json!({"targets":targets,"adapters":adapters}))
    }

    pub fn baseline_config(&self, options: &Value) -> Result<Value> {
        if let Some(source) = options.get("baseline_source") {
            ensure!(
                source
                    .as_str()
                    .is_some_and(|id| self.baseline_endpoints.iter().any(|b| b.id == id)),
                "baseline endpoint not configured on worker"
            );
        }
        Ok(json!(
            self.baseline_endpoints
                .iter()
                .map(|b| (
                    b.id.as_str(),
                    json!({"url":format!("{}/completions", b.endpoint), "model":b.model,
                "source_sha256":b.source_sha256, "runtime_id":b.runtime_id})
                ))
                .collect::<BTreeMap<_, _>>()
        ))
    }
}

pub fn eval_ready(policy: &Policy, env: &BTreeMap<String, String>) -> bool {
    policy.evaluation_allowed
        && env
            .get("BREWERY_ALLOW_GPU_LOAD")
            .is_some_and(|value| value == "1")
        && env
            .get("BREWERY_HALOGEN_IMAGE_ID")
            .and_then(|value| value.strip_prefix("sha256:"))
            .is_some_and(digest)
}

fn check_endpoint(url: &str, policy: &Policy) -> Result<()> {
    let port = url
        .strip_prefix("http://127.0.0.1:")
        .or_else(|| url.strip_prefix("http://[::1]:"))
        .and_then(|tail| tail.strip_suffix("/v1"))
        .filter(|port| {
            !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) && !port.starts_with('0')
        })
        .and_then(|port| port.parse::<u16>().ok());
    let port = port.context("endpoint must be http://127.0.0.1:PORT/v1 or http://[::1]:PORT/v1")?;
    ensure!(
        policy.allowed_ports.contains(&port),
        "endpoint port is not approved on this worker"
    );
    Ok(())
}

fn valid_target(key: &str, title: Option<&str>, model: &str) -> bool {
    id(key) && title.is_none_or(|title| label(title, 256)) && label(model, 256)
}

fn id(value: &str) -> bool {
    let valid = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit();
    value.len() <= 64
        && value.as_bytes().first().is_some_and(|&b| valid(b))
        && value.bytes().all(|b| valid(b) || matches!(b, b'_' | b'-'))
}
fn label(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max && !value.chars().any(char::is_control)
}
fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
