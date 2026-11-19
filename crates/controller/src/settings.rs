use crate::domain::{BaselineSource, BenchmarkChoice, Host};
use brewery_worker::operational::{self, Catalog, Policy, Settings};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

pub const KINDS: &[&str] = &["benchmark", "baseline", "evaluate", "generate"];

#[derive(Default)]
pub struct Managed {
    entries: BTreeMap<String, Option<PathBuf>>,
    local_configs: BTreeMap<String, PathBuf>,
}

#[derive(Deserialize)]
struct WorkerConfig {
    root: PathBuf,
    managed_settings: Option<Policy>,
    pinned_publication: Option<Value>,
    #[serde(default)]
    engine_env: BTreeMap<String, String>,
}

struct Profile {
    root: PathBuf,
    policy: Policy,
    catalog: Catalog,
    eval_ready: bool,
    settings: Settings,
}

#[derive(Serialize)]
pub struct View<'a> {
    id: &'a str,
    name: &'a str,
    managed: bool,
    ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
    approved_ports: Vec<u16>,
    evaluation_allowed: bool,
    baseline_allowed: bool,
    approved_suites: Vec<BenchmarkChoice>,
    #[serde(flatten)]
    settings: Settings,
}

pub enum PutError {
    Locked,
    NotReady,
    Conflict,
    Invalid(String),
}

impl Managed {
    pub fn from_config(config: Option<&Value>) -> Self {
        let Some(hosts) = config.and_then(|v| v["hosts"].as_object()) else {
            return Self::default();
        };
        let mut entries = BTreeMap::new();
        let mut local_configs = BTreeMap::new();
        for (id, host) in hosts {
            let path = if host["local"] == true && host["ssh"].is_null() {
                host["config"]
                    .as_str()
                    .map(PathBuf::from)
                    .filter(|p| p.is_absolute())
            } else {
                None
            };
            if let Some(path) = &path {
                local_configs.insert(id.clone(), path.clone());
            }
            if host["managed_settings"] == true {
                entries.insert(id.clone(), path);
            }
        }
        Self {
            entries,
            local_configs,
        }
    }

    pub fn enrolled(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    fn profile(&self, id: &str) -> Option<Profile> {
        let path = self.entries.get(id)?.as_ref()?;
        let config: WorkerConfig = operational::read(path).ok()?;
        if config.pinned_publication.is_some()
            || config
                .engine_env
                .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
            || std::env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_some()
        {
            return None;
        }
        // One managed alias per private config root; aliases would evade busy
        // checking. Separate workers must have disjoint state roots.
        for (other, path) in &self.local_configs {
            if other != id
                && let Ok(other) = operational::read::<WorkerConfig>(path)
                && (other.root.starts_with(&config.root) || config.root.starts_with(&other.root))
            {
                return None;
            }
        }
        let policy = config.managed_settings?;
        let catalog = policy.load_catalog().ok()?;
        let eval_ready = operational::eval_ready(&policy, &config.engine_env);
        let settings = Settings::load(&config.root).ok()?;
        Some(Profile {
            root: config.root,
            policy,
            catalog,
            eval_ready,
            settings,
        })
    }

    pub fn view<'a>(&self, host: &'a Host) -> View<'a> {
        let managed = self.enrolled(&host.id);
        match self.profile(&host.id) {
            Some(profile) => View {
                id: &host.id,
                name: &host.name,
                managed,
                ready: true,
                reason: None,
                approved_ports: profile.policy.allowed_ports,
                evaluation_allowed: profile.eval_ready,
                baseline_allowed: profile.policy.baseline_allowed,
                approved_suites: profile
                    .catalog
                    .adapters
                    .keys()
                    .map(|id| choice(id, None))
                    .collect(),
                settings: profile.settings,
            },
            None => View {
                id: &host.id,
                name: &host.name,
                managed,
                ready: false,
                reason: Some(if managed {
                    "Worker policy or suite catalog unavailable"
                } else {
                    "Worker settings are locked"
                }),
                approved_ports: Vec::new(),
                evaluation_allowed: false,
                baseline_allowed: false,
                approved_suites: Vec::new(),
                settings: Settings::default(),
            },
        }
    }

    pub fn put<'a>(&self, host: &'a Host, mut replacement: Settings) -> Result<View<'a>, PutError> {
        if !self.enrolled(&host.id) {
            return Err(PutError::Locked);
        }
        let profile = self.profile(&host.id).ok_or(PutError::NotReady)?;
        if replacement.revision != profile.settings.revision {
            return Err(PutError::Conflict);
        }
        replacement
            .validate(&profile.policy, &profile.catalog, profile.eval_ready)
            .map_err(|err| PutError::Invalid(err.to_string()))?;
        replacement.revision = replacement
            .revision
            .checked_add(1)
            .ok_or(PutError::Conflict)?;
        replacement
            .save(&profile.root)
            .map_err(|_| PutError::NotReady)?;
        Ok(self.view(host))
    }

    pub fn effective(&self, hosts: &[Host]) -> Vec<Host> {
        hosts
            .iter()
            .cloned()
            .map(|mut host| {
                if !self.enrolled(&host.id) {
                    return host;
                }
                let mut enabled = [false; 4];
                host.benchmark_targets.clear();
                host.benchmarks.clear();
                host.baseline_sources.clear();
                if let Some(profile) = self.profile(&host.id)
                    && profile
                        .settings
                        .validate(&profile.policy, &profile.catalog, profile.eval_ready)
                        .is_ok()
                {
                    let settings = profile.settings;
                    host.benchmark_targets = settings
                        .benchmark_targets
                        .iter()
                        .map(|t| choice(&t.id, t.label.clone()))
                        .collect();
                    host.benchmarks = settings
                        .enabled_suites
                        .iter()
                        .map(|id| choice(id, None))
                        .collect();
                    host.baseline_sources = settings
                        .baseline_endpoints
                        .into_iter()
                        .map(|b| BaselineSource {
                            id: b.id,
                            label: b.label,
                        })
                        .collect();
                    enabled = [
                        !host.benchmark_targets.is_empty() && !host.benchmarks.is_empty(),
                        !host.baseline_sources.is_empty(),
                        settings.evaluation_enabled,
                        settings.evaluation_enabled,
                    ];
                }
                for (&kind, enabled) in KINDS.iter().zip(enabled) {
                    if !enabled {
                        host.reasons.insert(
                            kind.into(),
                            "Managed worker has no ready settings for this job".into(),
                        );
                    } else {
                        host.reasons.remove(kind);
                    }
                    host.capabilities.insert(kind.into(), enabled);
                }
                host
            })
            .collect()
    }
}

fn choice(id: &str, label: Option<String>) -> BenchmarkChoice {
    BenchmarkChoice {
        id: id.into(),
        label,
    }
}
