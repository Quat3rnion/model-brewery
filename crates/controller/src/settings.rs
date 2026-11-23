use crate::domain::{BaselineSource, BenchmarkChoice, Host, Reason, standard_job_host_blocker};
use crate::store::Store;
use brewery_worker::operational::{self, Catalog, Policy, Settings};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};

pub const KINDS: &[&str] = &["benchmark", "baseline", "evaluate", "generate"];

pub(crate) fn trace_policy(
    config: Option<&Value>,
) -> Result<brewery_worker::trace::Policy, crate::Error> {
    let policy: brewery_worker::trace::Policy = config
        .and_then(|v| v.get("trace"))
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()?
        .unwrap_or_default();
    policy
        .check()
        .map_err(|e| -> crate::Error { e.to_string().into() })?;
    Ok(policy)
}

#[derive(Default)]
pub struct Managed {
    entries: BTreeMap<String, Value>,
    local_configs: BTreeMap<String, PathBuf>,
}

#[derive(Deserialize)]
struct WorkerConfig {
    root: Option<PathBuf>,
    managed_settings: Option<Value>,
    pinned_publication: Option<Value>,
    #[serde(default)]
    engine_env: BTreeMap<String, String>,
}

pub(crate) struct Profile {
    root: PathBuf,
    policy: Policy,
    catalog: Catalog,
    eval_ready: bool,
    settings: Settings,
}

pub(crate) enum Native {
    Managed(Profile),
    ProfileOnly {
        reason: &'static str,
        block_jobs: bool,
    },
}

impl Native {
    pub fn managed(&self) -> bool {
        matches!(self, Self::Managed(_))
    }
}

#[derive(Serialize)]
struct Management {
    href: String,
    available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
    state_url: &'static str,
    action_url: &'static str,
    actions: &'static [&'static str],
}

impl Management {
    fn for_host(host: &Host, available: bool) -> Self {
        Self {
            // Host IDs use only ASCII letters, digits, '.', '_' and '-'.
            href: format!("#/settings/infrastructure?host={}", host.id),
            available,
            reason: (!available).then_some("Infrastructure administration is not configured"),
            state_url: "/api/admin/state",
            action_url: "/api/admin/actions",
            actions: &["set_host_policy", "set_host_resources"],
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum EvaluationStatus {
    Managed,
    ProfileOnly,
}

#[derive(Serialize)]
struct Evaluation {
    status: EvaluationStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason_detail: Option<Reason>,
}

#[derive(Serialize)]
struct Readiness {
    ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<Reason>,
}

#[derive(Serialize)]
pub struct View<'a> {
    id: &'a str,
    name: &'a str,
    management: Management,
    evaluation: Evaluation,
    // Compatibility fields describe evaluation settings only, not host management.
    managed: bool,
    ready: bool,
    readiness: BTreeMap<&'static str, Readiness>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
    approved_ports: Vec<u16>,
    evaluation_allowed: bool,
    baseline_allowed: bool,
    approved_suites: Vec<BenchmarkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_binding: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    saved_profile: Option<Settings>,
    #[serde(flatten)]
    settings: Settings,
}

pub enum PutError {
    NotReady(&'static str),
    Storage(crate::Error),
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
            entries.insert(id.clone(), host.clone());
        }
        Self {
            entries,
            local_configs,
        }
    }

    pub fn native(&self, id: &str) -> Native {
        let explicit = self
            .entries
            .get(id)
            .is_some_and(|entry| entry["managed_settings"] == true);
        let profile_only = |reason, block_jobs| Native::ProfileOnly { reason, block_jobs };
        let Some(path) = self.local_configs.get(id) else {
            let reason = if id == crate::backend_hf::ID {
                "Saved on the controller only; the backend worker does not run evaluation"
            } else {
                "Saved on the controller only; this worker does not apply managed evaluation settings"
            };
            return profile_only(reason, explicit);
        };
        let Ok(config) = operational::read::<WorkerConfig>(path) else {
            return profile_only(
                "Saved on the controller only; check that the local worker config is readable and valid",
                true,
            );
        };
        let configured = config.managed_settings.is_some();
        if config.pinned_publication.is_some()
            || config
                .engine_env
                .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
            || std::env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_some()
        {
            return profile_only(
                "Saved on the controller only; source-pinned workers cannot apply managed settings",
                explicit || configured,
            );
        }
        let Some(raw_policy) = config.managed_settings else {
            return profile_only(
                "Saved on the controller only; no managed evaluation policy is configured",
                explicit,
            );
        };
        let Ok(policy) = serde_json::from_value::<Policy>(raw_policy) else {
            return profile_only(
                "Saved on the controller only; check the format of the worker's managed policy",
                true,
            );
        };
        let Some(root) = config.root else {
            return profile_only(
                "Saved on the controller only; managed evaluation requires a private worker state root",
                true,
            );
        };
        // One managed alias per private config root; aliases would evade busy
        // checking. Separate workers must have disjoint state roots.
        for (other, path) in &self.local_configs {
            if other != id
                && let Ok(other) = operational::read::<WorkerConfig>(path)
                && let Some(other_root) = other.root
                && (other_root.starts_with(&root) || root.starts_with(&other_root))
            {
                return profile_only(
                    "Saved on the controller only; managed evaluation requires a separate private state root for each worker",
                    true,
                );
            }
        }
        let Ok(catalog) = policy.load_catalog() else {
            return profile_only(
                "Saved on the controller only; check the worker policy's approved ports and suite catalog",
                true,
            );
        };
        let eval_ready = operational::eval_ready(&policy, &config.engine_env);
        let Ok(settings) = Settings::load(&root) else {
            return profile_only(
                "Saved on the controller only; check the worker's private state root and saved evaluation settings",
                true,
            );
        };
        Native::Managed(Profile {
            root,
            policy,
            catalog,
            eval_ready,
            settings,
        })
    }

    fn profile_binding(&self, host: &Host) -> String {
        if let Some(key) = &host.provision_binding {
            return key.clone();
        }
        // Bind to the dispatch route and local source, not labels, capabilities
        // or mutable policy. Adding or repairing policy keeps the same draft.
        let route: Vec<_> = self
            .entries
            .get(&host.id)
            .into_iter()
            .flat_map(|entry| {
                [
                    "local",
                    "worker",
                    "config",
                    "ssh",
                    "ssh_config",
                    "remote_worker",
                    "remote_config",
                ]
                .into_iter()
                .map(|field| &entry[field])
            })
            .collect();
        let source = self
            .local_configs
            .get(&host.id)
            .and_then(|path| operational::read::<WorkerConfig>(path).ok())
            .map(|cfg| {
                json!([
                    cfg.root,
                    cfg.pinned_publication,
                    cfg.engine_env.get("BREWERY_EXPECTED_SOURCE_SHA256")
                ])
            });
        binding(&json!([
            "controller",
            host.id,
            route,
            source,
            std::env::var("BREWERY_EXPECTED_SOURCE_SHA256").ok()
        ]))
    }

    pub fn view<'a>(
        &self,
        store: &Store,
        host: &'a Host,
        effective: Option<&Host>,
        management_available: bool,
    ) -> Result<View<'a>, crate::Error> {
        let binding = self.profile_binding(host);
        let draft = store.evaluation_profile(&binding)?;
        Ok(Self::render(
            host,
            effective,
            management_available,
            self.native(&host.id),
            binding,
            draft,
        ))
    }

    fn render<'a>(
        host: &'a Host,
        effective: Option<&Host>,
        management_available: bool,
        native: Native,
        binding: String,
        draft: Option<Settings>,
    ) -> View<'a> {
        let management = Management::for_host(host, management_available);
        let readiness = KINDS
            .iter()
            .map(|&kind| {
                let reason = match effective {
                    None => Some(Reason::settings(host, "policy", "Activate this worker")),
                    Some(host) => standard_job_host_blocker(host, kind),
                };
                (
                    kind,
                    Readiness {
                        ready: reason.is_none(),
                        reason,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let evaluation_reason = readiness["evaluate"].reason.clone();
        let ready = readiness.values().any(|row| row.ready);
        match native {
            Native::Managed(profile) => View {
                id: &host.id,
                name: &host.name,
                management,
                evaluation: Evaluation {
                    status: EvaluationStatus::Managed,
                    reason: evaluation_reason.as_ref().map(|reason| reason.text.clone()),
                    reason_detail: evaluation_reason,
                },
                managed: true,
                ready,
                readiness,
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
                profile_binding: None,
                saved_profile: draft,
                settings: profile.settings,
            },
            Native::ProfileOnly { reason, .. } => View {
                id: &host.id,
                name: &host.name,
                management,
                evaluation: Evaluation {
                    status: EvaluationStatus::ProfileOnly,
                    reason: Some(reason.into()),
                    reason_detail: evaluation_reason,
                },
                managed: false,
                ready,
                readiness,
                reason: Some(reason),
                approved_ports: Vec::new(),
                evaluation_allowed: false,
                baseline_allowed: false,
                approved_suites: Vec::new(),
                profile_binding: Some(binding),
                saved_profile: None,
                settings: draft.unwrap_or_default(),
            },
        }
    }

    pub fn put(
        &self,
        store: &Store,
        host: &Host,
        native: Native,
        mut replacement: Settings,
        request_binding: Option<&str>,
    ) -> Result<(), PutError> {
        let binding = self.profile_binding(host);
        let draft = store
            .evaluation_profile(&binding)
            .map_err(PutError::Storage)?;
        match native {
            Native::Managed(profile) => {
                if request_binding.is_some() {
                    return Err(PutError::Conflict);
                }
                advance_revision(&mut replacement, profile.settings.revision)?;
                replacement
                    .validate(&profile.policy, &profile.catalog, profile.eval_ready)
                    .map_err(|err| PutError::Invalid(err.to_string()))?;
                replacement.save(&profile.root).map_err(|_| {
                    PutError::NotReady(
                        "Could not save evaluation settings; check the worker's private state root",
                    )
                })?;
            }
            Native::ProfileOnly { .. } => {
                if request_binding != Some(binding.as_str()) {
                    return Err(PutError::Conflict);
                }
                advance_revision(&mut replacement, draft.as_ref().map_or(0, |d| d.revision))?;
                replacement
                    .validate_profile()
                    .map_err(|err| PutError::Invalid(err.to_string()))?;
                store
                    .save_evaluation_profile(&binding, &replacement)
                    .map_err(PutError::Storage)?;
            }
        }
        Ok(())
    }

    pub fn effective(&self, hosts: &[Host]) -> Vec<Host> {
        hosts
            .iter()
            .cloned()
            .map(|mut host| {
                let native = self.native(&host.id);
                if let Native::ProfileOnly {
                    block_jobs: false, ..
                } = native
                {
                    return host;
                };
                host.benchmark_targets.clear();
                host.benchmarks.clear();
                host.baseline_sources.clear();
                let validation = match &native {
                    Native::Managed(profile) => profile
                        .settings
                        .validate(&profile.policy, &profile.catalog, profile.eval_ready)
                        .map_err(|error| error.to_string()),
                    Native::ProfileOnly { reason, .. } => Err((*reason).into()),
                };
                if let Native::Managed(profile) = &native
                    && validation.is_ok()
                {
                    let settings = &profile.settings;
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
                        .iter()
                        .map(|b| BaselineSource {
                            id: b.id.clone(),
                            label: b.label.clone(),
                        })
                        .collect();
                }
                for &kind in KINDS {
                    let reason =
                        if self.entries.get(&host.id).is_some_and(|entry| {
                            entry["capabilities"][kind].as_bool() == Some(false)
                        }) {
                            Some(Reason::settings(
                                &host,
                                "policy",
                                &format!("Enable {kind} in the controller worker policy"),
                            ))
                        } else if let Native::Managed(profile) = &native {
                            profile.reason(&host, kind, &validation)
                        } else {
                            validation
                                .as_ref()
                                .err()
                                .map(|error| invalid_settings(&host, error))
                        };
                    host.capabilities.insert(kind.into(), reason.is_none());
                    if let Some(reason) = reason {
                        host.reasons.insert(kind.into(), reason.text.clone());
                        host.capability_reasons.insert(kind.into(), reason);
                    } else {
                        host.reasons.remove(kind);
                    }
                }
                host
            })
            .collect()
    }
}

impl Profile {
    fn reason(&self, host: &Host, kind: &str, validation: &Result<(), String>) -> Option<Reason> {
        let (section, action) = match kind {
            "evaluate" | "generate" if !self.policy.evaluation_allowed => {
                ("policy", "Allow evaluation in the worker policy")
            }
            "evaluate" | "generate" if !self.eval_ready => (
                "policy",
                "Allow GPU loading and pin a Halogen runtime image",
            ),
            "baseline" if !self.policy.baseline_allowed => {
                ("policy", "Allow baseline endpoints in the worker policy")
            }
            _ if validation.is_err() => {
                return validation
                    .as_ref()
                    .err()
                    .map(|error| invalid_settings(host, error));
            }
            "benchmark" if self.settings.benchmark_targets.is_empty() => {
                ("evaluation", "Add a benchmark target")
            }
            "benchmark" if self.settings.enabled_suites.is_empty() => {
                if self.catalog.adapters.is_empty() {
                    (
                        "policy",
                        "Add an approved benchmark suite to the worker catalog",
                    )
                } else {
                    ("evaluation", "Select a benchmark suite")
                }
            }
            "baseline" if self.settings.baseline_endpoints.is_empty() => {
                ("evaluation", "Add a baseline endpoint")
            }
            "evaluate" | "generate" if !self.settings.evaluation_enabled => {
                ("evaluation", "Allow GSM8K evaluation")
            }
            _ => return None,
        };
        Some(Reason::settings(host, section, action))
    }
}

fn invalid_settings(host: &Host, detail: &str) -> Reason {
    let mut reason = Reason::settings(host, "evaluation", "Correct the settings");
    reason.text = format!("{} ({detail}).", reason.text.trim_end_matches('.'));
    reason
}

fn advance_revision(settings: &mut Settings, current: u64) -> Result<(), PutError> {
    if settings.revision != current {
        return Err(PutError::Conflict);
    }
    settings.revision = current.checked_add(1).ok_or(PutError::Conflict)?;
    Ok(())
}

pub(crate) fn binding(source: &Value) -> String {
    format!("{:x}", Sha256::digest(source.to_string().as_bytes()))
}

fn choice(id: &str, label: Option<String>) -> BenchmarkChoice {
    BenchmarkChoice {
        id: id.into(),
        label,
    }
}
