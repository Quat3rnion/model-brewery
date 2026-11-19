use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
    time::{SystemTime, UNIX_EPOCH},
};

pub const KINDS: &[&str] = &[
    "compare",
    "inspect",
    "plan",
    "convert",
    "verify",
    "generate",
    "evaluate",
    "baseline",
    "benchmark",
    "transfer",
    "transfer_preview",
    "cache_cleanup",
    "hf_download",
    "weight_inspect",
    "weight_propose",
    "weight_tune",
    "weight_apply",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub host: String,
    pub path: String,
    pub kind: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NewModel {
    pub name: Option<String>,
    pub host: String,
    pub path: String,
    pub kind: String,
}

impl NewModel {
    pub fn validate(self) -> Result<Model, String> {
        let kind = if self.kind == "bf16-checkpoint" {
            "hf".into()
        } else {
            self.kind
        };
        let path = Path::new(&self.path);
        if self.path.len() > 4096
            || self.path.chars().any(char::is_control)
            || !path.is_absolute()
            || path.components().any(|c| matches!(c, Component::ParentDir))
            || path.file_name().is_none()
        {
            return Err("path must be an absolute worker path without parent traversal or control characters".into());
        }
        if !matches!(
            kind.as_str(),
            "gguf"
                | "hf"
                | "safetensors"
                | "hgn"
                | "brewery-bundle"
                | "hgn-v2"
                | "ht43"
                | "gsm8k-baseline"
                | "gsm8k-predictions"
                | "gsm8k-evaluation"
                | "assets"
                | "imatrix"
                | "weight-calibration"
                | "ngram"
                | "vision"
                | "mtp"
        ) {
            return Err("unknown model or input kind".into());
        }
        let name = self
            .name
            .unwrap_or_else(|| path.file_name().unwrap().to_string_lossy().into_owned());
        check_name(&name)?;
        Ok(Model {
            id: id(),
            name,
            host: self.host,
            path: self.path,
            kind,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct Prompt {
    pub id: String,
    pub name: String,
    pub text: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NewPrompt {
    pub name: String,
    pub text: String,
}

impl NewPrompt {
    pub fn validate(self) -> Result<Prompt, String> {
        check_name(&self.name)?;
        check_prompt(&self.text)?;
        Ok(Prompt {
            id: id(),
            name: self.name,
            text: self.text,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Copy, Clone, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl State {
    pub fn key(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
    pub fn terminal(self) -> bool {
        !matches!(self, Self::Queued | Self::Running)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Job {
    #[serde(default, skip_serializing_if = "crate::gguf::Format::is_hgn")]
    pub format: crate::gguf::Format,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_revision: Option<String>,
    pub id: String,
    pub kind: String,
    pub host: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_host: Option<String>,
    #[serde(default)]
    pub pool_slot: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_slot: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "brewery_worker::resources::Limits::is_empty"
    )]
    pub dispatch_resources: brewery_worker::resources::Limits,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub local_edit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory_revision: Option<u64>,
    pub state: State,
    pub progress: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_detail: Option<ProgressDetail>,
    pub input_ids: Vec<String>,
    pub options: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_id: Option<String>,
    pub cancel_requested: bool,
    pub has_result: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ProgressUnit {
    Bytes,
    Items,
    Tokens,
    Questions,
    Suites,
    Files,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ThreadActivity {
    slot: u8,
    task: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProgressDetail {
    phase: String,
    unit: ProgressUnit,
    completed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    current_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    files_completed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    files_total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    threads: Vec<ThreadActivity>,
    updated_at_ms: u64,
}

impl ProgressDetail {
    pub fn from_worker(raw: &serde_json::value::RawValue) -> Option<Self> {
        if raw.get().len() > 4096 {
            return None;
        }
        let detail: Self = serde_json::from_str(raw.get()).ok()?;
        let phase = detail.phase.as_bytes();
        (phase.len() <= 64
            && phase.first().is_some_and(u8::is_ascii_lowercase)
            && phase
                .iter()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_')
            && detail.updated_at_ms > 0
            && detail.threads.len() <= 16
            && detail.threads.windows(2).all(|rows| rows[0].slot < rows[1].slot)
            && detail.threads.iter().all(|row| (1..=64).contains(&row.slot)
                && !row.task.trim().is_empty() && row.task.len() <= 96
                && !matches!(row.task.as_str(), "." | "..")
                && !row.task.contains(['/', '\\'])
                && !row.task.chars().any(|c| c.is_control() || matches!(c, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')))
            && detail.total.is_none_or(|total| detail.completed <= total)
            && detail
                .files_completed
                .zip(detail.files_total)
                .is_none_or(|(done, total)| done <= total)
            && detail.input_count != Some(0)
            && detail.input_index.is_none_or(|index| {
                index > 0 && detail.input_count.is_none_or(|count| index <= count)
            })
            && detail.current_file.as_deref().is_none_or(|name| {
                !name.trim().is_empty()
                    && name.len() <= 255
                    && !matches!(name, "." | "..")
                    && !name.contains(['/', '\\'])
                    && !name.chars().any(char::is_control)
            }))
        .then_some(detail)
    }

    pub fn same_measurement(&self, other: &Self) -> bool {
        self.phase == other.phase
            && self.unit == other.unit
            && self.completed == other.completed
            && self.total == other.total
            && self.current_file == other.current_file
            && self.files_completed == other.files_completed
            && self.files_total == other.files_total
            && self.input_index == other.input_index
            && self.input_count == other.input_count
            && self.threads == other.threads
    }
}

#[derive(Serialize, Deserialize)]
pub struct Envelope {
    pub kind: String,
    pub inputs: Vec<Model>,
    pub options: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NewJob {
    pub kind: String,
    pub host: Option<String>,
    #[serde(default)]
    pub input_ids: Vec<String>,
    #[serde(default)]
    pub options: Map<String, Value>,
    pub prompt_id: Option<String>,
    pub prompt: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Event {
    pub at: u64,
    pub state: State,
    pub progress: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl From<&Job> for Event {
    fn from(job: &Job) -> Self {
        Self {
            at: job.updated_at,
            state: job.state,
            progress: job.progress.clone(),
            error: job.error.clone(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Host {
    #[serde(skip)]
    pub local_edit: bool,
    #[serde(skip)]
    pub(crate) provision_binding: Option<String>,
    pub id: String,
    pub name: String,
    pub capabilities: BTreeMap<String, bool>,
    pub component_cpu_only: bool,
    pub reasons: BTreeMap<String, String>,
    pub baseline_sources: Vec<BaselineSource>,
    pub benchmark_targets: Vec<BenchmarkChoice>,
    pub benchmarks: Vec<BenchmarkChoice>,
    pub quantization_pool: bool,
    pub quantization_backends: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub scheduling_paused: bool,
    #[serde(default)]
    pub resource_bounds: Option<brewery_worker::resources::Bounds>,
    #[serde(default)]
    pub resources: brewery_worker::resources::Limits,
}

impl Host {
    pub fn effective_resources(&self) -> brewery_worker::resources::Limits {
        brewery_worker::resources::Limits {
            concurrency: Some(if self.quantization_pool {
                self.resources.concurrency.unwrap_or(1)
            } else {
                1
            }),
            disk_reserve_bytes: if self.id == crate::backend_hf::ID {
                Some(16 << 30)
            } else {
                self.resources.disk_reserve_bytes
            },
            ..self.resources
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct BaselineSource {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkChoice {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

fn valid_benchmark_id(id: &str) -> bool {
    let alnum = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit();
    !id.is_empty()
        && id.len() <= 64
        && alnum(id.as_bytes()[0])
        && id.bytes().all(|b| alnum(b) || matches!(b, b'-' | b'_'))
}

fn benchmark_choices(config: &Value, key: &str) -> Result<Vec<BenchmarkChoice>, String> {
    let choices: Vec<BenchmarkChoice> = config
        .get(key)
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|e| format!("invalid {key}: {e}"))?
        .unwrap_or_default();
    let mut ids = BTreeSet::new();
    for choice in &choices {
        if !valid_benchmark_id(&choice.id) || !ids.insert(&choice.id) {
            return Err(format!("invalid or duplicate ID in {key}"));
        }
        if let Some(label) = &choice.label {
            check_name(label)?;
        }
    }
    Ok(choices)
}

impl Host {
    pub fn from_config(id: &str, config: &Value) -> Result<Self, String> {
        if !valid_token(id)
            || id == "auto"
            || id.contains(':')
            || !id.as_bytes()[0].is_ascii_alphanumeric()
            || !config.is_object()
        {
            return Err(format!("invalid dispatch host: {id}"));
        }
        let mut options: BTreeMap<String, bool> = config
            .get("capabilities")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| format!("invalid {id} capabilities: {e}"))?
            .unwrap_or_default();
        let mut reasons: BTreeMap<String, String> = config
            .get("reasons")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| format!("invalid {id} reasons: {e}"))?
            .unwrap_or_default();
        let benchmark_targets = benchmark_choices(config, "benchmark_targets")?;
        let benchmarks = benchmark_choices(config, "benchmarks")?;
        let benchmark_configured = !benchmark_targets.is_empty() && !benchmarks.is_empty();
        let mut capabilities = BTreeMap::new();
        for (capability, reason) in [
            (
                "weight_visual",
                "Visual weight reads are not configured on this worker",
            ),
            (
                "weight_hgn",
                "Native HGN weight jobs are not configured on this worker",
            ),
            (
                "weight_packed",
                "Packed weight jobs are not configured on this worker",
            ),
            (
                "weight_safetensors",
                "SafeTensors weight jobs are not configured on this worker",
            ),
        ] {
            let enabled = options.remove(capability).unwrap_or(false);
            capabilities.insert(capability.into(), enabled);
            if !enabled {
                reasons
                    .entry(capability.into())
                    .or_insert_with(|| reason.into());
            }
        }
        for capability in ["gguf_output", "gguf_hf_output"] {
            let enabled = options.remove(capability).unwrap_or(false)
                && (capability != "gguf_hf_output"
                    || capabilities.get("gguf_output") == Some(&true));
            capabilities.insert(capability.into(), enabled);
            if !enabled {
                reasons.entry(capability.into()).or_insert_with(|| {
                    "GGUF output is not configured for this source on this worker".into()
                });
            }
        }
        let custom_profiles = options.remove("custom_profiles").unwrap_or(false);
        capabilities.insert("custom_profiles".into(), custom_profiles);
        if !custom_profiles {
            reasons
                .entry("custom_profiles".into())
                .or_insert_with(|| "Custom profiles are not configured on this worker".into());
        }
        let convert_components = options.remove("convert_components").unwrap_or(false);
        let component_cpu_only = config
            .get("component_cpu_only")
            .map(|value| value.as_bool().ok_or("component_cpu_only must be a bool"))
            .transpose()?
            .unwrap_or(false);
        capabilities.insert("convert_components".into(), convert_components);
        if !convert_components {
            reasons
                .entry("convert_components".into())
                .or_insert_with(|| {
                    "HF component overrides require a separately configured component worker".into()
                });
        }
        for &kind in KINDS {
            let opt_in = matches!(
                kind,
                "generate" | "evaluate" | "baseline" | "benchmark" | "hf_download"
            ) || crate::weights::is_kind(kind);
            let enabled = options.remove(kind).unwrap_or(!opt_in)
                && (kind != "benchmark" || benchmark_configured);
            capabilities.insert(kind.into(), enabled);
            if !enabled {
                reasons.entry(kind.into()).or_insert_with(|| {
                    if kind == "benchmark" {
                        "Configure benchmark targets and suites and opt in on this worker"
                    } else if kind == "hf_download" {
                        "Configure Hugging Face downloads and opt in on this worker"
                    } else if crate::weights::is_kind(kind) {
                        "Weight editing is not configured on this worker"
                    } else if opt_in {
                        "Enable this workflow after configuring its evaluator on the worker"
                    } else {
                        "Disabled by the controller configuration"
                    }
                    .into()
                });
            }
        }
        let name = config
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(id)
            .to_owned();
        check_name(&name)?;
        let baseline_sources: Vec<BaselineSource> = config
            .get("baseline_sources")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| format!("invalid {id} baseline_sources: {e}"))?
            .unwrap_or_default();
        for source in &baseline_sources {
            if !valid_token(&source.id) {
                return Err(format!("invalid {id} baseline source ID"));
            }
            if let Some(label) = &source.label {
                check_name(label)?;
            }
        }
        let quantization_pool = config
            .get("quantization_pool")
            .map(|v| v.as_bool().ok_or("quantization_pool must be a bool"))
            .transpose()?
            .unwrap_or(false);
        let resource_bounds = if id == crate::backend_hf::ID
            || config.get("resource_bounds").is_some_and(Value::is_null)
        {
            None
        } else {
            let mut bounds: brewery_worker::resources::Bounds = config
                .get("resource_bounds")
                .map(|value| serde_json::from_value(value.clone()))
                .transpose()
                .map_err(|e| format!("invalid {id} resource_bounds: {e}"))?
                .unwrap_or_else(|| brewery_worker::resources::Bounds::standard(quantization_pool));
            if !quantization_pool && bounds.concurrency.is_some() {
                return Err("concurrency control requires a quantization-pool host".into());
            }
            // Non-pool scheduling remains globally exclusive.
            if !quantization_pool {
                bounds.concurrency = None;
            }
            bounds
                .check_ranges()
                .map_err(|e| format!("invalid {id} resource_bounds: {e}"))?;
            Some(bounds)
        };
        let quantization_backends: BTreeMap<String, Vec<String>> = config
            .get("quantization_backends")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| format!("invalid {id} quantization_backends: {e}"))?
            .unwrap_or_default();
        for (key, values) in &quantization_backends {
            if !matches!(key.as_str(), "ht" | "i4r" | "q4cp")
                || values.iter().any(|s| !valid_token(s))
            {
                return Err(format!("invalid {id} quantization backend allowlist"));
            }
        }
        Ok(Self {
            local_edit: false,
            provision_binding: None,
            id: id.into(),
            name,
            capabilities,
            component_cpu_only,
            reasons,
            baseline_sources,
            benchmark_targets,
            benchmarks,
            quantization_pool,
            quantization_backends,
            scheduling_paused: false,
            resource_bounds,
            resources: Default::default(),
        })
    }

    pub fn check_benchmark(&self, options: &Map<String, Value>) -> Result<(), String> {
        let target = options.get("benchmark_target").and_then(Value::as_str);
        let suites = options.get("benchmarks").and_then(Value::as_array);
        let (Some(target), Some(suites)) = (target, suites) else {
            return Err("benchmark options require benchmark_target and a benchmarks array".into());
        };
        if options.len() != 2 {
            return Err("benchmark options only accept benchmark_target and benchmarks".into());
        }
        if !valid_benchmark_id(target) || !(1..=8).contains(&suites.len()) {
            return Err("benchmark requires a lowercase target ID and 1–8 benchmark IDs".into());
        }
        if !self
            .benchmark_targets
            .iter()
            .any(|entry| entry.id == target)
        {
            return Err("benchmark_target is not configured on this host".into());
        }
        let mut selected = BTreeSet::new();
        for suite in suites {
            let Some(id) = suite.as_str().filter(|id| valid_benchmark_id(id)) else {
                return Err("benchmarks must contain lowercase benchmark IDs".into());
            };
            if !selected.insert(id) {
                return Err("benchmark IDs must be distinct".into());
            }
            if !self.benchmarks.iter().any(|entry| entry.id == id) {
                return Err("a benchmark ID is not configured on this host".into());
            }
        }
        Ok(())
    }

    pub fn supports_conversion(&self, options: &Map<String, Value>) -> bool {
        self.capabilities["convert"]
            && self
                .quantization_backends
                .iter()
                .all(|(encoder, backends)| {
                    match options.get(&format!("{encoder}_backend")) {
                        Some(value) => backends
                            .iter()
                            .any(|backend| value.as_str() == Some(backend.as_str())),
                        None => true, // Operator configures compatible worker defaults.
                    }
                })
    }
}

// Prefer the source host if eligible, avoiding a transfer. Selection of an
// *idle* host occurs in the scheduler, not during POST.
pub fn candidates<'a>(job: &Job, envelope: &Envelope, hosts: &'a [Host]) -> Vec<&'a Host> {
    if envelope.inputs.iter().any(|input| {
        hosts
            .iter()
            .any(|host| host.id == input.host && host.scheduling_paused)
    }) {
        return Vec::new();
    }
    let inputs = &envelope.inputs;
    let needs_components = inputs.first().is_some_and(|source| source.kind == "hf")
        && match job.kind.as_str() {
            "inspect" | "plan" | "convert" => inputs.len() > 1,
            "verify" => inputs.len() > 2,
            _ => false,
        };
    let components_allowed = |host: &Host| {
        if !job.format.is_hgn() {
            return host.capabilities["gguf_output"]
                && (inputs.first().is_none_or(|source| source.kind != "hf")
                    || host.capabilities["gguf_hf_output"]);
        }
        (!needs_components || host.capabilities["convert_components"])
            && (job.profile_revision.is_none() || host.capabilities["custom_profiles"])
            && (!matches!(job.kind.as_str(), "inspect" | "plan" | "convert" | "verify")
                || !host.component_cpu_only
                || [
                    "ht_backend",
                    "i4r_backend",
                    "q4cp_backend",
                    "decode_backend",
                ]
                .iter()
                .all(|key| job.options.get(*key).is_none_or(|v| v == "cpu")))
    };
    if job.kind != "convert" {
        let needs_hgn = crate::weights::is_kind(&job.kind)
            && inputs.first().is_some_and(|source| source.kind == "hgn");
        let needs_safetensors = crate::weights::is_kind(&job.kind)
            && inputs
                .first()
                .is_some_and(|source| crate::weights::is_safetensors(&source.kind));
        let needs_packed =
            crate::weights::is_kind(&job.kind) && crate::weights::is_packed(&envelope.options);
        return hosts
            .iter()
            .filter(|host| {
                host.id == job.host
                    && !host.scheduling_paused
                    && (!crate::weights::is_kind(&job.kind) || host.local_edit == job.local_edit)
                    && host.capabilities[&job.kind]
                    && (!needs_hgn || host.capabilities["weight_hgn"])
                    && (!needs_safetensors || host.capabilities["weight_safetensors"])
                    && (!needs_packed || host.capabilities["weight_packed"])
                    && (!crate::weights::needs_visual(&job.kind, &job.options)
                        || host.capabilities["weight_visual"])
                    && components_allowed(host)
                    && (job.kind != "benchmark" || host.check_benchmark(&job.options).is_ok())
            })
            .collect();
    }
    if job.host != "auto" {
        return hosts
            .iter()
            .filter(|host| {
                host.id == job.host
                    && !host.scheduling_paused
                    && host.supports_conversion(&job.options)
                    && components_allowed(host)
            })
            .collect();
    }
    let source = inputs.first().map(|model| model.host.as_str());
    let pool = hosts.iter().any(|host| host.quantization_pool);
    let mut eligible: Vec<_> = hosts
        .iter()
        .filter(|host| {
            (if pool {
                host.quantization_pool
            } else {
                Some(host.id.as_str()) == source
            }) && !host.scheduling_paused
                && host.supports_conversion(&job.options)
                && components_allowed(host)
        })
        .collect();
    eligible.sort_by_key(|host| Some(host.id.as_str()) != source);
    eligible
}

impl NewJob {
    pub fn build(
        self,
        host: String,
        inputs: Vec<Model>,
        prompt: Option<String>,
    ) -> Result<(Job, Envelope), String> {
        self.build_for(host, inputs, prompt, crate::gguf::Format::Hgn)
    }

    pub(crate) fn build_for(
        self,
        host: String,
        inputs: Vec<Model>,
        prompt: Option<String>,
        format: crate::gguf::Format,
    ) -> Result<(Job, Envelope), String> {
        if let Some(text) = &prompt {
            check_prompt(text)?;
            if self.kind == "compare" && text.len() > 8192 {
                return Err("comparison prompt cannot exceed 8192 bytes".into());
            }
        }
        let main = inputs.first().map_or("", |m| &m.kind);
        let second = inputs.get(1).map_or("", |m| &m.kind);
        let n = inputs.len();
        let hgn = |kind| matches!(kind, "hgn" | "brewery-bundle" | "hgn-v2" | "ht43");
        let reference = |kind| {
            matches!(
                kind,
                "gsm8k-predictions" | "gsm8k-baseline" | "gsm8k-evaluation"
            )
        };
        let weight_source = matches!(main, "gguf" | "hgn") || crate::weights::is_safetensors(main);
        let source = matches!(main, "hf" | "gguf" | "hgn");
        let baseline_source = self.options.contains_key("baseline_source");
        let valid = if !format.is_hgn() {
            let source = matches!(main, "gguf" | "hf");
            match self.kind.as_str() {
                "inspect" | "plan" | "convert" => {
                    source && (n == 1 || n == 2 && second == "imatrix")
                }
                "verify" => {
                    source && second == "gguf" && (n == 2 || n == 3 && inputs[2].kind == "imatrix")
                }
                _ => false,
            }
        } else {
            match self.kind.as_str() {
                "compare" => n == 2 && main == "gguf" && second == "gguf" && prompt.is_some(),
                "inspect" | "plan" | "convert" => (1..=5).contains(&n) && source,
                "verify" => (2..=6).contains(&n) && source && second == "hgn",
                "generate" => n == 1 && hgn(main) && prompt.is_some(),
                "evaluate" => {
                    hgn(main) && (n == 1 || (n == 2 && reference(second) && !baseline_source))
                }
                "baseline" => {
                    (n == 0 && baseline_source) || (n == 1 && !baseline_source && reference(main))
                }
                "transfer" | "transfer_preview" => (1..=32).contains(&n),
                "cache_cleanup" | "benchmark" | "hf_download" => n == 0,
                "weight_inspect" | "weight_propose" | "weight_apply" => weight_source && n == 1,
                "weight_tune" => {
                    weight_source
                        && n == 3
                        && inputs[1..].iter().all(|m| m.kind == "weight-calibration")
                        && inputs[1].id != inputs[2].id
                        && inputs[1].path != inputs[2].path
                }
                _ => false,
            }
        };
        if !valid {
            return Err(format!(
                "invalid inputs or missing prompt/baseline source for {}",
                self.kind
            ));
        }
        if prompt.is_some() && !matches!(self.kind.as_str(), "compare" | "generate") {
            return Err("this workflow does not accept a prompt".into());
        }
        if format.is_hgn() {
            if crate::weights::is_kind(&self.kind) {
                if inputs.iter().any(|input| input.host != host) {
                    return Err(
                        "all weight inputs must be registered on the destination worker".into(),
                    );
                }
                crate::weights::options(&self.kind, main, &Value::Object(self.options.clone()))?;
                if crate::weights::is_safetensors(main) && crate::weights::is_packed(&self.options)
                {
                    return Err(
                        "packed weight jobs do not support SafeTensors or HF sources".into(),
                    );
                }
            } else {
                check_options(&self.kind, &self.options, &inputs)?;
            }
        }
        let at = now();
        let envelope_kind = if self.kind == "cache_cleanup" {
            "cache-cleanup".into()
        } else {
            self.kind.clone()
        };
        let envelope = Envelope {
            kind: envelope_kind,
            inputs,
            options: self.options.clone(),
            prompt,
        };
        let requested_host = (host == "auto").then(|| host.clone());
        let job = Job {
            format,
            profile_revision: None,
            id: id(),
            kind: self.kind,
            host,
            state: State::Queued,
            progress: "Waiting for the compute slot".into(),
            progress_detail: None,
            requested_host,
            pool_slot: false,
            host_slot: None,
            dispatch_resources: Default::default(),
            local_edit: false,
            directory_revision: None,
            input_ids: self.input_ids,
            options: self.options,
            prompt_id: self.prompt_id,
            cancel_requested: false,
            has_result: false,
            error: None,
            created_at: at,
            updated_at: at,
        };
        Ok((job, envelope))
    }
}

fn check_options(job: &str, options: &Map<String, Value>, inputs: &[Model]) -> Result<(), String> {
    // Benchmark options are validated as a whole against the selected host
    // before enqueue and again by the scheduler before dispatch.
    if job == "benchmark" {
        return Ok(());
    }
    if job == "hf_download" {
        return brewery_worker::hf_download::Spec::parse(&Value::Object(options.clone()))
            .map(|_| ())
            .map_err(|_| "invalid Hugging Face download options".into());
    }
    for (key, value) in options {
        let key = key.as_str();
        let range = |min, max| value.as_u64().is_some_and(|n| (min..=max).contains(&n));
        let token = || value.as_str().is_some_and(valid_token);
        let valid = match (job, key) {
            ("compare", "n") => range(0, 128),
            ("compare", "labels") => range(0, 600),
            ("compare" | "plan" | "convert", "threads") => range(1, 64),
            ("compare", "raw" | "think") => value.is_boolean(),
            ("compare", "gpu_layers") => range(1, 4096) && options.contains_key("gpu_device"),
            ("compare", "gpu_device") => value.as_str().is_some_and(|device| {
                !device.is_empty()
                    && device.len() <= 64
                    && device
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
            }),
            ("inspect" | "plan" | "convert", "profile") => {
                matches!(value.as_str(), Some("v2" | "ht43"))
            }
            ("inspect" | "plan" | "convert", "ht_backend" | "i4r_backend" | "q4cp_backend")
            | ("verify", "decode_backend") => {
                matches!(value.as_str(), Some("cpu" | "hip" | "cuda"))
            }
            ("plan" | "convert", "seed") => range(0, u64::MAX),
            ("plan" | "convert", "ht_beam") => range(1, 256),
            ("inspect" | "plan" | "convert" | "verify", "assets" | "ngram" | "vision" | "mtp") => {
                value
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|i| *i >= if job == "verify" { 2 } else { 1 })
                    .and_then(|i| inputs.get(i))
                    .is_some_and(|input| input.kind == key)
            }
            ("evaluate" | "baseline", "baseline_source") => token(),
            ("generate", "max_tokens") => range(1, 512),
            ("cache_cleanup", "bundle_sha256") => value.as_str().is_some_and(|s| {
                s.len() == 64
                    && s.bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            }),
            _ => false,
        };
        if !valid {
            return Err(format!("unsupported or invalid {job} option: {key}"));
        }
    }
    if matches!(job, "inspect" | "plan" | "convert" | "verify") {
        let first_extra = if job == "verify" { 2 } else { 1 };
        for (i, input) in inputs.iter().enumerate().skip(first_extra) {
            if !matches!(input.kind.as_str(), "assets" | "ngram" | "vision" | "mtp")
                || options.get(&input.kind).and_then(Value::as_u64) != Some(i as u64)
            {
                return Err(
                    "each auxiliary input must be selected by its zero-based option index".into(),
                );
            }
        }
    }
    Ok(())
}

fn valid_token(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b':'))
}

fn check_name(name: &str) -> Result<(), String> {
    if name.trim().is_empty() || name.len() > 256 || name.chars().any(char::is_control) {
        Err("name must contain 1–256 bytes and no control characters".into())
    } else {
        Ok(())
    }
}

fn check_prompt(text: &str) -> Result<(), String> {
    if text.trim().is_empty() || text.len() > 65536 || text.contains('\0') {
        Err("prompt must contain 1–65536 bytes and no NUL".into())
    } else {
        Ok(())
    }
}

pub fn id() -> String {
    uuid::Uuid::new_v4().to_string()
}
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
