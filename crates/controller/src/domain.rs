use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
    time::{SystemTime, UNIX_EPOCH},
};

pub const KINDS: &[&str] = &[
    "compare",
    brewery_worker::trace::JOB,
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
    brewery_worker::exl3_layer::JOB,
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub host: String,
    pub path: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(default)]
    pub available: Option<bool>,
    #[serde(
        default = "unknown_availability_reason",
        skip_serializing_if = "Option::is_none"
    )]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

fn unknown_availability_reason() -> Option<String> {
    Some("The controller has not checked this path on the worker.".into())
}

impl Model {
    // Call only after the controller has accepted the producer's receipt.
    pub fn attest_output(&mut self, artifact: &Value) {
        if artifact["path"] == self.path
            && artifact["kind"] == self.kind
            && let Some(sha256) = artifact["sha256"]
                .as_str()
                .filter(|s| brewery_worker::is_digest(s))
            && let Some(bytes) = artifact["bytes"].as_u64().filter(|&n| n > 0)
        {
            self.sha256 = Some(sha256.into());
            self.bytes = Some(bytes);
        }
    }
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
                | brewery_worker::hgn_recipe::ARCHIVE
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
            sha256: None,
            bytes: None,
            available: None,
            reason: unknown_availability_reason(),
            job_id: None,
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
        self.validate_with_id(id())
    }

    pub fn validate_with_id(self, id: String) -> Result<Prompt, String> {
        check_name(&self.name)?;
        check_prompt(&self.text)?;
        Ok(Prompt {
            id,
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
    pub remote_editor: Option<uuid::Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub scratch_job: bool,
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
    // Library metadata is not a worker input or an attestation of current data.
    #[serde(serialize_with = "serialize_worker_inputs")]
    pub inputs: Vec<Model>,
    pub options: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scratch_pins: BTreeMap<String, String>,
}

fn serialize_worker_inputs<S: serde::Serializer>(
    inputs: &[Model],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct Input<'a> {
        id: &'a str,
        name: &'a str,
        host: &'a str,
        path: &'a str,
        kind: &'a str,
    }
    serializer.collect_seq(inputs.iter().map(|model| Input {
        id: &model.id,
        name: &model.name,
        host: &model.host,
        path: &model.path,
        kind: &model.kind,
    }))
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
    pub remote_editor: Option<uuid::Uuid>,
    #[serde(skip)]
    pub(crate) provision_binding: Option<String>,
    pub id: String,
    pub name: String,
    pub capabilities: BTreeMap<String, bool>,
    pub component_cpu_only: bool,
    pub reasons: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub capability_reasons: BTreeMap<String, Reason>,
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

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct Reason {
    pub text: String,
    pub fix_anchor: String,
}

impl Reason {
    pub fn link(text: &str, fix_anchor: &str) -> Self {
        Self {
            text: text.into(),
            fix_anchor: fix_anchor.into(),
        }
    }

    pub fn settings(host: &Host, section: &str, action: &str) -> Self {
        let (page, label) = match section {
            "evaluation" => ("evaluation", "Evaluation"),
            "directories" => ("infrastructure", "Model directories"),
            _ => ("infrastructure", "Admission and resource limits"),
        };
        let suffix = if page == "infrastructure" {
            format!("&section={section}")
        } else {
            String::new()
        };
        Self {
            text: format!("{action} in Settings > Hosts > {} > {label}.", host.name),
            fix_anchor: format!("#/settings/{page}?host={}{suffix}", host.id),
        }
    }
}

impl Host {
    pub fn capability_reason(&self, kind: &str) -> Reason {
        let detail = self.reasons.get(kind);
        if let Some(reason) = self.capability_reasons.get(kind)
            && detail == Some(&reason.text)
        {
            return reason.clone();
        }
        let action = match detail.map(String::as_str) {
            Some(
                "Disabled by the controller configuration"
                | "Disabled by infrastructure job policy"
                | "Disabled by infrastructure administration",
            ) => format!("Enable {kind} jobs in the worker policy"),
            _ if kind == "benchmark" && self.benchmark_targets.is_empty() => {
                "Configure a benchmark target and enable benchmark jobs".into()
            }
            _ if kind == "benchmark" && self.benchmarks.is_empty() => {
                "Configure a benchmark suite and enable benchmark jobs".into()
            }
            _ if matches!(kind, "generate" | "evaluate") => {
                "Configure the Halogen runtime and enable evaluation jobs".into()
            }
            _ if kind == "baseline" => {
                "Configure the baseline runtime and enable baseline jobs".into()
            }
            _ => format!("Check the {kind} runtime and worker policy"),
        };
        let mut reason = Reason::settings(self, "policy", &action);
        if let Some(detail) = detail {
            reason.text = format!("{}. {}", detail.trim_end_matches('.'), reason.text);
        }
        reason
    }

    pub fn refresh_capability_reasons(&mut self) {
        self.capability_reasons = self
            .capabilities
            .iter()
            .filter(|&(_, enabled)| !enabled)
            .map(|(kind, _)| (kind.clone(), self.capability_reason(kind)))
            .collect();
    }

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
        let mut capabilities = BTreeMap::<String, bool>::new();
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
        for capability in [
            "gguf_output",
            "gguf_hf_output",
            crate::gguf::STANDALONE_CAPABILITY,
            crate::gguf::ROCMFPX_CAPABILITY,
            crate::gguf::LAYER_RECIPE_CAPABILITY,
        ] {
            let enabled = options.remove(capability).unwrap_or(false)
                && (capability == "gguf_output" || capabilities.get("gguf_output") == Some(&true));
            capabilities.insert(capability.into(), enabled);
            if !enabled {
                reasons.entry(capability.into()).or_insert_with(|| {
                    "GGUF output is not configured for this source on this worker".into()
                });
            }
        }
        let native = brewery_worker::hgn_recipe::CAPABILITY;
        let native_enabled = options.remove(native).unwrap_or(false) && config["local"] == true;
        capabilities.insert(native.into(), native_enabled);
        if !native_enabled {
            reasons.entry(native.into()).or_insert_with(|| {
                "Native HT3/HT4 layer recipes require a configured local source worker".into()
            });
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
                "generate"
                    | "evaluate"
                    | "baseline"
                    | "benchmark"
                    | "hf_download"
                    | brewery_worker::trace::JOB
                    | brewery_worker::exl3_layer::JOB
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
                    } else if kind == brewery_worker::trace::JOB {
                        "Configure pinned CPU Lanternfish and opt in on this worker"
                    } else if kind == brewery_worker::exl3_layer::JOB {
                        "Configure a native EXL3 helper and opt in on this worker"
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
        let hy4 = options
            .remove(brewery_worker::hy4::CAPABILITY)
            .unwrap_or(false);
        let hy4_encode = options.remove(brewery_worker::hy4::ENCODE).unwrap_or(false);
        if hy4 && hy4_encode {
            return Err("Hy4 plan and encode require separate workers".into());
        }
        if hy4 || hy4_encode {
            for (capability, enabled) in &mut capabilities {
                if capability != if hy4 { "plan" } else { "convert" } {
                    *enabled = false;
                    reasons.insert(
                        capability.clone(),
                        "Dedicated Hy4 worker does not support this operation".into(),
                    );
                }
            }
        }
        capabilities.insert(brewery_worker::hy4::CAPABILITY.into(), hy4);
        capabilities.insert(brewery_worker::hy4::ENCODE.into(), hy4_encode);
        if !hy4_encode {
            reasons
                .entry(brewery_worker::hy4::ENCODE.into())
                .or_insert_with(|| "Hy4 gate encoding is not configured on this worker".into());
        }
        if !hy4 {
            reasons
                .entry(brewery_worker::hy4::CAPABILITY.into())
                .or_insert_with(|| {
                    "Hy4 metadata-only planning is not configured on this worker".into()
                });
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
        if hy4_encode && quantization_pool {
            return Err("Dedicated Hy4 encoding does not accept pool scheduling".into());
        }
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
            remote_editor: None,
            provision_binding: None,
            id: id.into(),
            name,
            capabilities,
            component_cpu_only,
            reasons,
            capability_reasons: BTreeMap::new(),
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

    fn conversion_backend_blocker(&self, options: &Map<String, Value>) -> Option<Reason> {
        self.quantization_backends
            .iter()
            .find_map(|(encoder, backends)| {
                // If the request omits the backend, the worker supplies its default.
                let backend = options.get(&format!("{encoder}_backend"))?.as_str();
                (!backends
                    .iter()
                    .any(|allowed| Some(allowed.as_str()) == backend))
                .then(|| {
                    let backend = backend.unwrap_or("requested");
                    Reason::settings(
                        self,
                        "policy",
                        &format!(
                            "Allow the {backend} backend for {encoder}, or select another backend"
                        ),
                    )
                })
            })
    }
}

pub fn input_blocker(job: &Job, envelope: &Envelope, hosts: &[Host]) -> Option<Reason> {
    hosts.iter().find_map(|host| {
        let source = envelope.inputs.iter().any(|input| input.host == host.id);
        if source && host.scheduling_paused {
            return Some(Reason::settings(
                host,
                "policy",
                "Resume job scheduling for the input worker",
            ));
        }
        if envelope.options.contains_key(brewery_worker::hy4::OPTION) {
            return None;
        }
        let hy4_encode = host.capabilities.get(brewery_worker::hy4::ENCODE) == Some(&true);
        let hy4_plan = host.capabilities.get(brewery_worker::hy4::CAPABILITY) == Some(&true);
        let remote_edit = crate::weights::is_kind(&job.kind)
            && host.id == job.host
            && job.remote_editor.is_some()
            && host.remote_editor == job.remote_editor;
        if (hy4_encode && (source || host.id == job.host)) || (source && hy4_plan && !remote_edit) {
            return Some(Reason::settings(
                host,
                "policy",
                "Use a worker that is not dedicated to Hy4",
            ));
        }
        None
    })
}

pub fn conversion_candidate_blocker(
    host: &Host,
    job: &Job,
    envelope: &Envelope,
    hosts: &[Host],
) -> Option<Reason> {
    let inputs = &envelope.inputs;
    let source = inputs.first();
    if job.host == "auto" {
        if hosts.iter().any(|host| host.quantization_pool) {
            if !host.quantization_pool {
                return Some(Reason::settings(
                    host,
                    "policy",
                    "Add this worker to the conversion pool",
                ));
            }
        } else if source.is_none_or(|source| source.host != host.id) {
            return Some(Reason::link(
                "Select the source worker or configure a conversion pool in Settings.",
                "#/settings/infrastructure",
            ));
        }
        if host.capabilities.get(brewery_worker::hy4::ENCODE) == Some(&true) {
            return Some(Reason::settings(
                host,
                "policy",
                "Use a worker that is not dedicated to Hy4",
            ));
        }
    }
    if host.scheduling_paused {
        return Some(Reason::settings(host, "policy", "Resume job scheduling"));
    }
    if !host.capabilities["convert"] {
        return Some(host.capability_reason("convert"));
    }
    host.conversion_backend_blocker(&job.options)
        .or_else(|| conversion_input_blocker(host, job, envelope))
}

fn conversion_input_blocker(host: &Host, job: &Job, envelope: &Envelope) -> Option<Reason> {
    let inputs = &envelope.inputs;
    let source = inputs.first();
    let local = || inputs.iter().all(|input| input.host == host.id);
    let capability = |kind: &str| (!host.capabilities[kind]).then(|| host.capability_reason(kind));
    let local_inputs = || {
        (!local())
            .then(|| Reason::link("Select the worker that holds the input files.", "#/library"))
    };
    if envelope
        .options
        .get("output_format")
        .and_then(Value::as_str)
        == Some(brewery_worker::hgn_recipe::FORMAT)
    {
        return capability(brewery_worker::hgn_recipe::CAPABILITY).or_else(local_inputs);
    }
    if !job.format.is_hgn() {
        return capability("gguf_output")
            .or_else(|| {
                envelope
                    .options
                    .contains_key("layer_recipe")
                    .then(|| capability(crate::gguf::LAYER_RECIPE_CAPABILITY).or_else(local_inputs))
                    .flatten()
            })
            .or_else(|| match envelope.options.get("encoder_family") {
                None => None,
                Some(family) if family == "rocmfpx" => capability(crate::gguf::ROCMFPX_CAPABILITY),
                Some(_) => Some(Reason::link(
                    "Select a supported GGUF encoder.",
                    "#/convert",
                )),
            })
            .or_else(|| match source.map(|source| source.kind.as_str()) {
                Some("hf") => capability("gguf_hf_output"),
                Some("safetensors") => {
                    capability(crate::gguf::STANDALONE_CAPABILITY).or_else(local_inputs)
                }
                Some("gguf") => None,
                _ => Some(Reason::link(
                    "Select a GGUF, HF or standalone SafeTensors source.",
                    "#/library",
                )),
            });
    }
    let needs_components = source.is_some_and(|source| source.kind == "hf")
        && match job.kind.as_str() {
            "inspect" | "plan" | "convert" => inputs.len() > 1,
            "verify" => inputs.len() > 2,
            _ => false,
        };
    if needs_components && let Some(reason) = capability("convert_components") {
        return Some(reason);
    }
    if job.profile_revision.is_some()
        && let Some(reason) = capability("custom_profiles")
    {
        return Some(reason);
    }
    (host.component_cpu_only
        && matches!(job.kind.as_str(), "inspect" | "plan" | "convert" | "verify")
        && [
            "ht_backend",
            "i4r_backend",
            "q4cp_backend",
            "decode_backend",
        ]
        .iter()
        .any(|key| job.options.get(*key).is_some_and(|v| v != "cpu")))
    .then(|| cpu_conversion_reason(host))
}

pub fn cpu_conversion_reason(host: &Host) -> Reason {
    Reason::link(
        &format!(
            "Select CPU backends or select another worker. {} uses CPU backends only.",
            host.name
        ),
        "#/convert",
    )
}

pub const SIMPLE_WORKFLOWS: &[&str] = &[
    "compare",
    "generate",
    "baseline",
    "evaluate",
    "benchmark",
    "transfer",
    "transfer_preview",
];

pub fn standard_job_host_blocker(host: &Host, kind: &str) -> Option<Reason> {
    if !host.capabilities[kind] {
        Some(host.capability_reason(kind))
    } else if host.scheduling_paused {
        Some(Reason::settings(host, "policy", "Resume job scheduling"))
    } else if host.capabilities.get(brewery_worker::hy4::CAPABILITY) == Some(&true)
        || host.capabilities.get(brewery_worker::hy4::ENCODE) == Some(&true)
    {
        Some(Reason::settings(
            host,
            "policy",
            "Use a worker that is not dedicated to Hy4",
        ))
    } else {
        None
    }
}

pub fn manual_registration_blocker(host: &str, kind: &str) -> Option<Reason> {
    if kind == brewery_worker::hgn_recipe::ARCHIVE {
        Some(Reason::link(
            "Run a verified recipe to register a native HGN archive.",
            "#/convert",
        ))
    } else if host == crate::backend_hf::ID {
        Some(Reason::link(
            "Select another worker in Library. Only verified jobs can register files on Backend downloads.",
            "#/library",
        ))
    } else {
        None
    }
}

// Prefer the source host if eligible, avoiding a transfer. Selection of an
// *idle* host occurs in the scheduler, not during POST.
pub fn candidates<'a>(job: &Job, envelope: &Envelope, hosts: &'a [Host]) -> Vec<&'a Host> {
    if job.scratch_job {
        return hosts
            .iter()
            .filter(|host| {
                host.id == job.host
                    && host.local_edit
                    && !host.scheduling_paused
                    && host.capabilities["weight_apply"]
                    && host.capabilities["weight_safetensors"]
                    && (job.kind != brewery_worker::trace::JOB
                        || host.capabilities[brewery_worker::trace::JOB])
            })
            .collect();
    }
    if input_blocker(job, envelope, hosts).is_some() {
        return Vec::new();
    }
    let inputs = &envelope.inputs;
    if envelope.options.contains_key(brewery_worker::hy4::OPTION) {
        let Some(capability) =
            brewery_worker::hy4::capability(&job.kind, &Value::Object(envelope.options.clone()))
        else {
            return Vec::new();
        };
        if inputs.len() != 1 || inputs[0].kind != "hf" || !job.format.is_hgn() {
            return Vec::new();
        }
        return hosts
            .iter()
            .filter(|host| {
                host.id == job.host
                    && host.id == inputs[0].host
                    && !host.scheduling_paused
                    && host.capabilities.get(capability) == Some(&true)
                    && host.capabilities[&job.kind]
            })
            .collect();
    }
    if job.kind != "convert" {
        if SIMPLE_WORKFLOWS.contains(&job.kind.as_str()) {
            return hosts
                .iter()
                .filter(|host| {
                    host.id == job.host
                        && standard_job_host_blocker(host, &job.kind).is_none()
                        && (job.kind != "benchmark" || host.check_benchmark(&job.options).is_ok())
                })
                .collect();
        }
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
                    && (!crate::weights::is_kind(&job.kind)
                        || (host.local_edit == job.local_edit
                            && host.remote_editor == job.remote_editor))
                    && host.capabilities[&job.kind]
                    && (host.capabilities.get(brewery_worker::hy4::CAPABILITY) != Some(&true)
                        || (crate::weights::is_kind(&job.kind) && job.remote_editor.is_some()))
                    && (!needs_hgn || host.capabilities["weight_hgn"])
                    && (!needs_safetensors || host.capabilities["weight_safetensors"])
                    && (!needs_packed || host.capabilities["weight_packed"])
                    && (job.kind != brewery_worker::exl3_layer::JOB
                        || (!host.local_edit
                            && host.remote_editor.is_none()
                            && (!host.component_cpu_only
                                || envelope.options.get("backend").is_some_and(|b| b == "cpu"))))
                    && (!crate::weights::needs_visual(&job.kind, &job.options)
                        || host.capabilities["weight_visual"])
                    && conversion_input_blocker(host, job, envelope).is_none()
            })
            .collect();
    }
    let source = inputs.first().map(|model| model.host.as_str());
    let mut eligible: Vec<_> = hosts
        .iter()
        .filter(|host| {
            (job.host == "auto" || job.host == host.id)
                && conversion_candidate_blocker(host, job, envelope, hosts).is_none()
        })
        .collect();
    if job.host == "auto" {
        eligible.sort_by_key(|host| Some(host.id.as_str()) != source);
    }
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
        self.build_inner(host, inputs, prompt, format, false)
    }

    pub(crate) fn build_hgn_recipe(
        self,
        host: String,
        inputs: Vec<Model>,
    ) -> Result<(Job, Envelope), String> {
        self.build_inner(host, inputs, None, crate::gguf::Format::Hgn, true)
    }

    fn build_inner(
        self,
        host: String,
        inputs: Vec<Model>,
        prompt: Option<String>,
        format: crate::gguf::Format,
        hgn_recipe: bool,
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
        let valid = if hgn_recipe {
            matches!(
                self.kind.as_str(),
                "inspect" | "plan" | "convert" | "verify"
            ) && inputs.iter().all(|input| input.host == host)
                && brewery_worker::hgn_recipe::layout(
                    self.kind == "verify",
                    inputs
                        .iter()
                        .map(|input| (input.kind.as_str(), input.host.as_str())),
                )
        } else if !format.is_hgn() {
            if main == "safetensors" && host != "auto" && inputs[0].host != host {
                return Err("standalone SafeTensors GGUF must run on the source worker".into());
            }
            matches!(
                self.kind.as_str(),
                "inspect" | "plan" | "convert" | "verify"
            ) && crate::gguf::input_layout(&inputs, self.kind == "verify").is_ok()
        } else {
            match self.kind.as_str() {
                "compare" => n == 2 && main == "gguf" && second == "gguf" && prompt.is_some(),
                brewery_worker::trace::JOB => {
                    n == 1
                        && matches!(main, "gguf" | brewery_worker::trace::HGN_KIND)
                        && host == inputs[0].host
                }
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
                brewery_worker::exl3_layer::JOB => {
                    n == 1 && main == "safetensors" && host == inputs[0].host && host != "auto"
                }
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
        if prompt.is_some()
            && !matches!(
                self.kind.as_str(),
                "compare" | "generate" | brewery_worker::trace::JOB
            )
        {
            return Err("this workflow does not accept a prompt".into());
        }
        if self.options.contains_key(brewery_worker::hy4::OPTION) {
            if !format.is_hgn()
                || brewery_worker::hy4::capability(&self.kind, &Value::Object(self.options.clone()))
                    .is_none()
                || n != 1
                || main != "hf"
                || inputs[0].host != host
                || prompt.is_some()
            {
                return Err("Hy4 jobs require one same-worker HF source and the exact operation-specific Q8_0 override".into());
            }
        } else if format.is_hgn() && !hgn_recipe {
            if self.kind == brewery_worker::exl3_layer::JOB {
                brewery_worker::exl3_layer::Spec::parse(&Value::Object(self.options.clone()))
                    .map_err(|e| format!("{e:#}"))?;
            } else if crate::weights::is_kind(&self.kind) {
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
        Ok(self.records(host, inputs, prompt, format))
    }

    pub(crate) fn records(
        self,
        host: String,
        inputs: Vec<Model>,
        prompt: Option<String>,
        format: crate::gguf::Format,
    ) -> (Job, Envelope) {
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
            scratch_pins: Default::default(),
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
            remote_editor: None,
            directory_revision: None,
            scratch_job: false,
            input_ids: self.input_ids,
            options: self.options,
            prompt_id: self.prompt_id,
            cancel_requested: false,
            has_result: false,
            error: None,
            created_at: at,
            updated_at: at,
        };
        (job, envelope)
    }
}

fn check_options(job: &str, options: &Map<String, Value>, inputs: &[Model]) -> Result<(), String> {
    // Benchmark options are validated as a whole against the selected host
    // before enqueue and again by the scheduler before dispatch.
    if matches!(job, "benchmark" | brewery_worker::trace::JOB) {
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
