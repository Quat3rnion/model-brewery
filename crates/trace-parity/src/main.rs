use serde::{Deserialize, Deserializer};
use std::collections::{BTreeMap, HashSet};
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

type Result<T> = std::result::Result<T, String>;

const SCHEMA_VERSION: u32 = 1;
const MAX_TOKENS: usize = 512;
const MAX_SCALARS: usize = 16_777_216;
const MAX_BYTES: u64 = 256 * 1024 * 1024;
const HELP: &str = "\
Usage: trace-parity [--require-moe] [--require-shared-gate] [--atol N] [--rtol N] CPU.json GPU.json

Compare two Lanternfish replay v1 JSON files of the same GGUF and fixed tokens.
The CPU file is the reference. Both prefill and singleton decode are required,
with every position, every layer residual, and full vocabulary logits.
Each position needs attention and FFN RMS coverage. A null layer signal declares
an omission, which must match the CPU; omitted keys are invalid.
MoE traces need routing IDs and weights on at least one layer at each position.
Routing IDs and order must match exactly. Each step needs observed matmul on
its requested backend. CPU traces must have zero GPU compute.

  --require-moe          Reject a dense model for a MoE test.
  --require-shared-gate  Require shared-gate coverage at each position.
  --atol N, --rtol N     Override one tolerance column for all numeric signals.
                         Diagnostic only; not valid for release evidence.

Fixed-v1 tolerances (absolute, relative):
  logits                    0.05,  0.02
  residual, attention/FFN   0.01,  0.02
  routing weights, gate     0.001, 0.02
For each finite scalar, |GPU - CPU| <= atol + rtol * |CPU|. Absolute tolerance
handles values near zero; relative tolerance allows drift at larger magnitude.
These are initial parity bounds, not measured guarantees for all devices.

Limits per file: 512 total tokens, 16,777,216 scalar slots, 256 MiB JSON.
Exit: 0 comparison passed; 1 evidence invalid or divergent; 2 usage error.
A pairwise result does not authenticate files or qualify a GPU architecture.
Record device/driver inventory and build provenance independently for release.
";

#[derive(Default)]
struct Options {
    require_moe: bool,
    require_shared_gate: bool,
    atol: Option<f32>,
    rtol: Option<f32>,
}

impl Options {
    fn diagnostic(&self) -> bool {
        self.atol.is_some() || self.rtol.is_some()
    }
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<(Options, PathBuf, PathBuf)> {
    let mut options = Options::default();
    let mut paths = Vec::new();
    let mut args = args.into_iter();
    let mut flags = true;
    while let Some(arg) = args.next() {
        match (flags, arg.to_str()) {
            (true, Some("--")) => flags = false,
            (true, Some("--require-moe")) => options.require_moe = true,
            (true, Some("--require-shared-gate")) => options.require_shared_gate = true,
            (true, Some(flag @ ("--atol" | "--rtol"))) => {
                let value = args
                    .next()
                    .and_then(|arg| arg.to_str()?.parse::<f32>().ok())
                    .filter(|n| n.is_finite() && *n >= 0.0)
                    .ok_or_else(|| format!("{flag} requires a finite, nonnegative number"))?;
                let slot = if flag == "--atol" {
                    &mut options.atol
                } else {
                    &mut options.rtol
                };
                if slot.replace(value).is_some() {
                    return Err(format!("{flag} may only be given once"));
                }
            }
            (true, Some(s)) if s.starts_with('-') => return Err(format!("unknown option: {s}")),
            _ => paths.push(PathBuf::from(arg)),
        }
    }
    let [cpu, gpu] = paths
        .try_into()
        .map_err(|_| "expected one CPU file and one GPU file")?;
    Ok((options, cpu, gpu))
}

#[derive(Deserialize)]
struct Trace {
    schema_version: u32,
    model: Model,
    build: Build,
    execution: Execution,
    prefill_tokens: Vec<i32>,
    decode_tokens: Vec<i32>,
    positions: Vec<Position>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Model {
    name: String,
    architecture: String,
    sha256: String,
    bytes: u64,
    n_layers: usize,
    hidden: usize,
    vocab: usize,
    n_experts: usize,
    top_k: usize,
}

#[derive(Deserialize)]
struct Build {
    lanternfish_version: String,
    lanternfish_sha256: String,
    llama_system_info: String,
    runtime_libraries: BTreeMap<PathBuf, String>,
}

#[derive(Deserialize, PartialEq, Eq, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Backend {
    Cpu,
    Gpu,
}

#[derive(Deserialize)]
struct Execution {
    backend: Backend,
    requested_gpu_layers: i32,
    #[serde(deserialize_with = "nullable")]
    device: Option<Device>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Device {
    name: String,
    description: String,
    backend: String,
    #[serde(rename = "id", deserialize_with = "nullable")]
    _id: Option<String>,
}

#[derive(Deserialize, PartialEq, Eq, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Prefill,
    Decode,
}

#[derive(Deserialize)]
struct Step {
    first: usize,
    n: usize,
    phase: Phase,
    gpu_compute_nodes: usize,
    gpu_matmul_nodes: usize,
    cpu_matmul_nodes: usize,
}

#[derive(Deserialize)]
struct Position {
    position: usize,
    phase: Phase,
    logits: Vec<f32>,
    layers: Vec<Layer>,
}

#[derive(Deserialize)]
struct Layer {
    layer: usize,
    residual: Vec<f32>,
    #[serde(deserialize_with = "nullable")]
    attn_rms: Option<f32>,
    #[serde(deserialize_with = "nullable")]
    ffn_rms: Option<f32>,
    #[serde(deserialize_with = "nullable")]
    expert_ids: Option<Vec<i32>>,
    #[serde(deserialize_with = "nullable")]
    expert_weights: Option<Vec<f32>>,
    #[serde(deserialize_with = "nullable")]
    shared_gate: Option<f32>,
}

// `Option` normally accepts a missing key. Capture must explicitly write null.
fn nullable<'de, D, T>(d: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(d)
}

fn read_trace(path: &Path) -> Result<Trace> {
    let load = || {
        let meta = path.metadata().map_err(|e| e.to_string())?;
        if !meta.is_file() || meta.len() > MAX_BYTES {
            return Err("not a regular file or exceeds 256 MiB".into());
        }
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mut input = BufReader::new(file).take(MAX_BYTES + 1);
        let result = serde_json::from_reader(&mut input).map_err(|e| e.to_string());
        if input.limit() == 0 {
            return Err("exceeds 256 MiB".into());
        }
        result
    };
    load().map_err(|e| format!("{}: {e}", path.display()))
}

fn require(condition: bool, message: impl FnOnce() -> String) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message())
    }
}

fn sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn validate_execution(run: &Trace, backend: Backend) -> Result<()> {
    let exec = &run.execution;
    require(exec.backend == backend, || {
        format!("expected backend {backend:?}, got {:?}", exec.backend)
    })?;
    match (backend, &exec.device) {
        (Backend::Cpu, None) if exec.requested_gpu_layers == 0 => {}
        (Backend::Gpu, Some(device))
            if (exec.requested_gpu_layers == -1 || exec.requested_gpu_layers > 0)
                && !device.name.trim().is_empty()
                && !device.description.trim().is_empty()
                && !device.backend.trim().is_empty()
                && !device.backend.trim().eq_ignore_ascii_case("cpu")
                && !device.backend.trim().eq_ignore_ascii_case("rpc") => {}
        _ => {
            return Err(format!(
                "invalid {backend:?} device or requested_gpu_layers"
            ))
        }
    }
    let prefill = run.prefill_tokens.len();
    require(exec.steps.len() == 1 + run.decode_tokens.len(), || {
        "execution.steps must cover the prefill batch and every decode token".into()
    })?;
    for (i, step) in exec.steps.iter().enumerate() {
        let expected = if i == 0 {
            (0, prefill, Phase::Prefill)
        } else {
            (prefill + i - 1, 1, Phase::Decode)
        };
        require((step.first, step.n, step.phase) == expected, || {
            format!("execution.steps[{i}] has incorrect phase or position coverage")
        })?;
        let evidence_ok = match backend {
            Backend::Cpu => {
                step.gpu_compute_nodes == 0
                    && step.gpu_matmul_nodes == 0
                    && step.cpu_matmul_nodes > 0
            }
            Backend::Gpu => {
                step.gpu_matmul_nodes > 0 && step.gpu_compute_nodes >= step.gpu_matmul_nodes
            }
        };
        require(evidence_ok, || {
            format!("execution.steps[{i}] has invalid {backend:?} compute/matmul evidence")
        })?;
    }
    Ok(())
}

fn validate_trace(run: &Trace, backend: Backend, options: &Options) -> Result<()> {
    require(run.schema_version == SCHEMA_VERSION, || {
        format!("unsupported schema_version {}", run.schema_version)
    })?;
    let m = &run.model;
    let b = &run.build;
    require(
        !m.name.trim().is_empty()
            && !m.architecture.trim().is_empty()
            && sha256(&m.sha256)
            && m.bytes > 0,
        || "invalid model identity".into(),
    )?;
    require(
        !b.lanternfish_version.trim().is_empty()
            && sha256(&b.lanternfish_sha256)
            && !b.llama_system_info.trim().is_empty()
            && b.runtime_libraries
                .iter()
                .all(|(path, hash)| path.is_absolute() && sha256(hash)),
        || "invalid build identity".into(),
    )?;
    require(
        m.n_layers > 0
            && m.hidden > 0
            && m.vocab > 0
            && m.top_k <= m.n_experts
            && (m.n_experts == 0) == (m.top_k == 0),
        || "invalid model dimensions".into(),
    )?;
    require(!options.require_moe || m.n_experts > 0, || {
        "--require-moe needs a routed model".into()
    })?;
    let count = run.prefill_tokens.len() + run.decode_tokens.len();
    require(
        !run.prefill_tokens.is_empty() && !run.decode_tokens.is_empty() && count <= MAX_TOKENS,
        || format!("need nonempty prefill and decode with at most {MAX_TOKENS} total tokens"),
    )?;
    // Same budget as capture: residual and logits, plus routed IDs, weights
    // and the three scalar lanes, including slots declared null.
    let slots = m
        .top_k
        .checked_mul(2)
        .and_then(|n| n.checked_add(3))
        .and_then(|n| n.checked_add(m.hidden))
        .and_then(|n| n.checked_mul(m.n_layers))
        .and_then(|n| n.checked_add(m.vocab))
        .and_then(|n| n.checked_mul(count));
    require(slots.is_some_and(|n| n <= MAX_SCALARS), || {
        format!("declared shapes exceed {MAX_SCALARS} scalar slots")
    })?;
    for (i, &token) in run
        .prefill_tokens
        .iter()
        .chain(&run.decode_tokens)
        .enumerate()
    {
        require((0..m.vocab).contains(&(token as usize)), || {
            format!("token at position {i} is outside the vocabulary")
        })?;
    }
    validate_execution(run, backend)?;
    require(run.positions.len() == count, || {
        "positions must cover every prefill and decode token".into()
    })?;
    for (i, p) in run.positions.iter().enumerate() {
        let phase = if i < run.prefill_tokens.len() {
            Phase::Prefill
        } else {
            Phase::Decode
        };
        require(p.position == i && p.phase == phase, || {
            format!("position {i} has incorrect index or phase")
        })?;
        require(
            p.logits.len() == m.vocab && p.layers.len() == m.n_layers,
            || format!("position {i}: logits or layer count does not match model dimensions"),
        )?;
        let (mut attn, mut ffn, mut routed, mut shared) = (false, false, false, false);
        for (j, layer) in p.layers.iter().enumerate() {
            require(layer.layer == j && layer.residual.len() == m.hidden, || {
                format!("position {i} layer {j}: incorrect layer index or residual width")
            })?;
            attn |= layer.attn_rms.is_some();
            ffn |= layer.ffn_rms.is_some();
            shared |= layer.shared_gate.is_some();
            match (&layer.expert_ids, &layer.expert_weights) {
                (None, None) => {}
                (Some(ids), Some(weights)) if m.top_k > 0 => {
                    require(ids.len() == m.top_k && weights.len() == m.top_k, || {
                        format!("position {i} layer {j}: expert IDs/weights do not match top_k")
                    })?;
                    let mut seen = HashSet::new();
                    require(
                        ids.iter().all(|&id| id >= 0 && (id as usize) < m.n_experts && seen.insert(id)),
                        || format!("position {i} layer {j}: out-of-range or duplicate expert ID"),
                    )?;
                    routed = true;
                }
                _ => {
                    return Err(format!(
                        "position {i} layer {j}: expert IDs and weights must both be present for routing"
                    ))
                }
            }
        }
        require(attn && ffn, || {
            format!("position {i}: missing attention or FFN RMS coverage")
        })?;
        require(m.n_experts == 0 || routed, || {
            format!("position {i}: missing MoE routing coverage")
        })?;
        require(!options.require_shared_gate || shared, || {
            format!("position {i}: missing shared_gate coverage")
        })?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Signal {
    Logits,
    Residual,
    AttnRms,
    FfnRms,
    ExpertWeights,
    SharedGate,
}

impl Signal {
    fn name(self) -> &'static str {
        match self {
            Self::Logits => "logits",
            Self::Residual => "residual",
            Self::AttnRms => "attn_rms",
            Self::FfnRms => "ffn_rms",
            Self::ExpertWeights => "expert_weights",
            Self::SharedGate => "shared_gate",
        }
    }

    fn atol(self) -> f64 {
        match self {
            Self::Logits => 0.05,
            Self::Residual | Self::AttnRms | Self::FfnRms => 0.01,
            Self::ExpertWeights | Self::SharedGate => 0.001,
        }
    }
}

#[derive(Clone, Copy)]
struct Location {
    position: usize,
    layer: Option<usize>,
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "position {}", self.position)?;
        if let Some(layer) = self.layer {
            write!(f, " layer {layer}")?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Summary {
    // Counts are per layer * position. Nulls are reported as omissions.
    attn: usize,
    ffn: usize,
    routed: usize,
    shared: usize,
}

fn numeric(
    options: &Options,
    location: Location,
    signal: Signal,
    cpu: &[f32],
    gpu: &[f32],
) -> Result<()> {
    // Shapes are checked against the same declared model before comparison.
    let atol = options.atol.map_or(signal.atol(), f64::from);
    let rtol = options.rtol.map_or(0.02, f64::from);
    for (i, (&cpu, &gpu)) in cpu.iter().zip(gpu).enumerate() {
        let (cpu, gpu) = (f64::from(cpu), f64::from(gpu));
        let delta = (gpu - cpu).abs();
        let bound = atol + rtol * cpu.abs();
        if !cpu.is_finite() || !gpu.is_finite() || delta > bound {
            return Err(format!(
                "{location} {}[{i}]: CPU={cpu:.8e} GPU={gpu:.8e}, |delta|={delta:.8e}, bound={bound:.8e}",
                signal.name(),
            ));
        }
    }
    Ok(())
}

fn optional<T>(
    location: Location,
    name: &str,
    cpu: Option<T>,
    gpu: Option<T>,
) -> Result<Option<(T, T)>> {
    match (cpu, gpu) {
        (Some(cpu), Some(gpu)) => Ok(Some((cpu, gpu))),
        (None, None) => Ok(None),
        _ => Err(format!(
            "{location} {name}: CPU/GPU signal coverage differs"
        )),
    }
}

fn compare(cpu: &Trace, gpu: &Trace, options: &Options) -> Result<Summary> {
    require(cpu.model == gpu.model, || {
        format!(
            "CPU/GPU model identity or dimensions differ: CPU={:?}, GPU={:?}",
            cpu.model, gpu.model
        )
    })?;
    require(
        cpu.prefill_tokens == gpu.prefill_tokens && cpu.decode_tokens == gpu.decode_tokens,
        || "CPU/GPU fixed token sequences differ".into(),
    )?;
    let mut summary = Summary::default();
    for (a, b) in cpu.positions.iter().zip(&gpu.positions) {
        let location = Location {
            position: a.position,
            layer: None,
        };
        numeric(options, location, Signal::Logits, &a.logits, &b.logits)?;
        for (a, b) in a.layers.iter().zip(&b.layers) {
            let location = Location {
                layer: Some(a.layer),
                ..location
            };
            numeric(
                options,
                location,
                Signal::Residual,
                &a.residual,
                &b.residual,
            )?;
            for (signal, a, b, count) in [
                (Signal::AttnRms, a.attn_rms, b.attn_rms, &mut summary.attn),
                (Signal::FfnRms, a.ffn_rms, b.ffn_rms, &mut summary.ffn),
                (
                    Signal::SharedGate,
                    a.shared_gate,
                    b.shared_gate,
                    &mut summary.shared,
                ),
            ] {
                if let Some((a, b)) = optional(location, signal.name(), a.as_ref(), b.as_ref())? {
                    numeric(
                        options,
                        location,
                        signal,
                        std::slice::from_ref(a),
                        std::slice::from_ref(b),
                    )?;
                    *count += 1;
                }
            }
            if let Some((a, b)) = optional(
                location,
                "expert_ids",
                a.expert_ids.as_deref(),
                b.expert_ids.as_deref(),
            )? {
                require(a == b, || {
                    let i = a
                        .iter()
                        .zip(b)
                        .position(|(a, b)| a != b)
                        .expect("equal ID shapes");
                    format!(
                        "{location} expert_ids[{i}]: CPU={} GPU={} (exact ordered IDs required)",
                        a[i], b[i]
                    )
                })?;
                summary.routed += 1;
            }
            if let Some((a, b)) = optional(
                location,
                "expert_weights",
                a.expert_weights.as_deref(),
                b.expert_weights.as_deref(),
            )? {
                numeric(options, location, Signal::ExpertWeights, a, b)?;
            }
        }
    }
    Ok(summary)
}

fn load_and_validate(path: &Path, backend: Backend, options: &Options) -> Result<Trace> {
    let run = read_trace(path)?;
    validate_trace(&run, backend, options).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(run)
}

fn run(options: Options, cpu_path: &Path, gpu_path: &Path) -> Result<()> {
    let cpu = load_and_validate(cpu_path, Backend::Cpu, &options)?;
    let gpu = load_and_validate(gpu_path, Backend::Gpu, &options)?;
    let summary = compare(&cpu, &gpu, &options)?;
    let slots = cpu.positions.len() * cpu.model.n_layers;
    let status = if options.diagnostic() {
        "DIAGNOSTIC PASS (not release evidence)"
    } else {
        "PASS (fixed-v1)"
    };
    let device = gpu.execution.device.as_ref().expect("validated GPU device");
    println!(
        "{status}: model_sha256={}, gpu_device={:?}, gpu_backend={:?}, prefill={}, decode={}, layers={}",
        cpu.model.sha256,
        device.name,
        device.backend,
        cpu.prefill_tokens.len(),
        cpu.decode_tokens.len(),
        cpu.model.n_layers,
    );
    println!(
        "Layer-position coverage: residual={slots}/{slots}, attn_rms={}/{slots}, ffn_rms={}/{slots}, routing={}/{slots}, shared_gate={}/{slots} (remaining slots explicitly null)",
        summary.attn, summary.ffn, summary.routed, summary.shared,
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if matches!(args.as_slice(), [arg] if arg == "--help" || arg == "-h") {
        print!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let (options, cpu, gpu) = match parse_args(args) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("trace-parity: {e}; see --help");
            return ExitCode::from(2);
        }
    };
    match run(options, &cpu, &gpu) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("trace-parity: FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}
