//! Bounded schema v1 exports and the read-only trace wire format.
use anyhow::{Context, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
};

pub mod diff;
pub mod routing;

pub const JOB: &str = "trace";
pub const EXECUTABLE: &str = "BREWERY_LANTERNFISH_TRACE";
pub const LIB_DIR: &str = "BREWERY_LANTERNFISH_TRACE_LIB_DIR";
pub const COMMIT: &str = "0ea30b37729a136a76b8412e7ad1ec91bd7de9b5";
pub const SHA256: &str = "0e4ca38c55a9afa0436eeadd06e89bf295eb201b986dd2f40305f95d3a61e46f";
pub const PROJECTION_EXECUTABLE: &str = "BREWERY_LANTERNFISH_PROJECTION";
pub const PROJECTION_LIB_DIR: &str = "BREWERY_LANTERNFISH_PROJECTION_LIB_DIR";
pub const PROJECTION_COMMIT: &str = "d8420857ed1ab2e29c527f1ffde09aeef19f36c4";
pub const PROJECTION_SHA256: &str =
    "3009f4b0c362ddc04d955e40efac9e1d88659b9a0bc393fdc8ea6da287bde010";
pub const ROUTING_EXECUTABLE: &str = "BREWERY_LANTERNFISH_ROUTING";
pub const ROUTING_LIB_DIR: &str = "BREWERY_LANTERNFISH_ROUTING_LIB_DIR";
pub const ROUTING_COMMIT: &str = "5699379af18d6428a6589c996f516c07a91e1a86";
pub const ROUTING_SHA256: &str = "fda55c709a0ef81920daae882147f7f066f506d27d73846785880c424c56a581";
pub const HGN_KIND: &str = "hgn";
pub const HGN_PYTHON: &str = "BREWERY_HGN_TRACE_PYTHON";
pub const HGN_PYTHON_SHA256: &str = "BREWERY_HGN_TRACE_PYTHON_SHA256";
pub const HGN_ASSETS: &str = "BREWERY_HGN_TRACE_ASSETS";
pub const HGN_RUNTIME_SHA256: &str = include_str!("../../../tools/hgn_trace/runtime.sha256");
pub const HGN_RUNTIME: &str = "hgn_cpu";
pub const HARD_BYTE_CAP: u64 = 32 << 20;
pub const RESPONSE_CAP: usize = 4 << 20;
const MAX_POSITIONS: usize = 4096;
const MAX_RETAIN: usize = 64;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Policy {
    pub default_byte_cap: u64,
    pub max_byte_cap: u64,
    pub retain_jobs: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            default_byte_cap: 2 << 20,
            max_byte_cap: 16 << 20,
            retain_jobs: 8,
        }
    }
}

impl Policy {
    pub fn check(self) -> Result<()> {
        ensure!(
            (1..=HARD_BYTE_CAP).contains(&self.max_byte_cap)
                && (1..=self.max_byte_cap).contains(&self.default_byte_cap)
                && (1..=MAX_RETAIN).contains(&self.retain_jobs),
            "invalid trace settings"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    LOut,
    FinalNorm,
    AttnOut,
    FfnInp,
    FfnOut,
    KqSoftMax,
    FfnMoeLogits,
    FfnMoeTopk,
    FfnMoeRouteWeights,
}

impl Role {
    fn routing(self) -> bool {
        matches!(
            self,
            Self::FfnMoeLogits | Self::FfnMoeTopk | Self::FfnMoeRouteWeights
        )
    }
    pub fn is_read_only(self) -> bool {
        matches!(self, Self::KqSoftMax | Self::FinalNorm)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capture {
    layers: BTreeSet<usize>,
    roles: BTreeSet<Role>,
    positions: BTreeSet<usize>,
    top_k: usize,
    pub byte_cap: u64,
    #[serde(default, skip_serializing_if = "is_false")]
    all_logits: bool,
}

impl Capture {
    // Internal circuits use either activation-only or final-position logits-only
    // exports. Unlike the public trace request, the latter reads the whole vocabulary.
    pub fn circuit(layers: &[usize], roles: &[Role], positions: &[usize], top_k: usize) -> Self {
        Self {
            layers: layers.iter().copied().collect(),
            roles: roles.iter().copied().collect(),
            positions: positions.iter().copied().collect(),
            top_k,
            byte_cap: HARD_BYTE_CAP,
            all_logits: false,
        }
    }
}
fn is_false(value: &bool) -> bool { !value }

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    layers: BTreeSet<usize>,
    roles: BTreeSet<Role>,
    positions: BTreeSet<usize>,
    top_k: usize,
    byte_cap: Option<u64>,
    prompts: Option<Vec<String>>,
    #[serde(default)]
    full_dump: bool,
    #[serde(default)]
    interventions: Vec<Rule>,
}

#[derive(Serialize, Deserialize)]
pub struct Rule {
    pub layer: usize,
    pub role: Role,
    pub positions: BTreeSet<usize>,
    #[serde(flatten)]
    pub op: Operation,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Zero,
    Scale { factor: f32 },
    Add { vector: Vec<f32> },
    Replace { vector: Vec<f32> },
    ZeroExpert { expert: usize },
    ProjectOut { vector: Vec<f32> },
    Subtract { vector: Vec<f32>, scale: f32 },
}

impl Operation {
    pub fn vector(&self) -> Option<&[f32]> {
        match self {
            Self::Add { vector }
            | Self::Replace { vector }
            | Self::ProjectOut { vector }
            | Self::Subtract { vector, .. } => Some(vector),
            _ => None,
        }
    }
}

impl Rule {
    pub fn check_all(rules: &[Self]) -> Result<()> {
        let mut edited = BTreeSet::new();
        ensure!(rules.len() <= 64, "too many trace interventions");
        for rule in rules {
            ensure!(
                rule.layer < 4096
                    && !rule.role.is_read_only()
                    && !rule.role.routing()
                    && !rule.positions.is_empty()
                    && rule
                        .positions
                        .iter()
                        .all(|&p| p < MAX_POSITIONS && edited.insert((rule.layer, rule.role, p))),
                "invalid or overlapping trace intervention"
            );
            if let Some(vector) = rule.op.vector() {
                ensure!(
                    (1..=16384).contains(&vector.len()) && vector.iter().all(|v| v.is_finite()),
                    "invalid trace intervention vector"
                );
            }
            match &rule.op {
                Operation::Scale { factor } => ensure!(factor.is_finite(), "invalid trace factor"),
                Operation::ProjectOut { vector } => ensure!(
                    vector.iter().any(|&v| v != 0.0),
                    "projection direction must be nonzero"
                ),
                Operation::ZeroExpert { expert } => ensure!(
                    rule.role == Role::FfnOut && *expert < 4096,
                    "zero_expert requires ffn_out and an expert ID below 4096"
                ),
                Operation::Subtract { vector, scale } => ensure!(
                    scale.is_finite()
                        && *scale >= 0.0
                        && vector.iter().all(|v| (v * scale).is_finite()),
                    "invalid subtraction scale or scaled vector"
                ),
                _ => {}
            }
        }
        Ok(())
    }
}

pub struct Request {
    pub prompts: Vec<String>,
    pub capture: Capture,
    pub interventions: Vec<Rule>,
    pub retain_jobs: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanternfishRuntime {
    Base,
    Projection,
    Routing,
}

impl LanternfishRuntime {
    pub fn pins(self) -> (&'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::Base => (EXECUTABLE, LIB_DIR, SHA256, COMMIT),
            Self::Projection => (PROJECTION_EXECUTABLE, PROJECTION_LIB_DIR, PROJECTION_SHA256, PROJECTION_COMMIT),
            Self::Routing => (ROUTING_EXECUTABLE, ROUTING_LIB_DIR, ROUTING_SHA256, ROUTING_COMMIT),
        }
    }
}

impl Request {
    pub fn lanternfish_runtime(&self) -> Result<LanternfishRuntime> {
        let projection = self.interventions.iter().any(|rule| {
            matches!(rule.op, Operation::ProjectOut { .. } | Operation::Subtract { .. })
        });
        let routing = self.routing();
        ensure!(!(projection && routing), "cannot combine MoE routing and projection in one trace");
        Ok(if projection {
            LanternfishRuntime::Projection
        } else if routing {
            LanternfishRuntime::Routing
        } else {
            LanternfishRuntime::Base
        })
    }

    fn routing(&self) -> bool {
        self.capture.roles.iter().any(|r| r.routing())
            || self
                .interventions
                .iter()
                .any(|r| matches!(r.op, Operation::ZeroExpert { .. }))
    }

    pub fn controller(options: &Value, prompt: Option<&str>, policy: Policy) -> Result<Self> {
        policy.check()?;
        let o: Options =
            serde_json::from_value(options.clone()).context("invalid trace options")?;
        let prompts = match (prompt, o.prompts) {
            (Some(p), None) => vec![p.to_owned()],
            (None, Some(prompts)) => prompts,
            _ => anyhow::bail!("provide either prompt, prompt_id, or options.prompts"),
        };
        ensure!(
            (1..=4).contains(&prompts.len())
                && prompts
                    .iter()
                    .all(|p| !p.trim().is_empty() && p.len() <= 16384 && !p.contains('\0')),
            "trace requires 1 to 4 prompts of at most 16384 bytes each"
        );
        let byte_cap = o.byte_cap.unwrap_or(policy.default_byte_cap);
        ensure!(
            (1..=policy.max_byte_cap).contains(&byte_cap),
            "trace byte_cap exceeds settings or is zero"
        );
        let rows = o.layers.len() * o.positions.len();
        ensure!(
            !o.positions.is_empty()
                && o.positions.iter().all(|&p| p < MAX_POSITIONS)
                && o.layers.iter().all(|&l| l < 4096)
                && o.top_k <= 100
                && (!o.layers.is_empty() || o.top_k > 0)
                && (o.roles.is_empty() || !o.layers.is_empty())
                && (!o.roles.is_empty() || o.top_k > 0),
            "invalid trace layers, roles, positions, or top_k"
        );
        ensure!(
            o.full_dump
                || (o.layers.len() <= 64
                    && o.positions.len() <= 128
                    && rows * o.roles.len().max(1) <= 2048),
            "large trace selections require full_dump"
        );
        let entries = rows * (o.roles.len() + usize::from(o.top_k > 0))
            + if o.top_k > 0 { o.positions.len() } else { 0 };
        ensure!(
            (entries as u64) * (prompts.len() as u64) <= byte_cap,
            "trace selection cannot fit byte_cap"
        );
        Rule::check_all(&o.interventions)?;
        Ok(Self {
            prompts,
            capture: Capture {
                layers: o.layers,
                roles: o.roles,
                positions: o.positions,
                top_k: o.top_k,
                byte_cap,
                all_logits: false,
            },
            interventions: o.interventions,
            retain_jobs: policy.retain_jobs,
        })
    }

    pub fn worker(options: &Value, prompt: Option<&str>) -> Result<Self> {
        let mut options = options
            .as_object()
            .context("invalid trace options")?
            .clone();
        let retain_jobs = serde_json::from_value(
            options
                .remove("retain_jobs")
                .context("missing trace retention")?,
        )?;
        if let Some(expected) = options.remove("expected_model_sha256") {
            ensure!(
                expected.as_str().is_some_and(crate::is_digest),
                "invalid expected trace model SHA-256"
            );
        }
        ensure!(
            options.get("byte_cap").is_some_and(Value::is_u64),
            "missing trace byte cap"
        );
        Self::controller(
            &Value::Object(options),
            prompt,
            Policy {
                max_byte_cap: HARD_BYTE_CAP,
                retain_jobs,
                ..Policy::default()
            },
        )
    }
}

pub fn check_model_revision(options: &Value, actual: &str) -> Result<()> {
    ensure!(
        options
            .get("expected_model_sha256")
            .is_none_or(|expected| expected.as_str() == Some(actual)),
        "trace model SHA-256 mismatch"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub struct Key {
    pub layer: usize,
    pub role: Role,
    pub position: usize,
}

#[derive(Deserialize)]
struct Tensor {
    #[serde(flatten)]
    key: Key,
    node: String,
    file: String,
    shape: Vec<u64>,
    dtype: String,
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: u64,
    kind: String,
    model: Value,
    spec: Capture,
    tokens: Vec<Token>,
    tensors: Vec<Tensor>,
    missing: Vec<Key>,
    interventions: Vec<Applied>,
    model_logits: Vec<Logits>,
    logit_lens: Vec<Logits>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Applied {
    #[serde(flatten)]
    key: Key,
    #[serde(skip_serializing_if = "Option::is_none")]
    expert: Option<usize>,
}
#[derive(Deserialize, Serialize, PartialEq)]
struct Token {
    position: usize,
    token_id: i32,
}

#[derive(Deserialize, Serialize)]
struct TopLogit {
    token_id: i32,
    logit: f32,
}

#[derive(Deserialize, Serialize)]
struct Logits {
    #[serde(skip_serializing_if = "Option::is_none")]
    layer: Option<usize>,
    position: usize,
    top: Vec<TopLogit>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Export {
    pub bytes: u64,
    pub manifest_sha256: String,
}

fn open_file(path: &Path, cap: u64) -> Result<(File, u64)> {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let meta = f.metadata()?;
    ensure!(
        meta.is_file() && meta.nlink() == 1 && meta.len() <= cap,
        "invalid trace file type or size"
    );
    Ok((f, meta.len()))
}

fn read_file(path: &Path, cap: u64) -> Result<Vec<u8>> {
    let (f, len) = open_file(path, cap)?;
    let mut data = Vec::with_capacity(len as usize);
    f.take(len + 1).read_to_end(&mut data)?;
    ensure!(data.len() as u64 == len, "trace file size changed");
    Ok(data)
}

fn manifest(dir: &Path, cap: u64) -> Result<(Manifest, Export)> {
    ensure!(
        fs::symlink_metadata(dir)?.is_dir(),
        "invalid trace directory"
    );
    let data = read_file(&dir.join("manifest.json"), cap)?;
    let manifest: Manifest = serde_json::from_slice(&data).context("invalid trace manifest")?;
    ensure!(
        manifest.schema_version == 1 && manifest.kind == JOB,
        "unsupported trace manifest schema"
    );
    let fingerprint = Export {
        bytes: data.len() as u64,
        manifest_sha256: format!("{:x}", Sha256::digest(&data)),
    };
    Ok((manifest, fingerprint))
}

pub fn verify(dir: &Path, spec: &Capture) -> Result<Export> {
    Ok(inspect(dir, spec)?.1)
}

fn inspect(dir: &Path, spec: &Capture) -> Result<(Manifest, Export)> {
    let (m, mut export) = manifest(dir, spec.byte_cap)?;
    ensure!(
        &m.spec == spec
            && m.model.is_object()
            && (1..=MAX_POSITIONS).contains(&m.tokens.len())
            && m.tokens
                .iter()
                .enumerate()
                .all(|(p, token)| p == token.position && token.token_id >= 0)
            && spec.positions.iter().all(|&p| p < m.tokens.len()),
        "trace manifest does not match capture spec"
    );
    for (rows, lens) in [(&m.model_logits, false), (&m.logit_lens, true)] {
        let mut seen = BTreeSet::new();
        let valid = |row: &Logits| -> bool {
            spec.positions.contains(&row.position)
                && (if lens {
                    row.layer.is_some_and(|l| spec.layers.contains(&l))
                } else {
                    row.layer.is_none()
                })
                && (if spec.all_logits {
                    !lens && (2..=262144).contains(&row.top.len())
                        && row.top.iter().enumerate().all(|(i, x)| x.token_id == i as i32 && x.logit.is_finite())
                } else {
                    seen_row_top(&row.top, spec.top_k)
                })
        };
        ensure!(
            rows.iter()
                .all(|row| valid(row) && seen.insert((row.layer, row.position)))
                && (spec.top_k != 0 || spec.all_logits || rows.is_empty())
                && (lens || (spec.top_k == 0 && !spec.all_logits) || rows.len() == spec.positions.len()),
            "invalid trace logits"
        );
    }
    let mut files = BTreeSet::from(["manifest.json"]);
    let mut rows = BTreeSet::new();
    let valid_key = |k: &Key| {
        spec.layers.contains(&k.layer)
            && spec.roles.contains(&k.role)
            && spec.positions.contains(&k.position)
    };
    for t in &m.tensors {
        let leaf = t
            .file
            .strip_prefix("tensor-")
            .and_then(|s| s.strip_suffix(".f32"));
        let shape_bytes = t.shape.iter().try_fold(4u64, |n, &d| n.checked_mul(d));
        ensure!(
            leaf.is_some_and(|s| s.len() == 6 && s.bytes().all(|b| b.is_ascii_digit()))
                && files.insert(t.file.as_str())
                && rows.insert(t.key)
                && valid_key(&t.key)
                && t.dtype == "f32_le"
                && (1..=2).contains(&t.shape.len())
                && t.bytes > 0
                && Some(t.bytes) == shape_bytes
                && crate::is_digest(&t.sha256),
            "invalid trace tensor descriptor"
        );
        ensure!(
            t.bytes <= spec.byte_cap - export.bytes,
            "trace export exceeds byte cap"
        );
        let (mut f, len) = open_file(&dir.join(&t.file), t.bytes)?;
        let mut sha = Sha256::new();
        let read = std::io::copy(&mut (&mut f).take(t.bytes + 1), &mut sha)?;
        ensure!(
            len == t.bytes && read == len && format!("{:x}", sha.finalize()) == t.sha256,
            "trace tensor size or SHA-256 mismatch"
        );
        export.bytes += t.bytes;
    }
    ensure!(
        m.missing.iter().all(|k| valid_key(k) && rows.insert(*k))
            && rows.len() == spec.layers.len() * spec.roles.len() * spec.positions.len(),
        "incomplete trace tensor inventory"
    );
    for entry in fs::read_dir(dir)? {
        ensure!(
            entry?.file_name().to_str().is_some_and(|s| files.remove(s)),
            "unexpected trace payload"
        );
    }
    ensure!(files.is_empty(), "missing trace payload");
    Ok((m, export))
}

pub struct CircuitTrace {
    pub activations: BTreeMap<Key, Vec<f32>>,
    pub metric: Option<f64>,
}

pub fn circuit_trace(
    dir: &Path,
    spec: &Capture,
    tokens: &[i32],
    width: usize,
    targets: (i32, Option<i32>),
    patch: Option<Key>,
) -> Result<CircuitTrace> {
    let (m, _) = inspect(dir, spec)?;
    ensure!(
        m.tokens
            .iter()
            .map(|t| t.token_id)
            .eq(tokens.iter().copied())
            && m.missing.is_empty()
            && m.interventions
                .iter()
                .map(|applied| (applied.key, applied.expert))
                .eq(patch.map(|key| (key, None)))
            && m.logit_lens.is_empty(),
        "circuit trace has missing rows, wrong tokens or wrong interventions"
    );
    let mut activations = BTreeMap::new();
    for tensor in m.tensors {
        ensure!(
            tensor.shape == [width as u64],
            "invalid circuit activation width"
        );
        let bytes = read_file(&dir.join(tensor.file), tensor.bytes)?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes)) == tensor.sha256,
            "circuit activation SHA-256 mismatch"
        );
        let values: Vec<_> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite circuit activation"
        );
        activations.insert(tensor.key, values);
    }
    let metric = if spec.top_k == 0 {
        None
    } else {
        let [row] = m.model_logits.as_slice() else {
            anyhow::bail!("expected one circuit logit row");
        };
        let mut seen = vec![false; spec.top_k];
        for s in &row.top {
            ensure!(
                seen.get_mut(s.token_id as usize)
                    .is_some_and(|v| !std::mem::replace(v, true)),
                "invalid circuit vocabulary logits"
            );
        }
        let logit = |id| {
            row.top
                .iter()
                .find(|s| s.token_id == id)
                .map(|s| f64::from(s.logit))
                .context("circuit objective token missing")
        };
        Some(logit(targets.0)? - targets.1.map(logit).transpose()?.unwrap_or(0.))
    };
    Ok(CircuitTrace {
        activations,
        metric,
    })
}

/// Read a new local export for W5 before it is published as a trace job.
pub fn verified_diff_input(dir: &Path, spec: &Capture) -> Result<Value> {
    let export = verify(dir, spec)?;
    let (m, seen) = manifest(dir, export.bytes)?;
    ensure!(
        seen.manifest_sha256 == export.manifest_sha256,
        "trace manifest changed"
    );
    let slice = Slice {
        view: View::DiffInput,
        prompt: 0,
        layer: None,
        role: None,
        start: None,
        end: None,
    };
    match diff::input(dir, m, &slice)? {
        ReadReply::Ok(value) => Ok(value),
        _ => bail!("verified trace exceeds the W5 diff input cap"),
    }
}

pub fn verify_interventions(dir: &Path, export: &Export, rules: &[Rule]) -> Result<()> {
    let (m, fingerprint) = manifest(dir, export.bytes)?;
    ensure!(
        fingerprint.manifest_sha256 == export.manifest_sha256,
        "trace manifest changed"
    );
    let mut expected = BTreeSet::new();
    for rule in rules {
        let expert = match rule.op {
            Operation::ZeroExpert { expert } => Some(expert),
            _ => None,
        };
        for &position in &rule.positions {
            expected.insert((
                Key {
                    layer: rule.layer,
                    role: rule.role,
                    position,
                },
                expert,
            ));
        }
    }
    for applied in m.interventions {
        ensure!(
            expected.remove(&(applied.key, applied.expert)),
            "unexpected trace intervention"
        );
    }
    ensure!(expected.is_empty(), "missing trace intervention");
    Ok(())
}

pub fn direction_probe() -> Capture {
    Capture {
        layers: BTreeSet::new(),
        roles: BTreeSet::new(),
        positions: BTreeSet::from([0]),
        top_k: 0,
        byte_cap: HARD_BYTE_CAP,
        all_logits: false,
    }
}

pub fn direction_capture(layers: &[usize], role: Role, position: usize) -> Capture {
    Capture {
        layers: layers.iter().copied().collect(),
        roles: BTreeSet::from([role, Role::LOut]),
        positions: BTreeSet::from([position]),
        top_k: 0,
        byte_cap: HARD_BYTE_CAP,
        all_logits: false,
    }
}

pub fn ablation_logits(position: usize) -> Capture {
    Capture {
        layers: BTreeSet::new(),
        roles: BTreeSet::new(),
        positions: BTreeSet::from([position]),
        top_k: 0,
        byte_cap: crate::ablation_search::EXPORT_CAP,
        all_logits: true,
    }
}

pub fn ablation_rows(dir: &Path, spec: &Capture, verified: &Export, rules: &[crate::weight_source::scratch::ablation::TraceRule]) -> Result<(Vec<i32>, Vec<f32>)> {
    let (m, found) = manifest(dir, spec.byte_cap)?;
    let expected: BTreeSet<_> = rules.iter().flat_map(|rule| rule.positions.iter().map(|&position| Key {
        layer: rule.layer, role: rule.role, position,
    })).collect();
    ensure!(found.manifest_sha256 == verified.manifest_sha256
        && m.model["arch"] == "qwen2"
        && m.interventions.len() == expected.len()
        && m.interventions.iter().all(|applied| applied.expert.is_none())
        && m.interventions.iter().map(|applied| applied.key).collect::<BTreeSet<_>>() == expected,
        "screen trace did not apply exactly the requested rules");
    let logits = if spec.all_logits {
        m.model_logits.into_iter().next().context("missing screen logits")?
            .top.into_iter().map(|score| score.logit).collect()
    } else {
        Vec::new()
    };
    Ok((m.tokens.into_iter().map(|token| token.token_id).collect(), logits))
}

/// Read the directions' rows and residual norms from a verified CPU trace. Keep token IDs in
/// memory to cross-check the probe and the activation run.
pub fn direction_rows(
    dir: &Path,
    spec: &Capture,
    verified: &Export,
    role: Role,
) -> Result<(Vec<i32>, Vec<crate::behavior_direction::Observation>)> {
    use crate::behavior_direction::{Observation, residual_norm};
    let (m, check) = manifest(dir, spec.byte_cap)?;
    ensure!(
        check.manifest_sha256 == verified.manifest_sha256
            && m.model["arch"] == "qwen2"
            && m.interventions.is_empty()
            && m.missing.is_empty(),
        "direction needs complete, unedited Qwen2 trace rows"
    );
    let width = m.model["hidden"]
        .as_u64()
        .context("missing Qwen2 hidden width")?;
    let mut tensors: std::collections::BTreeMap<_, _> = m
        .tensors
        .into_iter()
        .map(|t| ((t.key.layer, t.key.role), t))
        .collect();
    let mut read = |layer, role| -> Result<Vec<f32>> {
        let t = tensors
            .remove(&(layer, role))
            .context("missing direction tensor")?;
        ensure!(
            t.shape == [width] && (1..=16384).contains(&width),
            "invalid direction tensor shape"
        );
        let data = read_file(&dir.join(t.file), t.bytes)?;
        ensure!(
            data.len() as u64 == t.bytes && format!("{:x}", Sha256::digest(&data)) == t.sha256,
            "direction tensor size or SHA-256 mismatch"
        );
        Ok(data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect())
    };
    let rows = spec
        .layers
        .iter()
        .map(|&layer| {
            let activation = read(layer, role)?;
            if role == Role::LOut {
                Ok(Observation::residual(activation))
            } else {
                Ok(Observation {
                    activation,
                    residual_norm: residual_norm(&read(layer, Role::LOut)?),
                })
            }
        })
        .collect::<Result<_>>()?;
    Ok((m.tokens.into_iter().map(|t| t.token_id).collect(), rows))
}

fn seen_row_top(top: &[TopLogit], k: usize) -> bool {
    top.len() == k
        && top
            .iter()
            .all(|score| score.token_id >= 0 && score.logit.is_finite())
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum View {
    Manifest,
    Tensor,
    LogitLens,
    ModelLogits,
    /// Internal: joined, bounded slice used by controller trace diffs.
    DiffInput,
    /// Internal: validated rows for one routing atlas layer.
    RoutingInput,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Slice {
    pub view: View,
    #[serde(default)]
    pub prompt: usize,
    pub layer: Option<usize>,
    pub role: Option<Role>,
    pub start: Option<usize>,
    pub end: Option<usize>,
}

impl Slice {
    pub fn check(&self) -> Result<()> {
        let start = self.start.unwrap_or(0);
        let end = self.end.unwrap_or(MAX_POSITIONS);
        ensure!(
            self.prompt < 4
                && start < end
                && end <= MAX_POSITIONS
                && self.layer.is_none_or(|l| l < 4096),
            "invalid trace slice"
        );
        let valid = match self.view {
            View::Manifest => {
                self.layer.is_none()
                    && self.role.is_none()
                    && self.start.is_none()
                    && self.end.is_none()
            }
            View::Tensor => self.layer.is_some() && self.role.is_some(),
            View::LogitLens | View::DiffInput => self.role.is_none(),
            View::RoutingInput => {
                self.layer.is_some()
                    && self.role.is_none()
                    && self.start.is_some()
                    && self.end.is_some()
                    && end - start <= routing::MAX_ROWS
            }
            View::ModelLogits => self.layer.is_none() && self.role.is_none(),
        };
        ensure!(valid, "invalid query for trace view");
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "status", content = "data", rename_all = "snake_case")]
pub enum ReadReply {
    Ok(Value),
    Missing,
    TooLarge,
}

pub fn read(root: &Path, slice: &Slice, result: &Value) -> Result<ReadReply> {
    slice.check()?;
    let dir = root.join(slice.prompt.to_string());
    let Some(e) = result["prompts"]
        .as_array()
        .and_then(|v| v.get(slice.prompt))
    else {
        return Ok(ReadReply::Missing);
    };
    let expected: Export = serde_json::from_value(e.clone())?;
    ensure!(
        expected.bytes <= HARD_BYTE_CAP && fs::symlink_metadata(root)?.is_dir(),
        "invalid trace result"
    );
    let (m, export) = manifest(&dir, expected.bytes)?;
    ensure!(
        expected.manifest_sha256 == export.manifest_sha256,
        "trace manifest SHA-256 mismatch"
    );
    let in_range =
        |p: usize| (slice.start.unwrap_or(0)..slice.end.unwrap_or(MAX_POSITIONS)).contains(&p);
    let data = match slice.view {
        View::DiffInput => return diff::input(&dir, m, slice),
        View::RoutingInput => return routing::input(&dir, m, slice, result, &expected),
        View::Manifest => {
            let mut model = m.model;
            model
                .as_object_mut()
                .context("invalid trace model")?
                .remove("path");
            let mut summary = json!({"schema_version":1, "prompt":slice.prompt,"model":model,"model_id":result["model_id"],
                "model_sha256":result["model_sha256"],
                "bytes":expected.bytes,"manifest_sha256":expected.manifest_sha256,
                "spec":m.spec,"tokens":m.tokens,"tensor_count":m.tensors.len(),
                "missing_count":m.missing.len(),"intervention_count":m.interventions.len(),
                "logit_lens_count":m.logit_lens.len(),"model_logits_count":m.model_logits.len()});
            if result["runtime"] == HGN_RUNTIME {
                summary["runtime"] = json!(HGN_RUNTIME);
                summary["runtime_sha256"] = result["runtime_sha256"].clone();
            } else {
                summary["lanternfish_commit"] = json!(COMMIT);
            }
            summary
        }
        View::ModelLogits | View::LogitLens => {
            let rows = match slice.view {
                View::ModelLogits => m.model_logits,
                _ => m.logit_lens,
            };
            let mut size = 0;
            let mut selected = Vec::new();
            for row in rows.into_iter().filter(|row| {
                in_range(row.position) && slice.layer.is_none_or(|l| row.layer == Some(l))
            }) {
                size += serde_json::to_vec(&row)?.len() + 1;
                if size > RESPONSE_CAP {
                    return Ok(ReadReply::TooLarge);
                }
                selected.push(row);
            }
            json!({"prompt":slice.prompt,"rows":selected})
        }
        View::Tensor => {
            let mut rows = Vec::new();
            let mut size = 0;
            let selected = |k: &Key| {
                Some(k.layer) == slice.layer && Some(k.role) == slice.role && in_range(k.position)
            };
            for t in m.tensors.into_iter().filter(|t| selected(&t.key)) {
                size += (t.bytes as usize).div_ceil(3) * 4 + 256 + t.node.len();
                if size > RESPONSE_CAP {
                    return Ok(ReadReply::TooLarge);
                }
                let data = read_file(&dir.join(&t.file), t.bytes)?;
                ensure!(
                    data.len() as u64 == t.bytes
                        && format!("{:x}", Sha256::digest(&data)) == t.sha256,
                    "trace tensor size or SHA-256 mismatch"
                );
                rows.push(
                    json!({"position":t.key.position,"shape":t.shape,"node":t.node,
                    "sha256":t.sha256,"data_base64":STANDARD.encode(data)}),
                );
            }
            let missing: Vec<_> = m
                .missing
                .iter()
                .filter(|k| selected(k))
                .map(|k| k.position)
                .collect();
            if rows.is_empty() && missing.is_empty() {
                return Ok(ReadReply::Missing);
            }
            json!({"prompt":slice.prompt,"layer":slice.layer,"role":slice.role,"dtype":"f32_le","rows":rows,"missing_positions":missing})
        }
    };
    if serde_json::to_vec(&data)?.len() > RESPONSE_CAP {
        return Ok(ReadReply::TooLarge);
    }
    Ok(ReadReply::Ok(data))
}
