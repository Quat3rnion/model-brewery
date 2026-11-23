//! Bounded schema v1 exports and the read-only trace wire format.
use anyhow::{Context, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
};

pub mod diff;

pub const JOB: &str = "trace";
pub const EXECUTABLE: &str = "BREWERY_LANTERNFISH_TRACE";
pub const LIB_DIR: &str = "BREWERY_LANTERNFISH_TRACE_LIB_DIR";
pub const COMMIT: &str = "0ea30b37729a136a76b8412e7ad1ec91bd7de9b5";
pub const SHA256: &str = "0e4ca38c55a9afa0436eeadd06e89bf295eb201b986dd2f40305f95d3a61e46f";
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
}

impl Role {
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
}

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
}

impl Rule {
    pub fn check_all(rules: &[Self]) -> Result<()> {
        let mut edited = BTreeSet::new();
        ensure!(rules.len() <= 64, "too many trace interventions");
        for rule in rules {
            ensure!(
                rule.layer < 4096
                    && !rule.role.is_read_only()
                    && !rule.positions.is_empty()
                    && rule
                        .positions
                        .iter()
                        .all(|&p| p < MAX_POSITIONS && edited.insert((rule.layer, rule.role, p))),
                "invalid or overlapping trace intervention"
            );
            match &rule.op {
                Operation::Scale { factor } => ensure!(factor.is_finite(), "invalid trace factor"),
                Operation::Add { vector } | Operation::Replace { vector } => ensure!(
                    (1..=16384).contains(&vector.len()) && vector.iter().all(|v| v.is_finite()),
                    "invalid trace intervention vector"
                ),
                Operation::Zero => {}
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

impl Request {
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
struct Key {
    layer: usize,
    role: Role,
    position: usize,
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
    interventions: Vec<Key>,
    model_logits: Vec<Logits>,
    logit_lens: Vec<Logits>,
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
                && seen_row_top(&row.top, spec.top_k)
        };
        ensure!(
            rows.iter()
                .all(|row| valid(row) && seen.insert((row.layer, row.position)))
                && (spec.top_k != 0 || rows.is_empty())
                && (lens || spec.top_k == 0 || rows.len() == spec.positions.len()),
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
    Ok(export)
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
