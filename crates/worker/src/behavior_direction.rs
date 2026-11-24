//! Fit directions in activation space and score them on disjoint prompt pairs.
use crate::trace::{self, Role};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet, fs::OpenOptions, io::Read, os::unix::fs::OpenOptionsExt, path::Path,
};

pub const JOB: &str = "behavior_direction";
pub const VERSION: u32 = 2;
pub const CACHE: &str = "BREWERY_CONTRAST_CACHE";
pub const CAP: u64 = 2 << 20;
pub const MAX_LAST_K: usize = 16;
const FLOAT_CAP: usize = 65536;
const SCHEMA: &str = "brewery.contrast-prompts.v1";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Behavior {
    Slop,
    Sycophancy,
    ResponseLength,
}

impl Behavior {
    pub fn contrast_sha256(self) -> &'static str {
        match self {
            Self::Slop => "c844b9b32e5547473b0b30268d99dde456657a5c3d5f98d4e92bbeffd4ac7987",
            Self::Sycophancy => "3b255d8b4a6fceeaa8311202fe35f1ff3cb6cf85d5453c2bdffa576510d21021",
            Self::ResponseLength => {
                "f42e2aac3d2c0faad8d0f0e5e2a2c2a5c8580bb1f51ece3e6a30b26eac3c24b6"
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub behavior: Behavior,
    pub model_sha256: String,
    pub layers: Vec<usize>,
    #[serde(default = "l_out")]
    pub role: Role,
    pub position: Option<usize>,
    pub last_k: Option<usize>,
}

fn l_out() -> Role {
    Role::LOut
}

impl Request {
    pub fn parse(value: &Value) -> Result<Self> {
        let r: Self = serde_json::from_value(value.clone()).context("invalid direction options")?;
        ensure!(
            crate::is_digest(&r.model_sha256)
                && valid_selection(&r.layers, r.role)
                && *r.layers.last().unwrap() < 4096
                && r.position.is_none_or(|p| p < 4096)
                && r.last_k
                    .is_none_or(|k| (1..=MAX_LAST_K).contains(&k) && r.position.is_none()),
            "invalid direction revision, layer selection or token position"
        );
        Ok(r)
    }

    pub fn load_pairs(&self, cache: &Path) -> Result<Vec<Pair>> {
        load_pairs(cache, self.behavior, self.behavior.contrast_sha256())
    }

    pub fn check_input_kind(&self, kind: &str) -> Result<()> {
        match kind {
            "gguf" => ensure!(self.last_k.is_none(), "last_k requires HGN"),
            kind if trace::is_hgn(kind) => ensure!(
                self.role == Role::LOut && self.position.is_none_or(|pos| pos < 512),
                "HGN directions require l_out and a position below 512"
            ),
            _ => anyhow::bail!("direction needs a Library GGUF or HGN checkpoint"),
        }
        Ok(())
    }
}

pub fn load_pairs(cache: &Path, behavior: Behavior, pin: &str) -> Result<Vec<Pair>> {
    ensure!(crate::is_digest(pin), "invalid prompt set digest");
    let path = cache.join(format!("{pin}.json"));
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .context("approved direction prompt cache is missing")?;
    ensure!(f.metadata()?.is_file(), "invalid prompt cache file type");
    let mut bytes = Vec::new();
    f.take(CAP + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= CAP && format!("{:x}", Sha256::digest(&bytes)) == pin,
        "direction cache size or SHA-256 mismatch"
    );
    let set: PromptSet = serde_json::from_slice(&bytes).context("invalid direction cache JSON")?;
    let mut ids = BTreeSet::new();
    ensure!(
        set.schema == SCHEMA
            && set.behavior == behavior
            && (2..=64).contains(&set.rows.len())
            && set.rows.iter().all(|row| {
                crate::is_digest(&row.id)
                    && ids.insert(&row.id)
                    && row.positive.system != row.negative.system
                    && row.positive.user == row.negative.user
                    && row.positive.valid()
                    && row.negative.valid()
                    && match behavior {
                        Behavior::Sycophancy => {
                            matches!(row.expected_agreement.as_deref(), Some("(A)" | "(B)"))
                        }
                        _ => row.expected_agreement.is_none(),
                    }
            }),
        "invalid direction cache contents"
    );
    Ok(set.rows)
}

impl Request {
    pub fn verify_result(
        &self,
        result: &Value,
        model_id: &str,
        sha: &str,
        kind: &str,
    ) -> Result<()> {
        ensure!(
            current_runtime(result, kind),
            "direction result does not match the current runtime pin"
        );
        self.verify_version(result, model_id, sha, kind, VERSION)
    }

    pub fn verify_version(
        &self,
        result: &Value,
        model_id: &str,
        sha: &str,
        kind: &str,
        version: u32,
    ) -> Result<()> {
        self.check_input_kind(kind)?;
        let hgn = trace::is_hgn(kind);
        ensure!(
            matches!(version, 1 | VERSION)
                && !model_id.is_empty()
                && result["model_id"] == model_id
                && crate::is_digest(sha)
                && sha == self.model_sha256
                && result["model_sha256"] == sha
                && result["kind"] == JOB
                && result["schema_version"] == version
                && result["release_qualified"] == false
                && result["behavior"] == serde_json::to_value(self.behavior)?
                && result["contrast_sha256"] == self.behavior.contrast_sha256()
                && result["position"] == serde_json::to_value(self.position)?
                && (if hgn && self.position.is_none() {
                    result["last_k"] == self.last_k.unwrap_or(1)
                } else {
                    result.get("last_k").is_none()
                })
                && (current_runtime(result, kind) || (!hgn && original_runtime(result)))
                && (!hgn || version == VERSION),
            "direction result does not match the model, prompt or runtime pins"
        );
        if version == VERSION {
            ensure!(
                result["fit"]["layers"]
                    .as_array()
                    .is_some_and(|layers| layers
                        .iter()
                        .all(|row| row.get("held_out_cohens_d").is_some()
                            && row["mean_residual_norm"].is_number())),
                "missing direction separation or residual norm"
            );
        }
        let fit: Fit = serde_json::from_value(result["fit"].clone())?;
        fit.verify(self, version)
    }
}

fn current_runtime(result: &Value, kind: &str) -> bool {
    if trace::is_hgn(kind) {
        result.get("lanternfish_commit").is_none()
            && result["runtime"] == trace::HGN_RUNTIME
            && result["runtime_sha256"] == trace::HGN_RUNTIME_SHA256.trim_end()
            && result["executable_sha256"]
                .as_str()
                .is_some_and(crate::is_digest)
            && result["assets_sha256"].as_object().is_some_and(|assets| {
                assets.len() == 3
                    && ["sidecar", "config", "tokenizer"].into_iter().all(|name| {
                        assets
                            .get(name)
                            .and_then(Value::as_str)
                            .is_some_and(crate::is_digest)
                    })
            })
    } else {
        result.get("runtime").is_none()
            && result["lanternfish_commit"] == trace::COMMIT
            && result["executable_sha256"] == trace::SHA256
    }
}

// Older direction jobs used the projection runtime even without an intervention.
// Keep those receipts readable without accepting that pin for a new fit.
fn original_runtime(result: &Value) -> bool {
    result["lanternfish_commit"] == trace::PROJECTION_COMMIT
        && result["executable_sha256"] == trace::PROJECTION_SHA256
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptSet {
    schema: String,
    behavior: Behavior,
    rows: Vec<Pair>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub id: String,
    pub split: Split,
    pub positive: Prompt,
    pub negative: Prompt,
    pub expected_agreement: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prompt {
    system: String,
    user: String,
}

impl Prompt {
    fn valid(&self) -> bool {
        [&self.system, &self.user]
            .into_iter()
            .all(|s| !s.is_empty() && s.len() <= 3072 && !s.contains("<|") && !s.contains('\0'))
    }

    /// The causal prefix ends at the last user content token, not at a template
    /// delimiter. Trace only that prefix; later delimiters cannot change it.
    pub fn qwen_chat_prefix(&self) -> String {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}",
            self.system, self.user
        )
    }

    pub fn hgn_chat_prefix(&self) -> String {
        // Text-only chat template with enable_thinking=false and add_generation_prompt=true.
        self.qwen_chat_prefix() + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    pub fn user(&self) -> &str {
        &self.user
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    HeldOut,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub layer: usize,
    pub norm: f64,
    pub unit_vector: Option<Vec<f32>>,
    pub held_out_margin: Option<f64>,
    pub held_out_cohens_d: Option<f64>,
    pub mean_residual_norm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_mean: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fit {
    pub role: Role,
    pub train_pairs: usize,
    pub held_out_pairs: usize,
    pub selected_layer: Option<usize>,
    pub layers: Vec<Layer>,
}

impl Fit {
    fn verify(&self, request: &Request, version: u32) -> Result<()> {
        ensure!(
            self.role == request.role
                && self.train_pairs == 12
                && self.held_out_pairs == 8
                && self
                    .layers
                    .iter()
                    .map(|row| row.layer)
                    .eq(request.layers.iter().copied()),
            "invalid direction result selection or inventory"
        );
        let mut width = None;
        let mut best = (0.0, None);
        for row in &self.layers {
            if version == VERSION {
                ensure!(
                    row.mean_residual_norm
                        .is_some_and(|n| n.is_finite() && n >= 0.0)
                        && row.held_out_cohens_d.is_none_or(f64::is_finite),
                    "invalid direction separation or residual norm"
                );
            }
            if let Some(mean) = &row.negative_mean {
                ensure!(
                    version == VERSION
                        && (1..=16384).contains(&mean.len())
                        && mean.len() * self.layers.len() <= FLOAT_CAP
                        && mean.len() == *width.get_or_insert(mean.len())
                        && mean.iter().all(|x| x.is_finite()),
                    "invalid mean negative activation"
                );
            }
            match (&row.unit_vector, row.held_out_margin) {
                (None, None) if row.norm == 0.0 && row.held_out_cohens_d.is_none() => {}
                (Some(v), Some(margin)) => {
                    ensure!(
                        row.norm.is_finite()
                            && row.norm > 0.0
                            && margin.is_finite()
                            && (1..=16384).contains(&v.len())
                            && v.len() * self.layers.len() <= FLOAT_CAP
                            && v.len() == *width.get_or_insert(v.len())
                            && (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() - 1.0).abs()
                                <= 1e-5,
                        "invalid direction vector, norm or margin"
                    );
                    let score = if version == 1 {
                        Some(margin)
                    } else {
                        row.held_out_cohens_d
                    };
                    if let Some(score) = score.filter(|&score| score > best.0) {
                        best = (score, Some(row.layer));
                    }
                }
                _ => anyhow::bail!("inconsistent zero direction"),
            }
        }
        ensure!(
            self.selected_layer == best.1,
            "incorrect direction layer selection"
        );
        Ok(())
    }
}

fn valid_selection(layers: &[usize], role: Role) -> bool {
    (1..=64).contains(&layers.len())
        && layers.windows(2).all(|l| l[0] < l[1])
        && matches!(role, Role::LOut | Role::AttnOut | Role::FfnOut)
}

pub struct Observation {
    pub activation: Vec<f32>,
    pub residual_norm: f64,
}

pub fn residual_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt()
}

impl Observation {
    pub fn residual(activation: Vec<f32>) -> Self {
        let residual_norm = residual_norm(&activation);
        Self {
            activation,
            residual_norm,
        }
    }
}

/// Fit on training pairs, then select by the largest positive held-out Cohen's d.
/// An undefined score cannot win; equal scores select the lower layer.
pub fn compute(
    layers: &[usize],
    role: Role,
    splits: &[Split],
    mut trace: impl FnMut(usize, bool) -> Result<Vec<Observation>>,
) -> Result<Fit> {
    let train_pairs = splits.iter().filter(|&&p| p == Split::Train).count();
    let held_out_pairs = splits.len() - train_pairs;
    ensure!(
        train_pairs > 0 && held_out_pairs > 0,
        "direction needs train and held-out pairs"
    );
    ensure!(
        valid_selection(layers, role),
        "invalid direction layers or role"
    );
    let mut widths = vec![0; layers.len()];
    let mut checked_trace = |row, positive| {
        let observations = trace(row, positive)?;
        ensure!(
            observations.len() == layers.len(),
            "incomplete direction activation inventory"
        );
        for (obs, width) in observations.iter().zip(&mut widths) {
            let n = obs.activation.len();
            ensure!(
                (1..=16384).contains(&n)
                    && n * layers.len() <= FLOAT_CAP
                    && (*width == 0 || *width == n),
                "invalid direction activation width"
            );
            ensure!(
                obs.activation.iter().all(|v| v.is_finite())
                    && obs.residual_norm.is_finite()
                    && obs.residual_norm >= 0.0,
                "invalid direction activation or residual norm"
            );
            *width = n;
        }
        Ok::<_, anyhow::Error>(observations)
    };

    let mut train = vec![Vec::<f64>::new(); layers.len()];
    let mut negative_sums = train.clone();
    for (row, _) in splits
        .iter()
        .enumerate()
        .filter(|&(_, p)| *p == Split::Train)
    {
        let positive = checked_trace(row, true)?;
        let negative = checked_trace(row, false)?;
        for (((sum, negative_sum), p), n) in train
            .iter_mut()
            .zip(&mut negative_sums)
            .zip(positive)
            .zip(negative)
        {
            sum.resize(p.activation.len(), 0.0);
            negative_sum.resize(n.activation.len(), 0.0);
            for (((sum, negative_sum), p), n) in sum
                .iter_mut()
                .zip(negative_sum)
                .zip(p.activation)
                .zip(n.activation)
            {
                *sum += f64::from(p) - f64::from(n);
                *negative_sum += f64::from(n);
            }
        }
    }

    let mut fitted: Vec<_> = train
        .into_iter()
        .map(|mut mean| {
            for v in &mut mean {
                *v /= train_pairs as f64;
            }
            let norm = mean.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let unit = (norm > 0.0).then(|| mean.into_iter().map(|x| x / norm).collect::<Vec<_>>());
            (norm, unit, [Vec::<f64>::new(), Vec::new()], 0.0)
        })
        .collect();

    for (row, _) in splits
        .iter()
        .enumerate()
        .filter(|&(_, p)| *p == Split::HeldOut)
    {
        for (class, positive) in [true, false].into_iter().enumerate() {
            for ((_, unit, projections, residual_sum), obs) in
                fitted.iter_mut().zip(checked_trace(row, positive)?)
            {
                *residual_sum += obs.residual_norm;
                if let Some(unit) = unit {
                    projections[class].push(
                        unit.iter()
                            .zip(obs.activation)
                            .map(|(u, x)| u * f64::from(x))
                            .sum(),
                    );
                }
            }
        }
    }

    let mut best = (0.0, None);
    let layers = layers
        .iter()
        .zip(fitted)
        .zip(negative_sums)
        .map(|((&layer, stats), negative_sum)| {
            let (norm, unit, projections, residual_sum) = stats;
            let n = held_out_pairs as f64;
            let held_out_margin = unit.as_ref().map(|_| {
                projections[0]
                    .iter()
                    .zip(&projections[1])
                    .map(|(p, q)| p - q)
                    .sum::<f64>()
                    / n
            });
            let held_out_cohens_d = held_out_margin.and_then(|margin| {
                if held_out_pairs < 2 {
                    return None;
                }
                let within_class_ss = projections
                    .iter()
                    .map(|values| {
                        let mean = values.iter().sum::<f64>() / n;
                        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    })
                    .sum::<f64>();
                let pooled_sd = (within_class_ss / (2.0 * (n - 1.0))).sqrt();
                (pooled_sd > 0.0)
                    .then(|| margin / pooled_sd)
                    .filter(|d| d.is_finite())
            });
            if let Some(d) = held_out_cohens_d.filter(|&d| d > best.0) {
                best = (d, Some(layer));
            }
            Layer {
                layer,
                norm,
                unit_vector: unit.map(|v| v.into_iter().map(|x| x as f32).collect()),
                held_out_margin,
                held_out_cohens_d,
                mean_residual_norm: Some(residual_sum / (2.0 * n)),
                negative_mean: Some(
                    negative_sum
                        .into_iter()
                        .map(|x| (x / train_pairs as f64) as f32)
                        .collect(),
                ),
            }
        })
        .collect();
    Ok(Fit {
        role,
        train_pairs,
        held_out_pairs,
        selected_layer: best.1,
        layers,
    })
}
