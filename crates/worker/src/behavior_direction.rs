//! Fit directions in activation space and score them on disjoint prompt pairs.
use crate::trace::{self, Role};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet, fs::OpenOptions, io::Read, os::unix::fs::OpenOptionsExt, path::Path,
};

pub const JOB: &str = "behavior_direction";
pub const VERSION: u32 = 2;
pub const CACHE: &str = "BREWERY_CONTRAST_CACHE";
pub const CAP: u64 = 2 << 20;
pub const MAX_LAST_K: usize = 16;
pub const MAX_REGISTERED_ROWS: usize = 512;
const FLOAT_CAP: usize = 65536;
const SCHEMA: &str = "brewery.contrast-prompts.v1";
const REGISTERED_SCHEMA: &str = "brewery.registered-contrast.v1";

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

/// Reference to an operator-registered evaluation set used as a contrast source.
/// Fitting only ever uses the tune partition; the final gate stays unreachable.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Registered {
    pub name: String,
    #[serde(default)]
    pub sha256: String,
}

impl Registered {
    pub fn check(&self) -> Result<()> {
        ensure!(
            crate::scratch_eval::registered::name_valid(&self.name)
                && crate::is_digest(&self.sha256),
            "invalid registered contrast reference"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub behavior: Option<Behavior>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registered: Option<Registered>,
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
            r.behavior.is_some() != r.registered.is_some()
                && crate::is_digest(&r.model_sha256)
                && valid_selection(&r.layers, r.role)
                && *r.layers.last().unwrap() < 4096
                && r.position.is_none_or(|p| p < 4096)
                && r.last_k
                    .is_none_or(|k| (1..=MAX_LAST_K).contains(&k) && r.position.is_none()),
            "invalid direction revision, contrast source, layer selection or token position"
        );
        if let Some(registered) = &r.registered {
            registered.check()?;
        }
        Ok(r)
    }

    pub fn pin(&self) -> &str {
        match (&self.behavior, &self.registered) {
            (Some(behavior), None) => behavior.contrast_sha256(),
            (None, Some(registered)) => registered.sha256.as_str(),
            _ => "",
        }
    }

    pub fn cache_var(&self) -> &'static str {
        if self.registered.is_some() {
            crate::scratch_eval::registered::CACHE
        } else {
            CACHE
        }
    }

    pub fn load_pairs(&self, cache: &Path) -> Result<Vec<Pair>> {
        match &self.registered {
            Some(registered) => load_registered(cache, &registered.name, &registered.sha256),
            None => {
                let behavior = self.behavior.expect("parse checked the contrast source");
                load_pairs(cache, behavior, behavior.contrast_sha256())
            }
        }
    }

    pub fn check_input_kind(&self, kind: &str) -> Result<()> {
        match kind {
            "gguf" => (),
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
                && result["contrast_sha256"] == self.pin()
                && result["position"] == serde_json::to_value(self.position)?
                && (if hgn && self.position.is_none() {
                    result["last_k"] == self.last_k.unwrap_or(1)
                } else if let Some(k) = self.last_k {
                    result["last_k"] == k
                } else {
                    result.get("last_k").is_none()
                })
                && (current_runtime(result, kind)
                    || if hgn {
                        ORIGINAL_HGN_PINS.iter().any(|pin| hgn_runtime(result, pin))
                    } else {
                        original_runtime(result)
                    })
                && (!hgn || version == VERSION),
            "direction result does not match the model, prompt or runtime pins"
        );
        if let Some(registered) = &self.registered {
            ensure!(
                version == VERSION
                    && result["registered"] == json!({"name":registered.name,"sha256":registered.sha256}),
                "registered contrast fields do not match the request"
            );
        }
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

// These published version 2 receipts remain readable, but new fits must
// use the integrated runtime.
const ORIGINAL_HGN_PINS: &[&str] = &[
    "29c4699cfcf3170fdf5daeced397fc168f7488244ee8f92d24dc39613693b767", // H2
    "e09a61b5ef47004dcb6d95b5de24e6666d46d7896f5cf1a32d777755a6e40b00", // H3
    "8d4b6d5a685130b4603bbad5825f853c5d42b0275442541f4ea39fa327d8c0b3", // H5
    "1ac8e30b17ccd7b96ff84f840c3c6d911cb2cbded57d6a5cfec86217c417aef7", // Integrated H wave
];

fn current_runtime(result: &Value, kind: &str) -> bool {
    if trace::is_hgn(kind) {
        hgn_runtime(result, trace::HGN_RUNTIME_SHA256.trim_end())
    } else {
        result.get("runtime").is_none()
            && result["lanternfish_commit"] == trace::COMMIT
            && result["executable_sha256"] == trace::SHA256
    }
}

fn hgn_runtime(result: &Value, pin: &str) -> bool {
    result.get("lanternfish_commit").is_none()
        && result["runtime"] == trace::HGN_RUNTIME
        && result["runtime_sha256"] == pin
        && result["executable_sha256"]
            .as_str()
            .is_some_and(crate::is_digest)
        && serde_json::from_value::<trace::hgn_projection::AssetPins>(
            result["assets_sha256"].clone(),
        )
        .is_ok_and(|pins| pins.check().is_ok())
}

// Older direction jobs used the projection runtime even without an intervention.
// Keep those receipts readable without accepting that pin for a new fit.
fn original_runtime(result: &Value) -> bool {
    result["lanternfish_commit"] == trace::PROJECTION_COMMIT
        && result["executable_sha256"] == trace::PROJECTION_SHA256
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PromptSet {
    schema: String,
    behavior: Behavior,
    rows: Vec<Pair>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub id: String,
    pub split: Split,
    pub positive: Prompt,
    pub negative: Prompt,
    pub expected_agreement: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
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

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    HeldOut,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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
        let counts = if request.registered.is_some() {
            self.train_pairs >= 1
                && self.held_out_pairs >= 1
                && self.train_pairs + self.held_out_pairs <= MAX_REGISTERED_ROWS
        } else {
            self.train_pairs == 12 && self.held_out_pairs == 8
        };
        ensure!(
            self.role == request.role
                && counts
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

/// Open the content-addressed partition file for a registered set and return
/// its name and rows. Only the tune partition is accepted; a final-gate
/// partition file never reaches fitting.
pub fn load_registered_at(cache: &Path, pin: &str) -> Result<(String, Vec<Pair>)> {
    use crate::scratch_eval::registered::Partition;
    ensure!(crate::is_digest(pin), "invalid registered contrast pin");
    let path = cache.join(format!("{pin}.json"));
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .context("registered evaluation cache is missing the partition file")?;
    ensure!(f.metadata()?.is_file(), "invalid partition file type");
    let mut bytes = Vec::new();
    f.take(CAP + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= CAP && format!("{:x}", Sha256::digest(&bytes)) == pin,
        "registered cache size or SHA-256 mismatch"
    );
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RegisteredFile {
        schema: String,
        name: String,
        partition: Partition,
        rows: Vec<Pair>,
    }
    let file: RegisteredFile =
        serde_json::from_slice(&bytes).context("invalid registered contrast file")?;
    ensure!(
        file.schema == REGISTERED_SCHEMA && file.partition == Partition::Tune,
        "registered contrast file must carry the tune partition"
    );
    ensure!(
        crate::scratch_eval::registered::name_valid(&file.name),
        "invalid registered set name"
    );
    let mut ids = std::collections::BTreeSet::new();
    ensure!(
        (1..=MAX_REGISTERED_ROWS).contains(&file.rows.len())
            && file.rows.iter().all(|row| {
                crate::is_digest(&row.id)
                    && ids.insert(&row.id)
                    && row.positive.valid()
                    && row.negative.valid()
            }),
        "invalid registered contrast rows"
    );
    Ok((file.name, file.rows))
}

pub fn load_registered(cache: &Path, name: &str, pin: &str) -> Result<Vec<Pair>> {
    let (found, rows) = load_registered_at(cache, pin)?;
    ensure!(found == name, "registered set name does not match the pin");
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const PIN: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn rows(n: usize) -> Vec<Pair> {
        (0..n)
            .map(|i| Pair {
                id: format!("{:064x}", i + 1),
                split: if i % 4 == 3 { Split::HeldOut } else { Split::Train },
                positive: Prompt {
                    system: "Follow the instruction.".into(),
                    user: format!("Name plant {}.", i + 1),
                },
                negative: Prompt {
                    system: "Follow the instruction.".into(),
                    user: format!("Name color {}.", i + 1),
                },
                expected_agreement: None,
            })
            .collect()
    }

    fn write_registered_cache(dir: &Path, name: &str, partition: &str, rows: &[Pair]) -> String {
        let body = json!({"schema": REGISTERED_SCHEMA, "name": name, "partition": partition, "rows": rows});
        let bytes = serde_json::to_vec(&body).unwrap();
        let pin = format!("{:x}", Sha256::digest(&bytes));
        fs::write(dir.join(format!("{pin}.json")), &bytes).unwrap();
        pin
    }

    #[test]
    fn parse_accepts_a_registered_contrast_without_a_builtin_behavior() {
        let r = Request::parse(&json!({
            "registered": {"name": "garden", "sha256": PIN},
            "model_sha256": "a".repeat(64), "layers": [0, 6, 12],
        }))
        .unwrap();
        assert!(r.behavior.is_none());
        assert_eq!(r.registered.as_ref().unwrap().name, "garden");
        assert_eq!(r.pin(), PIN);
        assert_eq!(r.cache_var(), crate::scratch_eval::registered::CACHE);
        assert!(Request::parse(&json!({
            "behavior": "slop", "registered": {"name": "garden", "sha256": PIN},
            "model_sha256": "a".repeat(64), "layers": [0],
        }))
        .is_err());
        assert!(Request::parse(&json!({
            "model_sha256": "a".repeat(64), "layers": [0],
        }))
        .is_err());
        assert!(Request::parse(&json!({
            "registered": {"name": "Garden"}, "model_sha256": "a".repeat(64), "layers": [0],
        }))
        .is_err());
        assert!(Request::parse(&json!({
            "registered": {"name": "garden", "sha256": "zz"},
            "model_sha256": "a".repeat(64), "layers": [0],
        }))
        .is_err());
    }

    #[test]
    fn registered_cache_loader_pins_content_tune_partition_and_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let pin = write_registered_cache(dir.path(), "garden", "tune", &rows(3));
        assert_eq!(load_registered(dir.path(), "garden", &pin).unwrap().len(), 3);
        assert!(load_registered(dir.path(), "other", &pin).is_err());
        // 320 tune rows are accepted; a final_gate file never reaches fitting.
        let big = write_registered_cache(dir.path(), "large", "tune", &rows(320));
        assert_eq!(load_registered(dir.path(), "large", &big).unwrap().len(), 320);
        let gate = write_registered_cache(dir.path(), "garden", "final_gate", &rows(2));
        assert!(load_registered(dir.path(), "garden", &gate).is_err());
        let too_many = write_registered_cache(dir.path(), "huge", "tune", &rows(513));
        assert!(load_registered(dir.path(), "huge", &too_many).is_err());
        let with_delim = write_registered_cache(
            dir.path(),
            "bad",
            "tune",
            &[Pair {
                id: "c".repeat(64),
                split: Split::Train,
                positive: Prompt { system: "s".into(), user: "<|im_start|>".into() },
                negative: Prompt { system: "s".into(), user: "u".into() },
                expected_agreement: None,
            }],
        );
        assert!(load_registered(dir.path(), "bad", &with_delim).is_err());
        // An unknown file name must not load under a mismatched pin.
        assert!(load_registered(dir.path(), "garden", &pin).is_ok());
        assert!(pin != gate);
    }

    #[test]
    fn verify_result_accepts_registered_fields_and_rejects_wrong_pins() {
        let request = Request::parse(&json!({
            "registered": {"name": "garden", "sha256": PIN},
            "model_sha256": "a".repeat(64), "layers": [0, 6],
        }))
        .unwrap();
        let mut result = json!({
            "kind": JOB, "schema_version": VERSION, "release_qualified": false,
            "model_id": "m", "model_sha256": "a".repeat(64), "position": null,
            "last_k": 1, "contrast_sha256": PIN,
            "registered": {"name": "garden", "sha256": PIN},
            "runtime": trace::HGN_RUNTIME, "runtime_sha256": trace::HGN_RUNTIME_SHA256.trim_end(),
            "executable_sha256": "d".repeat(64),
            "assets_sha256": {"sidecar": "e".repeat(64), "config": "e".repeat(64), "tokenizer": "e".repeat(64)},
            "fit": {"role": "l_out", "train_pairs": 3, "held_out_pairs": 2, "selected_layer": 0,
                "layers": [
                    {"layer": 0, "norm": 2.0, "unit_vector": [1.0, 0.0], "held_out_margin": 0.5,
                     "held_out_cohens_d": 2.0, "mean_residual_norm": 1.0},
                    {"layer": 6, "norm": 0.0, "unit_vector": null, "held_out_margin": null,
                     "held_out_cohens_d": null, "mean_residual_norm": 1.0},
                ]},
        });
        request
            .verify_version(&result, "m", "a".repeat(64).as_str(), "hgn-v2", VERSION)
            .unwrap();
        result["contrast_sha256"] = json!("f".repeat(64));
        assert!(request
            .verify_version(&result, "m", "a".repeat(64).as_str(), "hgn-v2", VERSION)
            .is_err());
        result["contrast_sha256"] = json!(PIN);
        result["registered"] = json!({"name": "other", "sha256": PIN});
        assert!(request
            .verify_version(&result, "m", "a".repeat(64).as_str(), "hgn-v2", VERSION)
            .is_err());
    }

    #[test]
    fn builtin_requests_still_verify_with_their_pinned_digests() {
        for behavior in [Behavior::Slop, Behavior::Sycophancy, Behavior::ResponseLength] {
            let request = Request::parse(&json!({
                "behavior": serde_json::to_value(behavior).unwrap(),
                "model_sha256": "a".repeat(64), "layers": [0],
            }))
            .unwrap();
            assert_eq!(request.behavior, Some(behavior));
            assert!(request.registered.is_none());
            assert!(!request.pin().is_empty());
        }
    }
}
