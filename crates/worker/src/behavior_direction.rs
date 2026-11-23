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
pub const CACHE: &str = "BREWERY_CONTRAST_CACHE";
pub const CAP: u64 = 2 << 20;
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
                && r.position.is_none_or(|p| p < 4096),
            "invalid direction revision, layer selection or token position"
        );
        Ok(r)
    }

    pub fn load_pairs(&self, cache: &Path) -> Result<Vec<Pair>> {
        let pin = self.behavior.contrast_sha256();
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
        let set: PromptSet =
            serde_json::from_slice(&bytes).context("invalid direction cache JSON")?;
        let mut ids = BTreeSet::new();
        ensure!(
            set.schema == SCHEMA
                && set.behavior == self.behavior
                && (2..=64).contains(&set.rows.len())
                && set.rows.iter().all(|row| {
                    crate::is_digest(&row.id)
                        && ids.insert(&row.id)
                        && row.positive.system != row.negative.system
                        && row.positive.user == row.negative.user
                        && row.positive.valid()
                        && row.negative.valid()
                        && match set.behavior {
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

    pub fn verify_result(&self, result: &Value, model_id: &str, sha: &str) -> Result<()> {
        ensure!(
            !model_id.is_empty()
                && result["model_id"] == model_id
                && crate::is_digest(sha)
                && sha == self.model_sha256
                && result["model_sha256"] == sha
                && result["kind"] == JOB
                && result["schema_version"] == 1
                && result["release_qualified"] == false
                && result["behavior"] == serde_json::to_value(self.behavior)?
                && result["contrast_sha256"] == self.behavior.contrast_sha256()
                && result["position"] == serde_json::to_value(self.position)?
                && result["lanternfish_commit"] == trace::COMMIT
                && result["executable_sha256"] == trace::SHA256,
            "direction result does not match the model, prompt or runtime pins"
        );
        let fit: Fit = serde_json::from_value(result["fit"].clone())?;
        fit.verify(self)
    }
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
    id: String,
    pub split: Split,
    pub positive: Prompt,
    pub negative: Prompt,
    expected_agreement: Option<String>,
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
    fn verify(&self, request: &Request) -> Result<()> {
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
            match (&row.unit_vector, row.held_out_margin) {
                (None, None) if row.norm == 0.0 => {}
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
                    if margin > best.0 {
                        best = (margin, Some(row.layer));
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

/// Read activations in layer order, positive minus negative within each pair.
/// Select the layer with the largest signed mean held-out projection. Break
/// ties by layer number and do not select non-positive margins.
pub fn compute(
    layers: &[usize],
    role: Role,
    splits: &[Split],
    mut trace: impl FnMut(usize, bool) -> Result<Vec<Vec<f32>>>,
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
    let mut sums: [Vec<Vec<f64>>; 2] = Default::default();
    for (output, split) in sums.iter_mut().zip([Split::Train, Split::HeldOut]) {
        output.resize_with(layers.len(), Vec::new);
        for (row, _) in splits.iter().enumerate().filter(|&(_, p)| *p == split) {
            let positive = trace(row, true)?;
            let negative = trace(row, false)?;
            ensure!(
                positive.len() == layers.len() && negative.len() == layers.len(),
                "incomplete direction activation inventory"
            );
            for ((sum, p), n) in output.iter_mut().zip(positive).zip(negative) {
                ensure!(
                    (1..=16384).contains(&p.len())
                        && p.len() == n.len()
                        && p.len() * layers.len() <= FLOAT_CAP,
                    "invalid direction activation width"
                );
                if sum.is_empty() {
                    sum.resize(p.len(), 0.0);
                }
                ensure!(sum.len() == p.len(), "direction activation width changed");
                for ((sum, p), n) in sum.iter_mut().zip(p).zip(n) {
                    ensure!(
                        p.is_finite() && n.is_finite(),
                        "nonfinite direction activation"
                    );
                    *sum += f64::from(p) - f64::from(n);
                }
            }
        }
    }
    let [train, held_out] = sums;
    let mut fit = Fit {
        role,
        train_pairs,
        held_out_pairs,
        selected_layer: None,
        layers: Vec::with_capacity(layers.len()),
    };
    let mut best = 0.0;
    for ((&layer, train), held_out) in layers.iter().zip(train).zip(held_out) {
        ensure!(
            train.len() == held_out.len(),
            "held-out activation width changed"
        );
        let mean: Vec<_> = train.into_iter().map(|x| x / train_pairs as f64).collect();
        let norm = mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        let (unit_vector, held_out_margin) = if norm > 0.0 {
            let vector: Vec<_> = mean.into_iter().map(|x| x / norm).collect();
            let margin = vector.iter().zip(held_out).map(|(x, y)| x * y).sum::<f64>()
                / held_out_pairs as f64;
            if margin > best {
                best = margin;
                fit.selected_layer = Some(layer);
            }
            (
                Some(vector.into_iter().map(|x| x as f32).collect()),
                Some(margin),
            )
        } else {
            (None, None)
        };
        fit.layers.push(Layer {
            layer,
            norm,
            unit_vector,
            held_out_margin,
        });
    }
    Ok(fit)
}
