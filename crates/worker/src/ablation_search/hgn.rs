use crate::{
    behavior_direction::Behavior,
    trace::{
        self,
        hgn_projection::{AssetPins, Target, check_alpha, check_direction, direction_cosine},
    },
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

pub const JOB: &str = "screen_hgn_ablation";
pub const MAX_BYTES: u64 = 64 << 10;
pub const MAX_TRIALS: usize = 32;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub model_sha256: String,
    pub direction_id: String,
    pub source_layer: usize,
    pub first_layer: Option<usize>,
    pub last_layer: Option<usize>,
    pub targets: Vec<Target>,
    pub behavior: Behavior,
    pub seed: u32,
    pub vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alphas: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_source_layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_k_variant: Option<Variant>,
    pub assets_sha256: AssetPins,
    pub held_out_pairs: usize,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub direction_id: String,
    pub last_k: usize,
    pub vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_vector: Option<Vec<f32>>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    target: Target,
    direction_id: String,
    last_k: Option<usize>,
    alpha: f64,
    rank: usize,
    direction_cosine: Option<f64>,
}

impl Configuration {
    fn matches(&self, other: &Self) -> bool {
        self.target == other.target
            && self.direction_id == other.direction_id
            && self.last_k == other.last_k
            && self.alpha == other.alpha
            && self.rank == other.rank
            && match (self.direction_cosine, other.direction_cosine) {
                (Some(a), Some(b)) => (-1.0..=1.0).contains(&a) && (a - b).abs() <= 1e-12,
                (None, None) => true,
                _ => false,
            }
    }
}

impl Request {
    pub fn parse(value: &Value) -> Result<Self> {
        let r: Self = serde_json::from_value(value.clone())?;
        ensure!(
            crate::is_digest(&r.model_sha256)
                && !r.direction_id.is_empty()
                && r.direction_id.len() <= 128
                && r.source_layer < 4096
                && (1..=4).contains(&r.targets.len())
                && r.targets.iter().collect::<BTreeSet<_>>().len() == r.targets.len()
                && (1..=8).contains(&r.held_out_pairs),
            "invalid HGN screen pins or trial inventory"
        );
        ensure!(
            !r.alphas().is_empty()
                && r.alphas().len() <= 4
                && r.alphas()
                    .iter()
                    .enumerate()
                    .all(|(i, a)| !r.alphas()[..i].contains(a))
                && r.trial_count() <= MAX_TRIALS
                && r.last_k
                    .is_none_or(|k| (1..=crate::behavior_direction::MAX_LAST_K).contains(&k))
                && r.second_source_layer
                    == r.second_source_layer
                        .filter(|&i| i < 4096 && i != r.source_layer)
                && r.second_source_layer.is_some() == r.second_vector.is_some(),
            "invalid HGN screen alpha grid, fit position count, or second source layer"
        );
        for &alpha in r.alphas() {
            check_alpha(alpha)?;
        }
        if let Some(v) = &r.last_k_variant {
            ensure!(
                r.last_k == Some(1)
                    && (2..=crate::behavior_direction::MAX_LAST_K).contains(&v.last_k)
                    && !v.direction_id.is_empty()
                    && v.direction_id.len() <= 128
                    && v.direction_id != r.direction_id
                    && r.vector.len() == v.vector.len()
                    && r.second_vector.is_some() == v.second_vector.is_some(),
                "last_k_direction_id must supply a matching last-K fit alongside a single-position fit"
            );
        }
        for (_, _, first, second) in r.fits() {
            check_direction(first, r.first_layer, r.last_layer)?;
            if let Some(second) = second {
                check_direction(second, r.first_layer, r.last_layer)?;
                direction_cosine(first, second)?;
            }
        }
        r.assets_sha256.check()?;
        Ok(r)
    }

    fn alphas(&self) -> &[f64] {
        self.alphas.as_deref().unwrap_or(&[1.0])
    }

    fn fits(&self) -> impl Iterator<Item = (&str, Option<usize>, &[f32], Option<&[f32]>)> {
        std::iter::once((
            self.direction_id.as_str(),
            self.last_k,
            self.vector.as_slice(),
            self.second_vector.as_deref(),
        ))
        .chain(self.last_k_variant.iter().map(|v| {
            (
                v.direction_id.as_str(),
                Some(v.last_k),
                v.vector.as_slice(),
                v.second_vector.as_deref(),
            )
        }))
    }

    fn trial_count(&self) -> usize {
        self.targets.len() * self.alphas().len() * (1 + usize::from(self.last_k_variant.is_some()))
    }

    pub fn configurations(&self) -> Result<Vec<Configuration>> {
        let mut configs = Vec::with_capacity(self.trial_count());
        for (id, last_k, vector, second) in self.fits() {
            let cosine = second.map(|v| direction_cosine(vector, v)).transpose()?;
            for &alpha in self.alphas() {
                for &target in &self.targets {
                    configs.push(Configuration {
                        target,
                        direction_id: id.into(),
                        last_k,
                        alpha,
                        rank: 1 + usize::from(second.is_some()),
                        direction_cosine: cosine,
                    });
                }
            }
        }
        Ok(configs)
    }

    pub fn trial_specs(&self) -> Result<Vec<Value>> {
        // Share one seeded basis across the matched fits, scales and targets.
        let seed = u64::from(self.seed) ^ self.source_layer as u64;
        let random = super::control_vector(&self.vector, seed);
        let second_random = if self.second_vector.is_some() {
            let v = (1..=8)
                .map(|i| super::control_vector(&self.vector, seed.wrapping_add(i)))
                .find(|v| direction_cosine(&random, v).is_ok())
                .ok_or_else(|| {
                    anyhow::anyhow!("could not draw an independent HGN random control")
                })?;
            Some(v)
        } else {
            None
        };
        let mut specs = Vec::with_capacity(self.trial_count());
        for (_, _, vector, second) in self.fits() {
            for &alpha in self.alphas() {
                for target in &self.targets {
                    let mut spec = serde_json::json!({
                        "target":target, "first_layer":self.first_layer, "last_layer":self.last_layer,
                        "vector":vector, "random":random, "alpha":alpha,
                    });
                    if let Some(second) = second {
                        spec["second_vector"] = serde_json::json!(second);
                        spec["second_random"] = serde_json::json!(second_random);
                    }
                    specs.push(spec);
                }
            }
        }
        Ok(specs)
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scores {
    pub marker: String,
    pub marker_tokens: Vec<usize>,
    pub prompts: usize,
    pub parent_hits: usize,
    pub parent_mean: f64,
    pub control_mean: f64,
    pub trials: Vec<Trial>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Trial {
    reduction: f64,
    random_reduction: f64,
    control_shift: f64,
    random_control_shift: f64,
    kl: f64,
    random_kl: f64,
    wins_over_random: usize,
}

impl Scores {
    pub fn frontier(&self) -> (Vec<usize>, Vec<usize>) {
        let pareto = super::pareto_by(&self.trials, |t| (t.reduction, t.kl));
        let selected = pareto
            .iter()
            .copied()
            .filter(|&i| {
                let t = &self.trials[i];
                self.parent_hits > 0
                    && t.reduction > t.control_shift
                    && t.reduction > t.random_reduction
                    && t.wins_over_random > self.prompts / 2
            })
            .collect();
        (pareto, selected)
    }

    pub fn verify(&self, req: &Request) -> Result<()> {
        ensure!(
            self.prompts == req.held_out_pairs
                && self.parent_hits <= self.prompts
                && (1..=128).contains(&self.marker.len())
                && (1..=16).contains(&self.marker_tokens.len())
                && self.marker_tokens.iter().all(|&id| id < 262144)
                && self.marker_tokens.iter().collect::<BTreeSet<_>>().len()
                    == self.marker_tokens.len()
                && self.parent_mean.is_finite()
                && self.control_mean.is_finite()
                && self.trials.len() == req.trial_count()
                && self
                    .trials
                    .iter()
                    .all(|t| t.wins_over_random <= self.prompts
                        && t.reduction.is_finite()
                        && t.random_reduction.is_finite()
                        && [t.kl, t.random_kl, t.control_shift, t.random_control_shift]
                            .into_iter()
                            .all(|v| v.is_finite() && v >= 0.0)),
            "invalid HGN screen marker, trial or score"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub kind: String,
    pub schema_version: u32,
    pub release_qualified: bool,
    pub model_sha256: String,
    pub direction_id: String,
    pub source_layer: usize,
    pub first_layer: Option<usize>,
    pub last_layer: Option<usize>,
    pub targets: Vec<Target>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_source_layer: Option<usize>,
    pub configurations: Vec<Configuration>,
    pub behavior: Behavior,
    pub seed: u32,
    pub runtime: String,
    pub runtime_sha256: String,
    pub executable_sha256: String,
    pub assets_sha256: AssetPins,
    pub contrast_sha256: String,
    pub scores: Scores,
    pub pareto: Vec<usize>,
    pub selected: Vec<usize>,
}

impl Report {
    pub fn verify(&self, r: &Request, executable: Option<&str>) -> Result<()> {
        self.scores.verify(r)?;
        let configurations = r.configurations()?;
        ensure!(
            self.kind == JOB
                && self.schema_version == 1
                && !self.release_qualified
                && self.model_sha256 == r.model_sha256
                && self.direction_id == r.direction_id
                && self.source_layer == r.source_layer
                && self.first_layer == r.first_layer
                && self.last_layer == r.last_layer
                && self.targets == r.targets
                && self.second_source_layer == r.second_source_layer
                && self.configurations.len() == configurations.len()
                && self
                    .configurations
                    .iter()
                    .zip(configurations)
                    .all(|(a, b)| a.matches(&b))
                && self.behavior == r.behavior
                && self.seed == r.seed
                && self.runtime == trace::HGN_RUNTIME
                && self.runtime_sha256 == trace::HGN_RUNTIME_SHA256.trim_end()
                && crate::is_digest(&self.executable_sha256)
                && executable.is_none_or(|pin| self.executable_sha256 == pin)
                && self.assets_sha256 == r.assets_sha256
                && self.contrast_sha256 == r.behavior.contrast_sha256()
                && self.scores.frontier() == (self.pareto.clone(), self.selected.clone()),
            "HGN screen receipt does not match its request and Pareto frontier"
        );
        Ok(())
    }
}
