use crate::{
    behavior_direction::Behavior,
    trace::{
        self,
        hgn_projection::{AssetPins, Target, check_direction},
    },
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

pub const JOB: &str = "screen_hgn_ablation";
pub const MAX_BYTES: u64 = 64 << 10;

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
    pub assets_sha256: AssetPins,
    pub held_out_pairs: usize,
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
        check_direction(&r.vector, r.first_layer, r.last_layer)?;
        r.assets_sha256.check()?;
        Ok(r)
    }

    pub fn trial_specs(&self) -> Vec<Value> {
        // One random draw for this source direction, shared between matched
        // writer/range settings, so control comparisons are reproducible.
        let random = super::control_vector(
            &self.vector,
            u64::from(self.seed) ^ self.source_layer as u64,
        );
        self.targets
            .iter()
            .map(|target| {
                serde_json::json!({
                    "target":target, "first_layer":self.first_layer, "last_layer":self.last_layer,
                    "vector":self.vector, "random":random,
                })
            })
            .collect()
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
                && self.trials.len() == req.targets.len()
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
