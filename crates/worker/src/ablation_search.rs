//! Bounded search receipts and deterministic proposal policy shared by engine and controller.
use crate::{
    behavior_direction::{Behavior, Fit},
    trace::Role,
    weight_source::scratch::ablation::{Kernel, Plan, Settings, Shape, Variant},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const JOB: &str = "search_ablation";
pub const SCREEN: &str = "screen_ablation";
pub const EXECUTABLE: &str = "BREWERY_LANTERNFISH_ABLATION";
pub const COMMIT: &str = "1bcb4667b3dc1d46d5597d43dc4ca04c9120335b";
pub const SHA256: &str = "d2c38124fad896149456869b02adccc78827ba9dcc9657f88efa6e869d8d2b92";
pub const EXPORT_CAP: u64 = 16 << 20;
pub const RESERVE: u64 = EXPORT_CAP + (16 << 20);
pub const MAX_TRIALS: usize = 32;
pub const MAX_CHILDREN: usize = 3;
pub const MIN_TRIALS: usize = 3;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenRequest {
    pub model_sha256: String,
    pub direction_id: String,
    pub fit: Fit,
    pub behavior: Behavior,
    pub seed: u32,
    pub trial_budget: usize,
    pub max_children: usize,
    pub confirmation_sha256: String,
}

impl ScreenRequest {
    pub fn parse(value: &Value) -> Result<Self> {
        let request: Self = serde_json::from_value(value.clone())?;
        request.check()?;
        Ok(request)
    }

    pub fn check(&self) -> Result<()> {
        ensure!(
            crate::is_digest(&self.model_sha256)
                && crate::is_digest(&self.confirmation_sha256)
                && !self.direction_id.is_empty()
                && matches!(self.behavior, Behavior::Slop | Behavior::Sycophancy)
                && (MIN_TRIALS..=MAX_TRIALS).contains(&self.trial_budget)
                && (1..=MAX_CHILDREN.min(self.trial_budget)).contains(&self.max_children)
                && self.fit.role == Role::LOut
                && (3..=32).contains(&self.fit.layers.len())
                && self
                    .fit
                    .layers
                    .windows(2)
                    .all(|pair| pair[0].layer < pair[1].layer)
                && self.fit.layers.last().unwrap().layer < 4096
                && self
                    .fit
                    .layers
                    .iter()
                    .filter(|row| row.unit_vector.is_some())
                    .count()
                    >= 3,
            "invalid screen fit, behavior, pins or trial limits"
        );
        Ok(())
    }

    pub fn plan(&self, candidate: Candidate, random: bool) -> Result<Plan> {
        let fitted = self
            .fit
            .layers
            .iter()
            .find(|r| r.layer == candidate.source_layer)
            .context("missing screen source layer")?;
        let direction = fitted
            .unit_vector
            .as_deref()
            .context("zero screen direction")?;
        let control;
        let vector = if random {
            control = control_vector(
                direction,
                u64::from(self.seed) ^ candidate.source_layer as u64,
            );
            &control
        } else {
            direction
        };
        Plan::new(
            &candidate.settings(),
            vector,
            fitted.negative_mean.as_deref(),
            |layer| {
                self.fit
                    .layers
                    .iter()
                    .find(|r| r.layer == layer)?
                    .negative_mean
                    .as_deref()
            },
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Positions {
    UserLast,
    UserLast4,
    UserAndAnswerLast,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub source_layer: usize,
    pub first_layer: usize,
    pub last_layer: usize,
    pub peak: usize,
    pub width: u8,
    pub variant: Variant,
    pub alpha: f64,
    pub positions: Positions,
}

impl Candidate {
    pub fn settings(self) -> Settings {
        Settings {
            first_layer: self.first_layer,
            last_layer: self.last_layer,
            kernel: Kernel {
                shape: Shape::Gaussian,
                peak: self.peak,
                width: f64::from(self.width),
                min_weight: 0.0,
            },
            variant: self.variant,
            alpha: self.alpha,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Trial {
    pub candidate: Candidate,
    pub reduction: f64,
    pub random_reduction: f64,
    pub kl: f64,
    pub random_kl: f64,
    pub wins_over_random: usize,
}

impl Trial {
    pub fn moves_beyond_control(&self, n: usize) -> bool {
        self.reduction > 0.0
            && self.reduction > self.random_reduction
            && self.wins_over_random > n / 2
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenReport {
    pub schema_version: u32,
    pub model_sha256: String,
    pub direction_id: String,
    pub behavior: Behavior,
    pub seed: u32,
    pub lanternfish_sha256: String,
    pub screen_sha256: String,
    pub confirmation_sha256: String,
    pub marker: String,
    pub marker_tokens: Vec<usize>,
    pub parent_mean: f64,
    pub parent_hits: usize,
    pub prompts: usize,
    pub trials: Vec<Trial>,
    pub pareto: Vec<usize>,
    pub selected: Vec<usize>,
}

impl ScreenReport {
    pub fn verify(&self, req: &ScreenRequest) -> Result<()> {
        ensure!(
            self.schema_version == 1
                && self.model_sha256 == req.model_sha256
                && self.direction_id == req.direction_id
                && self.behavior == req.behavior
                && self.seed == req.seed
                && self.lanternfish_sha256 == SHA256
                && self.screen_sha256 == req.behavior.contrast_sha256()
                && self.confirmation_sha256 == req.confirmation_sha256
                && self.prompts == req.fit.held_out_pairs
                && self.parent_hits <= self.prompts
                && !self.marker.is_empty()
                && !self.marker_tokens.is_empty()
                && self.parent_mean.is_finite()
                && self.trials.len() == req.trial_budget
                && self.trials.iter().all(|t| t.kl.is_finite()
                    && t.kl >= 0.0
                    && t.random_kl.is_finite()
                    && t.random_kl >= 0.0
                    && t.reduction.is_finite()
                    && t.random_reduction.is_finite()
                    && t.wins_over_random <= self.prompts),
            "screen receipt does not match the request"
        );
        let mut proposals = Proposals::new(req);
        for (index, trial) in self.trials.iter().enumerate() {
            ensure!(
                proposals.next(req, &self.trials[..index])? == trial.candidate,
                "screen receipt has unexpected parameters"
            );
        }
        let (pareto, selected) = frontier(
            &self.trials,
            self.prompts,
            self.parent_hits,
            req.max_children,
        );
        ensure!(
            self.pareto == pareto && self.selected == selected,
            "invalid screen frontier"
        );
        Ok(())
    }
}

/// Keep the unconstrained screen/KL frontier. Only admit points with a stock
/// signal and a positive effect beyond their paired random control.
pub fn frontier(trials: &[Trial], n: usize, hits: usize, limit: usize) -> (Vec<usize>, Vec<usize>) {
    let keep = |i: usize| {
        let t = &trials[i];
        !trials.iter().any(|other| {
            other.kl <= t.kl
                && other.reduction >= t.reduction
                && (other.kl < t.kl || other.reduction > t.reduction)
        })
    };
    let mut front: Vec<_> = (0..trials.len()).filter(|&i| keep(i)).collect();
    front.sort_by(|&a, &b| {
        trials[b]
            .reduction
            .total_cmp(&trials[a].reduction)
            .then(trials[a].kl.total_cmp(&trials[b].kl))
            .then(a.cmp(&b))
    });
    let selected = front
        .iter()
        .copied()
        .filter(|&i| hits > 0 && trials[i].moves_beyond_control(n))
        .take(limit)
        .collect();
    (front, selected)
}

pub struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let z = self.0;
        let z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn control_vector(target: &[f32], seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut values: Vec<_> = target
        .iter()
        .map(|_| {
            // Twelve uniforms approximate a centered normal, with no platform-dependent trig.
            (0..12)
                .map(|_| (rng.next() >> 40) as f64 / (1 << 24) as f64)
                .sum::<f64>()
                - 6.0
        })
        .collect();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let scale = target
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt()
        / norm(&values);
    values.iter_mut().for_each(|v| *v *= scale);
    values.into_iter().map(|v| v as f32).collect()
}

pub struct Proposals {
    rng: Rng,
    layers: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontier_keeps_cost_tradeoff_and_only_admits_effects_beyond_control() {
        let candidate = Candidate {
            source_layer: 2,
            first_layer: 2,
            last_layer: 2,
            peak: 2,
            width: 1,
            variant: Variant::Plain,
            alpha: 1.0,
            positions: Positions::UserLast,
        };
        let trial = |reduction, random_reduction, kl, wins_over_random| Trial {
            candidate,
            reduction,
            random_reduction,
            kl,
            random_kl: 0.01,
            wins_over_random,
        };
        let trials = [
            trial(0.4, 0.7, 0.04, 3),
            trial(0.3, 0.1, 0.02, 6),
            trial(0.1, 0.05, 0.01, 5),
            trial(0.08, 0.01, 0.03, 6),
        ];
        assert_eq!(frontier(&trials, 8, 5, 1), (vec![0, 1, 2], vec![1]));
        assert_eq!(frontier(&trials, 8, 5, 3), (vec![0, 1, 2], vec![1, 2]));
        assert_eq!(frontier(&trials, 8, 0, 3), (vec![0, 1, 2], vec![]));
    }
}

impl Proposals {
    pub fn new(req: &ScreenRequest) -> Self {
        Self {
            rng: Rng(u64::from(req.seed)),
            layers: req
                .fit
                .layers
                .iter()
                .filter(|r| r.unit_vector.is_some())
                .map(|r| r.layer)
                .collect(),
        }
    }

    pub fn next(&mut self, req: &ScreenRequest, history: &[Trial]) -> Result<Candidate> {
        let n = self.layers.len();
        let ordinal = history.len();
        let mut best = None;
        for _ in 0..96 {
            let source = if ordinal < 3 {
                (ordinal * (n - 1) / 2 + req.seed as usize % n) % n
            } else {
                self.rng.pick(n)
            };
            let peak = if ordinal < 3 {
                source
            } else {
                self.rng.pick(n)
            };
            let radius = self.rng.pick(3);
            let idx = self.rng.pick(3);
            let variant = [Variant::Plain, Variant::Projected, Variant::NormPreserving]
                [if ordinal < 3 { ordinal } else { idx }];
            let peak_layer = self.layers[peak];
            let candidate = Candidate {
                source_layer: self.layers[source],
                peak: peak_layer,
                first_layer: peak_layer.saturating_sub(radius),
                last_layer: (peak_layer + radius).min(self.layers[n - 1]),
                width: [1, 2, 4][self.rng.pick(3)],
                variant,
                alpha: if ordinal < 3 {
                    [1.0, 1.5, 0.5][ordinal]
                } else {
                    [0.5, 1.0, 1.5, 2.0][self.rng.pick(4)]
                },
                positions: [
                    Positions::UserLast,
                    Positions::UserLast4,
                    Positions::UserAndAnswerLast,
                ][if ordinal < 3 {
                    ordinal
                } else {
                    self.rng.pick(3)
                }],
            };
            if history.iter().any(|t| t.candidate == candidate)
                || req.plan(candidate, false).is_err()
                || req.plan(candidate, true).is_err()
            {
                continue;
            }
            if ordinal < 3 || ordinal % 4 == 3 {
                return Ok(candidate);
            }
            // Kernel-weighted posterior estimate with a unit Gaussian prior.
            // Alternate the KL penalty so the proposals explore both objectives.
            let lambda = [0.1, 1.0, 10.0][ordinal % 3];
            let (sum, mass) = history.iter().fold((0.0, 1.0), |(sum, mass), t| {
                let d = t.candidate;
                let distance = (d.source_layer.abs_diff(candidate.source_layer)
                    + d.peak.abs_diff(candidate.peak)
                    + d.first_layer.abs_diff(candidate.first_layer)
                    + d.last_layer.abs_diff(candidate.last_layer))
                    as f64
                    / 4.0
                    + (d.alpha - candidate.alpha).abs()
                    + f64::from(d.variant != candidate.variant)
                    + f64::from(d.positions != candidate.positions)
                    + f64::from(d.width.abs_diff(candidate.width)) / 4.0;
                let w = (-distance).exp();
                (sum + w * (t.reduction - lambda * t.kl), mass + w)
            });
            let acquisition = sum / mass + 0.1 / mass.sqrt();
            if best.as_ref().is_none_or(|&(score, _)| acquisition > score) {
                best = Some((acquisition, candidate));
            }
        }
        best.map(|(_, candidate)| candidate)
            .context("no valid screen candidates for this fit")
    }
}
