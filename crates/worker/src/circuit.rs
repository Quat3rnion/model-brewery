//! Request and durable result for a single clean-to-corrupt attribution sweep.
use crate::{
    is_digest,
    trace::{Key, Role},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

pub const JOB: &str = "circuit";
pub const MAX_PROMPT_BYTES: usize = 4096;
pub const MAX_TOKENS: usize = 128;
pub const MAX_CALLS: usize = 256;
pub const MAX_PATCHES: usize = 32;
pub const QUALIFICATION: &str = "not release-qualified";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub model_sha256: String,
    pub clean_prompt: String,
    pub corrupt_prompt: String,
    pub target_id: i32,
    pub contrast_id: Option<i32>,
    pub layers: Vec<usize>,
    pub positions: Vec<usize>,
    pub roles: Vec<Role>,
    pub top_k: usize,
    #[serde(default)]
    pub sample_lower: usize,
}

fn distinct<T: Ord>(items: &[T]) -> bool {
    items.iter().collect::<BTreeSet<_>>().len() == items.len()
}

impl Request {
    pub fn parse(options: &Value) -> Result<Self> {
        let r: Self = serde_json::from_value(options.clone()).context("invalid circuit options")?;
        ensure!(is_digest(&r.model_sha256), "invalid circuit model SHA-256");
        ensure!(
            r.clean_prompt != r.corrupt_prompt
                && [&r.clean_prompt, &r.corrupt_prompt].into_iter().all(|p| {
                    !p.trim().is_empty() && p.len() <= MAX_PROMPT_BYTES && !p.contains('\0')
                }),
            "select two distinct non-empty prompts of at most 4096 bytes each"
        );
        ensure!(
            r.target_id >= 0 && r.contrast_id.is_none_or(|t| t >= 0 && t != r.target_id),
            "invalid circuit target or contrast token ID"
        );
        ensure!(
            (1..=128).contains(&r.layers.len())
                && distinct(&r.layers)
                && r.layers.iter().all(|&l| l < 4096)
                && (1..=MAX_TOKENS).contains(&r.positions.len())
                && distinct(&r.positions)
                && r.positions.iter().all(|&p| p < MAX_TOKENS)
                && (1..=3).contains(&r.roles.len())
                && distinct(&r.roles)
                && r.roles
                    .iter()
                    .all(|r| matches!(r, Role::LOut | Role::AttnOut | Role::FfnOut)),
            "select distinct circuit layers, positions and supported roles"
        );
        let n = r.layers.len() * r.positions.len() * r.roles.len();
        let patches = r.top_k.checked_add(r.sample_lower);
        ensure!(
            n * (1 + usize::from(r.contrast_id.is_some())) <= MAX_CALLS
                && r.top_k > 0
                && patches.is_some_and(|p| p <= n.min(MAX_PATCHES)),
            "circuit exceeds the gradient, site or patch budget"
        );
        Ok(r)
    }

    pub fn check_model(&self, actual: &str) -> Result<()> {
        ensure!(
            actual == self.model_sha256,
            "circuit model SHA-256 mismatch"
        );
        Ok(())
    }

    pub fn check_tokens(&self, clean: &[i32], corrupt: &[i32]) -> Result<()> {
        ensure!(
            (1..=MAX_TOKENS).contains(&clean.len())
                && clean.len() == corrupt.len()
                && clean != corrupt
                && clean.iter().chain(corrupt).all(|&t| t >= 0)
                && self.positions.iter().all(|&p| p < clean.len()),
            "circuit needs distinct equal-length token sequences and in-range positions"
        );
        Ok(())
    }

    pub fn keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.layers.iter().flat_map(move |&layer| {
            self.positions.iter().flat_map(move |&position| {
                self.roles.iter().map(move |&role| Key {
                    layer,
                    position,
                    role,
                })
            })
        })
    }

    /// Select unique lower ranks with a fixed seed so confirmation is repeatable.
    pub fn patch_ranks(&self, n: usize) -> BTreeSet<usize> {
        let mut lower: Vec<_> = (self.top_k..n).collect();
        let mut state: u64 = 5;
        for i in 0..self.sample_lower {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = i + ((state >> 32) % (lower.len() - i) as u64) as usize;
            lower.swap(i, j);
        }
        (0..self.top_k)
            .chain(lower.into_iter().take(self.sample_lower))
            .collect()
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Site {
    #[serde(flatten)]
    pub key: Key,
    pub attribution: f64,
    pub patched_delta: Option<f64>,
}

impl Site {
    pub fn rank(&self, other: &Self) -> std::cmp::Ordering {
        other
            .attribution
            .abs()
            .total_cmp(&self.attribution.abs())
            .then(self.key.cmp(&other.key))
    }
}

/// Pearson and Spearman correlations on signed, measured pairs only.
pub fn correlations(sites: &[Site]) -> (Option<f64>, Option<f64>) {
    let (x, y): (Vec<_>, Vec<_>) = sites
        .iter()
        .filter_map(|s| Some((s.attribution, s.patched_delta?)))
        .unzip();
    (pearson(&x, &y), pearson(&ranks(&x), &ranks(&y)))
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.; values.len()];
    let mut start = 0;
    while start < indices.len() {
        let count = indices[start..]
            .iter()
            .take_while(|&&i| values[i] == values[indices[start]])
            .count();
        let end = start + count;
        let rank = (start + end - 1) as f64 / 2.;
        for &i in &indices[start..end] {
            ranks[i] = rank;
        }
        start = end;
    }
    ranks
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() < 2 || x.iter().all(|v| *v == x[0]) || y.iter().all(|v| *v == y[0]) {
        return None;
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mx, my) = (mean(x), mean(y));
    let (cov, xx, yy) = x.iter().zip(y).fold((0., 0., 0.), |(cov, xx, yy), (a, b)| {
        let (dx, dy) = (a - mx, b - my);
        (cov + dx * dy, xx + dx * dx, yy + dy * dy)
    });
    (xx > 0. && yy > 0.).then(|| (cov / (xx.sqrt() * yy.sqrt())).clamp(-1., 1.))
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub schema_version: u32,
    pub kind: String,
    pub qualification: String,
    pub model_id: String,
    pub model_sha256: String,
    pub backend: crate::locate_edit::Backend,
    pub clean_prompt: String,
    pub corrupt_prompt: String,
    pub clean_tokens: Vec<i32>,
    pub corrupt_tokens: Vec<i32>,
    pub target_id: i32,
    pub contrast_id: Option<i32>,
    pub clean_metric: f64,
    pub corrupt_metric: f64,
    pub sites: Vec<Site>,
    pub top_k: usize,
    pub sample_lower: usize,
    pub pearson_r: Option<f64>,
    pub spearman_rho: Option<f64>,
    pub executable_sha256: String,
    pub manifest_sha256: String,
}

impl Artifact {
    pub fn check(&self, req: &Request, model_id: &str) -> Result<()> {
        req.check_model(&self.model_sha256)?;
        req.check_tokens(&self.clean_tokens, &self.corrupt_tokens)?;
        let logit_bound = f64::from(f32::MAX) * 2.;
        // Bounds follow from finite F32 vectors/logits and a maximum width of 16384.
        let dot_bound = logit_bound * logit_bound * 16384.;
        ensure!(
            self.schema_version == 1
                && self.kind == JOB
                && self.qualification == QUALIFICATION
                && self.model_id == model_id
                && self.backend == crate::locate_edit::Backend::Cpu
                && self.clean_prompt == req.clean_prompt
                && self.corrupt_prompt == req.corrupt_prompt
                && self.target_id == req.target_id
                && self.contrast_id == req.contrast_id
                && self.top_k == req.top_k
                && self.sample_lower == req.sample_lower
                && self.clean_metric.abs() <= logit_bound
                && self.corrupt_metric.abs() <= logit_bound
                && is_digest(&self.executable_sha256)
                && is_digest(&self.manifest_sha256),
            "circuit artifact does not match the request"
        );
        let keys: BTreeSet<_> = self.sites.iter().map(|s| s.key).collect();
        let patch_ranks = req.patch_ranks(req.keys().count());
        ensure!(
            keys.len() == self.sites.len()
                && keys == req.keys().collect()
                && self.sites.windows(2).all(|w| w[0].rank(&w[1]).is_le())
                && self.sites.iter().enumerate().all(|(i, s)| {
                    s.attribution.abs() <= dot_bound
                        && if patch_ranks.contains(&i) {
                            s.patched_delta.is_some_and(|d| d.abs() <= logit_bound * 2.)
                        } else {
                            s.patched_delta.is_none()
                        }
                }),
            "invalid circuit site inventory, rank or confirmation measurements"
        );
        let (r, rho) = correlations(&self.sites);
        let check = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(a), Some(b)) => (-1. ..=1.).contains(&b) && (a - b).abs() <= 1e-12,
            (None, None) => true,
            _ => false,
        };
        ensure!(
            check(r, self.pearson_r) && check(rho, self.spearman_rho),
            "invalid circuit correlation"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_is_undefined_for_constant_measured_values() {
        let sites = [1., 2., 3.].map(|attribution| Site {
            key: Key {
                layer: 0,
                role: Role::LOut,
                position: 0,
            },
            attribution,
            patched_delta: Some(0.1),
        });
        assert_eq!(correlations(&sites), (None, None));
    }

    #[test]
    fn spearman_averages_tied_ranks_and_excludes_unmeasured_sites() {
        let sites = [
            (1., Some(10.)),
            (1., Some(20.)),
            (3., Some(20.)),
            (4., None),
        ]
        .map(|(attribution, patched_delta)| Site {
            key: Key {
                layer: 0,
                role: Role::LOut,
                position: 0,
            },
            attribution,
            patched_delta,
        });
        let (_, rho) = correlations(&sites);
        assert!((rho.unwrap() - 0.5).abs() < 1e-12);
    }
}
