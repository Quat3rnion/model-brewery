use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Residual,
    Dense,
    Experts,
    All,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AssetPins {
    pub sidecar: String,
    pub config: String,
    pub tokenizer: String,
}

impl AssetPins {
    pub fn check(&self) -> Result<()> {
        ensure!(
            [&self.sidecar, &self.config, &self.tokenizer]
                .into_iter()
                .all(|s| crate::is_digest(s)),
            "invalid HGN companion asset pins"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    pub target: Target,
    pub first_layer: Option<usize>,
    pub last_layer: Option<usize>,
    pub vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_k: Option<usize>,
    pub assets_sha256: AssetPins,
}

impl Projection {
    pub fn check(&self) -> Result<()> {
        check_direction(&self.vector, self.first_layer, self.last_layer)?;
        check_alpha(self.alpha.unwrap_or(1.0))?;
        if let Some(second) = &self.second_vector {
            check_direction(second, self.first_layer, self.last_layer)?;
            direction_cosine(&self.vector, second)?;
        }
        ensure!(
            self.last_k
                .is_none_or(|k| (1..=crate::behavior_direction::MAX_LAST_K).contains(&k)),
            "invalid HGN direction fit position count"
        );
        self.assets_sha256.check()
    }

    pub fn runtime_spec(&self) -> serde_json::Value {
        let mut spec = serde_json::json!({
            "target":self.target, "first_layer":self.first_layer,
            "last_layer":self.last_layer, "vector":self.vector,
        });
        if let Some(second) = &self.second_vector {
            spec["second_vector"] = serde_json::json!(second);
        }
        if let Some(alpha) = self.alpha {
            spec["alpha"] = serde_json::json!(alpha);
        }
        if let Some(last_k) = self.last_k {
            spec["last_k"] = serde_json::json!(last_k);
        }
        spec
    }
}

pub fn check_alpha(alpha: f64) -> Result<()> {
    ensure!(
        alpha > 0.0 && alpha <= 4.0,
        "projection alpha must be in (0,4]"
    );
    Ok(())
}

pub fn direction_cosine(first: &[f32], second: &[f32]) -> Result<f64> {
    ensure!(
        first.len() == second.len(),
        "projection direction width mismatch"
    );
    let norm = crate::behavior_direction::residual_norm;
    let (a, b) = (norm(first), norm(second));
    let cosine = (first
        .iter()
        .zip(second)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>()
        / (a * b))
        .clamp(-1.0, 1.0);
    let remaining = first
        .iter()
        .zip(second)
        .map(|(&x, &y)| (f64::from(y) / b - cosine * f64::from(x) / a).powi(2))
        .sum::<f64>();
    ensure!(remaining > 1e-10, "HGN projection directions are collinear");
    Ok(cosine)
}

pub fn check_direction(vector: &[f32], first: Option<usize>, last: Option<usize>) -> Result<()> {
    ensure!(
        matches!((first, last), (None, None))
            || matches!((first, last), (Some(a), Some(b)) if a <= b && b < 4096),
        "provide both layer endpoints in order, or neither for all layers"
    );
    let norm_sq: f64 = vector.iter().map(|&x| f64::from(x).powi(2)).sum();
    ensure!(
        (2..=16384).contains(&vector.len()) && (norm_sq - 1.0).abs() < 1e-3,
        "expected a unit HGN direction"
    );
    Ok(())
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Applied {
    pub target: Target,
    pub first_layer: usize,
    pub last_layer: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction_cosine: Option<f64>,
}
