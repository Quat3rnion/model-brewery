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
    pub assets_sha256: AssetPins,
}

impl Projection {
    pub fn check(&self) -> Result<()> {
        check_direction(&self.vector, self.first_layer, self.last_layer)?;
        self.assets_sha256.check()
    }

    pub fn runtime_spec(&self) -> serde_json::Value {
        serde_json::json!({
            "target":self.target, "first_layer":self.first_layer,
            "last_layer":self.last_layer, "vector":self.vector,
        })
    }
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
}
