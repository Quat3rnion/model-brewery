use super::{MAX_POSITIONS, Operation, Role, Rule};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    pub direction_id: String,
    pub source_layer: usize,
    pub layer: usize,
    pub positions: BTreeSet<usize>,
    pub alpha: f64,
    pub last_k: Option<usize>,
    pub vector: Vec<f32>,
}

impl Projection {
    pub fn check(&self) -> Result<()> {
        super::hgn_projection::check_alpha(self.alpha)?;
        ensure!(
            !self.direction_id.is_empty()
                && self.direction_id.len() <= 128
                && self.source_layer < 4096
                && self.layer < 4096
                && (1..=128).contains(&self.positions.len())
                && self.positions.iter().all(|&p| p < MAX_POSITIONS)
                && self
                    .last_k
                    .is_none_or(|k| (1..=crate::behavior_direction::MAX_LAST_K).contains(&k))
                && (1..=16384).contains(&self.vector.len())
                && self.vector.iter().all(|v| v.is_finite())
                && (crate::behavior_direction::residual_norm(&self.vector) - 1.0).abs() < 1e-3,
            "invalid GGUF direction or projection selection"
        );
        Ok(())
    }

    pub fn runtime_spec(&self) -> Value {
        json!([{
            "layer":self.layer, "role":Role::LOut, "positions":self.positions,
            "op":"project_out", "vector":self.vector, "scale":self.alpha,
        }])
    }

    pub fn applied_sites(&self) -> [Rule; 1] {
        [Rule {
            layer: self.layer,
            role: Role::LOut,
            positions: self.positions.clone(),
            // Lanternfish attests sites, not operation parameters.
            op: Operation::Zero,
        }]
    }

    pub fn receipt(&self) -> Value {
        json!({
            "direction_id":self.direction_id, "source_layer":self.source_layer,
            "layer":self.layer, "positions":self.positions, "alpha":self.alpha,
            "last_k":self.last_k, "rank":1,
        })
    }
}
