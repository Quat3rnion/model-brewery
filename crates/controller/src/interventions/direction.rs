use super::{Binding, Result};
use crate::{directions, store::Store, web::ApiError};
use brewery_worker::trace::{Operation, Role, Rule};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Serialize, Deserialize)]
pub struct DirectionRule {
    direction_id: String,
    direction_version: u32,
    source_layer: usize,
    layer: usize,
    role: Role,
    positions: BTreeSet<usize>,
    #[serde(flatten)]
    operation: DirectionOperation,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum DirectionOperation {
    ProjectOut,
    Subtract { scale: f32 },
}

impl DirectionRule {
    pub(super) fn resolve(&self, store: &Store, binding: &Binding) -> Result<Rule> {
        let artifact = directions::artifact(
            store,
            &self.direction_id,
            self.direction_version,
            Some(self.source_layer),
        )?;
        if artifact["model_id"] != binding.model_id
            || artifact["model_sha256"] != binding.model_sha256
        {
            return Err(ApiError::bad(
                "Direction is bound to a different model or GGUF hash",
            ));
        }
        let vector = serde_json::from_value(artifact["fit"]["layers"][0]["unit_vector"].clone())
            .map_err(|_| ApiError::bad("The requested layer has no nonzero direction"))?;
        let op = match self.operation {
            DirectionOperation::ProjectOut => Operation::ProjectOut { vector },
            DirectionOperation::Subtract { scale } => Operation::Subtract { vector, scale },
        };
        Ok(Rule {
            layer: self.layer,
            role: self.role,
            positions: self.positions.clone(),
            op,
        })
    }
}
