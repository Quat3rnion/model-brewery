use super::{Artifact, Result};
use crate::web::ApiError;
use brewery_worker::trace::{Operation, Role};
use std::collections::BTreeMap;

const ALIGNMENT: usize = 32;
const GGUF_STRING: u32 = 8;
const GGUF_INT32: u32 = 5;
const GGML_F32: u32 = 0;

pub(super) fn encode(artifact: &Artifact) -> Result<Vec<u8>> {
    let mut directions = BTreeMap::new();
    for rule in &artifact.interventions {
        if matches!(&rule.op, Operation::ProjectOut { .. }) {
            return Err(ApiError::bad(
                "llama.cpp control vectors cannot express a projection",
            ));
        }
        if rule.layer == 0 {
            return Err(ApiError::bad(
                "llama.cpp control vectors cannot target layer 0",
            ));
        }
        if rule.role != Role::LOut {
            return Err(ApiError::bad(
                "Control vector export requires the l_out role",
            ));
        }
        let (vector, multiplier) = match &rule.op {
            Operation::Add { vector } => (vector, 1.0),
            Operation::Subtract { vector, scale } => (vector, -*scale),
            _ => {
                return Err(ApiError::bad(
                    "Control vector export requires add or subtract",
                ));
            }
        };
        if directions
            .insert(rule.layer, (vector, multiplier))
            .is_some()
        {
            return Err(ApiError::bad(
                "Control vector export needs one direction per layer",
            ));
        }
    }
    let width = directions
        .values()
        .next()
        .expect("nonempty artifact")
        .0
        .len();
    if directions.values().any(|(vector, _)| vector.len() != width) {
        return Err(ApiError::bad(
            "Control vector export requires one shared vector width",
        ));
    }

    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes()); // GGUF version
    out.extend_from_slice(&(directions.len() as u64).to_le_bytes());
    out.extend_from_slice(&3u64.to_le_bytes()); // metadata count
    for (key, value) in [
        ("general.architecture", "controlvector"),
        ("controlvector.model_hint", artifact.model_hint.as_str()),
    ] {
        string(&mut out, key);
        out.extend_from_slice(&GGUF_STRING.to_le_bytes());
        string(&mut out, value);
    }
    string(&mut out, "controlvector.layer_count");
    out.extend_from_slice(&GGUF_INT32.to_le_bytes());
    out.extend_from_slice(&(directions.len() as i32).to_le_bytes());

    let stride = (width * 4).next_multiple_of(ALIGNMENT);
    for (i, &layer) in directions.keys().enumerate() {
        // llama.cpp indexes direction tensors by the zero-based block index;
        // its control vector adapter explicitly excludes block 0.
        string(&mut out, &format!("direction.{layer}"));
        out.extend_from_slice(&1u32.to_le_bytes()); // dimensions
        out.extend_from_slice(&(width as u64).to_le_bytes());
        out.extend_from_slice(&GGML_F32.to_le_bytes());
        out.extend_from_slice(&((i * stride) as u64).to_le_bytes());
    }
    pad(&mut out);
    for (vector, multiplier) in directions.values() {
        out.extend(vector.iter().flat_map(|v| (v * multiplier).to_le_bytes()));
        pad(&mut out);
    }
    Ok(out)
}

fn pad(out: &mut Vec<u8>) {
    out.resize(out.len().next_multiple_of(ALIGNMENT), 0);
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}
