//! Wire contract for the read-only quantization damage map.
use crate::{is_digest, locate_edit::Backend, sensitivity};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};

pub const JOB: &str = "quantization_damage";
pub const MAX_MODEL_BYTES: u64 = 2 << 30;
pub const MAX_GRADIENT_BYTES: u64 = 768 << 20;
pub const MAX_TENSORS: usize = 512;

fn one() -> usize {
    1
}
fn three() -> usize {
    3
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub reference_sha256: String,
    pub quantized_sha256: String,
    pub gguf_type: String,
    #[serde(default)]
    pub backend: Backend,
    #[serde(default = "one")]
    pub observations: usize,
    #[serde(default = "three")]
    pub swaps: usize,
    #[serde(default)]
    pub tensors: Vec<String>,
}

impl Request {
    pub fn parse(value: &Value) -> Result<Self> {
        let r: Self =
            serde_json::from_value(value.clone()).context("invalid damage map options")?;
        ensure!(
            is_digest(&r.reference_sha256)
                && is_digest(&r.quantized_sha256)
                && r.reference_sha256 != r.quantized_sha256
                && matches!(r.gguf_type.as_str(), "Q4_K_M")
                && (1..=4).contains(&r.observations)
                && (1..=5).contains(&r.swaps)
                && r.tensors.len() <= MAX_TENSORS
                && r.tensors.iter().all(|n| sensitivity::valid_tensor(n))
                && r.tensors.iter().collect::<BTreeSet<_>>().len() == r.tensors.len(),
            "select two distinct pinned GGUFs, a Q4_K_M base, 1 to 4 prompts and 1 to 5 swaps"
        );
        Ok(r)
    }

    pub fn hashes(&self) -> [&str; 2] {
        [&self.reference_sha256, &self.quantized_sha256]
    }
}

pub fn configured(
    env: &std::collections::BTreeMap<String, String>,
    r: &Request,
    admission: Option<&Path>,
) -> bool {
    sensitivity::configured(env, r.backend, admission)
}

pub fn check_result(request: &Request, result: &Value, ids: [&str; 2]) -> Result<()> {
    let tensors = result["tensors"]
        .as_array()
        .context("missing damage map tensors")?;
    let observations = result["observations"]
        .as_array()
        .context("missing damage map observations")?;
    let proposal = &result["recipe_proposal"];
    let recipe = |key: &str, profile_key: &str| -> Result<()> {
        let options = &proposal[key];
        let profile = &proposal[profile_key];
        let types = options["tensor_types"]
            .as_object()
            .context("missing damage map overrides")?;
        ensure!(
            options.as_object().is_some_and(|o| o.len() == 3)
                && options["output_format"] == "gguf"
                && options["gguf_type"] == request.gguf_type
                && profile.as_object().is_some_and(|p| p.len() == 4)
                && profile["format"] == "gguf"
                && profile["name"]
                    .as_str()
                    .is_some_and(|n| !n.is_empty() && n.len() <= 256)
                && profile["preset"] == options["gguf_type"]
                && profile["overrides"] == options["tensor_types"]
                && types.len() <= 256
                && types.iter().all(|(n, t)| sensitivity::valid_tensor(n)
                    && matches!(
                        t.as_str(),
                        Some("Q4_0" | "Q5_0" | "Q8_0" | "Q4_K" | "Q5_K" | "Q6_K")
                    )),
            "invalid damage map recipe options"
        );
        Ok(())
    };
    ensure!(
        result["kind"] == JOB
            && result["schema_version"] == 1
            && result["qualification"] == sensitivity::QUALIFICATION
            && result["backend"] == request.backend.key()
            && result["reference_id"] == ids[0]
            && result["quantized_id"] == ids[1]
            && result["reference_sha256"] == request.reference_sha256
            && result["quantized_sha256"] == request.quantized_sha256
            && result["corpus_sha256"] == sensitivity::CORPUS_SHA256
            && ["manifest_sha256", "executable_sha256"]
                .iter()
                .all(|k| result[k].as_str().is_some_and(is_digest))
            && (1..=MAX_TENSORS).contains(&tensors.len())
            && observations.len() == request.observations
            && proposal["applied"] == false
            && proposal["reference_sha256"] == request.reference_sha256
            && proposal["baseline_sha256"] == request.quantized_sha256
            && proposal["payload_bytes_delta"] == 0
            && result["layers"].is_array(),
        "invalid quantization damage artifact"
    );
    let mut seen = BTreeSet::new();
    let valid_values = |key: &str, t: &Value| -> bool {
        t[key].as_array().is_some_and(|rows| {
            rows.len() == observations.len()
                && rows.iter().all(|x| x.as_f64().is_some_and(f64::is_finite))
        })
    };
    for (i, row) in tensors.iter().enumerate() {
        let name = row["name"]
            .as_str()
            .filter(|n| sensitivity::valid_tensor(n))
            .context("invalid damage map tensor name")?;
        ensure!(
            seen.insert(name)
                && row["layer"] == serde_json::json!(sensitivity::layer(name))
                && row["mean_absolute_damage"]
                    .as_f64()
                    .is_some_and(|f| f.is_finite() && f >= 0.)
                && valid_values("predicted_recovery", row)
                && if i < request.swaps {
                    valid_values("measured_recovery", row)
                } else {
                    row["measured_recovery"].is_null()
                },
            "invalid damage map tensor row"
        );
    }
    ensure!(
        request.tensors.is_empty()
            || request
                .tensors
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == seen,
        "damage map result tensor selection mismatch"
    );
    ensure!(
        observations
            .iter()
            .all(|o| o["w5"]["model_logits"].is_array()
                && [
                    "reference_logit",
                    "quantized_logit",
                    "measured_drift",
                    "predicted_recovery_sum"
                ]
                .iter()
                .all(|k| o[k].as_f64().is_some_and(f64::is_finite))),
        "invalid damage map observations"
    );
    recipe("options", "profile")?;
    recipe("uniform_options", "uniform_profile")?;
    let proposed = proposal["options"]["tensor_types"].as_object().unwrap();
    let baseline = proposal["uniform_options"]["tensor_types"]
        .as_object()
        .unwrap();
    ensure!(
        proposed.keys().eq(baseline.keys()),
        "damage map recipe inventory differs from baseline"
    );
    Ok(())
}
