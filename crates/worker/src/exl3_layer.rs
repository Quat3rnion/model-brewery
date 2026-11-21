use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

pub const JOB: &str = "exl3_layer";
pub const HELPER: &str = "BREWERY_EXL3_CHECKPOINT";
pub const SCHEMA: &str = "brewery.exl3-layer.v1";
pub const KIND: &str = "exl3-layer";
pub const FILE: &str = "exl3-layer.safetensors";
pub const SCRATCH: &str = ".exl3-layer-";
pub const DIM: usize = 128;
pub const MAX_SOURCE: u64 = 64 << 20;
pub const MAX_OUTPUT: u64 = 32 << 10;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub source_sha256: String,
    pub tensor: String,
    pub shape: [usize; 2],
    pub codebook: Codebook,
    pub twice_bits: u8,
    pub backend: Backend,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Codebook {
    #[serde(rename = "3inst")]
    ThreeInst,
    Mcg,
    Mul1,
}

impl Codebook {
    pub fn name(self) -> &'static str {
        match self {
            Self::ThreeInst => "3inst",
            Self::Mcg => "mcg",
            Self::Mul1 => "mul1",
        }
    }
    pub fn marker(self) -> Option<u32> {
        match self {
            Self::ThreeInst => None,
            Self::Mcg => Some(0xcbac1fed),
            Self::Mul1 => Some(0x83dcd12d),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cpu,
    Cuda,
    Rocm,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
        }
    }
}

impl Spec {
    pub fn parse(options: &Value) -> Result<Self> {
        let value: Self = serde_json::from_value(options.clone())?;
        ensure!(
            crate::is_digest(&value.source_sha256),
            "invalid EXL3 source SHA-256"
        );
        ensure!(
            crate::weight_source::native_tensor_name(&value.tensor),
            "invalid EXL3 tensor name"
        );
        ensure!(
            value.shape == [DIM, DIM],
            "EXL3 requires exactly one 128x128 tensor"
        );
        let rate = value.twice_bits;
        ensure!(
            ((2..=16).contains(&rate) && rate.is_multiple_of(2))
                || (value.codebook == Codebook::Mul1 && matches!(rate, 3 | 5 | 7)),
            "unsupported EXL3 codebook and twice_bits"
        );
        Ok(value)
    }

    pub fn check_receipt(&self, host: &str, result: &Value) -> Result<()> {
        let report = &result["report"];
        let artifact = &result["artifact"];
        let path = artifact["path"].as_str().map(Path::new);
        ensure!(
            result["kind"] == JOB
                && result["host"] == host
                && result["source_format"] == "safetensors"
                && result["source_sha256"] == self.source_sha256
                && result["verified"] == true
                && result["release_qualified"] == false
                && result.get("library_model_id").is_none()
                && result["backend_receipt"]["tool"] == HELPER
                && result["backend_receipt"]["executable_sha256"]
                    .as_str()
                    .is_some_and(crate::is_digest)
                && report["schema"] == SCHEMA
                && report["verification"] == "independent-decode-and-forward"
                && report["module"] == "layer"
                && matches!(
                    report["source_dtype"].as_str(),
                    Some("F16" | "BF16" | "F32")
                )
                && serde_json::to_value(self)?
                    .as_object()
                    .unwrap()
                    .iter()
                    .all(|(key, value)| report.get(key) == Some(value))
                && [
                    "relative_l2_error",
                    "max_abs_error",
                    "backend_max_abs_error"
                ]
                .iter()
                .all(|key| report[key]
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value >= 0.))
                && report["relative_l2_error"]
                    .as_f64()
                    .is_some_and(|v| v <= 1.)
                && artifact["kind"] == KIND
                && path.is_some_and(|p| crate::absolute(p).is_ok() && p.ends_with(FILE))
                && artifact["sha256"].as_str().is_some_and(crate::is_digest)
                && artifact["bytes"]
                    .as_u64()
                    .is_some_and(|bytes| (1..=MAX_OUTPUT).contains(&bytes))
                && artifact.get("files").is_none()
                && artifact.get("bundle_sha256").is_none(),
            "invalid EXL3 layer result receipt"
        );
        Ok(())
    }
}
