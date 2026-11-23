//! Shared request and bounded native receipt for private locate-then-edit jobs.
use crate::{is_digest, weight_source::scratch::Operation};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const JOB: &str = "scratch_locate_edit";
pub const EXECUTABLE: &str = "BREWERY_LANTERNFISH_EDIT";
pub const LIB_DIR: &str = "BREWERY_LANTERNFISH_EDIT_LIB_DIR";
pub const COMMIT: &str = "f8bbba37318cc6d5a11462662ab165b22bc791bf";
pub const SHA256: &str = "f21adb9c95eef7b0a9ccc0a367df1d9daf242fd9bc69a2254e8d016aefa4e5d1";
pub const LIBRARIES: [(&str, &str); 4] = [
    (
        "libllama.so.0",
        "e5ac3b4879c27219ef0ed919e80a552e5e935495babd69f3384c925b45c38799",
    ),
    (
        "libggml.so.0",
        "3e121f67ae15c7c3e0dae38c666a62836b3160f49d7723f1d2ff798935ea77fd",
    ),
    (
        "libggml-base.so.0",
        "97d1ff9c0e72562b7b82c595ef8024884a186f09ea081bd6fef4cca674a4eb66",
    ),
    (
        "libggml-cpu.so.0",
        "a2b68b5eab8eec8daa9bba9ee8e56b6d93ca6c2c7ca31f1b0b9533f3791977e6",
    ),
];
pub const INPUTS: [&str; 2] = ["gguf", "safetensors"];
pub const SAMPLE_SHA: &str = "21ffd89a9f728b1b760433afd385852589cf5012aac10336ac33b2ad23609877";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    Rome,
    Memit,
    Alphaedit,
}

impl Method {
    pub const fn key(self) -> &'static str {
        match self {
            Self::Rome => "rome",
            Self::Memit => "memit",
            Self::Alphaedit => "alphaedit",
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fact {
    subject: String,
    prompt: String,
    target: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    pub layers: Vec<usize>,
    facts: Vec<Fact>,
    #[serde(default = "steps")]
    steps: usize,
}

fn steps() -> usize {
    16
}

#[derive(Deserialize, Serialize)]
pub struct Request {
    pub method: Method,
    #[serde(flatten)]
    pub edit: Edit,
}

impl Request {
    pub fn parse(value: &Value) -> Result<Self> {
        let request: Self = serde_json::from_value(value.clone())?;
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<()> {
        let Edit {
            layers,
            facts,
            steps,
        } = &self.edit;
        ensure!(
            (1..=32).contains(steps)
                && (1..=8).contains(&facts.len())
                && (1..=3).contains(&layers.len())
                && layers.windows(2).all(|x| x[0] < x[1])
                && layers.iter().all(|&layer| layer < 1024),
            "use 1-8 facts, 1-3 increasing layer indices and 1-32 steps"
        );
        match self.method {
            Method::Rome => ensure!(
                layers.len() == 1 && facts.len() == 1,
                "ROME needs one layer and fact"
            ),
            Method::Memit => ensure!(layers.len() >= 2, "MEMIT needs two or three layers"),
            Method::Alphaedit => (),
        }
        for f in facts {
            ensure!(
                !f.subject.trim().is_empty()
                    && f.subject.len() <= 256
                    && !f.target.trim().is_empty()
                    && f.target.len() <= 256
                    && f.prompt.len() <= 1024
                    && f.prompt.matches("{}").count() == 1,
                "each fact needs a subject, target and one {{}} slot in the prompt"
            );
        }
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Update {
    layer: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Factors {
    schema_version: u8,
    model_sha256: String,
    pub covariance_sha256: String,
    method: Method,
    pub source_dtype: String,
    pub gradient_dtype: String,
    pub losses: Vec<[f32; 2]>,
    updates: Vec<Update>,
}

impl Factors {
    pub fn verify(&self, request: &Request, gguf: &str) -> Result<()> {
        ensure!(
            self.schema_version == 1
                && is_digest(gguf)
                && self.model_sha256 == gguf
                && self.method == request.method
                && self.covariance_sha256 == SAMPLE_SHA
                && matches!(self.source_dtype.as_str(), "f32" | "bf16")
                && self.gradient_dtype == "f32"
                && self.losses.len() == request.edit.facts.len()
                && self
                    .losses
                    .iter()
                    .all(|[a, b]| a.is_finite() && b.is_finite() && *b >= 0.0 && b <= a)
                && self
                    .updates
                    .iter()
                    .map(|u| u.layer)
                    .eq(request.edit.layers.iter().copied()),
            "edit factors do not match the pinned input and request"
        );
        let rank = request.edit.facts.len();
        let mut touched = 0usize;
        for u in &self.updates {
            ensure!(
                u.keys.len() == rank && u.values.len() == rank,
                "invalid update rank"
            );
            let width = u.keys[0].len();
            let height = u.values[0].len();
            ensure!(
                (1..=32768).contains(&width)
                    && (1..=32768).contains(&height)
                    && u.keys.iter().all(|k| k.len() == width)
                    && u.values.iter().all(|v| v.len() == height)
                    && u.keys
                        .iter()
                        .chain(&u.values)
                        .flatten()
                        .all(|v| v.is_finite()),
                "non-finite or invalid low-rank factors"
            );
            // Standalone conversion always emits BF16, but the source may be F32.
            touched += width * height * 2;
        }
        ensure!(
            touched as u64 <= crate::weight_source::scratch::MAX_EDIT_BYTES,
            "edit exceeds scratch tensor budget"
        );
        Ok(())
    }

    pub fn operations(&self) -> Vec<Operation> {
        self.updates
            .iter()
            .map(|u| Operation::LowRankBatch {
                tensor: format!("model.layers.{}.mlp.down_proj.weight", u.layer),
                u: u.values.clone(),
                v: u.keys.clone(),
            })
            .collect()
    }
}
