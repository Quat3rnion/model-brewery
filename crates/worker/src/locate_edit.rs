//! Shared request and bounded native receipt for private locate-then-edit jobs.
use crate::{is_digest, weight_source::scratch::Operation};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::LazyLock;

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
pub const CUDA_EXECUTABLE: &str = "BREWERY_LANTERNFISH_EDIT_CUDA";
pub const CUDA_LIB_DIR: &str = "BREWERY_LANTERNFISH_EDIT_CUDA_LIB_DIR";
pub const ROCM_EXECUTABLE: &str = "BREWERY_LANTERNFISH_EDIT_ROCM";
pub const ROCM_LIB_DIR: &str = "BREWERY_LANTERNFISH_EDIT_ROCM_LIB_DIR";
const ROCM_MANIFEST: &str = include_str!("../../../qualification/lanternfish-rocm.sha256");
type LibraryPins = Vec<(&'static str, &'static str)>;
static ROCM_PIN: LazyLock<(&str, LibraryPins)> = LazyLock::new(|| {
    let mut records = ROCM_MANIFEST.lines().map(|line| {
        let (sha, path) = line
            .split_once("  ")
            .expect("invalid embedded ROCm manifest");
        assert!(is_digest(sha), "invalid embedded ROCm digest");
        (path, sha)
    });
    let (binary, sha256) = records.next().expect("empty embedded ROCm manifest");
    assert_eq!(binary, "lanternfish");
    let libraries = records
        .map(|(path, sha)| {
            (
                path.strip_prefix("lib/")
                    .expect("invalid embedded ROCm path"),
                sha,
            )
        })
        .collect();
    (sha256, libraries)
});
pub const GPU_GATE: &str = "BREWERY_ALLOW_GPU_LOAD";
pub const CUDA_DEVICES: &str = "CUDA_VISIBLE_DEVICES";
const CUDA_LIBRARIES: [(&str, &str); 5] = [
    (
        "libggml-base.so.0",
        "c9b00aa527999534c80b90c8459d00d7975c6a9f844409e1753d25b1844bd773",
    ),
    (
        "libggml-cpu.so.0",
        "db9e2e0c9a18e2ef1b66e1629dc38ed33b29eec1b9bc9ffbc001b2efe50f24ae",
    ),
    (
        "libggml-cuda.so.0",
        "b9a2c9c43421da2ad63cc9b9172ff6c06532b8665ece3296ee8cc2d04772326b",
    ),
    (
        "libggml.so.0",
        "e18423ed800f5ab61ae3e88c5bffcb26d49207dcf977ee888ee77f7c13e694ec",
    ),
    (
        "libllama.so.0",
        "7c42ebe0f25be95472dc4a13f1dffa351c5b12281935baa205bc3d02d89dad0c",
    ),
];
pub const INPUTS: [&str; 2] = ["gguf", "safetensors"];
pub const SAMPLE_SHA: &str = "21ffd89a9f728b1b760433afd385852589cf5012aac10336ac33b2ad23609877";

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Cpu,
    Cuda,
    Rocm,
}

pub struct Pin {
    pub executable: &'static str,
    pub lib_dir: &'static str,
    pub commit: &'static str,
    pub sha256: &'static str,
    pub libraries: &'static [(&'static str, &'static str)],
}

impl Backend {
    fn is_cpu(&self) -> bool {
        *self == Self::Cpu
    }

    pub const fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    pub const fn runtime_stage_reserve(self) -> u64 {
        match self {
            Self::Rocm => 768 << 20,
            _ => 0,
        }
    }

    pub const fn key(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
        }
    }

    pub const fn visibility_vars(self) -> &'static [&'static str] {
        match self {
            Self::Cpu => &[],
            Self::Cuda => &[CUDA_DEVICES],
            Self::Rocm => &["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"],
        }
    }

    pub fn env_keys(self) -> impl Iterator<Item = &'static str> {
        let pin = self.pin();
        [pin.executable, pin.lib_dir]
            .into_iter()
            .chain(self.is_gpu().then_some(GPU_GATE))
            .chain(self.visibility_vars().iter().copied())
    }

    pub fn pin(self) -> Pin {
        match self {
            Self::Cpu => Pin {
                executable: EXECUTABLE,
                lib_dir: LIB_DIR,
                commit: COMMIT,
                sha256: SHA256,
                libraries: &LIBRARIES,
            },
            Self::Cuda => Pin {
                executable: CUDA_EXECUTABLE,
                lib_dir: CUDA_LIB_DIR,
                commit: "3b59809c42e8301222ca4a2bc3121266c12a3532",
                sha256: "816a74eaf076648c97e7d86950a57a3da6b22dbc7940285b955edb15e9c0c959",
                libraries: &CUDA_LIBRARIES,
            },
            Self::Rocm => Pin {
                executable: ROCM_EXECUTABLE,
                lib_dir: ROCM_LIB_DIR,
                commit: "a55898fd807d78291d2195abed1e8ca171bc2d16",
                sha256: ROCM_PIN.0,
                libraries: &ROCM_PIN.1,
            },
        }
    }
}

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
    #[serde(default, skip_serializing_if = "Backend::is_cpu")]
    pub backend: Backend,
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
            ..
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
pub struct Provenance {
    pub backend: Backend,
    pub device: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    updates: Vec<Update>,
}

impl Factors {
    pub fn verify(&self, request: &Request, gguf: &str) -> Result<()> {
        let backend = request.edit.backend;
        ensure!(
            match &self.provenance {
                None => backend == Backend::Cpu,
                Some(audit) => {
                    audit.backend == backend
                        && if backend.is_gpu() {
                            audit
                                .device
                                .strip_prefix(backend.label())
                                .is_some_and(|index| {
                                    index.bytes().all(|b| b.is_ascii_digit())
                                        && index.parse::<u32>().is_ok()
                                })
                        } else {
                            audit.device == backend.label()
                        }
                }
            },
            "edit factor provenance does not match requested {} backend",
            backend.key()
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn gpu_receipt_requires_matching_device_provenance() {
        for (backend, device) in [("cuda", "CUDA0"), ("rocm", "ROCm0")] {
            verify_device(backend, device);
        }
    }

    fn verify_device(backend: &str, device: &str) {
        let model = "a".repeat(64);
        let mut options = json!({"method":"rome","backend":backend,"layers":[12],
            "facts":[{"subject":"Apple A5","prompt":"{} was created by","target":"Google"}]});
        let request = Request::parse(&options).unwrap();
        let old_receipt = json!({"schema_version":1,"model_sha256":model,
            "covariance_sha256":SAMPLE_SHA,"method":"rome","source_dtype":"bf16",
            "gradient_dtype":"f32","losses":[[2.0,0.25]],
            "updates":[{"layer":12,"keys":[[0.5]],"values":[[0.25]]}]});
        let mut receipt = old_receipt.clone();
        receipt["provenance"] = json!({"backend":backend,"device":device});
        let parsed: Factors = serde_json::from_value(receipt.clone()).unwrap();
        parsed.verify(&request, &model).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), receipt);
        let old_factors: Factors = serde_json::from_value(old_receipt).unwrap();
        assert!(old_factors.verify(&request, &model).is_err());
        for audit in [
            json!({"backend":"cpu","device":"CPU"}),
            json!({"backend":backend,"device":""}),
            json!({"backend":backend,"device":"CPU"}),
            json!({"backend":backend,"device":format!("{device}extra")}),
            json!({"backend":"cuda","device":"ROCm0"}),
            json!({"backend":"rocm","device":"CUDA0"}),
        ] {
            receipt["provenance"] = audit;
            let parsed: Factors = serde_json::from_value(receipt.clone()).unwrap();
            assert!(parsed.verify(&request, &model).is_err());
        }
        options.as_object_mut().unwrap().remove("backend");
        let default = Request::parse(&options).unwrap();
        old_factors.verify(&default, &model).unwrap();
        assert!(
            serde_json::to_value(default)
                .unwrap()
                .get("backend")
                .is_none()
        );
    }
}
