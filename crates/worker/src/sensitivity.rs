//! Shared request and artifact contracts for empirical Fisher measurements.
use crate::{is_digest, locate_edit::Backend};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

pub const JOB: &str = "sensitivity";
pub const PIN: &str = "BREWERY_LANTERNFISH_SENSITIVITY_PIN";
pub const MANIFEST_SHA: &str = "BREWERY_LANTERNFISH_SENSITIVITY_MANIFEST_SHA256";
pub const CORPUS_SHA256: &str = "9678f9afdc31b2a9be7fec2045997e2360693ee72b893287c0d539983bff4923";
pub const QUALIFICATION: &str = "not release-qualified";
pub const REVISION_PIN: &str = "revision_sha256";
pub const MAX_TENSORS: usize = 32;
pub const MAX_CALLS: usize = 256;
pub const MAX_GRADIENT_BYTES: u64 = 128 << 20;
pub const MAX_TOTAL_BYTES: u64 = 2 << 30;
pub const MAX_OBSERVATIONS: usize = 64;
pub const RUNTIME_RESERVE: u64 = 1 << 30;

fn default_observations() -> usize {
    16
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub model_sha256: String,
    pub corpus_sha256: String,
    #[serde(default)]
    pub backend: Backend,
    #[serde(default)]
    pub layers: Vec<u32>,
    #[serde(default)]
    pub tensors: Vec<String>,
    #[serde(default = "default_observations")]
    pub observations: usize,
}

pub fn layer(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split_once('.')?.0.parse().ok()
}

pub fn valid_tensor(name: &str) -> bool {
    (1..=63).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

impl Request {
    pub fn parse(options: &Value) -> Result<Self> {
        let r: Self =
            serde_json::from_value(options.clone()).context("invalid sensitivity options")?;
        ensure!(
            is_digest(&r.model_sha256),
            "invalid sensitivity model SHA-256"
        );
        ensure!(
            r.corpus_sha256 == CORPUS_SHA256,
            "select the pinned W4 held-out corpus SHA-256"
        );
        ensure!(
            (1..=MAX_OBSERVATIONS).contains(&r.observations)
                && r.layers.is_empty() != r.tensors.is_empty()
                && options.get("layers").is_some() == !r.layers.is_empty()
                && options.get("tensors").is_some() == !r.tensors.is_empty()
                && r.layers.len() <= 16
                && r.layers.iter().all(|&i| i < 4096)
                && r.layers.iter().collect::<BTreeSet<_>>().len() == r.layers.len()
                && r.tensors.len() <= MAX_TENSORS
                && r.tensors.iter().all(|n| valid_tensor(n))
                && r.tensors.iter().collect::<BTreeSet<_>>().len() == r.tensors.len(),
            "select distinct layers (1 to 16) or exact weight names (1 to 32), with 1 to 64 observations"
        );
        r.calls(r.tensors.len().max(r.layers.len()))?;
        Ok(r)
    }

    pub fn resolved(options: &Value) -> Result<(Self, String)> {
        let mut options = options
            .as_object()
            .context("invalid sensitivity options")?
            .clone();
        let revision = options.remove(REVISION_PIN);
        let request = Self::parse(&Value::Object(options))?;
        let revision = match revision {
            None => request.model_sha256.clone(),
            Some(Value::String(sha)) if is_digest(&sha) => sha,
            _ => anyhow::bail!("invalid sensitivity revision SHA-256"),
        };
        Ok((request, revision))
    }

    pub fn calls(&self, tensors: usize) -> Result<usize> {
        let calls = self
            .observations
            .checked_mul(tensors)
            .filter(|&n| n <= MAX_CALLS);
        ensure!(
            (1..=MAX_TENSORS).contains(&tensors),
            "sensitivity needs 1 to 32 tensors"
        );
        calls.context("sensitivity exceeds the 256 gradient call budget")
    }

    pub fn check_model(&self, actual: &str) -> Result<()> {
        ensure!(
            self.model_sha256 == actual,
            "sensitivity model SHA-256 mismatch"
        );
        Ok(())
    }
}

pub fn configured(
    env: &BTreeMap<String, String>,
    backend: Backend,
    admission: Option<&Path>,
) -> bool {
    env.get(PIN).is_some_and(|p| Path::new(p).is_absolute())
        && env.get(MANIFEST_SHA).is_some_and(|sha| is_digest(sha))
        && (!backend.is_gpu()
            || (admission.is_some()
                && env
                    .get(crate::locate_edit::GPU_GATE)
                    .is_some_and(|v| v == "1")))
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub schema_version: u32,
    pub kind: String,
    pub qualification: String,
    pub model_id: String,
    pub model_sha256: String,
    pub revision_sha256: String,
    pub corpus_sha256: String,
    pub backend: Backend,
    pub observation_count: usize,
    pub observations_sha256: String,
    pub executable_sha256: String,
    pub manifest_sha256: String,
    pub tensors: Vec<Tensor>,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub name: String,
    pub layer: Option<u32>,
    pub shape: Vec<u64>,
    pub fisher: f64,
}

impl Tensor {
    pub fn bytes(&self) -> Result<u64> {
        let bytes = self.shape.iter().try_fold(4u64, |n, &d| n.checked_mul(d));
        ensure!(
            (1..=4).contains(&self.shape.len()),
            "invalid sensitivity tensor rank"
        );
        bytes
            .filter(|n| (4..=MAX_GRADIENT_BYTES).contains(n))
            .context("sensitivity tensor exceeds the gradient byte budget")
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub layer: u32,
    pub elements: u64,
    pub fisher: f64,
}

pub fn aggregate(tensors: &[Tensor]) -> Result<Vec<Layer>> {
    let mut rows: BTreeMap<u32, (u64, f64)> = BTreeMap::new();
    for t in tensors {
        if let Some(layer) = t.layer {
            let elements = t.bytes()? / 4;
            let (count, sum) = rows.entry(layer).or_default();
            *count += elements;
            *sum += t.fisher * elements as f64;
        }
    }
    Ok(rows
        .into_iter()
        .map(|(layer, (elements, sum))| Layer {
            layer,
            elements,
            fisher: sum / elements as f64,
        })
        .collect())
}

impl Artifact {
    pub fn check(&self, req: &Request, model_id: &str, revision: &str) -> Result<()> {
        req.calls(self.tensors.len())?;
        ensure!(
            self.schema_version == 1
                && self.kind == JOB
                && self.qualification == QUALIFICATION
                && self.model_id == model_id
                && self.backend == req.backend
                && self.model_sha256 == req.model_sha256
                && self.revision_sha256 == revision
                && self.corpus_sha256 == req.corpus_sha256
                && self.observation_count == req.observations
                && [
                    &self.observations_sha256,
                    &self.executable_sha256,
                    &self.manifest_sha256
                ]
                .into_iter()
                .all(|s| is_digest(s)),
            "sensitivity artifact does not match the request"
        );
        let mut names = BTreeSet::new();
        let mut layers = BTreeSet::new();
        let mut bytes = 0;
        for t in &self.tensors {
            ensure!(
                valid_tensor(&t.name)
                    && t.layer == layer(&t.name)
                    && names.insert(&t.name)
                    && (0. ..=f64::from(f32::MAX).powi(2)).contains(&t.fisher)
                    && (req.layers.is_empty() || t.shape.len() == 2),
                "invalid sensitivity tensor result"
            );
            layers.extend(t.layer);
            bytes += t.bytes()?;
        }
        ensure!(
            bytes * self.observation_count as u64 <= MAX_TOTAL_BYTES
                && (if req.tensors.is_empty() {
                    req.layers.iter().copied().collect::<BTreeSet<_>>() == layers
                        && self.tensors.iter().all(|t| t.layer.is_some())
                } else {
                    req.tensors.iter().collect::<BTreeSet<_>>() == names
                }),
            "sensitivity tensor results do not match the selection or budget"
        );
        let expected = aggregate(&self.tensors)?;
        ensure!(
            self.layers.len() == expected.len()
                && self.layers.iter().zip(expected).all(|(a, b)| {
                    a.layer == b.layer
                        && a.elements == b.elements
                        && a.fisher >= 0.
                        && (a.fisher - b.fisher).abs() <= b.fisher * 1e-12
                }),
            "invalid sensitivity layer aggregates"
        );
        Ok(())
    }
}
