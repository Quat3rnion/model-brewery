//! Protocol for the private HGN stored-file checks job.
use crate::{Input, Job, hgn_edit, is_digest, token, trace};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

pub use crate::weight_source::scratch::hgn_ablation::audit;

pub const JOB: &str = "scratch_hgn_checks";
pub const GRANT: &str = "hgn_check_grant";
const DOMAIN: &[u8] = b"brewery.hgn-checks.v1\0";
pub const SCHEMA: &str = "brewery.hgn-stored-checks.v1";
pub const SEED: u32 = 26842;
pub const CONTROLS: usize = 5;
pub const PROMPTS: [&str; 2] = ["The capital of France is", "Two plus two equals"];

pub(crate) fn gate(
    leftover: f64,
    max_expert_leftover: Option<f64>,
    control_shifts: &[f64; CONTROLS],
) -> bool {
    (0.0..=0.10).contains(&leftover)
        && max_expert_leftover.is_none_or(|v| (0.0..=0.10).contains(&v))
        && control_shifts.iter().all(|v| (0.0..=0.05).contains(v))
}

pub fn agrees(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6 * (1.0 + b.abs())
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub parent_sha256: String,
    pub revision_sha256: String,
    pub root_sha256: String,
    pub parent_producer: Option<String>,
    pub child_producer: String,
    pub first_layer: usize,
    pub last_layer: usize,
}

impl Request {
    pub fn verify(&self) -> Result<()> {
        token(&self.child_producer)?;
        if let Some(id) = &self.parent_producer {
            token(id)?;
        }
        ensure!(
            [
                &self.parent_sha256,
                &self.revision_sha256,
                &self.root_sha256
            ]
            .into_iter()
            .all(|s| is_digest(s))
                && self.parent_sha256 != self.revision_sha256
                && self.parent_producer.as_ref() != Some(&self.child_producer)
                && (self.parent_producer.is_some() || self.root_sha256 == self.parent_sha256)
                && self.first_layer <= self.last_layer
                && self.last_layer < 4096
                && self.last_layer - self.first_layer < 64,
            "invalid stored HGN check pins"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Claim {
    job_id: String,
    host: String,
    input_id: String,
    path: PathBuf,
    kind: String,
    request: Request,
}

pub fn grant(id: &str, input: &Input, req: Request, key: &[u8; 32]) -> Result<String> {
    req.verify()?;
    hgn_edit::sign(
        DOMAIN,
        &Claim {
            job_id: id.into(),
            host: input.host.clone(),
            input_id: input.extra["id"]
                .as_str()
                .context("missing HGN root ID")?
                .into(),
            path: input.path.clone(),
            kind: input.kind.clone(),
            request: req,
        },
        key,
    )
}

pub fn authorize(config: &hgn_edit::Config, id: &str, action: &str, job: &Job) -> Result<Request> {
    let [input] = job.inputs.as_slice() else {
        anyhow::bail!("expected one direct HGN root input");
    };
    ensure!(
        action == "start"
            && job.kind == JOB
            && job.prompt.is_none()
            && job.resources.is_empty()
            && input.host == config.host
            && hgn_edit::is_hgn(&input.kind)
            && job.options.as_object().is_some_and(|v| v.len() == 1),
        "expected same-host private HGN check"
    );
    let grant = job.options[GRANT]
        .as_str()
        .filter(|v| v.len() <= 8192)
        .context("missing controller HGN check grant")?;
    let claim: Claim = hgn_edit::verify_grant(DOMAIN, grant, &config.grant_key_file)?;
    ensure!(
        claim.job_id == id
            && claim.host == input.host
            && claim.path == input.path
            && claim.kind == input.kind
            && input.extra["id"] == claim.input_id,
        "HGN check grant does not match the job"
    );
    claim.request.verify()?;
    Ok(claim.request)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EngineOptions {
    pub request: Request,
    pub feedback_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorAudit {
    pub leftover: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_expert_leftover: Option<f64>,
    pub parent_mean_abs_component: f64,
    pub rel_error: f64,
    /// Absolute relative movement in the mean magnitude along each control.
    pub control_shifts: [f64; CONTROLS],
    pub passed: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptScores {
    pub tokens: usize,
    pub kl_parent_child: f64,
    pub top1_agrees: bool,
    pub mean_abs_logit_delta: f64,
    pub max_abs_logit_delta: f64,
    pub parent_top1_logit_delta: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub schema: String,
    pub parent_sha256: String,
    pub revision_sha256: String,
    pub root_sha256: String,
    pub seed: u32,
    pub tensors: BTreeMap<String, TensorAudit>,
    pub prompts: [PromptScores; PROMPTS.len()],
    pub runtime: String,
    pub runtime_sha256: String,
    pub executable_sha256: String,
    pub assets_sha256: BTreeMap<String, String>,
}

impl Report {
    pub fn verify(&self, req: &Request, feedback: &hgn_edit::Report) -> Result<()> {
        req.verify()?;
        ensure!(
            self.schema == SCHEMA
                && self.seed == SEED
                && self.parent_sha256 == req.parent_sha256
                && self.revision_sha256 == req.revision_sha256
                && self.root_sha256 == req.root_sha256
                && self.runtime == trace::HGN_RUNTIME
                && self.runtime_sha256 == trace::HGN_RUNTIME_SHA256.trim_end()
                && is_digest(&self.executable_sha256)
                && self.assets_sha256.len() == 3
                && ["sidecar", "config", "tokenizer"]
                    .iter()
                    .all(|k| self.assets_sha256.get(*k).is_some_and(|v| is_digest(v)))
                && !self.tensors.is_empty()
                && self.tensors.len() == feedback.tensors.len(),
            "stored HGN checks do not match the pinned inputs"
        );
        for (name, row) in &self.tensors {
            let expected = feedback
                .tensors
                .get(name)
                .context("unknown HGN audit tensor")?;
            ensure!(
                agrees(row.leftover, expected.leftover_after)
                    && agrees(row.rel_error, expected.rel_error)
                    && agrees(
                        row.parent_mean_abs_component,
                        expected.parent_mean_abs_component
                    )
                    && [row.leftover, row.rel_error, row.parent_mean_abs_component]
                        .iter()
                        .chain(&row.control_shifts)
                        .all(|v| v.is_finite() && *v >= 0.)
                    && row.max_expert_leftover.is_some() == (expected.shape.len() == 3)
                    && row
                        .max_expert_leftover
                        .is_none_or(|v| v.is_finite() && v >= 0. && v + 1e-9 >= row.leftover)
                    && row.passed
                        == gate(row.leftover, row.max_expert_leftover, &row.control_shifts),
                "invalid decoded HGN audit metrics"
            );
        }
        for row in &self.prompts {
            ensure!(
                (1..=512).contains(&row.tokens)
                    && [
                        row.kl_parent_child,
                        row.mean_abs_logit_delta,
                        row.max_abs_logit_delta
                    ]
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.)
                    && row.mean_abs_logit_delta <= row.max_abs_logit_delta
                    && row.parent_top1_logit_delta.abs() <= row.max_abs_logit_delta,
                "invalid HGN CPU forward metrics"
            );
        }
        Ok(())
    }
}
