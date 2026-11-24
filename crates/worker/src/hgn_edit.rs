//! Controller-scoped grants for same-host private native HGN jobs.
use crate::{Input, Job, is_digest, token, weight_source::open_regular};
use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::{
    io::Read,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

pub use crate::weight_source::scratch::hgn_ablation::{MODEL, REPORT, Report};
pub const ENCODER_SHA256: &str = bf16_to_hgn::ENCODER_SHA256;
pub const JOB: &str = "scratch_hgn_ablation";
pub const GRANT: &str = "hgn_edit_grant";
pub const PRIVATE: &str = "brewery-private-hgn-scratch";
pub const KEY_FILE: &str = "BREWERY_HGN_EDIT_KEY_FILE";
pub const CACHE: &str = "BREWERY_HGN_EDIT_CACHE";
pub const RESERVE: &str = "BREWERY_HGN_EDIT_RESERVE_BYTES";

pub fn is_hgn(kind: &str) -> bool {
    matches!(kind, "hgn" | "hgn-v2")
}

pub fn private_path(path: &Path) -> bool {
    path.components().any(|c| c.as_os_str() == PRIVATE)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub parent_sha256: String,
    pub producer: Option<String>,
    pub first_layer: usize,
    pub last_layer: usize,
    pub direction: Vec<f32>,
}

impl Request {
    pub fn check(&self) -> Result<()> {
        ensure!(is_digest(&self.parent_sha256), "invalid HGN parent SHA-256");
        ensure!(
            self.first_layer <= self.last_layer
                && self.last_layer < 4096
                && self.last_layer - self.first_layer < 64,
            "invalid HGN layer range"
        );
        ensure!(
            (1..=16384).contains(&self.direction.len())
                && self.direction.iter().all(|v| v.is_finite())
                && self.direction.iter().any(|&v| v != 0.0),
            "invalid HGN direction"
        );
        if let Some(id) = &self.producer {
            token(id)?;
        }
        Ok(())
    }

    pub fn verify(&self, report: &Report) -> Result<()> {
        let norm = self
            .direction
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        ensure!(
            report.schema == "brewery.hgn-feedback.v1"
                && report.encoder_sha256 == ENCODER_SHA256
                && report.parent_sha256 == self.parent_sha256
                && is_digest(&report.revision_sha256)
                && report.revision_sha256 != self.parent_sha256
                && report.direction.len() == self.direction.len()
                && report
                    .direction
                    .iter()
                    .zip(&self.direction)
                    .all(|(a, b)| { (a - f64::from(*b) / norm).abs() < 1e-12 })
                && (1..=256).contains(&report.tensors.len()),
            "HGN edit receipt does not match the request"
        );
        for row in report.tensors.values() {
            let tol = match (row.store, row.variant) {
                (0, 0) => 0.01,
                (5, 2) | (7, 0) | (23, 128) => 0.05,
                (16, 0x1206 | 0x1208) => 0.08,
                _ => anyhow::bail!("invalid HGN edit receipt store"),
            };
            ensure!(
                row.tolerance == tol
                    && row.bytes > 0
                    && row.iterations <= 32
                    && matches!(row.shape.len(), 2 | 3)
                    && row.shape[row.shape.len() - 2] == self.direction.len() as u64
                    && (0.0..=tol).contains(&row.leftover_after)
                    && [
                        row.leftover_before,
                        row.parent_mean_abs_component,
                        row.rel_error
                    ]
                    .into_iter()
                    .all(|x| x.is_finite() && x >= 0.0)
                    && is_digest(&row.payload_sha256),
                "invalid HGN tensor receipt"
            );
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub host: String,
    pub grant_key_file: PathBuf,
    pub build_admission: PathBuf,
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

pub fn read_key(path: &Path) -> Result<[u8; 32]> {
    ensure!(
        path.is_absolute() && path.canonicalize()? == path,
        "invalid HGN edit key path"
    );
    let mut file = open_regular(path)?;
    let meta = file.metadata()?;
    ensure!(
        meta.uid() == unsafe { libc::geteuid() }
            && meta.nlink() == 1
            && meta.mode() & 0o077 == 0
            && meta.len() == 32,
        "invalid HGN edit key file"
    );
    let mut key = [0; 32];
    file.read_exact(&mut key)?;
    Ok(key)
}

pub fn grant(id: &str, input: &Input, request: Request, key: &[u8; 32]) -> Result<String> {
    let claim = Claim {
        job_id: id.into(),
        host: input.host.clone(),
        input_id: input.extra["id"]
            .as_str()
            .context("missing HGN model ID")?
            .into(),
        path: input.path.clone(),
        kind: input.kind.clone(),
        request,
    };
    let json = serde_json::to_vec(&claim)?;
    let mut mac = Hmac::<Sha256>::new_from_slice(key)?;
    mac.update(b"brewery.hgn-edit.v1\0");
    mac.update(&json);
    Ok(format!(
        "{}.{}",
        URL_SAFE_NO_PAD.encode(&json),
        URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes())
    ))
}

impl Config {
    pub fn authorize(&self, id: &str, action: &str, job: &Job) -> Result<Request> {
        let [input] = job.inputs.as_slice() else {
            anyhow::bail!("expected one direct HGN input");
        };
        ensure!(
            action == "start"
                && job.kind == JOB
                && job.prompt.is_none()
                && job.resources.is_empty()
                && input.host == self.host
                && is_hgn(&input.kind),
            "expected same-host private HGN edit"
        );
        let token = job
            .options
            .get(GRANT)
            .and_then(|x| x.as_str())
            .filter(|t| t.len() <= 600_000)
            .context("missing controller HGN edit grant")?;
        ensure!(
            job.options.as_object().is_some_and(|v| v.len() == 1),
            "unexpected HGN options"
        );
        let (bytes, sig) = token.split_once('.').context("invalid HGN edit grant")?;
        let json = URL_SAFE_NO_PAD.decode(bytes)?;
        let mut mac = Hmac::<Sha256>::new_from_slice(&read_key(&self.grant_key_file)?)?;
        mac.update(b"brewery.hgn-edit.v1\0");
        mac.update(&json);
        mac.verify_slice(&URL_SAFE_NO_PAD.decode(sig)?)
            .map_err(|_| anyhow::anyhow!("invalid controller HGN edit grant"))?;
        let claim: Claim = serde_json::from_slice(&json)?;
        ensure!(
            claim.job_id == id
                && claim.host == input.host
                && claim.kind == input.kind
                && claim.path == input.path
                && input.extra["id"] == claim.input_id,
            "HGN edit grant does not match the job"
        );
        claim.request.check()?;
        Ok(claim.request)
    }
}
