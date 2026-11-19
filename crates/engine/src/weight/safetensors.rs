use crate::{converter, integrity::Identity, Input};
use anyhow::{ensure, Context, Result};
use brewery_worker::weight_source::{self, Verified};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

pub(super) const SCRATCH_PREFIX: &str = ".weight-safetensors-";
pub(super) const MEMORY_BYTES: u64 = 2 << 30;
pub(super) const DISK_RESERVE: &str = "BREWERY_WEIGHT_SAFETENSORS_DISK_RESERVE_BYTES";

pub(super) struct Source {
    pub path: PathBuf,
    // The measured descriptors stay open across all helper reads.
    _proof: Option<Verified>,
}

impl Source {
    pub fn prepare(input: &Input, id: &Identity, work: &Path, apply: bool) -> Result<Self> {
        ensure!(
            input
                .source_host
                .as_ref()
                .is_none_or(|host| host == &input.host),
            "SafeTensors weight jobs require a direct local source"
        );
        let proof = weight_source::configured_publication(&input.path, &input.kind)?;
        let needed = if let Some(proof) = &proof {
            ensure!(
                proof.bundle.digest == id.bundle_sha256
                    && proof.bundle.files.len() == id.files.len()
                    && proof
                        .bundle
                        .files
                        .iter()
                        .zip(&id.files)
                        .all(|(a, b)| a.path == b.path
                            && a.bytes == b.bytes
                            && a.sha256 == b.sha256),
                "SafeTensors publication differs from resolved source manifest"
            );
            if apply {
                id.bytes
            } else {
                0
            }
        } else {
            let limit =
                weight_source::verify_limit(env::var(weight_source::VERIFY_BYTES).ok().as_deref())?;
            ensure!(
                id.bytes <= limit,
                "SafeTensors source exceeds explicit verification limit"
            );
            id.bytes
                .checked_mul(if apply { 2 } else { 1 })
                .context("weight disk size overflow")?
        };
        if needed != 0 {
            let reserve = env::var(DISK_RESERVE).map_or(Ok(0), |v| v.parse::<u64>())?;
            let preflight = converter::disk_preflight(
                work,
                needed
                    .checked_add(reserve)
                    .context("weight disk size overflow")?,
            )?;
            ensure!(
                preflight["fits"] == true,
                "insufficient disk for private SafeTensors output/snapshot and reserve"
            );
        }
        let manifest = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(work.join("source-manifest.json"))?;
        let path = match &proof {
            Some(verified) => {
                serde_json::to_writer(&manifest, &verified.measured)?;
                input.path.clone()
            }
            None => {
                let hf = input.kind == "hf";
                let path = work.join(if hf {
                    "source.hf"
                } else {
                    "source.safetensors"
                });
                if hf {
                    fs::create_dir(&path)?;
                }
                for member in &id.files {
                    let (from, to) = if hf {
                        (input.path.join(&member.path), path.join(&member.path))
                    } else {
                        (input.path.clone(), path.clone())
                    };
                    let mut source = weight_source::open_regular(&from)?;
                    ensure!(
                        source.metadata()?.len() == member.bytes,
                        "SafeTensors source size changed"
                    );
                    let mut output = fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .mode(0o400)
                        .open(to)?;
                    let mut hash = Sha256::new();
                    let mut remaining = member.bytes;
                    let mut chunk = vec![0; 1 << 20];
                    while remaining != 0 {
                        let length = remaining.min(chunk.len() as u64) as usize;
                        source.read_exact(&mut chunk[..length])?;
                        hash.update(&chunk[..length]);
                        output.write_all(&chunk[..length])?;
                        remaining -= length as u64;
                    }
                    ensure!(
                        source.read(&mut chunk[..1])? == 0
                            && format!("{:x}", hash.finalize()) == member.sha256,
                        "SafeTensors source bytes differ from verified manifest"
                    );
                }
                serde_json::to_writer(&manifest, &id.files)?;
                path
            }
        };
        ensure!(
            manifest.metadata()?.len() <= weight_source::LIMIT,
            "SafeTensors helper manifest too large"
        );
        id.check()?;
        Ok(Self {
            path,
            _proof: proof,
        })
    }
}

pub(super) fn check_output(
    path: &Path,
    kind: &str,
    source: &Identity,
    output: &Identity,
) -> Result<()> {
    weight_source::inventory(path, kind)?;
    ensure!(
        source.bytes == output.bytes
            && source.files.len() == output.files.len()
            && (kind != "hf"
                || source.files.iter().zip(&output.files).all(|(a, b)| {
                    a.path == b.path
                        && a.bytes == b.bytes
                        && (a.path.ends_with(".safetensors") || a.sha256 == b.sha256)
                })),
        "edited SafeTensors output did not preserve complete source membership and assets"
    );
    Ok(())
}
