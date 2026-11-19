use crate::{converter, integrity::Identity, Input};
use anyhow::{ensure, Context, Result};
use brewery_worker::weight_source::{self, Proof};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::{
            fs::{MetadataExt, OpenOptionsExt},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::Command,
};

pub(super) const SCRATCH_PREFIX: &str = ".weight-safetensors-";
pub(super) const MEMORY_BYTES: u64 = 2 << 30;
pub(super) const DISK_RESERVE: &str = "BREWERY_WEIGHT_SAFETENSORS_DISK_RESERVE_BYTES";

pub(super) struct Source {
    pub path: PathBuf,
    // For verity, measured descriptors stay open across all helper reads.
    proof: Option<Proof>,
}

impl Source {
    pub fn prepare(input: &Input, id: &Identity, work: &Path, apply: bool) -> Result<Self> {
        let meminfo = fs::read_to_string("/proc/meminfo")?;
        let available_kib = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemAvailable:"))
            .and_then(|line| line.split_ascii_whitespace().next())
            .context("cannot determine memory available for SafeTensors worker")?
            .parse::<u64>()?;
        ensure!(
            available_kib >= 256 * 1024,
            "insufficient available memory for bounded SafeTensors helper"
        );
        ensure!(
            input
                .source_host
                .as_ref()
                .is_none_or(|host| host == &input.host),
            "SafeTensors weight jobs require a direct local source"
        );
        let proof = weight_source::proof(&input.path, &input.kind, |key| env::var_os(key))?;
        let needed = if let Some(proof) = &proof {
            let bundle = proof.bundle();
            ensure!(
                bundle.digest == id.bundle_sha256
                    && bundle.files.len() == id.files.len()
                    && bundle
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
                verified.write_manifest(&manifest)?;
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
            manifest.metadata()?.len() <= weight_source::CHUNK_INDEX_LIMIT,
            "SafeTensors helper manifest too large"
        );
        id.check()?;
        Ok(Self { path, proof })
    }

    pub fn has_proof(&self) -> bool {
        self.proof.is_some()
    }

    pub fn pin_for_sandbox(
        &self,
        cmd: &mut Command,
        id: &Identity,
        hf: bool,
    ) -> Result<&'static Path> {
        let proof = self
            .proof
            .as_ref()
            .context("missing SafeTensors source proof")?;
        let mut pins = Vec::new();
        cmd.arg("--pinned-safetensors")
            .arg(id.files.len().to_string());
        for (index, entry) in id.files.iter().enumerate() {
            let path = if hf {
                self.path.join(&entry.path)
            } else {
                self.path.clone()
            };
            let file = weight_source::open_regular(&path)?;
            id.check_open_weight_member(index, &file)?;
            if let Proof::Verity(verified) = proof {
                let meta = file.metadata()?;
                let measured = &verified.measured[index];
                ensure!(
                    (meta.dev(), meta.ino()) == (measured.device, measured.inode),
                    "SafeTensors member differs from measured inode"
                );
            }
            cmd.arg(file.as_raw_fd().to_string()).arg(&entry.path);
            pins.push(file);
        }
        id.check()?;
        cmd.arg("--");
        // Only these read-only pins leave CLOEXEC, only in the helper child.
        // Bubblewrap consumes and closes them before starting the editor.
        unsafe {
            cmd.pre_exec(move || {
                for file in &pins {
                    if libc::fcntl(file.as_raw_fd(), libc::F_SETFD, 0) < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        Ok(Path::new(if hf {
            "/weight-source/source.hf"
        } else {
            "/weight-source/source.safetensors"
        }))
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
