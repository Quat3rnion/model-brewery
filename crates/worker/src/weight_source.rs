//! Local managed SafeTensors source policy. This is not the qualification pin.
use crate::{Bundle, Entry, absolute, decode_limited, is_digest, relative, tree_digest};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
};

pub const GATE: &str = "BREWERY_WEIGHT_SAFETENSORS";
pub const PUBLICATION: &str = "BREWERY_WEIGHT_SAFETENSORS_PUBLICATION";
pub const VERIFY_BYTES: &str = "BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES";
pub const INDEX: &str = "model.safetensors.index.json";
pub const LIMIT: u64 = 4 << 20;
pub const MAX_BYTES: u64 = 2 << 40;

pub fn supports(kind: &str) -> bool {
    matches!(kind, "hf" | "safetensors")
}

pub fn job(kind: &str) -> bool {
    matches!(
        kind,
        "weight_inspect" | "weight_propose" | "weight_tune" | "weight_apply"
    )
}

pub fn verify_limit(value: Option<&str>) -> Result<u64> {
    let limit: u64 = value
        .context("SafeTensors requires an immutable publication or explicit bounded verification")?
        .parse()?;
    ensure!(
        (1..=64 << 20).contains(&limit),
        "SafeTensors verification limit must be 1–67108864 bytes"
    );
    Ok(limit)
}

pub fn inventory(path: &Path, kind: &str) -> Result<Vec<(String, fs::Metadata)>> {
    absolute(path)?;
    ensure!(
        path.canonicalize()? == path,
        "noncanonical SafeTensors path"
    );
    let mut members = Vec::new();
    if kind == "hf" {
        ensure!(
            path.is_dir(),
            "HF weight source must be an indexed directory"
        );
        for child in fs::read_dir(path)? {
            ensure!(members.len() < 4096, "too many HF bundle members");
            let child = child?;
            let name = child
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("invalid HF member name"))?;
            relative(&name)?;
            let meta = fs::symlink_metadata(child.path())?;
            ensure!(
                meta.is_file() && meta.nlink() == 1,
                "HF requires flat regular files without links"
            );
            members.push((name, meta));
        }
        ensure!(
            members.iter().any(|(name, _)| name == INDEX)
                && members
                    .iter()
                    .any(|(name, _)| name.ends_with(".safetensors")),
            "HF requires model.safetensors.index.json and SafeTensors shards"
        );
        members.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    } else {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("invalid SafeTensors filename")?;
        relative(name)?;
        let meta = fs::symlink_metadata(path)?;
        ensure!(
            name.ends_with(".safetensors") && meta.is_file() && meta.nlink() == 1,
            "SafeTensors source must be one regular .safetensors file without links"
        );
        members.push((name.into(), meta));
    }
    Ok(members)
}

pub fn bounded_inventory(path: &Path, kind: &str, limit: u64) -> Result<()> {
    let members = inventory(path, kind)?;
    let total = members
        .iter()
        .try_fold(0u64, |n, (_, meta)| n.checked_add(meta.len()))
        .context("bundle size overflow")?;
    ensure!(
        total <= limit,
        "SafeTensors source exceeds explicit verification limit"
    );
    Ok(())
}

// This config file is an operator trust root: the publisher must compute the
// canonical file SHA-256 and the kernel's fs-verity digest of the SAME bytes.
// A library request cannot name this file or supply verity measurements.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Publication {
    schema: String,
    source: PathBuf,
    bundle_sha256: String,
    files: Vec<PublishedFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublishedFile {
    path: String,
    bytes: u64,
    sha256: String,
    fsverity_sha256: String,
}

#[derive(Serialize)]
pub struct MeasuredFile {
    #[serde(flatten)]
    pub entry: Entry,
    pub device: u64,
    pub inode: u64,
}

pub struct Verified {
    pub bundle: Bundle,
    pub measured: Vec<MeasuredFile>,
    // Keep the measured inode live until the helper exits. The helper compares
    // its open fd's device and inode, not path metadata.
    _files: Vec<fs::File>,
}

pub fn publication(path: &Path, kind: &str, receipt: &Path) -> Result<Option<Verified>> {
    absolute(receipt)?;
    ensure!(
        receipt.canonicalize()? == receipt,
        "noncanonical SafeTensors publication receipt"
    );
    let file = open_regular(receipt)?;
    let before = file.metadata()?;
    ensure!(
        before.mode() & 0o022 == 0
            && (before.uid() == 0 || before.uid() == unsafe { libc::geteuid() }),
        "untrusted SafeTensors publication receipt permissions"
    );
    let pin: Publication = decode_limited(&file, LIMIT)?;
    let after = file.metadata()?;
    ensure!(
        before.ctime() == after.ctime() && before.ctime_nsec() == after.ctime_nsec(),
        "SafeTensors publication receipt changed during read"
    );
    ensure!(
        pin.schema == "brewery.weight-publication.v1",
        "invalid SafeTensors publication schema"
    );
    if pin.source != path {
        return Ok(None);
    }
    let members = inventory(path, kind)?;
    ensure!(
        members.len() == pin.files.len(),
        "SafeTensors publication membership changed"
    );
    let mut files = Vec::with_capacity(members.len());
    let mut measured = Vec::with_capacity(members.len());
    for ((name, meta), member) in members.into_iter().zip(pin.files) {
        ensure!(
            name == member.path && meta.len() == member.bytes && is_digest(&member.fsverity_sha256),
            "invalid SafeTensors publication member"
        );
        let physical = if kind == "hf" {
            path.join(&name)
        } else {
            path.to_owned()
        };
        let fd = open_regular(&physical)?;
        let opened = fd.metadata()?;
        ensure!(
            opened.dev() == meta.dev()
                && opened.ino() == meta.ino()
                && opened.len() == member.bytes,
            "SafeTensors source changed during measurement"
        );
        let actual = measure(&fd)?;
        ensure!(
            actual == member.fsverity_sha256,
            "SafeTensors fs-verity digest mismatch"
        );
        measured.push(MeasuredFile {
            entry: Entry {
                path: name,
                bytes: member.bytes,
                sha256: member.sha256,
            },
            device: opened.dev(),
            inode: opened.ino(),
        });
        files.push(fd);
    }
    let bundle = Bundle {
        entry: if kind == "hf" {
            ".".into()
        } else {
            measured[0].entry.path.clone()
        },
        digest: pin.bundle_sha256,
        files: measured.iter().map(|m| m.entry.clone()).collect(),
    };
    bundle.validate()?;
    ensure!(
        bundle.bytes()? <= MAX_BYTES,
        "SafeTensors publication exceeds model limit"
    );
    ensure!(
        bundle.digest == tree_digest(&bundle.files),
        "incorrect SafeTensors publication digest"
    );
    Ok(Some(Verified {
        bundle,
        measured,
        _files: files,
    }))
}

pub fn configured_publication(path: &Path, kind: &str) -> Result<Option<Verified>> {
    match env::var_os(PUBLICATION) {
        Some(receipt) => publication(path, kind, Path::new(&receipt)),
        None => Ok(None),
    }
}

pub fn open_regular(path: &Path) -> Result<fs::File> {
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let meta = file.metadata()?;
    ensure!(
        meta.is_file() && meta.nlink() == 1,
        "expected a regular file without links"
    );
    Ok(file)
}

fn measure(fd: &fs::File) -> Result<String> {
    #[repr(C)]
    struct Measurement {
        algorithm: u16,
        size: u16,
        digest: [u8; 32],
    }
    // FS_IOC_MEASURE_VERITY = _IOWR('f', 134, struct fsverity_digest)
    // with the four-byte header; the trailing digest is a flexible array.
    #[cfg(not(target_arch = "x86_64"))]
    anyhow::bail!("SafeTensors fs-verity measurement is unsupported on this architecture");
    #[cfg(target_arch = "x86_64")]
    {
        let mut output = Measurement {
            algorithm: 0,
            size: 32,
            digest: [0; 32],
        };
        if unsafe { libc::ioctl(fd.as_raw_fd(), 0xc004_6686, &mut output) } < 0 {
            return Err(std::io::Error::last_os_error())
                .context("SafeTensors immutable fs-verity proof unavailable");
        }
        ensure!(
            output.algorithm == 1 && output.size == 32,
            "SafeTensors requires SHA-256 fs-verity"
        );
        Ok(output.digest.iter().map(|b| format!("{b:02x}")).collect())
    }
}
