#![cfg(target_os = "linux")]

mod runtime;
pub use runtime::{prove, Proof};
mod generation;
pub use generation::{
    generate, generate_observed, qualify_generation, qualify_generation_observed, Generation,
    GenerationBundle,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Component, Path, PathBuf},
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    HgnV2,
    Ht43,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
#[serde(rename_all = "kebab-case")]
enum Role {
    Checkpoint,
    Lookup,
    Vision,
    Draft,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: Format,
    files: Vec<ManifestFile>,
    #[serde(default)]
    source_sha256: Option<String>,
    #[serde(default)]
    tokenizer: Vec<generation::Asset>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestFile {
    role: Role,
    path: PathBuf,
    sha256: String,
}

#[derive(Debug, Serialize)]
pub struct Qualification {
    schema: &'static str,
    format: Format,
    files: Vec<QualifiedFile>,
}

#[derive(Debug, Serialize)]
struct QualifiedFile {
    role: Role,
    path: PathBuf,
    sha256: String,
    hgn_model_id: String,
    identity: FileIdentity,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct FileIdentity {
    dev_major: u32,
    dev_minor: u32,
    inode: u64,
    bytes: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl FileIdentity {
    fn read(meta: &fs::Metadata) -> Result<Self> {
        if !meta.is_file() {
            return Err("bundle members must be regular files".into());
        }
        Ok(Self {
            dev_major: libc::major(meta.dev()),
            dev_minor: libc::minor(meta.dev()),
            inode: meta.ino(),
            bytes: meta.len(),
            modified: (meta.mtime(), meta.mtime_nsec()),
            changed: (meta.ctime(), meta.ctime_nsec()),
        })
    }
}

impl QualifiedFile {
    fn check_unchanged(&self) -> Result<()> {
        if self.path.canonicalize()? != self.path
            || FileIdentity::read(&fs::symlink_metadata(&self.path)?)? != self.identity
        {
            return Err(format!("bundle member changed: {}", self.path.display()).into());
        }
        Ok(())
    }
}

/// Check a manifest's exact file identities and HGN headers without using a runtime.
/// This does not validate every tensor or prove that an engine can load the bundle.
pub fn qualify(manifest: &Path) -> Result<Qualification> {
    let manifest = manifest.canonicalize()?;
    let root = manifest
        .parent()
        .ok_or("manifest has no parent directory")?;
    let mut json = Vec::new();
    File::open(&manifest)?.take(65537).read_to_end(&mut json)?;
    if json.len() > 65536 {
        return Err("bundle manifest exceeds 64 KiB".into());
    }
    qualify_members(root, serde_json::from_slice(&json)?, &mut |_, _| {})
}

fn qualify_members(
    root: &Path,
    manifest: Manifest,
    observe: &mut dyn FnMut(&Path, u64),
) -> Result<Qualification> {
    let Manifest { format, files, .. } = manifest;
    let mut roles = BTreeSet::new();
    if !files.iter().all(|f| roles.insert(f.role)) || !roles.contains(&Role::Checkpoint) {
        return Err("bundle needs exactly one checkpoint and no repeated roles".into());
    }

    let mut identities = BTreeSet::new();
    let mut qualified = Vec::with_capacity(files.len());
    for member in files {
        if member.path.as_os_str().is_empty()
            || member
                .path
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err("bundle member paths must be relative with no traversal".into());
        }
        if !is_sha256(&member.sha256) {
            return Err(format!("invalid SHA-256 for {}", member.path.display()).into());
        }
        let path = root.join(&member.path);
        if path.canonicalize()? != path {
            return Err(format!("bundle member path is not canonical: {}", path.display()).into());
        }
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)?;
        let identity = FileIdentity::read(&file.metadata()?)?;
        if !identities.insert((identity.dev_major, identity.dev_minor, identity.inode)) {
            return Err("two bundle roles refer to the same file".into());
        }
        let hgn_model_id = check_header(&mut file, identity.bytes, format, member.role)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        file.seek(SeekFrom::Start(0))?;
        observe(&path, 0);
        let mut hash = Sha256::new();
        let mut bytes = 0;
        let mut buf = vec![0; 1 << 20];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
            bytes += n as u64;
            observe(&path, n as u64);
        }
        if bytes != identity.bytes || FileIdentity::read(&file.metadata()?)? != identity {
            return Err(format!("bundle member changed while reading: {}", path.display()).into());
        }
        if format!("{:x}", hash.finalize()) != member.sha256 {
            return Err(format!("SHA-256 mismatch: {}", path.display()).into());
        }
        let checked = QualifiedFile {
            role: member.role,
            path,
            sha256: member.sha256,
            hgn_model_id,
            identity,
        };
        checked.check_unchanged()?;
        qualified.push(checked);
    }
    Ok(Qualification {
        schema: "halogen-hgn-preflight-v1",
        format,
        files: qualified,
    })
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn check_header(file: &mut File, bytes: u64, format: Format, role: Role) -> Result<String> {
    let mut header = [0; 104];
    file.read_exact(&mut header)?;
    if &header[..8] != b"HGN1\x02\x00\x00\x00" {
        return Err("expected HGN1 magic with little-endian schema version 2".into());
    }
    let tensors = u32::from_le_bytes(header[8..12].try_into()?);
    let dir_offset = u64::from_le_bytes(header[16..24].try_into()?);
    let data_offset = u64::from_le_bytes(header[24..32].try_into()?);
    let declared_size = u64::from_le_bytes(header[32..40].try_into()?);
    if tensors == 0
        || dir_offset != 104
        || data_offset < dir_offset + u64::from(tensors) * 160
        || data_offset >= bytes
        || declared_size != bytes
    {
        return Err("invalid HGN header bounds or file size".into());
    }
    let id = header[40..]
        .split(|b| *b == 0)
        .next()
        .ok_or("missing HGN model id")?;
    let id = std::str::from_utf8(id)?;
    let expected = match (role, format) {
        (Role::Checkpoint, Format::HgnV2) => "qwen3.8-flash-next-v2",
        (Role::Checkpoint, Format::Ht43) => "qwen3.8-flash-next-ht43",
        (Role::Lookup, _) => "qwen3.8-flash-next-ngram",
        (Role::Vision, _) => "qwen3.8-flash-next-vision",
        // The external native drafter in 0.20.6 has a separate HGN header.
        // Do not guess its model id: this role is not supported until verified.
        (Role::Draft, _) => {
            return Err("external draft HGN is not supported; use the embedded head".into())
        }
    };
    if id != expected {
        return Err(format!("HGN model id {id:?} does not match {expected:?}").into());
    }
    Ok(id.to_owned())
}
