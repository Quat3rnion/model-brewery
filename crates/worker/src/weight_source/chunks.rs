//! Trust-on-operator-pin chunk witnesses for files without kernel verity.
use super::{MAX_BYTES, inventory, open_regular};
use crate::{Bundle, Entry, absolute, decode_limited, is_digest, tree_digest};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

pub const CHUNK_INDEX: &str = "BREWERY_WEIGHT_CHUNK_INDEX";
pub const CHUNK_INDEX_SHA256: &str = "BREWERY_WEIGHT_CHUNK_INDEX_SHA256";
pub const CHUNK_INDEX_LIMIT: u64 = 64 << 20;
const CHUNK: usize = 1 << 20;
const SCHEMA: &str = "brewery.weight-chunk-index.v1";

#[derive(Deserialize)]
struct Published {
    schema: String,
    model_dir: PathBuf,
    files: BTreeMap<String, Member>,
}

#[derive(Deserialize)]
struct Member {
    size: u64,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: String,
    source: PathBuf,
    bundle_sha256: String,
    files: Vec<Witness>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Witness {
    path: String,
    bytes: u64,
    sha256: String,
    chunk_sha256: Vec<String>,
}

pub struct Index {
    pub bundle: Bundle,
    pub(super) files: Vec<Witness>,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hf_bundle(files: Vec<Entry>, digest: String) -> Result<Bundle> {
    let bundle = Bundle {
        entry: ".".into(),
        digest,
        files,
    };
    bundle.validate()?;
    ensure!(
        bundle.bytes()? <= MAX_BYTES,
        "chunk index source exceeds model limit"
    );
    Ok(bundle)
}

fn members(path: &Path, bundle: &Bundle) -> Result<Vec<(String, fs::Metadata)>> {
    let members = inventory(path, "hf")?;
    ensure!(
        members.len() == bundle.files.len()
            && members
                .iter()
                .zip(&bundle.files)
                .all(|((n, m), f)| *n == f.path && m.len() == f.bytes),
        "chunk index source differs from pinned membership or size"
    );
    Ok(members)
}

pub(super) fn verified(
    path: &Path,
    kind: &str,
    index: &Path,
    expected: &str,
) -> Result<Option<Index>> {
    ensure!(is_digest(expected), "invalid chunk index SHA-256 pin");
    let file = open_regular(index)?;
    ensure!(
        file.metadata()?.len() <= CHUNK_INDEX_LIMIT,
        "chunk index exceeds byte limit"
    );
    let mut bytes = vec![];
    file.take(CHUNK_INDEX_LIMIT + 1).read_to_end(&mut bytes)?;
    // Parse exactly the bytes we authenticated. Source metadata, ownership, and
    // matching stats are not proof of content.
    ensure!(
        bytes.len() as u64 <= CHUNK_INDEX_LIMIT && sha(&bytes) == expected,
        "SafeTensors chunk index SHA-256 mismatch"
    );
    let doc: Document = serde_json::from_slice(&bytes)?;
    ensure!(doc.schema == SCHEMA, "unsupported chunk index schema");
    if doc.source != path || kind != "hf" {
        return Ok(None);
    }
    let bundle = hf_bundle(
        doc.files
            .iter()
            .map(|f| Entry {
                path: f.path.clone(),
                bytes: f.bytes,
                sha256: f.sha256.clone(),
            })
            .collect(),
        doc.bundle_sha256,
    )?;
    members(path, &bundle)?;
    ensure!(
        doc.files.iter().all(|f| {
            f.chunk_sha256.len() as u64 == f.bytes.div_ceil(CHUNK as u64)
                && f.chunk_sha256.iter().all(|h| is_digest(h))
        }),
        "invalid SafeTensors chunk witnesses"
    );
    Ok(Some(Index {
        bundle,
        files: doc.files,
    }))
}

/// Explicit offline operation. The independent tree pin authenticates the
/// whole-file hashes; each witness is derived from the same stream.
pub fn build_chunk_index(receipt: &Path, expected_tree: &str, output: &Path) -> Result<String> {
    absolute(output)?;
    let published: Published = decode_limited(open_regular(receipt)?, 128 << 10)?;
    ensure!(
        published.schema == "model-brewery.source-publication.v1",
        "invalid source publication receipt"
    );
    let source = published.model_dir;
    let bundle = hf_bundle(
        published
            .files
            .into_iter()
            .map(|(path, m)| Entry {
                path,
                bytes: m.size,
                sha256: m.sha256,
            })
            .collect(),
        expected_tree.into(),
    )?;
    ensure!(
        tree_digest(&bundle.files) == expected_tree,
        "source publication tree SHA-256 mismatch"
    );
    let members = members(&source, &bundle)?;
    let parent = output
        .parent()
        .context("chunk index output needs a parent directory")?;
    let parent_meta = fs::symlink_metadata(parent)?;
    ensure!(
        parent.canonicalize()? == parent
            && parent_meta.is_dir()
            && parent_meta.uid() == unsafe { libc::geteuid() }
            && parent_meta.mode() & 0o077 == 0
            && !output.starts_with(&source),
        "chunk index output must be in a separate private directory outside the source"
    );
    ensure!(
        matches!(fs::symlink_metadata(output), Err(e) if e.kind() == std::io::ErrorKind::NotFound),
        "chunk index output already exists"
    );
    // Reject an oversized witness table before reading any tensor payload.
    let chunk_bytes: u64 = bundle
        .files
        .iter()
        .map(|f| 67 * f.bytes.div_ceil(CHUNK as u64))
        .sum();
    ensure!(
        chunk_bytes <= CHUNK_INDEX_LIMIT,
        "chunk index would exceed byte limit"
    );
    let mut witnesses = Vec::with_capacity(members.len());
    let mut buffer = vec![0; CHUNK];
    for ((name, expected), entry) in members.into_iter().zip(&bundle.files) {
        let path = source.join(&name);
        let mut input = open_regular(&path)?;
        let before = input.metadata()?;
        ensure!(
            stamp(&before) == stamp(&expected),
            "source changed before reading {name}"
        );
        let mut whole = Sha256::new();
        let mut chunks = Vec::new();
        let mut left = entry.bytes;
        while left != 0 {
            let len = left.min(CHUNK as u64) as usize;
            let part = &mut buffer[..len];
            input.read_exact(part)?;
            whole.update(&*part);
            chunks.push(sha(part));
            left -= len as u64;
        }
        ensure!(
            input.read(&mut buffer[..1])? == 0
                && stamp(&input.metadata()?) == stamp(&before)
                && format!("{:x}", whole.finalize()) == entry.sha256,
            "source SHA-256 mismatch: {name}"
        );
        witnesses.push(Witness {
            path: name,
            bytes: entry.bytes,
            sha256: entry.sha256.clone(),
            chunk_sha256: chunks,
        });
    }
    let doc = Document {
        schema: SCHEMA.into(),
        source,
        bundle_sha256: bundle.digest,
        files: witnesses,
    };
    let bytes = serde_json::to_vec(&doc)?;
    ensure!(
        bytes.len() as u64 <= CHUNK_INDEX_LIMIT,
        "chunk index exceeds byte limit"
    );
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(&bytes)?;
    temp.as_file().sync_all()?;
    temp.persist_noclobber(output)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(sha(&bytes))
}

fn stamp(m: &fs::Metadata) -> (u64, u64, u64, i64, i64) {
    (m.dev(), m.ino(), m.len(), m.ctime(), m.ctime_nsec())
}
