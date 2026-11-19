use crate::{cache, progress::Notify};
use anyhow::{Context, Result, ensure};
use brewery_worker::*;
use serde::{Deserialize, Deserializer, de};
use std::{
    collections::BTreeMap,
    fmt, fs,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    path::{Path, PathBuf},
};

const MAX_RECEIPT_BYTES: u64 = 128 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    pub source: PathBuf,
    receipt: PathBuf,
    repo_id: String,
    revision: String,
    bundle_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    model_dir: PathBuf,
    repo_id: String,
    revision: String,
    shard_count: usize,
    weight_bytes: u64,
    #[serde(deserialize_with = "unique_files")]
    files: BTreeMap<String, Member>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    size: u64,
    sha256: String,
}

fn unique_files<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<BTreeMap<String, Member>, D::Error> {
    struct Files;
    impl<'de> de::Visitor<'de> for Files {
        type Value = BTreeMap<String, Member>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("publication files with unique names")
        }

        fn visit_map<M: de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> std::result::Result<Self::Value, M::Error> {
            let mut files = BTreeMap::new();
            while let Some((name, member)) = map.next_entry()? {
                if files.insert(name, member).is_some() {
                    return Err(de::Error::custom("duplicate publication file name"));
                }
            }
            Ok(files)
        }
    }
    d.deserialize_map(Files)
}

impl Pin {
    pub fn manifest(&self, roots: &[PathBuf], progress: &mut Notify) -> Result<Bundle> {
        progress.phase("pinned_manifest", ProgressUnit::Files, None)?;
        ensure!(
            is_digest(&self.bundle_sha256),
            "invalid pinned bundle digest"
        );
        ensure!(
            self.revision.len() == 40
                && self
                    .revision
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                && self
                    .repo_id
                    .split_once('/')
                    .is_some_and(|(owner, model)| token(owner).is_ok() && token(model).is_ok()),
            "invalid pinned publication identity"
        );
        let source = cache::authorized(&self.source, roots)?;
        ensure!(
            source.as_os_str() == self.source.as_os_str(),
            "noncanonical pinned source"
        );
        let receipt_file = cache::regular(&self.receipt)?;
        let meta = receipt_file.metadata()?;
        ensure!(
            meta.nlink() == 1,
            "publication receipt must not have file links"
        );
        ensure!(
            meta.len() <= MAX_RECEIPT_BYTES,
            "publication receipt is too large"
        );
        let receipt: Receipt = decode_limited(&receipt_file, MAX_RECEIPT_BYTES)?;
        ensure!(
            cache::same_file(&meta, &receipt_file.metadata()?),
            "publication receipt changed during read"
        );
        ensure!(
            receipt.schema == "model-brewery.source-publication.v1"
                && receipt.model_dir.as_os_str() == self.source.as_os_str()
                && receipt.repo_id == self.repo_id
                && receipt.revision == self.revision,
            "publication receipt does not match the configured identity"
        );
        let mut shards = 0;
        let mut weight_bytes = 0_u64;
        let mut files = Vec::with_capacity(receipt.files.len());
        for (path, member) in receipt.files {
            relative(&path)?;
            ensure!(
                !path.contains('/'),
                "publication receipt must have flat membership"
            );
            if path.ends_with(".safetensors") {
                shards += 1;
                weight_bytes = weight_bytes
                    .checked_add(member.size)
                    .context("weight size overflow")?;
            }
            files.push(Entry {
                path,
                bytes: member.size,
                sha256: member.sha256,
            });
        }
        ensure!(
            shards > 0 && shards == receipt.shard_count && weight_bytes == receipt.weight_bytes,
            "publication receipt shard summary mismatch"
        );
        let bundle = Bundle {
            entry: ".".into(),
            digest: self.bundle_sha256.clone(),
            files,
        };
        bundle.validate()?;

        let dir = cache::no_symlink_open(&source, libc::O_RDONLY | libc::O_DIRECTORY)?;
        let dir_meta = dir.metadata()?;
        let total = bundle.files.len() as u64;
        progress.detail.total = Some(total);
        progress.detail.files_total = Some(total);
        progress.detail.files_completed = Some(0);
        for child in fs::read_dir(format!("/proc/self/fd/{}", dir.as_raw_fd()))? {
            let name = child?.file_name();
            let name = name.to_str().context("publication path is not UTF-8")?;
            let index = bundle
                .files
                .binary_search_by(|f| f.path.as_str().cmp(name))
                .map_err(|_| anyhow::anyhow!("unexpected file in pinned source"))?;
            let file = cache::readable_regular(cache::no_symlink_open_at(
                &dir,
                Path::new(name),
                libc::O_PATH,
            )?)?;
            let meta = file.metadata()?;
            ensure!(
                meta.len() == bundle.files[index].bytes && meta.nlink() == 1,
                "pinned source size or file links differ from publication"
            );
            // Opening tests readability without reading payload bytes. Only the
            // engine can verify the subsequent payload hashes against the pin.
            progress.detail.completed += 1;
            progress.detail.files_completed = Some(progress.detail.completed);
            progress.file(name);
            progress.tick(false)?;
        }
        ensure!(
            progress.detail.completed == total,
            "missing file in pinned source"
        );
        let current = cache::no_symlink_open(&source, libc::O_PATH | libc::O_DIRECTORY)?;
        ensure!(
            cache::same_file(&dir_meta, &dir.metadata()?)
                && cache::same_file(&dir_meta, &current.metadata()?),
            "pinned directory changed during inventory"
        );
        progress.tick(true)?;
        Ok(bundle)
    }
}
