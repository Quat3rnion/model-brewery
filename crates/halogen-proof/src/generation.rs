use super::{is_sha256, qualify_members, runtime, Format, Manifest, Qualification, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Asset {
    path: PathBuf,
    sha256: String,
}

pub struct GenerationBundle {
    pub(crate) qualification: Qualification,
    source_sha256: String,
    manifest_sha256: String,
    pub(crate) assets: Vec<Asset>,
    pub(crate) stage: tempfile::TempDir,
}

impl GenerationBundle {
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    pub fn format_name(&self) -> &'static str {
        match self.qualification.format {
            Format::HgnV2 => "hgn-v2",
            Format::Ht43 => "ht43",
        }
    }
}

#[derive(Serialize)]
pub struct GenerationProof<'a> {
    load_proof: runtime::Proof<'a>,
    source_sha256: &'a str,
    manifest_sha256: &'a str,
    tokenizer: &'a [Asset],
}

pub struct Generation<'a> {
    pub proof: GenerationProof<'a>,
    pub texts: Vec<String>,
}

fn read_asset(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if path.canonicalize()? != path {
        return Err("symlinks are not allowed in generation bundle assets".into());
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err("generation bundle asset must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("generation bundle asset exceeded size limit".into());
    }
    Ok(bytes)
}

/// Qualify the complete generation manifest, including its immutable source
/// identity and flat tokenizer directory. Stage exact tokenizer bytes privately.
pub fn qualify_generation(directory: &Path) -> Result<GenerationBundle> {
    qualify_generation_observed(directory, &mut |_, _| {})
}

/// Observe bytes after hashing; zero starts a new member.
pub fn qualify_generation_observed(
    directory: &Path,
    observe: &mut dyn FnMut(&Path, u64),
) -> Result<GenerationBundle> {
    let directory = directory.canonicalize()?;
    let manifest_bytes = read_asset(&directory.join("manifest.json"), 64 * 1024)?;
    let mut manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    let source_sha256 = manifest
        .source_sha256
        .take()
        .filter(|s| is_sha256(s))
        .ok_or("missing or invalid source_sha256")?;
    let assets = std::mem::take(&mut manifest.tokenizer);
    if assets.is_empty() || assets.len() > 32 {
        return Err("generation manifest needs 1..=32 explicit tokenizer assets".into());
    }
    let relative_tokenizer = assets[0]
        .path
        .parent()
        .ok_or("missing tokenizer directory")?;
    if ![Path::new("tokenizer"), Path::new("native/tokenizer")].contains(&relative_tokenizer) {
        return Err("tokenizer assets must be in tokenizer/ or native/tokenizer/".into());
    }
    let tokenizer = directory.join(relative_tokenizer);
    if tokenizer.canonicalize()? != tokenizer {
        return Err("tokenizer directory must not be a symlink".into());
    }
    let mut names = BTreeSet::new();
    for asset in &assets {
        let name = asset
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|n| {
                n.len() <= 128
                    && n.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
            })
            .ok_or("invalid tokenizer filename")?;
        if asset.path != relative_tokenizer.join(name)
            || !names.insert(name)
            || !is_sha256(&asset.sha256)
        {
            return Err("invalid, duplicate or unhashed tokenizer asset".into());
        }
    }
    if !names.contains("tokenizer.json") {
        return Err("tokenizer.json is required".into());
    }
    for entry in fs::read_dir(&tokenizer)? {
        let entry = entry?;
        if !entry.file_type()?.is_file()
            || !entry
                .file_name()
                .to_str()
                .is_some_and(|name| names.contains(name))
        {
            return Err("unlisted or non-regular tokenizer asset".into());
        }
    }
    let stage = tempfile::tempdir()?;
    for asset in &assets {
        let bytes = read_asset(&directory.join(&asset.path), 64 * 1024 * 1024)?;
        if format!("{:x}", Sha256::digest(&bytes)) != asset.sha256 {
            return Err("tokenizer asset SHA-256 mismatch".into());
        }
        File::create(
            stage
                .path()
                .join(asset.path.file_name().ok_or("missing asset filename")?),
        )?
        .write_all(&bytes)?;
    }
    Ok(GenerationBundle {
        qualification: qualify_members(&directory, manifest, observe)?,
        source_sha256,
        manifest_sha256: format!("{:x}", Sha256::digest(&manifest_bytes)),
        assets,
        stage,
    })
}

/// Use one qualified, network-isolated Halogen container for the whole batch.
/// Invoke this only after the worker operator admits the workload under its GPU gate.
pub fn generate<'a>(
    bundle: &'a GenerationBundle,
    image_id: &str,
    prompts: &[String],
    max_tokens: u32,
) -> Result<Generation<'a>> {
    generate_observed(bundle, image_id, prompts, max_tokens, &mut |_| {})
}

/// Observe validated completion counts; zero marks the verified runtime load.
pub fn generate_observed<'a>(
    bundle: &'a GenerationBundle,
    image_id: &str,
    prompts: &[String],
    max_tokens: u32,
    observe: &mut dyn FnMut(u64),
) -> Result<Generation<'a>> {
    if prompts.is_empty()
        || prompts.len() > 1319
        || !(1..=512).contains(&max_tokens)
        || prompts
            .iter()
            .any(|p| p.trim().is_empty() || p.len() > 16_384)
    {
        return Err("invalid isolated generation batch".into());
    }
    let (load_proof, texts) = runtime::run(
        &bundle.qualification,
        image_id,
        900,
        Some(runtime::Completion {
            bundle,
            prompts,
            max_tokens,
            observe,
        }),
    )?;
    Ok(Generation {
        proof: GenerationProof {
            load_proof,
            source_sha256: &bundle.source_sha256,
            manifest_sha256: &bundle.manifest_sha256,
            tokenizer: &bundle.assets,
        },
        texts,
    })
}
