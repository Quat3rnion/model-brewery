use crate::{progress::Progress, Input};
use anyhow::{ensure, Context, Result};
use brewery_hash_pool::ProgressEvent;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, Metadata},
    io::{self, BufRead, BufReader, Read},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Member {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(PartialEq, Eq)]
pub struct Stamp {
    dev: u64,
    ino: u64,
    len: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

impl Stamp {
    fn new(meta: &Metadata) -> Self {
        Self {
            dev: meta.dev(),
            ino: meta.ino(),
            len: meta.len(),
            mtime: (meta.mtime(), meta.mtime_nsec()),
            ctime: (meta.ctime(), meta.ctime_nsec()),
        }
    }

    pub fn check(&self, path: &Path) -> Result<()> {
        ensure!(
            path.canonicalize()? == path && *self == Self::new(&fs::symlink_metadata(path)?),
            "input changed: {}",
            path.display()
        );
        Ok(())
    }
}

pub fn canonical(path: &Path) -> Result<Metadata> {
    ensure!(path.is_absolute(), "resolved paths must be absolute");
    ensure!(
        path.canonicalize()
            .with_context(|| format!("open {}", path.display()))?
            == path,
        "resolved path must be canonical and symlink-free: {}",
        path.display()
    );
    let meta = fs::symlink_metadata(path)?;
    ensure!(
        meta.is_file() || meta.is_dir(),
        "not a regular file or directory"
    );
    Ok(meta)
}

pub fn file_digest(path: &Path, label: String) -> Result<(Member, Stamp)> {
    let metadata = canonical(path)?;
    ensure!(metadata.is_file(), "not a regular file: {}", path.display());
    let stamp = Stamp::new(&metadata);
    let member = digest(path, label, &stamp, &|_| Ok(()))?;
    Ok((member, stamp))
}

fn digest(
    path: &Path,
    label: String,
    before: &Stamp,
    advance: &dyn Fn(u64) -> Result<()>,
) -> Result<Member> {
    let mut file = BufReader::with_capacity(1 << 20, File::open(path)?);
    ensure!(
        *before == Stamp::new(&file.get_ref().metadata()?),
        "file changed before hashing"
    );
    let mut hash = Sha256::new();
    let mut bytes = 0;
    loop {
        let chunk = match file.fill_buf() {
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            read => read?,
        };
        let n = chunk.len();
        if n == 0 {
            break;
        }
        ensure!(
            n as u64 <= before.len - bytes,
            "file size changed while hashing"
        );
        hash.update(chunk);
        bytes += n as u64;
        file.consume(n);
        advance(n as u64)?;
    }
    before.check(path)?;
    ensure!(bytes == before.len, "file size changed while hashing");
    Ok(Member {
        path: label,
        bytes,
        sha256: format!("{:x}", hash.finalize()),
    })
}

pub struct Identity {
    pub files: Vec<Member>,
    pub bytes: u64,
    pub sha256: String,
    pub bundle_sha256: String,
    stamps: Vec<(PathBuf, Stamp)>,
    entry: bool,
}

impl Identity {
    // For managed SafeTensors only: the caller must independently verify
    // the manifest against fs-verity or hash all bytes into a private snapshot.
    pub(crate) fn weight_manifest(input: &Input) -> Result<Self> {
        let files = input
            .files
            .clone()
            .context("missing weight source manifest")?;
        let members = brewery_worker::weight_source::inventory(&input.path, &input.kind)?;
        ensure!(
            members.len() == files.len(),
            "weight source membership changed"
        );
        let entry = input.kind != "hf";
        let mut stamps = Vec::new();
        if !entry {
            stamps.push((input.path.clone(), Stamp::new(&canonical(&input.path)?)));
        }
        let mut bytes = 0u64;
        for ((name, meta), member) in members.into_iter().zip(&files) {
            ensure!(
                member.path == name && member.bytes == meta.len() && sha256_string(&member.sha256),
                "invalid weight source manifest member"
            );
            let path = if entry {
                input.path.clone()
            } else {
                input.path.join(name)
            };
            stamps.push((path, Stamp::new(&meta)));
            bytes = bytes
                .checked_add(member.bytes)
                .context("weight size overflow")?;
        }
        let bundle_sha256 = manifest_hash(&files);
        let sha256 = if entry {
            files[0].sha256.clone()
        } else {
            bundle_sha256.clone()
        };
        ensure!(
            input.bundle_sha256.as_ref() == Some(&bundle_sha256)
                && input.bytes == Some(bytes)
                && if entry || input.sha256.is_some() {
                    input.sha256.as_ref() == Some(&sha256)
                } else {
                    true
                },
            "weight source manifest digest or size mismatch"
        );
        let id = Self {
            files,
            bytes,
            sha256,
            bundle_sha256,
            stamps,
            entry,
        };
        id.check()?;
        Ok(id)
    }

    // weight_manifest stores the HF directory stamp before its sorted member stamps.
    pub(crate) fn check_open_weight_member(&self, index: usize, file: &File) -> Result<()> {
        ensure!(
            self.stamps[index + usize::from(!self.entry)].1 == Stamp::new(&file.metadata()?),
            "SafeTensors member changed before sandbox pin"
        );
        Ok(())
    }

    // Hash the same private bytes the caller will parse. The expected stamp
    // and extent come from weight_manifest, not from a replacement file.
    pub(crate) fn snapshot_single_weight_member(
        &self,
        limit: u64,
        progress: &Progress,
    ) -> Result<Vec<u8>> {
        ensure!(
            self.entry && self.files.len() == 1,
            "expected one weight member"
        );
        let (path, stamp) = &self.stamps[0];
        let expected = &self.files[0];
        ensure!(
            expected.bytes <= limit,
            "SafeTensors source exceeds explicit verification limit"
        );
        let mut file = brewery_worker::weight_source::open_regular(path)?;
        self.check_open_weight_member(0, &file)?;
        let len = usize::try_from(expected.bytes).context("weight source extent overflow")?;
        let mut snapshot = Vec::new();
        snapshot.try_reserve_exact(len)?;
        snapshot.resize(len, 0);
        let mut phase = progress.start("verify_input", "bytes", Some(expected.bytes));
        let mut hash = Sha256::new();
        for chunk in snapshot.chunks_mut(1 << 20) {
            file.read_exact(chunk)?;
            hash.update(&*chunk);
            phase.advance(chunk.len() as u64);
        }
        ensure!(
            *stamp == Stamp::new(&file.metadata()?),
            "weight source changed while reading"
        );
        self.check()?;
        ensure!(
            format!("{:x}", hash.finalize()) == expected.sha256,
            "weight source SHA-256 changed"
        );
        Ok(snapshot)
    }

    pub fn with_added_file(mut self, root: &Path, name: &str) -> Result<Self> {
        ensure!(
            !self.entry && !self.files.iter().any(|f| f.path == name),
            "invalid additional bundle file"
        );
        // Runtime manifest creation changes the root directory metadata, but
        // not the contents or identities of any pre-existing bundle members.
        self.stamps.retain(|(path, _)| path != root);
        self.check()?;
        self.stamps
            .push((root.into(), Stamp::new(&canonical(root)?)));
        entry_file(&root.join(name), root, &mut self)?;
        self.files.sort_by(|a, b| a.path.cmp(&b.path));
        self.bundle_sha256 = manifest_hash(&self.files);
        self.sha256 = self.bundle_sha256.clone();
        self.check()?;
        Ok(self)
    }

    pub fn check(&self) -> Result<()> {
        for (path, stamp) in &self.stamps {
            stamp.check(path)?;
        }
        Ok(())
    }

    pub fn artifact(&self, path: &Path, kind: &str) -> Value {
        if self.entry && self.files.len() == 1 {
            json!({"path":path,"kind":kind,"sha256":self.sha256,"bytes":self.bytes})
        } else {
            json!({"path":path,"kind":kind,"sha256":self.sha256,"bytes":self.bytes,
                   "bundle_sha256":self.bundle_sha256,"files":self.files})
        }
    }

    pub fn input(&self, input: &Input) -> Value {
        json!({"id":input.id,"name":input.name,"path":input.path,"kind":input.kind,
               "host":input.host,"source_host":input.source_host.as_ref().unwrap_or(&input.host),
               "sha256":self.sha256,"bundle_sha256":self.bundle_sha256,"bytes":self.bytes,
               "files":self.files})
    }
}

fn basename(path: &Path) -> Result<String> {
    label(Path::new(path.file_name().context("expected filename")?))
}

fn label(path: &Path) -> Result<String> {
    let label = path.to_str().context("bundle names must be UTF-8")?;
    ensure!(
        !label.is_empty()
            && label.len() <= 4096
            && !label.contains('\\')
            && !label.chars().any(char::is_control),
        "invalid bundle member name"
    );
    Ok(label.into())
}

fn entry_file(path: &Path, root: &Path, ids: &mut Identity) -> Result<()> {
    let (member, stamp) = file_digest(path, label(path.strip_prefix(root)?)?)?;
    ids.bytes = ids
        .bytes
        .checked_add(member.bytes)
        .context("bundle size overflow")?;
    ids.files.push(member);
    ensure!(ids.files.len() <= 16_384, "too many files in input bundle");
    ids.stamps.push((path.into(), stamp));
    Ok(())
}

fn walk(
    path: &Path,
    depth: usize,
    dirs: &mut Vec<(PathBuf, Stamp)>,
    files: &mut Vec<(PathBuf, Stamp)>,
) -> Result<()> {
    ensure!(
        depth <= 16 && dirs.len() + files.len() < 32_768,
        "input bundle is too large or nested"
    );
    let meta = canonical(path)?;
    if meta.is_dir() {
        dirs.push((path.into(), Stamp::new(&meta)));
        for child in fs::read_dir(path)? {
            walk(&child?.path(), depth + 1, dirs, files)?;
        }
    } else {
        ensure!(files.len() < 16_384, "too many files in input bundle");
        files.push((path.into(), Stamp::new(&meta)));
    }
    Ok(())
}

fn gguf_shards(path: &Path) -> Result<Vec<PathBuf>> {
    let name = basename(path)?;
    let stem = name
        .strip_suffix(".gguf")
        .context("GGUF filename must end in .gguf")?;
    let shards = match stem.rsplit_once("-of-") {
        Some((left, total)) => match left.rsplit_once('-') {
            Some((prefix, index))
                if total.len() == 5
                    && index.len() == 5
                    && total.bytes().all(|b| b.is_ascii_digit())
                    && index.bytes().all(|b| b.is_ascii_digit()) =>
            {
                let total: usize = total.parse()?;
                ensure!(
                    (2..=1024).contains(&total) && index == "00001",
                    "select the first valid numbered GGUF shard"
                );
                (1..=total)
                    .map(|n| path.with_file_name(format!("{prefix}-{n:05}-of-{total:05}.gguf")))
                    .collect()
            }
            _ => vec![path.into()],
        },
        None => vec![path.into()],
    };
    for shard in &shards {
        ensure!(canonical(shard)?.is_file(), "GGUF shard is not a file");
        let mut magic = [0; 4];
        File::open(shard)?.read_exact(&mut magic)?;
        ensure!(&magic == b"GGUF", "invalid GGUF magic: {}", shard.display());
    }
    Ok(shards)
}

pub fn identify(path: &Path, gguf: bool) -> Result<Identity> {
    identify_inner(path, gguf, None)
}

pub fn identify_output(path: &Path, progress: &Progress) -> Result<Identity> {
    identify_inner(
        path,
        false,
        Some(HashProgress {
            progress,
            phase: "identify_output",
            input: None,
        }),
    )
}

struct HashProgress<'a> {
    progress: &'a Progress,
    phase: &'static str,
    input: Option<(u64, u64)>,
}

fn identify_inner(path: &Path, gguf: bool, progress: Option<HashProgress<'_>>) -> Result<Identity> {
    let meta = canonical(path)?;
    let mut files = Vec::new();
    let mut ids = Identity {
        files: Vec::new(),
        stamps: Vec::new(),
        bytes: 0,
        sha256: String::new(),
        bundle_sha256: String::new(),
        entry: !meta.is_dir(),
    };
    if gguf {
        ensure!(meta.is_file(), "GGUF input must be a file");
        for shard in gguf_shards(path)? {
            let metadata = canonical(&shard)?;
            ensure!(metadata.is_file(), "GGUF shard is not a file");
            files.push((shard, Stamp::new(&metadata)));
        }
    } else if meta.is_dir() {
        walk(path, 0, &mut ids.stamps, &mut files)?;
    } else {
        files.push((path.into(), Stamp::new(&meta)));
    }
    ensure!(!files.is_empty(), "empty input bundle");
    let root = if meta.is_dir() {
        path
    } else {
        path.parent().context("file has no parent")?
    };
    let total = files
        .iter()
        .try_fold(0u64, |sum, (_, stamp)| sum.checked_add(stamp.len))
        .context("bundle size overflow")?;
    let mut phase = progress.map(|meter| {
        meter.progress.start_for(
            meter.phase,
            "bytes",
            Some(total),
            Some(files.len() as u64),
            meter.input,
        )
    });
    let hashed = brewery_hash_pool::map_observed(
        &files,
        |_, (file, stamp), advance| digest(file, label(file.strip_prefix(root)?)?, stamp, advance),
        |slot, index, event| {
            if let Some(phase) = phase.as_mut() {
                match event {
                    ProgressEvent::Started => {
                        let file = &files[index].0;
                        phase.file(file);
                        if let Some(name) = file.file_name().and_then(|name| name.to_str()) {
                            phase.assign(slot, name);
                        }
                    }
                    ProgressEvent::Advanced(bytes) => phase.advance(bytes),
                    ProgressEvent::Finished => {
                        phase.unassign(slot);
                        phase.file_done();
                    }
                    ProgressEvent::Stopped => phase.unassign(slot),
                }
            }
            Ok(())
        },
    );
    if let Some(phase) = phase {
        phase.finish();
    }
    ids.files = hashed?;
    ids.stamps.extend(files);
    ids.bytes = total;
    ids.files.sort_by(|a, b| a.path.cmp(&b.path));
    ids.bundle_sha256 = manifest_hash(&ids.files);
    ids.sha256 = if ids.entry {
        let name = basename(path)?;
        ids.files
            .iter()
            .find(|m| m.path == name)
            .expect("entry included")
            .sha256
            .clone()
    } else {
        ids.bundle_sha256.clone()
    };
    ids.check()?;
    Ok(ids)
}

fn manifest_hash(files: &[Member]) -> String {
    let mut hash = Sha256::new();
    for m in files {
        hash.update(format!("{}\0{}\0{}\n", m.path, m.bytes, m.sha256).as_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn sha256_string(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn verification_phase(input: &Input, index: usize, host: &str) -> Result<&'static str> {
    let Some(crate::IntegrityBasis::PinnedPublication) = input.integrity_basis else {
        return Ok("verify_input");
    };
    ensure!(
        index == 0
            && input.kind == "hf"
            && input
                .source_host
                .as_deref()
                .is_none_or(|source| source == host)
            && canonical(&input.path)?.is_dir(),
        "pinned publication requires the direct HF source directory as the first input"
    );
    ensure!(
        input.bundle_sha256.as_deref().is_some_and(sha256_string),
        "pinned publication requires a valid bundle SHA-256"
    );
    let files = input
        .files
        .as_deref()
        .filter(|files| !files.is_empty())
        .context("pinned publication requires a nonempty file manifest")?;
    ensure!(
        files.iter().all(|file| sha256_string(&file.sha256))
            && files.windows(2).all(|pair| pair[0].path < pair[1].path),
        "pinned publication requires a sorted manifest with valid file SHA-256 digests"
    );
    Ok("verify_pinned_input")
}

pub fn check_inputs(inputs: &[Input], progress: &Progress) -> Result<Vec<Identity>> {
    let host = &inputs.first().context("no input was selected")?.host;
    ensure!(
        !host.is_empty() && inputs.iter().all(|i| i.host == *host),
        "all resolved inputs must be on one selected host"
    );
    inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let phase = verification_phase(input, index, host)?;
            let id = identify_inner(
                &input.path,
                input.kind == "gguf",
                Some(HashProgress {
                    progress,
                    phase,
                    input: Some((index as u64 + 1, inputs.len() as u64)),
                }),
            )?;
            if let Some(expected) = &input.sha256 {
                ensure!(
                    *expected == id.sha256,
                    "SHA-256 mismatch: {}",
                    input.path.display()
                );
            }
            if let Some(expected) = &input.bundle_sha256 {
                ensure!(
                    *expected == id.bundle_sha256,
                    "bundle SHA-256 mismatch: {}",
                    input.path.display()
                );
            }
            if let Some(expected) = &input.files {
                ensure!(
                    *expected == id.files,
                    "bundle file list mismatch: {}",
                    input.path.display()
                );
            }
            if input.integrity_basis.is_some() {
                ensure!(
                    input.bytes.is_none_or(|expected| expected == id.bytes),
                    "pinned publication bundle size mismatch: {}",
                    input.path.display()
                );
            }
            if input
                .source_host
                .as_ref()
                .is_some_and(|original| original != host)
            {
                ensure!(
                    input.bundle_sha256.is_some()
                        || id.entry && id.files.len() == 1 && input.sha256.is_some(),
                    "cross-host input lacks a verified staging digest"
                );
            }
            Ok(id)
        })
        .collect()
}

pub fn source_hash(inputs: &[Input], ids: &[Identity], roles: &[(&str, usize)]) -> String {
    // A role-qualified digest identifies all physical source dependencies. Do
    // not confuse it with the converter's per-tensor canonical-source hashes.
    let mut hash = Sha256::new();
    hash.update(b"brewery-canonical-source-v1\n");
    for (role, i) in roles {
        hash.update(format!(
            "{}\0{}\0{}\n",
            role, inputs[*i].kind, ids[*i].bundle_sha256
        ));
    }
    format!("{:x}", hash.finalize())
}
