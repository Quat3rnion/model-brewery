use anyhow::{Context, Result, ensure};
pub mod behavior_direction;
pub mod compiled_program;
pub mod exl3_layer;
pub mod hf_download;
pub mod hgn_recipe;
pub mod hy4;
pub mod locate_edit;
pub mod operational;
pub mod resources;
pub mod scratch_eval;
pub mod source_grants;
pub mod trace;
pub mod weight_source;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

pub const CHUNK: usize = 8 * 1024 * 1024;
pub const LIMIT: u64 = 16 * 1024 * 1024;
pub const SUGGEST_PREFIX_BYTES: usize = 4096;
pub const SUGGEST_ITEMS: usize = 32;
pub const SUGGEST_PATH_BYTES: usize = 16 * 1024;
pub const SUGGEST_JSON_LIMIT: u64 = 48 * 1024;
pub const PROGRESS_LIMIT: u64 = 4096;
pub const PROGRESS_THREADS: usize = 16;
pub const ROCMFPX_REVISION: &str = "fb08d7cdb67082ddda15a25e64dc7ec413afc41c";

pub fn object() -> Value {
    json!({})
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Job {
    pub kind: String,
    pub inputs: Vec<Input>,
    #[serde(default = "object")]
    pub options: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "resources::Limits::is_empty")]
    pub resources: resources::Limits,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub host: String,
    pub path: PathBuf,
    pub kind: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Reply {
    pub state: String,
    pub progress: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_detail: Option<ProgressDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub resources_enforced: bool,
}

impl Reply {
    pub fn new(state: &str, progress: &str) -> Self {
        Self {
            state: state.into(),
            progress: progress.into(),
            progress_detail: None,
            result: None,
            error: None,
            resources_enforced: false,
        }
    }
    pub fn success(result: Value) -> Self {
        Self {
            result: Some(result),
            ..Self::new("succeeded", "complete")
        }
    }
    pub fn error(error: impl ToString) -> Self {
        Self {
            error: Some(error.to_string()),
            ..Self::new("failed", "failed")
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProgressUnit {
    Bytes,
    Items,
    Tokens,
    Questions,
    Suites,
    Files,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProgressDetail {
    pub phase: String,
    pub unit: ProgressUnit,
    pub completed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_completed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_count: Option<u64>,
    pub updated_at_ms: u64,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "decode_threads"
    )]
    pub threads: Vec<ProgressThread>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProgressThread {
    pub slot: usize,
    pub task: String,
}

pub fn valid_thread_task(task: &str) -> bool {
    task.len() <= 96 && !task.trim().is_empty() && valid_basename(task)
}

fn decode_threads<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<ProgressThread>, D::Error> {
    let value = Value::deserialize(d)?;
    let Ok(rows) = serde_json::from_value::<Vec<ProgressThread>>(value) else {
        return Ok(Vec::new());
    };
    if rows.len() > PROGRESS_THREADS
        || rows
            .iter()
            .any(|r| !(1..=64).contains(&r.slot) || !valid_thread_task(&r.task))
        || rows.windows(2).any(|r| r[0].slot >= r[1].slot)
    {
        return Ok(Vec::new());
    }
    Ok(rows)
}

/// Fit the actual serialized frame, including escaping and its framing bytes.
pub fn bound_progress(mut detail: ProgressDetail, frame_bytes: usize) -> ProgressDetail {
    while !detail.threads.is_empty()
        && serde_json::to_vec(&detail)
            .is_ok_and(|json| json.len() + frame_bytes > PROGRESS_LIMIT as usize)
    {
        detail.threads.pop();
    }
    detail
}

#[derive(Default)]
pub struct PoolSlots(std::collections::BTreeMap<usize, String>);

impl PoolSlots {
    pub fn update(&mut self, slot: usize, event: brewery_hash_pool::ProgressEvent, task: &str) {
        use brewery_hash_pool::ProgressEvent::*;
        match event {
            Started if (1..=64).contains(&slot) => {
                self.0.insert(
                    slot,
                    if valid_thread_task(task) {
                        task.into()
                    } else {
                        "File".into()
                    },
                );
            }
            Finished | Stopped => {
                self.0.remove(&slot);
            }
            _ => {}
        }
    }

    pub fn rows(&self) -> Vec<ProgressThread> {
        self.0
            .iter()
            .take(PROGRESS_THREADS)
            .map(|(&slot, task)| ProgressThread {
                slot,
                task: task.clone(),
            })
            .collect()
    }
}
pub fn valid_basename(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 255
        && !matches!(s, "." | "..")
        && !s.contains(['/', '\\'])
        && !s.chars().any(|ch| {
            ch.is_control() || matches!(ch, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        })
}

impl ProgressDetail {
    pub fn valid(&self) -> bool {
        self.updated_at_ms > 0
            && self.total.is_none_or(|n| self.completed <= n)
            && self.current_file.as_deref().is_none_or(valid_basename)
            && self
                .files_total
                .is_none_or(|n| self.files_completed.unwrap_or(0) <= n)
            && self.input_count.is_none_or(|n| n > 0)
            && self
                .input_index
                .is_none_or(|i| i != 0 && self.input_count.is_none_or(|n| i <= n))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuggestRequest {
    pub prefix: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StatRequest {
    pub path: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum PathState {
    File { bytes: u64 },
    Directory,
    Missing,
    OutsideRoots,
}

pub fn validate_suggest_prefix(prefix: &str) -> Result<()> {
    ensure!(
        prefix.len() <= SUGGEST_PREFIX_BYTES
            && (prefix.is_empty() || prefix.starts_with('/'))
            && !prefix.contains("//")
            && !prefix.split('/').any(|part| part.starts_with('.'))
            && !prefix.chars().any(char::is_control),
        "invalid path suggestion prefix"
    );
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PathSuggestions {
    pub items: Vec<PathSuggestion>,
    pub truncated: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PathSuggestion {
    pub path: String,
    pub kind: PathKind,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PathKind {
    File,
    Directory,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Entry {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Bundle {
    pub entry: String,
    pub files: Vec<Entry>,
    pub digest: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Rpc {
    Prepare {
        id: String,
        action: String,
        job: Job,
    },
    Status {
        id: String,
    },
    Cancel {
        id: String,
    },
    Finish {
        id: String,
        failure: Option<String>,
        result: Option<Value>,
    },
    Resolve {
        id: String,
        index: usize,
        input: Input,
        bundle: Option<Bundle>,
    },
    Start {
        id: String,
    },
    Cleanup {
        id: String,
    },
    Describe {
        input: Input,
    },
    DescribeProgress {
        input: Input,
        index: usize,
        count: usize,
    },
    StageProgress {
        id: String,
        detail: ProgressDetail,
    },
    SuggestPaths(SuggestRequest),
    TraceRead {
        id: String,
        slice: trace::Slice,
    },
    Stat(StatRequest),
    Read {
        root: PathBuf,
        entry: Entry,
        offset: u64,
    },
    Plan {
        id: String,
        bundle: Bundle,
        preview: bool,
    },
    Write {
        id: String,
        digest: String,
        entry: Entry,
        offset: u64,
        data: String,
    },
    Seal {
        id: String,
        digest: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Description {
    pub root: PathBuf,
    pub bundle: Bundle,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Plan {
    pub path: PathBuf,
    pub cached: bool,
    pub available_bytes: u64,
    pub offsets: Vec<u64>,
}

impl Bundle {
    pub fn bytes(&self) -> Result<u64> {
        self.files.iter().try_fold(0u64, |a, f| {
            a.checked_add(f.bytes).context("bundle size overflow")
        })
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.files.is_empty() && self.files.len() <= 16384,
            "empty or oversized file manifest"
        );
        ensure!(
            self.entry == "." || self.files.iter().any(|f| f.path == self.entry),
            "invalid entry"
        );
        for f in &self.files {
            relative(&f.path)?;
            ensure!(is_digest(&f.sha256), "invalid file digest");
        }
        ensure!(
            self.files.windows(2).all(|w| w[0].path < w[1].path),
            "unsorted or duplicate paths"
        );
        ensure!(
            self.digest == tree_digest(&self.files),
            "manifest digest mismatch"
        );
        self.bytes()?;
        Ok(())
    }
}

pub fn token(s: &str) -> Result<()> {
    ensure!(
        !s.is_empty()
            && s.len() <= 128
            && s.as_bytes()[0].is_ascii_alphanumeric()
            && s.bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c)),
        "invalid identifier"
    );
    Ok(())
}

pub fn relative(s: &str) -> Result<()> {
    ensure!(
        !s.is_empty()
            && s.len() <= 4096
            && !s.contains('\\')
            && !s.chars().any(char::is_control)
            && s.split('/').all(|p| !p.is_empty() && p != "." && p != "..")
            && Path::new(s)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "unsafe relative path"
    );
    Ok(())
}

pub fn absolute(p: &Path) -> Result<()> {
    ensure!(
        p.is_absolute()
            && p.components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_))),
        "expected an absolute path"
    );
    Ok(())
}

pub fn is_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

// RPC errors can include paths or arbitrary error text. Only closed categories
// may enter the job's externally visible error field.
pub fn staging_error(detail: &str) -> &'static str {
    if detail.contains("Worker does not support requested resource controls") {
        "Worker does not support requested resource controls"
    } else if detail.contains("SafeTensors immutable publication invalid") {
        "SafeTensors immutable publication could not be verified"
    } else if detail.contains("SafeTensors requires an immutable publication")
        || detail.contains("SafeTensors source exceeds explicit verification limit")
        || detail.contains("SafeTensors verification limit must")
    {
        "SafeTensors source requires a valid immutable publication or an explicit verification allowance within the size limit"
    } else if detail.contains("SafeTensors weight jobs require a direct local source") {
        "SafeTensors weight jobs require a direct local source"
    } else if detail.contains("Pinned publication manifest unavailable") {
        "Pinned publication manifest unavailable; check the configured receipt and source inventory"
    } else if detail.contains("Source input unavailable")
        || detail.contains("outside allowed source roots")
    {
        "Source input unavailable; check the registered location and worker source roots"
    } else if detail.contains("insufficient target disk")
        || detail.contains("target disk reserve")
        || detail.contains("No space left on device")
    {
        "Target disk space is insufficient; check free space and the configured reserve"
    } else if detail.contains("trace model SHA-256 mismatch") {
        "Trace model SHA-256 mismatch; use the model revision bound to the intervention"
    } else if detail.contains("scratch input SHA-256 mismatch")
        || detail.contains("hash mismatch")
        || detail.contains("content mismatch")
        || detail.contains("changed while hashing")
    {
        "Input checksum verification failed; check source integrity before retrying"
    } else {
        "Worker could not prepare the inputs; check source access, target storage and worker configuration"
    }
}

pub fn hash(file: &Path) -> Result<String> {
    let mut file = File::open(file).with_context(|| format!("open {}", file.display()))?;
    let mut sha = Sha256::new();
    std::io::copy(&mut file, &mut sha)?;
    Ok(format!("{:x}", sha.finalize()))
}

pub fn tree_digest(files: &[Entry]) -> String {
    let mut sha = Sha256::new();
    for f in files {
        sha.update(format!("{}\0{}\0{}\n", f.path, f.bytes, f.sha256));
    }
    format!("{:x}", sha.finalize())
}

pub fn decode<T: serde::de::DeserializeOwned>(r: impl Read) -> Result<T> {
    decode_limited(r, LIMIT)
}

pub fn decode_limited<T: serde::de::DeserializeOwned>(r: impl Read, limit: u64) -> Result<T> {
    let mut buf = Vec::new();
    r.take(limit + 1).read_to_end(&mut buf)?;
    ensure!(buf.len() as u64 <= limit, "JSON exceeds limit");
    Ok(serde_json::from_slice(&buf)?)
}

pub fn load<T: serde::de::DeserializeOwned>(p: &Path) -> Result<T> {
    decode(File::open(p).with_context(|| p.display().to_string())?)
}

pub fn atomic<T: Serialize>(p: &Path, value: &T) -> Result<()> {
    let parent = p.parent().context("no parent")?;
    let tmp = parent.join(format!(".{}.new", p.file_name().unwrap().to_string_lossy()));
    let mut f = File::create(&tmp)?;
    serde_json::to_writer(&mut f, value)?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    fs::rename(tmp, p)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub fn lock(p: &Path) -> Result<File> {
    let f = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(p)?;
    fs2::FileExt::lock_exclusive(&f)?;
    Ok(f)
}

pub fn available(p: &Path) -> Result<u64> {
    Ok(fs2::available_space(p)?)
}
pub mod ablation_search;
