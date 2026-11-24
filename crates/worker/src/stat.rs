use crate::{Config, cache};
use anyhow::{Result, ensure};
use brewery_worker::{
    Input, LIMIT, PathState, SUGGEST_PREFIX_BYTES, absolute, decode_limited, is_digest,
    scratch_eval, token,
};
use serde::Deserialize;
use std::{collections::BTreeMap, fs::File, io::ErrorKind, os::unix::fs::MetadataExt, path::Path};

pub fn get(path: &str) -> Result<PathState> {
    if path.len() > SUGGEST_PREFIX_BYTES
        || !path.starts_with('/')
        || path.chars().any(char::is_control)
        || absolute(Path::new(path)).is_err()
    {
        return Ok(PathState::OutsideRoots);
    }
    // Path::strip_prefix in no_symlink_open drops these directory suffixes.
    // Execution still uses the original path for the source metadata check.
    let flags = libc::O_PATH
        | if path.ends_with('/') || path.ends_with("/.") {
            libc::O_DIRECTORY
        } else {
            0
        };
    let path = Path::new(path);
    let (cfg, _) = Config::load()?;
    if cfg.scratch_access.is_some() || brewery_worker::hgn_edit::private_path(path) {
        return Ok(PathState::OutsideRoots);
    }
    if cfg.configured_source_root(path).is_none()
        && !(cfg.cache_source_root(path).is_some() && private_root(&cfg.root)?.is_some())
        && !cfg
            .hy4_plan
            .as_ref()
            .is_some_and(|plan| path == plan.source)
        && !cfg
            .backend_weight_calibration_root
            .as_ref()
            .is_some_and(|root| {
                path.parent() == Some(root)
                    && path.extension().is_some_and(|ext| ext == "json")
                    && path
                        .file_stem()
                        .and_then(|name| name.to_str())
                        .is_some_and(is_digest)
            })
        && !published(&cfg.root, path)?
    {
        return Ok(PathState::OutsideRoots);
    }
    let fd = match cache::no_symlink_open(path, flags) {
        Ok(fd) => fd,
        Err(error) => {
            return match error.downcast_ref::<std::io::Error>() {
                Some(e) if matches!(e.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory) => {
                    Ok(PathState::Missing)
                }
                Some(e) if e.raw_os_error() == Some(libc::ELOOP) => Ok(PathState::OutsideRoots),
                _ => Err(error),
            };
        }
    };
    let meta = fd.metadata()?;
    Ok(if meta.is_file() {
        PathState::File { bytes: meta.len() }
    } else if meta.is_dir() {
        PathState::Directory
    } else {
        PathState::OutsideRoots
    })
}

// Private worker state is not a source root. Accept exactly the path
// named by a successful job's publication, not the whole work directory.
fn published(root: &Path, path: &Path) -> Result<bool> {
    let jobs = root.join("jobs");
    let Ok(relative) = path.strip_prefix(jobs) else {
        return Ok(false);
    };
    let mut parts = relative.iter();
    let Some(id) = parts.next().and_then(|part| part.to_str()) else {
        return Ok(false);
    };
    if token(id).is_err() || !parts.next().is_some_and(|part| part == "work") {
        return Ok(false);
    }
    let Some(fd) = private_root(root)? else {
        return Ok(false);
    };
    let relative_record = Path::new("jobs").join(id).join("state.json");
    let record = match cache::no_symlink_open_at(&fd, &relative_record, libc::O_PATH) {
        Ok(fd) => cache::readable_regular(fd)?,
        Err(error) if missing(&error) => return Ok(false),
        Err(error) => return Err(error),
    };
    #[derive(Deserialize)]
    struct Record {
        action: String,
        reply: Reply,
        job: Option<Job>,
        #[serde(default)]
        inputs: BTreeMap<usize, Input>,
    }
    #[derive(Deserialize)]
    struct Job {
        kind: String,
    }
    #[derive(Deserialize)]
    struct Reply {
        state: String,
        result: Option<ResultArtifact>,
    }
    #[derive(Deserialize)]
    struct ResultArtifact {
        artifact: Option<Artifact>,
    }
    #[derive(Deserialize)]
    struct Artifact {
        path: Option<String>,
    }
    let record: Record = decode_limited(record, LIMIT)?;
    Ok(record.action == "start"
        && record.reply.state == "succeeded"
        && record
            .job
            .is_none_or(|job| !scratch_eval::private_output(&job.kind))
        && !record.inputs.values().any(scratch_eval::granted)
        && record
            .reply
            .result
            .and_then(|result| result.artifact)
            .and_then(|artifact| artifact.path)
            .is_some_and(|published| Path::new(&published) == path))
}

fn private_root(root: &Path) -> Result<Option<File>> {
    let fd = match cache::no_symlink_open(root, libc::O_DIRECTORY | libc::O_PATH) {
        Ok(fd) => fd,
        Err(error) if missing(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    let metadata = fd.metadata()?;
    ensure!(
        metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o077 == 0,
        "worker root must be private and owned by the worker"
    );
    Ok(Some(fd))
}

fn missing(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|e| e.kind() == ErrorKind::NotFound)
}
