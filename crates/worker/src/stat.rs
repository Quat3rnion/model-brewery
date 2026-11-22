use crate::{Config, cache};
use anyhow::{Result, ensure};
use brewery_worker::{LIMIT, PathState, SUGGEST_PREFIX_BYTES, decode_limited, is_digest, token};
use serde::Deserialize;
use std::{io::ErrorKind, os::unix::fs::MetadataExt, path::Path};

pub fn get(path: &str) -> Result<PathState> {
    if path.len() > SUGGEST_PREFIX_BYTES
        || !path.starts_with('/')
        || path.chars().any(char::is_control)
        || path.split('/').any(|part| matches!(part, "." | ".."))
    {
        return Ok(PathState::OutsideRoots);
    }
    let path = Path::new(path);
    let (cfg, _) = Config::load()?;
    let in_sources = cfg.source_roots.iter().any(|root| {
        root.is_absolute()
            && root != Path::new("/")
            && !root.starts_with(&cfg.root)
            && !cfg.root.starts_with(root)
            && path.starts_with(root)
    });
    if !in_sources
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
    let fd = match cache::no_symlink_open(path, libc::O_PATH) {
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
    let fd = match cache::no_symlink_open(root, libc::O_DIRECTORY | libc::O_PATH) {
        Ok(fd) => fd,
        Err(error) if missing(&error) => return Ok(false),
        Err(error) => return Err(error),
    };
    let metadata = fd.metadata()?;
    ensure!(
        metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o077 == 0,
        "worker root must be private and owned by the worker"
    );
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
            .reply
            .result
            .and_then(|result| result.artifact)
            .and_then(|artifact| artifact.path)
            .is_some_and(|published| Path::new(&published) == path))
}

fn missing(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|e| e.kind() == ErrorKind::NotFound)
}
