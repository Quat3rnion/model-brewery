use crate::{Config, cache::no_symlink_open};
use anyhow::{Result, ensure};
use brewery_worker::*;
use std::{
    collections::BTreeMap,
    fs,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    path::Path,
    sync::mpsc,
    time::{Duration, Instant},
};

const SCAN_LIMIT: usize = 4096;
const SCAN_TIME: Duration = Duration::from_millis(200);

pub fn run(prefix: String) -> Result<PathSuggestions> {
    validate_suggest_prefix(&prefix)?;
    within_limit(move || suggest(&prefix)).map(|reply| {
        reply.unwrap_or(PathSuggestions {
            items: vec![],
            truncated: true,
        })
    })
}

pub(crate) fn within_limit<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<Option<T>> {
    // The worker runs one RPC per process. A watchdog also bounds a lookup
    // stalled in kernel filesystem I/O, where the scan budget cannot be checked.
    let (send, receive) = mpsc::sync_channel(1);
    std::thread::Builder::new().spawn(move || {
        let _ = send.send(work());
    })?;
    match receive.recv_timeout(Duration::from_secs(1)) {
        Ok(result) => result.map(Some),
        Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[derive(Default)]
struct Matches {
    paths: BTreeMap<String, PathKind>,
    bytes: usize,
    truncated: bool,
}

impl Matches {
    fn push(&mut self, path: &str, kind: PathKind) -> bool {
        if self.paths.contains_key(path) {
            return true;
        }
        if path.len() > SUGGEST_PREFIX_BYTES {
            self.truncated = true;
            return true;
        }
        if self.paths.len() == SUGGEST_ITEMS || self.bytes + path.len() > SUGGEST_PATH_BYTES {
            self.truncated = true;
            return false;
        }
        self.bytes += path.len();
        self.paths.insert(path.to_owned(), kind);
        true
    }
}

fn directory(path: &Path) -> Result<fs::File> {
    no_symlink_open(path, libc::O_RDONLY | libc::O_DIRECTORY)
}

fn suggest(prefix: &str) -> Result<PathSuggestions> {
    let (cfg, _) = Config::load()?;
    // Validate existing state without creating it or following a symlink. A
    // non-existent state root is normal before the worker's first job.
    match no_symlink_open(&cfg.root, libc::O_PATH | libc::O_DIRECTORY) {
        Ok(fd) => {
            let md = fd.metadata()?;
            ensure!(
                md.uid() == unsafe { libc::geteuid() } && md.mode() & 0o077 == 0,
                "worker root must be private and owned by the worker"
            );
        }
        Err(e)
            if e.downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) => {}
        Err(_) => anyhow::bail!("unsafe worker state root"),
    }

    let deadline = Instant::now() + SCAN_TIME;
    let mut result = Matches::default();
    let split = prefix
        .rsplit_once('/')
        .map(|(parent, name)| (if parent.is_empty() { "/" } else { parent }, name));
    let mut scan = false;
    for (i, root) in cfg.source_roots.iter().enumerate() {
        if i == SCAN_LIMIT || Instant::now() >= deadline {
            result.truncated = true;
            break;
        }
        let Some(path) = root.to_str() else {
            continue;
        };
        if path == "/"
            || path.ends_with('/')
            || validate_suggest_prefix(path).is_err()
            || !root.is_absolute()
            || root.starts_with(&cfg.root)
            || cfg.root.starts_with(root)
        {
            continue;
        }
        if split.is_some_and(|(parent, _)| Path::new(parent).starts_with(root)) {
            scan = true;
        }
        // Return configured roots for partial ancestors without listing any
        // ancestor or inspecting paths constructed from such a prefix.
        if path.starts_with(prefix)
            && directory(root).is_ok()
            && !result.push(path, PathKind::Directory)
        {
            break;
        }
    }

    if scan && let Some((parent, needle)) = split {
        scan_directory(parent, needle, deadline, &mut result);
    }
    Ok(PathSuggestions {
        items: result
            .paths
            .into_iter()
            .map(|(path, kind)| PathSuggestion { path, kind })
            .collect(),
        truncated: result.truncated,
    })
}

fn scan_directory(parent: &str, prefix: &str, deadline: Instant, result: &mut Matches) {
    if Instant::now() >= deadline {
        result.truncated = true;
        return;
    }
    // Hold the openat2 descriptor while enumerating. Never reopen the
    // user-selected path by name and never follow any listed entry.
    let Ok(fd) = directory(Path::new(parent)) else {
        return;
    };
    let Ok(mut entries) = fs::read_dir(format!("/proc/self/fd/{}", fd.as_raw_fd())) else {
        return;
    };
    for _ in 0..SCAN_LIMIT {
        if Instant::now() >= deadline {
            break;
        }
        let entry = match entries.next() {
            Some(Ok(entry)) => entry,
            Some(Err(_)) => break,
            None => return,
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(prefix) || name.starts_with('.') || name.chars().any(char::is_control)
        {
            continue;
        }
        let kind = match entry.file_type() {
            Ok(ty) if ty.is_file() => PathKind::File,
            Ok(ty) if ty.is_dir() => PathKind::Directory,
            _ => continue,
        };
        if !result.push(&format!("{parent}/{name}"), kind) {
            return;
        }
    }
    result.truncated = true;
}
