use crate::progress::Notify;
use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use brewery_hash_pool::ProgressEvent;
use brewery_worker::*;
use sha2::{Digest, Sha256};
use std::{
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt},
        },
    },
    path::{Path, PathBuf},
};

pub(crate) fn no_symlink_open(path: &Path, flags: i32) -> Result<File> {
    absolute(path)?;
    let root = File::open("/")?;
    no_symlink_open_at(&root, path.strip_prefix("/")?, flags)
        .with_context(|| path.display().to_string())
}

pub(crate) fn no_symlink_open_at(root: &File, path: &Path, flags: i32) -> Result<File> {
    let name = CString::new(path.as_os_str().as_bytes())?;
    #[repr(C)]
    struct How {
        flags: u64,
        mode: u64,
        resolve: u64,
    }
    // Linux openat2: RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS. Unlike a
    // canonicalize-then-open check, this also covers mutable parent directories.
    let mut flags = flags | libc::O_CLOEXEC;
    if flags & libc::O_PATH == 0 {
        flags |= libc::O_NONBLOCK;
    }
    let how = How {
        flags: flags as u64,
        mode: 0,
        resolve: 0x08 | 0x04,
    };
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            root.as_raw_fd(),
            name.as_ptr(),
            &how,
            std::mem::size_of::<How>(),
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(unsafe { File::from_raw_fd(fd as i32) })
}

pub fn regular(path: &Path) -> Result<File> {
    readable_regular(no_symlink_open(path, libc::O_PATH)?)
}

pub(crate) fn readable_regular(f: File) -> Result<File> {
    ensure!(f.metadata()?.is_file(), "not a regular file");
    Ok(File::open(format!("/proc/self/fd/{}", f.as_raw_fd()))?)
}

pub fn authorized(path: &Path, roots: &[PathBuf]) -> Result<PathBuf> {
    absolute(path)?;
    let p = path
        .canonicalize()
        .with_context(|| path.display().to_string())?;
    ensure!(
        p == path && roots.iter().any(|r| p.starts_with(r)),
        "path outside source roots or follows a symlink"
    );
    Ok(p)
}

fn text(p: &Path) -> Result<String> {
    let s = p.to_str().context("path is not UTF-8")?;
    relative(s)?;
    Ok(s.to_owned())
}

fn walk(
    root: &Path,
    dir: &Path,
    files: &mut Vec<Inventory>,
    depth: u8,
    progress: &mut Notify,
) -> Result<()> {
    ensure!(depth < 64, "bundle tree too deep");
    let fd = no_symlink_open(dir, libc::O_RDONLY | libc::O_DIRECTORY)?;
    for d in fs::read_dir(format!("/proc/self/fd/{}", fd.as_raw_fd()))? {
        progress.tick(false)?;
        let d = d?;
        let p = dir.join(d.file_name());
        if d.file_type()?.is_dir() {
            walk(root, &p, files, depth + 1, progress)?;
        } else {
            ensure!(
                d.file_type()?.is_file(),
                "bundle contains a symlink or special file"
            );
            push(root, &p, files, progress)?;
        }
    }
    Ok(())
}

struct Inventory {
    path: String,
    metadata: fs::Metadata,
}

fn push(root: &Path, file: &Path, files: &mut Vec<Inventory>, progress: &mut Notify) -> Result<()> {
    ensure!(files.len() < 16384, "too many bundle files");
    let p = text(file.strip_prefix(root)?)?;
    let metadata = no_symlink_open(file, libc::O_PATH)?.metadata()?;
    ensure!(metadata.is_file(), "not a regular file");
    files.push(Inventory { path: p, metadata });
    progress.detail.completed += 1;
    progress.file(
        file.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(""),
    );
    progress.tick(false)
}

pub fn describe(
    p: &Path,
    kind: &str,
    boundary: &Path,
    phase: &str,
    progress: &mut Notify,
) -> Result<Description> {
    describe_bounded(p, kind, boundary, phase, progress, u64::MAX)
}

pub fn describe_bounded(
    p: &Path,
    kind: &str,
    boundary: &Path,
    phase: &str,
    progress: &mut Notify,
    max_bytes: u64,
) -> Result<Description> {
    describe_inner(p, kind, boundary, phase, progress, max_bytes, false)
}

pub fn describe_exact_file(p: &Path, progress: &mut Notify) -> Result<Description> {
    let boundary = p.parent().context("source has no parent")?;
    describe_inner(p, "", boundary, "hash_input", progress, u64::MAX, true)
}

fn describe_inner(
    p: &Path,
    kind: &str,
    boundary: &Path,
    phase: &str,
    progress: &mut Notify,
    max_bytes: u64,
    exact_file: bool,
) -> Result<Description> {
    progress.phase(
        if phase == "verify_output" {
            "inventory_output"
        } else {
            "inventory_input"
        },
        ProgressUnit::Files,
        None,
    )?;
    let mut files = Vec::new();
    let root;
    let entry;
    if no_symlink_open(p, libc::O_PATH)?.metadata()?.is_dir() {
        ensure!(!exact_file, "scratch input must be a regular file");
        root = p.to_owned();
        entry = ".".into();
        ensure!(
            root.starts_with(boundary),
            "bundle outside the configured source root"
        );
        walk(&root, &root, &mut files, 0, progress)?;
    } else {
        root = p.parent().context("source has no parent")?.to_owned();
        ensure!(
            root.starts_with(boundary),
            "bundle outside the configured source root"
        );
        entry = text(Path::new(p.file_name().context("source has no basename")?))?;
        if matches!(entry.as_str(), "manifest.json" | "bundle.json")
            && matches!(kind, "hgn" | "hgn-v2" | "ht43" | "hf" | "brewery-bundle")
        {
            walk(&root, &root, &mut files, 0, progress)?;
        } else if kind == "gguf"
            && let Some((stem, n, count)) = split_gguf(&entry)
        {
            ensure!(
                n == 1 && count > 1 && count <= 1024,
                "register the first shard of a bounded split GGUF"
            );
            for i in 1..=count {
                push(
                    &root,
                    &root.join(format!("{stem}-{i:05}-of-{count:05}.gguf")),
                    &mut files,
                    progress,
                )?;
            }
        } else {
            push(&root, p, &mut files, progress)?;
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let total = files.iter().try_fold(0_u64, |total, f| {
        total.checked_add(f.metadata.len()).context("size overflow")
    })?;
    ensure!(
        total <= max_bytes,
        "SafeTensors source exceeds explicit verification limit"
    );
    progress.reset(phase, ProgressUnit::Bytes, Some(total));
    progress.detail.files_completed = Some(0);
    progress.detail.files_total = Some(files.len() as u64);
    progress.tick(true)?;
    let result = brewery_hash_pool::map_observed(
        &files,
        |_, f, advance| {
            let (bytes, sha256) = regular_hash(
                &root.join(&f.path),
                Some(&f.metadata),
                f.metadata.len(),
                &mut vec![0; HASH_BUFFER],
                advance,
            )?;
            Ok(Entry {
                path: f.path.clone(),
                bytes,
                sha256,
            })
        },
        |slot, index, event| hash_progress(progress, &files[index].path, slot, event),
    );
    let files = progress.pooled(result)?;
    let bundle = Bundle {
        entry,
        digest: tree_digest(&files),
        files,
    };
    bundle.validate()?;
    Ok(Description { root, bundle })
}

pub(crate) fn split_gguf(name: &str) -> Option<(&str, usize, usize)> {
    let (left, right) = name.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (stem, n) = left.rsplit_once('-')?;
    (right.len() == 5 && n.len() == 5).then_some(())?;
    Some((stem, n.parse().ok()?, right.parse().ok()?))
}

pub fn read(root: &Path, roots: &[PathBuf], entry: &Entry, offset: u64) -> Result<String> {
    relative(&entry.path)?;
    let p = authorized(&root.join(&entry.path), roots)?;
    let mut f = regular(&p)?;
    ensure!(
        offset < entry.bytes && f.metadata()?.len() == entry.bytes,
        "source changed or invalid offset"
    );
    f.seek(SeekFrom::Start(offset))?;
    let mut b = vec![0; (entry.bytes - offset).min(CHUNK as u64) as usize];
    f.read_exact(&mut b)?;
    Ok(STANDARD.encode(b))
}

pub struct Cache {
    pub complete: PathBuf,
    pub partial: PathBuf,
    pub reserve: u64,
}

impl Cache {
    pub fn ready(&self, digest: &str) -> PathBuf {
        self.complete.join(digest)
    }
    pub fn staged(&self, digest: &str) -> PathBuf {
        self.partial.join(digest)
    }

    pub fn plan(&self, b: &Bundle, preview: bool, progress: &mut Notify) -> Result<Plan> {
        progress.phase(
            "plan_transfer",
            ProgressUnit::Files,
            Some(b.files.len() as u64),
        )?;
        b.validate()?;
        let ready = self.ready(&b.digest);
        let staged = self.staged(&b.digest);
        let cached = ready.is_dir();
        let mut offsets = vec![];
        if cached {
            let existing: Bundle = load(&ready.join("manifest.json"))?;
            ensure!(existing.files == b.files, "cached manifest mismatch");
            verify(&ready.join("data"), b, progress)?;
            offsets.extend(b.files.iter().map(|f| f.bytes));
        } else {
            for (i, f) in b.files.iter().enumerate() {
                let p = staged.join(i.to_string());
                let offset = if p.exists() {
                    regular(&p)?.metadata()?.len()
                } else {
                    0
                };
                ensure!(offset <= f.bytes, "partial size exceeds expected length");
                offsets.push(offset);
                progress.detail.completed += 1;
                progress.tick(false)?;
            }
        }
        let free = available(&self.complete)?;
        let remaining = b.bytes()? - offsets.iter().sum::<u64>();
        if !preview {
            ensure!(
                remaining == 0 || free.saturating_sub(self.reserve) >= remaining,
                "insufficient target disk: {remaining} missing bytes, {free} available, {} reserved",
                self.reserve
            );
            if !cached {
                fs::create_dir_all(&staged)?;
                atomic(&staged.join("manifest.json"), b)?;
            }
        }
        Ok(Plan {
            path: if b.entry == "." {
                ready.join("data")
            } else {
                ready.join("data").join(&b.entry)
            },
            available_bytes: free,
            offsets,
            cached,
        })
    }

    pub fn write(
        &self,
        digest: &str,
        entry: &Entry,
        offset: u64,
        data: String,
        progress: Option<&mut Notify<'_>>,
    ) -> Result<(u64, u64)> {
        ensure!(is_digest(digest), "invalid cache digest");
        // A concurrent seal may already have made the content available.
        if self.ready(digest).is_dir() {
            return Ok((entry.bytes, 0));
        }
        let staged = self.staged(digest);
        let b: Bundle = load(&staged.join("manifest.json"))?;
        let i = b
            .files
            .iter()
            .position(|f| f == entry)
            .context("file not in manifest")?;
        let bytes = STANDARD.decode(data)?;
        ensure!(
            offset < entry.bytes && bytes.len() as u64 == (entry.bytes - offset).min(CHUNK as u64),
            "invalid chunk length"
        );
        let p = staged.join(i.to_string());
        let mut f = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&p)?;
        let current = f.metadata()?.len();
        ensure!(current <= entry.bytes && current >= offset, "offset gap");
        let (next, added) = if current > offset {
            // A short disk write or disconnect can leave part of a chunk.
            // Return the persisted length; the sender resumes from that byte.
            (current, 0)
        } else {
            ensure!(
                available(&self.complete)?.saturating_sub(self.reserve) >= bytes.len() as u64,
                "target disk reserve reached"
            );
            f.seek(SeekFrom::Start(offset))?;
            f.write_all(&bytes)?;
            f.sync_data()?;
            (offset + bytes.len() as u64, bytes.len() as u64)
        };
        if let Some(progress) = progress {
            let mut staged_bytes = 0;
            let mut files_completed = 0;
            for (i, e) in b.files.iter().enumerate() {
                let path = staged.join(i.to_string());
                let size = if path.exists() {
                    regular(&path)?.metadata()?.len()
                } else {
                    0
                };
                ensure!(size <= e.bytes, "partial size exceeds expected length");
                staged_bytes += size;
                files_completed += u64::from(size == e.bytes);
            }
            progress.reset("copy_input", ProgressUnit::Bytes, Some(b.bytes()?));
            progress.detail.completed = staged_bytes;
            progress.detail.files_completed = Some(files_completed);
            progress.detail.files_total = Some(b.files.len() as u64);
            progress.file(
                Path::new(&entry.path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(""),
            );
            progress.tick(true)?;
        }
        Ok((next, added))
    }

    pub fn seal(&self, digest: &str, progress: &mut Notify) -> Result<()> {
        ensure!(is_digest(digest), "invalid cache digest");
        let ready = self.ready(digest);
        let staged = self.staged(digest);
        match fs::symlink_metadata(&ready) {
            Ok(metadata) => {
                ensure!(metadata.is_dir(), "unsafe cache destination");
                let existing: Bundle = decode(regular(&ready.join("manifest.json"))?)?;
                existing.validate()?;
                ensure!(existing.digest == digest, "cached manifest mismatch");
                verify(&ready.join("data"), &existing, progress)?;
                return self.finish_seal(&staged);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        let b: Bundle = load(&staged.join("manifest.json"))?;
        b.validate()?;
        ensure!(b.digest == digest, "staged manifest mismatch");
        progress.reset("verify_staged", ProgressUnit::Bytes, Some(b.bytes()?));
        progress.detail.files_completed = Some(0);
        progress.detail.files_total = Some(b.files.len() as u64);
        progress.tick(true)?;
        // The cache lock is still exclusive. Create only missing empty
        // partials here; readers never mutate the staging area.
        for (i, f) in b.files.iter().enumerate() {
            if f.bytes == 0 {
                let p = staged.join(i.to_string());
                match File::create_new(&p) {
                    Ok(file) => file.sync_all()?,
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(e) => return Err(e.into()),
                }
                progress.tick(false)?;
            }
        }
        let result = brewery_hash_pool::map_observed(
            &b.files,
            |i, f, advance| {
                let p = staged.join(i.to_string());
                if regular(&p)?.metadata()?.len() != f.bytes
                    || regular_hash(&p, None, f.bytes, &mut vec![0; HASH_BUFFER], advance)?.1
                        != f.sha256
                {
                    return Err(BadPartial(i).into());
                }
                Ok(())
            },
            |slot, index, event| hash_progress(progress, &b.files[index].path, slot, event),
        );
        if let Err(error) = progress.pooled(result) {
            if let Some(BadPartial(i)) = error.downcast_ref::<BadPartial>() {
                // All readers have joined. Keep good partials for a retry.
                fs::remove_file(staged.join(i.to_string()))?;
                File::open(&staged)?.sync_all()?;
                anyhow::bail!("staged file hash mismatch; bad partial removed");
            }
            return Err(error);
        }
        let publish = staged.join("publish");
        if publish.exists() {
            fs::remove_dir_all(&publish)?;
        }
        fs::create_dir_all(publish.join("data"))?;
        for (i, f) in b.files.iter().enumerate() {
            let p = publish.join("data").join(&f.path);
            fs::create_dir_all(p.parent().unwrap())?;
            fs::hard_link(staged.join(i.to_string()), p)?;
            progress.tick(false)?;
        }
        atomic(&publish.join("manifest.json"), &b)?;
        sync_dirs(&publish)?;
        progress.tick(true)?;
        rename_noreplace(&publish, &ready)?;
        self.finish_seal(&staged)
    }

    fn finish_seal(&self, staged: &Path) -> Result<()> {
        // Persist the published name before discarding the resumable source.
        File::open(&self.complete)?.sync_all()?;
        match fs::remove_dir_all(staged) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        File::open(&self.partial)?.sync_all()?;
        Ok(())
    }
}

#[derive(Debug)]
struct BadPartial(usize);

impl std::fmt::Display for BadPartial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("staged file hash mismatch")
    }
}

impl std::error::Error for BadPartial {}

fn rename_noreplace(from: &Path, to: &Path) -> Result<()> {
    let from = CString::new(from.as_os_str().as_bytes())?;
    let to = CString::new(to.as_os_str().as_bytes())?;
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error()).context("publish verified cache bundle");
    }
    Ok(())
}

fn sync_dirs(p: &Path) -> Result<()> {
    for f in fs::read_dir(p)? {
        let f = f?;
        if f.file_type()?.is_dir() {
            sync_dirs(&f.path())?;
        }
    }
    File::open(p)?.sync_all()?;
    Ok(())
}

fn verify(root: &Path, bundle: &Bundle, progress: &mut Notify) -> Result<()> {
    progress.reset("verify_cache", ProgressUnit::Bytes, Some(bundle.bytes()?));
    progress.detail.files_completed = Some(0);
    progress.detail.files_total = Some(bundle.files.len() as u64);
    progress.tick(true)?;
    let boundary = [root.to_owned()];
    let result = brewery_hash_pool::map_observed(
        &bundle.files,
        |_, f, advance| {
            let p = authorized(&root.join(&f.path), &boundary)?;
            let (bytes, hash) =
                regular_hash(&p, None, f.bytes, &mut vec![0; HASH_BUFFER], advance)?;
            ensure!(
                bytes == f.bytes && hash == f.sha256,
                "cache content mismatch"
            );
            Ok(())
        },
        |slot, index, event| hash_progress(progress, &bundle.files[index].path, slot, event),
    );
    progress.pooled(result).map(|_| ())
}

// The pool delivers events on the caller thread. At most one current basename
// is displayed; the byte and completed-file counts include all readers.
fn hash_progress(
    progress: &mut Notify,
    path: &str,
    slot: usize,
    event: ProgressEvent,
) -> Result<()> {
    let basename = path.rsplit('/').next().unwrap_or("");
    progress.slot(slot, event, basename);
    if progress.detail.current_file.as_deref() != Some(basename) {
        progress.file(basename);
    }
    match event {
        ProgressEvent::Advanced(bytes) => progress.detail.completed += bytes,
        ProgressEvent::Finished => *progress.detail.files_completed.get_or_insert(0) += 1,
        ProgressEvent::Started | ProgressEvent::Stopped => {}
    }
    progress.tick(false)
}

const HASH_BUFFER: usize = 1024 * 1024;

fn regular_hash(
    p: &Path,
    expected: Option<&fs::Metadata>,
    expected_bytes: u64,
    buffer: &mut [u8],
    mut advance: impl FnMut(u64) -> Result<()>,
) -> Result<(u64, String)> {
    let mut f = regular(p)?;
    let m = f.metadata()?;
    ensure!(
        m.len() == expected_bytes && expected.is_none_or(|e| same_file(e, &m)),
        "source changed while hashing"
    );
    let mut hash = Sha256::new();
    let mut bytes = 0_u64;
    loop {
        let n = match f.read(buffer) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if n == 0 {
            break;
        }
        bytes += n as u64;
        ensure!(bytes <= expected_bytes, "source changed while hashing");
        hash.update(&buffer[..n]);
        advance(n as u64)?;
    }
    ensure!(
        m.len() == bytes && same_file(&m, &f.metadata()?),
        "source changed while hashing"
    );
    Ok((bytes, format!("{:x}", hash.finalize())))
}

pub(crate) fn same_file(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    (
        a.dev(),
        a.ino(),
        a.len(),
        a.mtime(),
        a.mtime_nsec(),
        a.ctime(),
        a.ctime_nsec(),
    ) == (
        b.dev(),
        b.ino(),
        b.len(),
        b.mtime(),
        b.mtime_nsec(),
        b.ctime(),
        b.ctime_nsec(),
    )
}
