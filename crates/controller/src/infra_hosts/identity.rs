use serde::Serialize;
use std::{
    env,
    ffi::CString,
    fs::{self, File, Metadata},
    io::{BufRead as _, BufReader, Read as _},
    os::{
        fd::{AsRawFd as _, FromRawFd as _},
        unix::fs::MetadataExt as _,
    },
};

const MAX_PATH: usize = 512;
const MAX_ENTRIES: usize = 256;
const MAX_KEY_SIZE: u64 = 64 * 1024;

#[derive(Serialize)]
pub(super) struct Identity {
    path: String,
    label: String,
}

fn safe_components(path: &str) -> Option<&str> {
    let relative = path.strip_prefix('/')?;
    (path.len() <= MAX_PATH
        && relative.split('/').all(|component| {
            !matches!(component, "" | "." | "..")
                && component
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
        }))
    .then_some(relative)
}

pub(super) fn safe_path(path: &str) -> bool {
    safe_components(path).is_some_and(|relative| {
        let name = relative.rsplit('/').next().unwrap_or_default();
        !name.ends_with(".pub")
            && !matches!(
                name,
                "config" | "authorized_keys" | "authorized_keys2" | "known_hosts" | "known_hosts2"
            )
    })
}

pub(super) fn configured_default() -> Option<String> {
    env::var("BREWERY_PROVISION_IDENTITY").ok()
}

pub(super) fn catalog() -> Vec<Identity> {
    let Some(dir) = env::var("HOME")
        .ok()
        .map(|home| format!("{}/.ssh", home.trim_end_matches('/')))
        .filter(|dir| safe_components(dir).is_some())
        .and_then(|path| Some((open_dir(path.strip_prefix('/')?)?, path)))
    else {
        return Vec::new();
    };
    let (dir, path) = dir;
    let Ok(entries) = fs::read_dir(fd_path(&dir)) else {
        return Vec::new();
    };
    let mut identities: Vec<_> = entries
        .take(MAX_ENTRIES)
        .filter_map(|entry| {
            let label = entry.ok()?.file_name().into_string().ok()?;
            let path = format!("{path}/{label}");
            (safe_path(&path) && eligible_at(&dir, &label)).then_some(Identity { path, label })
        })
        .collect();
    identities.sort_unstable_by(|a, b| a.label.cmp(&b.label));
    identities
}

pub(super) fn usable(path: &str) -> bool {
    if !safe_path(path) {
        return false;
    }
    let relative = &path[1..];
    let (parent, name) = relative.rsplit_once('/').unwrap_or(("", relative));
    open_dir(parent).is_some_and(|dir| eligible_at(&dir, name))
}

fn open_dir(relative: &str) -> Option<File> {
    let mut dir = File::open("/").ok()?;
    // Every ancestor must be trustworthy while OpenSSH later resolves the
    // path again. Root-owned sticky directories also protect existing entries.
    if !trusted_directory(&dir.metadata().ok()?) {
        return None;
    }
    if !relative.is_empty() {
        for component in relative.split('/') {
            dir = open_at(&dir, component, libc::O_PATH | libc::O_DIRECTORY)?;
            if !trusted_directory(&dir.metadata().ok()?) {
                return None;
            }
        }
    }
    Some(dir)
}

fn trusted_directory(md: &Metadata) -> bool {
    let owner = md.uid();
    // SAFETY: geteuid has no arguments and does not dereference memory.
    (owner == 0 || owner == unsafe { libc::geteuid() })
        && (md.mode() & 0o022 == 0 || (owner == 0 && md.mode() & 0o1000 != 0))
}

fn open_at(dir: &File, name: &str, flags: libc::c_int) -> Option<File> {
    let name = CString::new(name).ok()?;
    // SAFETY: both the fd and C string remain valid for this call.
    let fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    // SAFETY: a successful openat returns a new owned descriptor.
    (fd >= 0).then(|| unsafe { File::from_raw_fd(fd) })
}

fn fd_path(file: &File) -> String {
    format!("/proc/self/fd/{}", file.as_raw_fd())
}

fn eligible_at(dir: &File, name: &str) -> bool {
    let Some(entry) = open_at(dir, name, libc::O_PATH) else {
        return false;
    };
    let Ok(md) = entry.metadata() else {
        return false;
    };
    if !md.is_file()
        || md.uid() != unsafe { libc::geteuid() }
        || md.nlink() != 1
        || md.mode() & 0o7177 != 0
        || md.mode() & 0o400 == 0
        || !(128..=MAX_KEY_SIZE).contains(&md.len())
    {
        return false;
    }
    // Read only from the checked inode; opening a named FIFO or device would
    // have side effects. Keep the read to a small prefix and never return it.
    let Ok(file) = File::open(fd_path(&entry)) else {
        return false;
    };
    let mut header = Vec::new();
    BufReader::with_capacity(64, file.take(64))
        .read_until(b'\n', &mut header)
        .is_ok()
        && matches!(
            header.as_slice(),
            b"-----BEGIN OPENSSH PRIVATE KEY-----\n"
                | b"-----BEGIN RSA PRIVATE KEY-----\n"
                | b"-----BEGIN DSA PRIVATE KEY-----\n"
                | b"-----BEGIN EC PRIVATE KEY-----\n"
                | b"-----BEGIN PRIVATE KEY-----\n"
                | b"-----BEGIN ENCRYPTED PRIVATE KEY-----\n"
        )
}
