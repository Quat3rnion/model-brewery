//! Local worker policy for controller-managed model-directory grants.
//!
//! This module is an exact-path check, not a discovery API. The controller
//! must first select an explicit local worker that runs as its own account
//! and pass the trusted protection paths for that worker.
//!
//! Linux `openat2` and `statx` birth times are required. Bind aliases and
//! nested mounts are refused. The directory is pinned, not the model bytes:
//! existing worker/engine integrity checks still verify each selected input.
//! This is not a sandbox against root or another process running as the worker.
//! SSH support needs authenticated validation and revalidation on the owning
//! worker, its own protected-path inventory and a trusted grant delivery path.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::{self, File},
    io::{self, Read},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{ffi::OsStrExt, fs::MetadataExt},
    },
    path::{Path, PathBuf},
};

const SYSTEM: &[&str] = &[
    "/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/nix", "/proc",
    "/root", "/run", "/sbin", "/sys", "/tmp", "/usr", "/var",
];
const PRIVATE_COMPONENTS: &[&str] = &[
    ".aws",
    ".azure",
    ".config",
    ".docker",
    ".git",
    ".gnupg",
    ".kube",
    ".local",
    ".password-store",
    ".pki",
    ".ssh",
    ".terraform",
    ".mozilla",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceGrant {
    pub host: String,
    pub path: PathBuf,
    identity: Identity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    worker: String,
    device: (u32, u32),
    inode: u64,
    birth: (i64, u32),
}

/// Validate only the directory the operator named; never list or open its
/// entries. Return a persistable identity for confirmation and worker use.
pub fn validate(host: &str, path: &Path, protected: &[PathBuf]) -> Result<SourceGrant> {
    crate::token(host)?;
    let root = checked_directory(path, protected)?;
    Ok(SourceGrant {
        host: host.into(),
        path: path.into(),
        identity: root.identity,
    })
}

pub fn revalidate(host: &str, grant: &SourceGrant, protected: &[PathBuf]) -> Result<()> {
    pinned(host, grant, protected).map(drop)
}

fn pinned(host: &str, grant: &SourceGrant, protected: &[PathBuf]) -> Result<Directory> {
    ensure!(
        host == grant.host,
        "The grant belongs to a different worker"
    );
    let root = checked_directory(&grant.path, protected)?;
    ensure!(
        root.identity == grant.identity,
        "The directory or worker identity changed; save a new grant"
    );
    Ok(root)
}

fn overlaps(a: &Path, b: &Path) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

fn conservative_path(path: &Path) -> Result<PathBuf> {
    // A protected directory may not exist yet. Resolve its deepest existing
    // ancestor so a configured symlink cannot hide a future protected path.
    crate::absolute(path)?;
    let mut parent = path;
    loop {
        match parent.canonicalize() {
            Ok(canonical) => return Ok(canonical.join(path.strip_prefix(parent)?)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                ensure!(
                    fs::symlink_metadata(parent)
                        .is_err_and(|err| err.kind() == io::ErrorKind::NotFound),
                    "Protected path could not be resolved"
                );
                parent = parent
                    .parent()
                    .context("Protected path could not be resolved")?;
            }
            Err(_) => anyhow::bail!("Protected path could not be resolved"),
        }
    }
}

fn exact_path(path: &Path) -> Result<&str> {
    let text = path
        .to_str()
        .and_then(|s| s.strip_prefix('/'))
        .context("Enter an exact absolute directory path")?;
    // Inspect the raw spelling before Path methods normalize `//` and `/.`.
    ensure!(
        text.len() < 4096 && text.split('/').all(crate::valid_basename),
        "Use an exact absolute path without aliases"
    );
    ensure!(
        !text
            .split('/')
            .any(|part| PRIVATE_COMPONENTS.contains(&part)),
        "Choose a model path outside credential and configuration directories"
    );
    Ok(text)
}

struct Directory {
    file: File,
    identity: Identity,
}

fn checked_directory(path: &Path, protected: &[PathBuf]) -> Result<Directory> {
    let text = exact_path(path)?;
    let mut parts = text.split('/');
    let top = parts.next().unwrap_or_default();
    let second = parts.next();
    ensure!(
        !(SYSTEM.iter().any(|p| overlaps(path, Path::new(p)))
            || top == "home" && (second.is_none() || parts.next().is_none())),
        "Choose a model directory outside the system and home roots"
    );
    if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
        let canonical = conservative_path(&home)?;
        ensure!(
            !home.starts_with(path) && !canonical.starts_with(path),
            "A home directory or its parent cannot be granted"
        );
        for name in [
            ".cache/huggingface/token",
            ".cache/huggingface/stored_tokens",
            ".huggingface/token",
            ".local/share/model-brewery",
            ".config/model-brewery",
        ] {
            ensure_unprotected(path, &home.join(name))?;
        }
    }
    for key in [
        "HF_TOKEN_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_SHARED_CREDENTIALS_FILE",
        "KUBECONFIG",
    ] {
        if let Some(value) = env::var_os(key) {
            let paths = if key == "KUBECONFIG" {
                env::split_paths(&value).collect()
            } else {
                vec![value.into()]
            };
            for config in paths.into_iter().filter(|p| !p.as_os_str().is_empty()) {
                ensure_unprotected(path, &std::path::absolute(config)?)?;
            }
        }
    }
    if let Some(home) = env::var_os("HF_HOME") {
        let home = std::path::absolute(home)?;
        ensure_unprotected(path, &home.join("token"))?;
        ensure_unprotected(path, &home.join("stored_tokens"))?;
    }
    for protected in protected {
        ensure_unprotected(path, protected)?;
    }
    let namespace_root = File::open("/")?;
    let file = open_at(
        &namespace_root,
        Path::new(text),
        libc::O_RDONLY | libc::O_DIRECTORY,
        false,
    )
    .map_err(|_| {
        anyhow::anyhow!("The exact directory must already exist and be readable without symlinks")
    })?;
    let stat = stat(&file)?;
    mount_check(path, &stat)?;
    trusted_path(&namespace_root, Path::new(text), &file, false)?;
    let id =
        fs::read_to_string("/etc/machine-id").context("Worker machine identity unavailable")?;
    let id = id.trim();
    ensure!(
        id.len() == 32
            && id.bytes().any(|b| b != b'0')
            && id.bytes().all(|b| b.is_ascii_hexdigit()),
        "Worker machine identity unavailable"
    );
    let worker = format!(
        "{:x}",
        Sha256::digest(format!("brewery-source-grant-v1\0{id}\0{}", unsafe {
            libc::geteuid()
        }))
    );
    Ok(Directory {
        file,
        identity: Identity {
            worker,
            device: (stat.stx_dev_major, stat.stx_dev_minor),
            inode: stat.stx_ino,
            birth: (stat.stx_btime.tv_sec, stat.stx_btime.tv_nsec),
        },
    })
}

fn ensure_unprotected(path: &Path, protected: &Path) -> Result<()> {
    ensure!(
        !overlaps(path, protected) && !overlaps(path, &conservative_path(protected)?),
        "Choose a directory outside protected Brewery and credential paths"
    );
    Ok(())
}

fn open_at(root: &File, path: &Path, flags: i32, same_mount: bool) -> Result<File> {
    let name = std::ffi::CString::new(path.as_os_str().as_bytes())?;
    // openat2 RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS, plus NO_XDEV for
    // descendants. No path-normalizing or symlink-following fallback.
    let flags = flags
        | libc::O_CLOEXEC
        | if flags & libc::O_PATH == 0 {
            libc::O_NONBLOCK
        } else {
            0
        };
    let how = [flags as u64, 0, 0x08 | 0x04 | u64::from(same_mount)];
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            root.as_raw_fd(),
            name.as_ptr(),
            &how,
            size_of_val(&how),
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(unsafe { File::from_raw_fd(fd as i32) })
}

fn stat(file: &File) -> Result<libc::statx> {
    let mut st = unsafe { std::mem::zeroed::<libc::statx>() };
    let mask = libc::STATX_INO | libc::STATX_BTIME | libc::STATX_MNT_ID;
    ensure!(
        unsafe {
            libc::statx(
                file.as_raw_fd(),
                c"".as_ptr(),
                libc::AT_EMPTY_PATH,
                mask,
                &mut st,
            )
        } == 0
            && st.stx_mask & mask == mask,
        "The filesystem cannot supply a stable directory identity"
    );
    Ok(st)
}

fn mount_field(value: &str) -> Result<PathBuf> {
    use std::os::unix::ffi::OsStringExt;
    let mut bytes = Vec::with_capacity(value.len());
    let mut input = value.as_bytes();
    while let Some((&b, rest)) = input.split_first() {
        if b == b'\\' {
            let escape = rest.get(..3).context("Invalid mount path")?;
            let byte = match escape {
                b"040" => b' ',
                b"011" => b'\t',
                b"012" => b'\n',
                b"134" => b'\\',
                _ => anyhow::bail!("Invalid mount path"),
            };
            bytes.push(byte);
            input = &rest[3..];
        } else {
            bytes.push(b);
            input = rest;
        }
    }
    ensure!(bytes.first() == Some(&b'/'), "Invalid mount path");
    Ok(std::ffi::OsString::from_vec(bytes).into())
}

fn mount_check(path: &Path, stat: &libc::statx) -> Result<()> {
    const LIMIT: u64 = 4 << 20;
    let mut info = String::new();
    File::open("/proc/self/mountinfo")?
        .take(LIMIT + 1)
        .read_to_string(&mut info)?;
    ensure!(
        info.len() as u64 <= LIMIT,
        "Worker mount table could not be verified"
    );
    let device = format!("{}:{}", stat.stx_dev_major, stat.stx_dev_minor);
    let mut found = false;
    let mut whole_device_mounts = std::collections::BTreeMap::<&str, usize>::new();
    let mut ancestors = Vec::new();
    let invalid = "Choose a directory outside bind aliases and nested mounts";
    for line in info.lines() {
        let mut fields = line.split_ascii_whitespace();
        let mount: u64 = fields.next().context(invalid)?.parse()?;
        let backing = fields.nth(1).context(invalid)?;
        let source = fields.next().context(invalid)?;
        let target = mount_field(fields.next().context(invalid)?)?;
        let is_root = target == Path::new("/");
        if source == "/" {
            *whole_device_mounts.entry(backing).or_default() += 1;
        }
        if mount == stat.stx_mnt_id {
            ensure!(backing == device && path.starts_with(&target), "{invalid}");
            found = true;
        } else {
            ensure!(!target.starts_with(path), "{invalid}");
        }
        if !is_root && path.starts_with(&target) {
            ensure!(source == "/", "{invalid}");
            ancestors.push(backing);
        }
    }
    ensure!(
        found
            && ancestors
                .iter()
                .all(|device| whole_device_mounts.get(device) == Some(&1)),
        "{invalid}"
    );
    Ok(())
}

fn trusted_path(root: &File, rel: &Path, file: &File, same_mount: bool) -> Result<()> {
    let trusted = |file: &File| -> Result<()> {
        let md = file.metadata()?;
        ensure!(
            (md.uid() == 0 || md.uid() == unsafe { libc::geteuid() }) && md.mode() & 0o022 == 0,
            "Model paths must not be writable by untrusted accounts"
        );
        Ok(())
    };
    trusted(file)?;
    for parent in rel
        .ancestors()
        .skip(1)
        .filter(|p| !p.as_os_str().is_empty())
    {
        trusted(&open_at(
            root,
            parent,
            libc::O_PATH | libc::O_DIRECTORY,
            same_mount,
        )?)?;
    }
    Ok(())
}

/// Prove that an editor input is of an expected model type and lies inside the
/// pinned directory. Hold the directory FD through the caller's operation and
/// revalidate before passing the path to another process.
pub fn editor_input(
    host: &str,
    grant: &SourceGrant,
    path: &Path,
    kind: &str,
    protected: &[PathBuf],
) -> Result<File> {
    exact_path(path)?;
    let relative = path
        .strip_prefix(&grant.path)
        .context("Input is outside the approved model directory")?;
    let root = pinned(host, grant, protected)?;
    let entry = if relative.as_os_str().is_empty() {
        Path::new(".")
    } else {
        relative
    };
    if kind == "hf" {
        let dir = open_at(&root.file, entry, libc::O_RDONLY | libc::O_DIRECTORY, true)?;
        trusted_path(&root.file, entry, &dir, true)?;
        // The existing HF editor requires a flat indexed SafeTensors bundle.
        // Validate each member by descriptor (no mount crossings/hard links)
        // before the worker inventory hashes it.
        let mut has_index = false;
        let mut has_shard = false;
        for (i, child) in fs::read_dir(format!("/proc/self/fd/{}", dir.as_raw_fd()))?.enumerate() {
            ensure!(i < 4096, "Too many HF model files");
            let name = child?.file_name();
            let name = name.to_str().context("Invalid HF model filename")?;
            ensure!(
                crate::valid_basename(name)
                    && !name.starts_with('.')
                    && (name.ends_with(".safetensors")
                        || matches!(
                            name,
                            "config.json"
                                | "generation_config.json"
                                | "tokenizer.json"
                                | "tokenizer_config.json"
                                | "special_tokens_map.json"
                                | "added_tokens.json"
                                | "vocab.json"
                                | "chat_template.json"
                                | crate::weight_source::INDEX
                        )
                        || matches!(
                            name,
                            "merges.txt"
                                | "vocab.txt"
                                | "tokenizer.model"
                                | "spiece.model"
                                | "sentencepiece.bpe.model"
                                | "chat_template.jinja"
                        )),
                "The HF model directory contains a non-model file"
            );
            regular_member(&dir, Path::new(name))?;
            has_index |= name == crate::weight_source::INDEX;
            has_shard |= name.ends_with(".safetensors");
        }
        ensure!(
            has_index && has_shard,
            "Select an indexed HF SafeTensors checkpoint"
        );
    } else {
        ensure!(
            matches!(kind, "gguf" | "hgn" | "safetensors")
                && path.extension().is_some_and(|ext| ext == kind),
            "Select a supported model file"
        );
        let file = regular_member(&root.file, entry)?;
        if matches!(kind, "gguf" | "hgn") {
            let mut magic = [0; 4];
            File::open(format!("/proc/self/fd/{}", file.as_raw_fd()))?.read_exact(&mut magic)?;
            ensure!(
                &magic == if kind == "gguf" { b"GGUF" } else { b"HGN1" },
                "Select a valid model file"
            );
        }
        if kind == "gguf" {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("Invalid GGUF name")?;
            if let Some((left, count)) = name.strip_suffix(".gguf").unwrap().rsplit_once("-of-")
                && let Some((prefix, first)) = left.rsplit_once('-')
                && first.len() == 5
                && count.len() == 5
                && let (Ok(first), Ok(count)) = (first.parse::<u16>(), count.parse::<u16>())
            {
                ensure!(
                    first == 1 && (2..=1024).contains(&count),
                    "Select the first GGUF shard"
                );
                for i in 2..=count {
                    regular_member(
                        &root.file,
                        &entry.with_file_name(format!("{prefix}-{i:05}-of-{count:05}.gguf")),
                    )?;
                }
            }
        }
    }
    Ok(root.file)
}

fn regular_member(dir: &File, path: &Path) -> Result<File> {
    let file = open_at(dir, path, libc::O_PATH, true)
        .map_err(|_| anyhow::anyhow!("A model file is missing or uses an alias"))?;
    let md = file.metadata()?;
    ensure!(
        md.is_file() && md.nlink() == 1,
        "Model files must be regular files without aliases"
    );
    trusted_path(dir, path, &file, true)?;
    Ok(file)
}
