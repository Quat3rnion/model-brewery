use crate::integrity;
use anyhow::{ensure, Context, Result};
use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

// Lanternfish 0ea30b3 and scratch-check-logits-v1 use the same CPU W0 build.
pub(crate) const PINS: &[(&str, &str)] = &[
    (
        "libllama.so.0",
        "bb94467edcc18c8fb875f86628d89c112adedcfea1e8f6c5549c706adf5f25cc",
    ),
    (
        "libggml.so.0",
        "04a3a199429b02e6e4bce9a8cbc465530bcffe4b0539be561dedfcfd476957be",
    ),
    (
        "libggml-base.so.0",
        "1fef6e5b16b4b03e8255cf5f3c446cc76005515e0bb261b4d0ee333ea399fbad",
    ),
    (
        "libggml-cpu.so.0",
        "29390b353b7931bce11926598773eb052bbe53506b903f2140a13ea1f54ead6e",
    ),
];

pub(crate) fn stage(work: &Path, key: &str, label: &str, pins: &[(&str, &str)]) -> Result<PathBuf> {
    let configured = PathBuf::from(env::var_os(key).with_context(|| format!("{key} is required"))?);
    ensure!(
        integrity::canonical(&configured)
            .with_context(|| format!("invalid {key}"))?
            .is_dir(),
        "{key} must be a directory"
    );
    let libs = work.join("libs");
    // Do not let the ELF loader split or expand this single search path.
    ensure!(
        !libs
            .as_os_str()
            .as_encoded_bytes()
            .iter()
            .any(|b| matches!(b, b':' | b';' | b'$')),
        "job library directory must not contain ':', ';' or '$'"
    );
    fs::create_dir(&libs)?;
    for &(name, sha) in pins {
        let source = configured
            .join(name)
            .canonicalize()
            .with_context(|| format!("open {label} library {name} in {key}"))?;
        ensure!(
            source.parent() == Some(configured.as_path())
                && integrity::canonical(&source)?.is_file(),
            "{label} library {name} in {key} must resolve to a regular file in that directory"
        );
        let staged = libs.join(name);
        fs::copy(source, &staged).with_context(|| format!("stage {label} library {name}"))?;
        fs::set_permissions(&staged, fs::Permissions::from_mode(0o400))?;
        ensure!(
            integrity::file_digest(&staged, name.into())?.0.sha256 == sha,
            "{label} library {name} SHA-256 mismatch"
        );
    }
    Ok(libs)
}
