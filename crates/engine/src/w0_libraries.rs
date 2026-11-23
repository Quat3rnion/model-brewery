use crate::integrity;
use anyhow::{ensure, Context, Result};
use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Component, Path, PathBuf},
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
    stage_from(work, &configured, key, label, pins)
}

pub(crate) fn stage_from(
    work: &Path,
    configured: &Path,
    key: &str,
    label: &str,
    pins: &[(&str, &str)],
) -> Result<PathBuf> {
    ensure!(
        integrity::canonical(configured)
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
        let relative = Path::new(name);
        ensure!(
            !name.is_empty()
                && relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "invalid {label} runtime member {name}"
        );
        let expected = configured.join(relative);
        let source = expected
            .canonicalize()
            .with_context(|| format!("open {label} library {name} in {key}"))?;
        ensure!(
            source.parent() == expected.parent() && integrity::canonical(&source)?.is_file(),
            "{label} library {name} in {key} must resolve to a regular file in that directory"
        );
        let staged = libs.join(relative);
        fs::create_dir_all(staged.parent().context("missing staging directory")?)?;
        fs::copy(source, &staged).with_context(|| format!("stage {label} library {name}"))?;
        fs::set_permissions(&staged, fs::Permissions::from_mode(0o400))?;
        ensure!(
            integrity::file_digest(&staged, name.into())?.0.sha256 == sha,
            "{label} library {name} SHA-256 mismatch"
        );
    }
    Ok(libs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn stages_pinned_nested_assets_and_rejects_tampering() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path().canonicalize().unwrap();
        let source = root.join("source");
        let nested = "rocblas/library/kernel.co";
        let member = source.join(nested);
        fs::create_dir_all(member.parent().unwrap()).unwrap();
        fs::write(&member, b"abc").unwrap();
        let pins = [(
            nested,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )];
        let stage = |name| {
            let work = root.join(name);
            fs::create_dir(&work).unwrap();
            stage_from(&work, &source, "fixture", "ROCm", &pins)
        };
        let staged = stage("valid").unwrap();
        assert_eq!(fs::read(staged.join(nested)).unwrap(), b"abc");
        fs::write(&member, b"def").unwrap();
        assert!(stage("tampered")
            .unwrap_err()
            .to_string()
            .contains("SHA-256 mismatch"));

        fs::remove_dir_all(source.join("rocblas")).unwrap();
        let elsewhere = root.join("rocblas/library");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::write(elsewhere.join("kernel.co"), b"abc").unwrap();
        symlink(root.join("rocblas"), source.join("rocblas")).unwrap();
        assert!(stage("escaped").is_err());
    }
}
