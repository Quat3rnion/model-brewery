//! Stage the Brewery-owned HGN runtime and verify the model's companion assets.
use crate::{gguf_conversion::Tool, integrity};
use anyhow::{Context, Result, ensure};
use brewery_worker::trace::{
    HGN_ASSETS, HGN_PYTHON, HGN_PYTHON_SHA256, HGN_RUNTIME, HGN_RUNTIME_SHA256,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const SOURCES: &[(&str, &[u8])] = &[
    (
        "forward.py",
        include_bytes!("../../../../tools/hgn_trace/forward.py"),
    ),
    (
        "launch.py",
        include_bytes!("../../../../tools/hgn_trace/launch.py"),
    ),
    (
        "requirements.lock.txt",
        include_bytes!("../../../../tools/hgn_trace/requirements.lock.txt"),
    ),
    (
        "trace.py",
        include_bytes!("../../../../tools/hgn_trace/trace.py"),
    ),
    (
        "wsp/hgn.py",
        include_bytes!("../../../../tools/hgn_trace/wsp/hgn.py"),
    ),
    (
        "wsp/vendor/LICENSE",
        include_bytes!("../../../../tools/hgn_trace/wsp/vendor/LICENSE"),
    ),
    (
        "wsp/vendor/hgnenc.py",
        include_bytes!("../../../../tools/hgn_trace/wsp/vendor/hgnenc.py"),
    ),
    (
        "wsp/vendor/hgnht.py",
        include_bytes!("../../../../tools/hgn_trace/wsp/vendor/hgnht.py"),
    ),
    (
        "wsp/vendor/hgnlib.py",
        include_bytes!("../../../../tools/hgn_trace/wsp/vendor/hgnlib.py"),
    ),
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Assets {
    sidecar: Pin,
    config: Pin,
    tokenizer: Pin,
}

pub(super) struct Runtime {
    python: Tool,
    entry: PathBuf,
    assets: Vec<(&'static str, Pin, integrity::Stamp)>,
}

impl Runtime {
    pub(super) fn stage(work: &Path, model_sha: &str) -> Result<Self> {
        let configured =
            PathBuf::from(env::var_os(HGN_PYTHON).context("HGN trace Python is not configured")?);
        ensure!(configured.is_absolute(), "{HGN_PYTHON} must be absolute");
        ensure!(
            configured.canonicalize()? == configured,
            "{HGN_PYTHON} must be a canonical, copied venv executable"
        );
        let python = Tool::from_path(configured)?;
        ensure!(
            python.sha
                == env::var(HGN_PYTHON_SHA256)
                    .context("HGN trace Python hash is not configured")?,
            "HGN trace Python SHA-256 mismatch"
        );

        let root = work.join("hgn-runtime");
        let mut hash = Sha256::new();
        for &(name, bytes) in SOURCES {
            let path = root.join(name);
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(&path, bytes)?;
            let (member, _) = integrity::file_digest(&path, name.to_owned())?;
            hash.update(format!("{name}\0{}\n", member.sha256));
        }
        ensure!(
            format!("{:x}", hash.finalize()) == HGN_RUNTIME_SHA256.trim_end(),
            "HGN trace runtime SHA-256 mismatch"
        );
        let entry = root.join("launch.py");

        let registry_path =
            PathBuf::from(env::var_os(HGN_ASSETS).context("HGN trace assets are not configured")?);
        let meta = integrity::canonical(&registry_path)?;
        ensure!(
            meta.is_file() && meta.len() <= 1 << 20,
            "invalid HGN trace asset registry"
        );
        let mut registry: BTreeMap<String, Assets> =
            serde_json::from_slice(&fs::read(registry_path)?)?;
        let pins = registry
            .remove(model_sha)
            .context("HGN trace assets are not pinned for this model SHA-256")?;
        let assets = [
            ("sidecar", pins.sidecar),
            ("config", pins.config),
            ("tokenizer", pins.tokenizer),
        ]
        .into_iter()
        .map(|(name, pin)| {
            ensure!(
                brewery_worker::is_digest(&pin.sha256),
                "invalid HGN {name} SHA-256 pin"
            );
            let (member, stamp) = integrity::file_digest(&pin.path, name.into())?;
            ensure!(member.sha256 == pin.sha256, "HGN {name} SHA-256 mismatch");
            Ok((name, pin, stamp))
        })
        .collect::<Result<_>>()?;
        Ok(Self {
            python,
            entry,
            assets,
        })
    }

    pub(super) fn command(&self, work: &Path, threads: usize) -> Command {
        let mut cmd = self.python.command(work, threads);
        cmd.env("OPENBLAS_NUM_THREADS", "1")
            .env("TOKENIZERS_PARALLELISM", "false")
            .args(["-I", "-B"])
            .arg(&self.entry);
        for (name, pin, _) in &self.assets {
            cmd.arg(format!("--{name}")).arg(&pin.path);
        }
        cmd
    }

    pub(super) fn capture(&self, cmd: &mut Command, work: &Path) -> Result<()> {
        let check = || {
            for (_, pin, stamp) in &self.assets {
                stamp.check(&pin.path)?;
            }
            Ok::<_, anyhow::Error>(())
        };
        check()?;
        self.python
            .capture(cmd, work)
            .context("HGN CPU trace failed")?;
        check()
    }

    pub(super) fn receipt(&self) -> Value {
        let assets: BTreeMap<_, _> = self
            .assets
            .iter()
            .map(|(label, pin, _)| (*label, &pin.sha256))
            .collect();
        json!({"runtime":HGN_RUNTIME, "runtime_sha256":HGN_RUNTIME_SHA256.trim_end(),
               "executable_sha256":self.python.sha, "assets_sha256":assets})
    }
}
