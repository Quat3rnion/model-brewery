//! Local companion processes for the controller-owned download lane.
use crate::{
    Error,
    domain::{Host, KINDS},
};
use serde_json::{Value, json};
use std::{
    env,
    fs::{self, DirBuilder},
    io,
    os::unix::fs::{DirBuilderExt as _, MetadataExt as _},
    path::{Path, PathBuf},
};

pub const ID: &str = "backend-hf";
// The engine enforces a disk budget from the resolved file sizes rather
// than imposing a model-size ceiling independent of the storage volume.
const MAX_BYTES: u64 = i64::MAX as u64;
const DISK_RESERVE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

pub struct Lane {
    pub dispatch: PathBuf,
    pub dispatch_config: PathBuf,
    pub edit_config: Option<PathBuf>,
    pub calibration: Option<PathBuf>,
    pub edit_formats: std::collections::BTreeSet<String>,
    local: Value,
}

impl Lane {
    pub async fn prepare(data: &Path, dist: &Path) -> Result<Option<Self>, Error> {
        let installed = env::current_exe()?;
        let dir = installed
            .parent()
            .ok_or("controller executable has no directory")?;
        let (Some(dispatch), Some(worker), Some(engine), Some(hfr)) = (
            executable(dir, "BREWERY_HF_DISPATCH", "brewery-dispatch"),
            executable(dir, "BREWERY_HF_WORKER", "brewery-worker"),
            executable(dir, "BREWERY_HF_ENGINE", "brewery-engine"),
            executable(dir, "BREWERY_HF_HFR", "hfr"),
        ) else {
            return Ok(None);
        };
        let root = data.canonicalize()?.join(ID);
        let dist = dist.canonicalize().or_else(|_| std::path::absolute(dist))?;
        if root.starts_with(dist) {
            return Err("download storage must be outside served web assets".into());
        }
        private_dir(&root)?;
        let worker_root = root.join("worker");
        private_dir(&worker_root)?;
        let credentials = root.join("credentials");
        private_dir(&credentials)?;
        let token = credentials.join("token");
        let config = root.join("worker.json");
        let dispatch_config = root.join("dispatch.json");
        let editor = if let Some(path) =
            executable(dir, "BREWERY_HF_WEIGHT_EDIT", "brewery-weight-edit")
        {
            crate::local_edit::probe(&path, &["weight_inspect", "weight_propose", "weight_apply"])
                .await
                .map(|caps| (path, caps))
        } else {
            None
        };
        let edit_formats = editor
            .as_ref()
            .map(|(_, caps)| {
                caps.formats
                    .iter()
                    .filter(|name| matches!(name.as_str(), "gguf" | "hf" | "safetensors"))
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>()
            })
            .unwrap_or_default();
        let editor = editor.filter(|_| !edit_formats.is_empty());
        let tuner = if let (Some((_, edit)), Some(path)) = (
            &editor,
            executable(dir, "BREWERY_HF_WEIGHT_TUNE", "brewery-weight-tune"),
        ) {
            crate::local_edit::probe(&path, &["weight_tune"])
                .await
                .filter(|tune| {
                    edit_formats.is_subset(&tune.formats)
                        && (!edit.visual || tune.visual)
                        && (!edit.packed || !edit_formats.contains("gguf") || tune.packed)
                })
                .map(|_| path)
        } else {
            None
        };
        let visual = editor.as_ref().is_some_and(|(_, caps)| caps.visual);
        let packed = editor
            .as_ref()
            .is_some_and(|(_, caps)| caps.packed && edit_formats.contains("gguf"));
        let safetensors = edit_formats.contains("hf") || edit_formats.contains("safetensors");
        let calibration = tuner.as_ref().map(|_| root.join("calibration"));
        if let Some(path) = &calibration {
            private_dir(path)?;
        }
        let edit_config = if let Some((editor, _)) = &editor {
            let edit_root = root.join("editor");
            private_dir(&edit_root)?;
            let config = root.join("editor.json");
            let mut options = json!({
                "root":edit_root,"engine":engine,"disk_reserve_bytes":DISK_RESERVE_BYTES,
                "backend_weight_download_root":worker_root,
                "engine_env":{
                    "BREWERY_WEIGHT_EDIT":editor,
                    "BREWERY_WEIGHT_PACKED": if packed {"1"} else {"0"},
                    "BREWERY_WEIGHT_VISUAL": if visual {"1"} else {"0"},
                    "BREWERY_WEIGHT_SAFETENSORS": if safetensors {"1"} else {"0"},
                    "BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES":"67108864",
                    "BREWERY_CPU_THREADS":"2",
                    "OPENBLAS_NUM_THREADS":"2","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
                    "PYTHONNOUSERSITE":"1"
                }
            });
            if let Some(tuner) = &tuner {
                options["engine_env"]["BREWERY_WEIGHT_TUNE"] = json!(tuner);
                options["backend_weight_calibration_root"] = json!(calibration);
            }
            brewery_worker::atomic(&config, &options)?;
            let dispatch = root.join("editor-dispatch.json");
            brewery_worker::atomic(
                &dispatch,
                &json!({
                    "hosts":{ID:{"local":true,"worker":worker,"config":config}}
                }),
            )?;
            Some(dispatch)
        } else {
            None
        };
        let capabilities = KINDS
            .iter()
            .map(|&kind| {
                (
                    kind.into(),
                    json!(
                        kind == "hf_download"
                            || (kind == "weight_tune" && tuner.is_some())
                            || (editor.is_some()
                                && matches!(
                                    kind,
                                    "weight_inspect" | "weight_propose" | "weight_apply"
                                ))
                    ),
                )
            })
            .collect::<serde_json::Map<String, Value>>();
        let local = json!({
            "name": "Backend downloads",
            "capabilities": capabilities,
            "local": true,
            "worker": worker,
            "config": config,
            "hf_token_file": token
        });
        let mut local = local;
        local["capabilities"]["weight_visual"] = json!(visual);
        local["capabilities"]["weight_safetensors"] = json!(safetensors);
        local["capabilities"]["weight_packed"] = json!(packed);
        brewery_worker::atomic(
            &config,
            &json!({
                "root": worker_root,
                "engine": engine,
                "disk_reserve_bytes": DISK_RESERVE_BYTES,
                "hf_download": {"hfr": hfr, "token_file": token, "max_bytes": MAX_BYTES}
            }),
        )?;
        brewery_worker::atomic(&dispatch_config, &json!({"hosts": {ID: local}}))?;
        Ok(Some(Self {
            dispatch,
            dispatch_config,
            edit_config,
            calibration,
            edit_formats,
            local,
        }))
    }

    pub fn host(&self) -> Result<Host, String> {
        Host::from_config(ID, &self.local)
    }

    pub fn local_config(&self) -> &Value {
        &self.local
    }

    pub fn config_for(&self, kind: &str) -> Option<&Path> {
        match kind {
            "hf_download" => Some(&self.dispatch_config),
            "weight_inspect" | "weight_propose" | "weight_apply" => self.edit_config.as_deref(),
            "weight_tune" if self.calibration.is_some() => self.edit_config.as_deref(),
            _ => None,
        }
    }
}

fn executable(dir: &Path, key: &str, sibling: &str) -> Option<PathBuf> {
    let path = env::var_os(key).map_or_else(|| dir.join(sibling), PathBuf::from);
    brewery_worker::absolute(&path).ok()?;
    let path = path.canonicalize().ok()?;
    let m = path.metadata().ok()?;
    (m.is_file()
        && (m.uid() == 0 || m.uid() == unsafe { libc::geteuid() })
        && m.mode() & 0o6022 == 0
        && m.mode() & 0o100 != 0)
        .then_some(path)
}

fn private_dir(path: &Path) -> Result<(), Error> {
    match DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    let m = fs::symlink_metadata(path)?;
    if !m.is_dir() || m.mode() & 0o7777 != 0o700 || m.uid() != unsafe { libc::geteuid() } {
        return Err("backend download directory must be private and controller-owned".into());
    }
    Ok(())
}
