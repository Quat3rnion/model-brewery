//! A controller-owned CPU worker per configured local source host.
use crate::{Error, dispatch::read_limited, domain::Host};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::Read,
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{process::Command, time::timeout};

pub const CAPS: &[&str] = &[
    "weight_inspect",
    "weight_propose",
    "weight_apply",
    "weight_tune",
    "weight_visual",
    "weight_hgn",
    "weight_packed",
    "weight_safetensors",
];
const EDIT: &[&str] = &["weight_inspect", "weight_propose", "weight_apply"];

#[derive(Default)]
pub struct Lanes {
    pub dispatch: PathBuf,
    lanes: BTreeMap<String, Lane>,
}

pub struct Lane {
    pub config: PathBuf,
    pub capabilities: BTreeMap<String, bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Receipt {
    schema: String,
    jobs: BTreeSet<String>,
    pub formats: BTreeSet<String>,
    pub visual: bool,
    pub packed: bool,
}

impl Lanes {
    pub async fn prepare(data: &Path, dist: &Path, config: Option<&Value>) -> Result<Self, Error> {
        let mut result = Self::default();
        let Some(hosts) = config.and_then(|v| v["hosts"].as_object()) else {
            return Ok(result);
        };
        if !hosts
            .values()
            .any(|host| host["local"] == true && host["config"].is_string())
        {
            return Ok(result);
        }
        let exe = env::current_exe()?;
        let parent = exe
            .parent()
            .ok_or("controller executable has no directory")?;
        let (Some(dispatch), Some(worker), Some(engine), Some(helper)) = (
            executable(parent, "DISPATCH", "brewery-dispatch"),
            executable(parent, "WORKER", "brewery-worker"),
            executable(parent, "ENGINE", "brewery-engine"),
            executable(parent, "HELPER", "brewery-weight-edit"),
        ) else {
            return Ok(result);
        };
        let Some(edit) = probe(&helper, EDIT)
            .await
            .filter(|r| r.formats.contains("gguf"))
        else {
            return Ok(result);
        };
        let tuner = if let Some(path) = executable(parent, "TUNER", "brewery-weight-tune") {
            probe(&path, &["weight_tune"])
                .await
                .filter(|tune| {
                    edit.formats.is_subset(&tune.formats)
                        && (!edit.visual || tune.visual)
                        && (!edit.packed || tune.packed)
                })
                .map(|_| path)
        } else {
            None
        };
        let dir = data.canonicalize()?.join("local-edit");
        let assets = dist.canonicalize().or_else(|_| std::path::absolute(dist))?;
        if dir.starts_with(assets) {
            return Err("local editor storage must be outside served assets".into());
        }
        private_dir(&dir)?;
        result.dispatch = dispatch;
        for (id, profile) in hosts {
            if profile["local"] != true
                || !profile["ssh"].is_null()
                || !profile["managed_ssh"].is_null()
            {
                continue;
            }
            let Some(path) = profile["config"].as_str() else {
                continue;
            };
            let Some(original) = trusted_config(Path::new(path)) else {
                continue;
            };
            let Some(roots) = source_roots(&original, &dir) else {
                continue;
            };
            // The original executor is never reused or reconfigured.
            let lane_dir = dir.join(id);
            let root = lane_dir.join("worker");
            private_dir(&lane_dir)?;
            private_dir(&root)?;
            let config = lane_dir.join("worker.json");
            let mut engine_env = json!({
                "BREWERY_WEIGHT_EDIT":helper,
                "BREWERY_CPU_THREADS":"2","OPENBLAS_NUM_THREADS":"2","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
                "BREWERY_ALLOW_GPU_LOAD":"0","CUDA_VISIBLE_DEVICES":"","HIP_VISIBLE_DEVICES":"",
                "ROCR_VISIBLE_DEVICES":"","PYTHONNOUSERSITE":"1"
            });
            if let Some(tuner) = &tuner {
                engine_env["BREWERY_WEIGHT_TUNE"] = json!(tuner);
            }
            let hgn = edit.formats.contains("hgn");
            let safetensors = edit.formats.contains("safetensors") && edit.formats.contains("hf");
            if safetensors {
                // Match backend-hf: verify small files in full, require
                // operator-provided proof for larger checkpoints.
                engine_env["BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES"] = json!("67108864");
            }
            for (name, enabled) in [
                ("BREWERY_WEIGHT_VISUAL", edit.visual),
                ("BREWERY_WEIGHT_PACKED", edit.packed),
                ("BREWERY_WEIGHT_HGN", hgn),
                ("BREWERY_WEIGHT_SAFETENSORS", safetensors),
            ] {
                if enabled {
                    engine_env[name] = json!("1");
                }
            }
            // Only the operator's explicit editor proof controls can carry over.
            for name in [
                "BREWERY_WEIGHT_SAFETENSORS_PUBLICATION",
                "BREWERY_WEIGHT_CHUNK_INDEX",
                "BREWERY_WEIGHT_CHUNK_INDEX_SHA256",
            ] {
                if let Some(value) = original["engine_env"][name].as_str() {
                    engine_env[name] = json!(value);
                }
            }
            brewery_worker::atomic(
                &config,
                &json!({
                    "root":root,"engine":engine,"source_roots":roots,"engine_env":engine_env,
                    "disk_reserve_bytes":original["disk_reserve_bytes"].as_u64().unwrap_or(1 << 30)
                }),
            )?;
            let dispatch = lane_dir.join("dispatch.json");
            brewery_worker::atomic(
                &dispatch,
                &json!({
                    "hosts":{id:{"local":true,"worker":worker,"config":config}}
                }),
            )?;
            let capabilities = CAPS
                .iter()
                .map(|&key| {
                    (
                        key.into(),
                        match key {
                            "weight_tune" => tuner.is_some(),
                            "weight_visual" => edit.visual,
                            "weight_hgn" => hgn,
                            "weight_packed" => edit.packed,
                            "weight_safetensors" => safetensors,
                            _ => true,
                        },
                    )
                })
                .collect();
            result.lanes.insert(
                id.clone(),
                Lane {
                    config: dispatch,
                    capabilities,
                },
            );
        }
        Ok(result)
    }

    pub fn get(&self, id: &str) -> Option<&Lane> {
        self.lanes.get(id)
    }
}

impl Lane {
    pub fn advertise(&self, host: &mut Host) {
        host.local_edit = true;
        for (cap, enabled) in &self.capabilities {
            host.capabilities.insert(cap.clone(), *enabled);
            if *enabled {
                host.reasons.remove(cap);
            } else {
                host.reasons.insert(
                    cap.clone(),
                    "Not supported by the installed CPU editor".into(),
                );
            }
        }
    }
}

pub fn disable(host: &mut Host) {
    for &cap in CAPS {
        host.capabilities.insert(cap.into(), false);
        host.reasons
            .insert(cap.into(), "Weight editor disabled in Settings".into());
    }
}

pub(crate) async fn probe(path: &Path, jobs: &[&str]) -> Option<Receipt> {
    let mut child = Command::new(path)
        .arg("--capabilities")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    let (bytes, status) = timeout(Duration::from_secs(5), async {
        tokio::try_join!(read_limited(stdout, 4096), child.wait())
    })
    .await
    .ok()?
    .ok()?;
    let receipt: Receipt = serde_json::from_slice(&bytes).ok()?;
    (status.success()
        && receipt.schema == "brewery.weight-runtime.v1"
        && receipt
            .jobs
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == jobs.iter().copied().collect()
        && !receipt.formats.is_empty()
        && receipt
            .formats
            .iter()
            .all(|f| matches!(f.as_str(), "gguf" | "hgn" | "hf" | "safetensors")))
    .then_some(receipt)
}

fn executable(dir: &Path, key: &str, sibling: &str) -> Option<PathBuf> {
    let path = env::var_os(format!("BREWERY_LOCAL_EDIT_{key}"))
        .map_or_else(|| dir.join(sibling), PathBuf::from);
    if !path.is_absolute() {
        return None;
    }
    let path = path.canonicalize().ok()?;
    let meta = path.metadata().ok()?;
    (trusted(&meta) && meta.mode() & 0o100 != 0).then_some(path)
}

fn trusted(meta: &fs::Metadata) -> bool {
    meta.is_file()
        && (meta.uid() == 0 || meta.uid() == unsafe { libc::geteuid() })
        && meta.mode() & 0o6022 == 0
}

fn trusted_config(path: &Path) -> Option<Value> {
    if !path.is_absolute() || path.canonicalize().ok()? != path {
        return None;
    }
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .ok()?;
    if !trusted(&file.metadata().ok()?) {
        return None;
    }
    let mut data = Vec::new();
    file.take(128 * 1024 + 1).read_to_end(&mut data).ok()?;
    (data.len() <= 128 * 1024)
        .then(|| serde_json::from_slice(&data).ok())
        .flatten()
}

fn source_roots(config: &Value, state: &Path) -> Option<Vec<PathBuf>> {
    let values = config["source_roots"].as_array()?;
    if values.is_empty() || values.len() > 64 {
        return None;
    }
    values
        .iter()
        .map(|value| {
            let path = PathBuf::from(value.as_str()?);
            (path.is_absolute()
                && path.canonicalize().ok()? == path
                && path.is_dir()
                && !path.starts_with(state)
                && !state.starts_with(&path))
            .then_some(path)
        })
        .collect()
}

fn private_dir(path: &Path) -> Result<(), Error> {
    match fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_dir() || meta.mode() & 0o7777 != 0o700 || meta.uid() != unsafe { libc::geteuid() } {
        return Err("local editor storage must be private and controller-owned".into());
    }
    Ok(())
}
