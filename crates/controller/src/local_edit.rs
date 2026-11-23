//! A controller-owned CPU worker per configured local source host.
use crate::{Error, dispatch::read_limited, domain::Host};
use brewery_worker::locate_edit::{self, Backend};
use brewery_worker::{scratch_eval, trace};
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
    reasons: BTreeMap<String, &'static str>,
    pub protected: Vec<PathBuf>,
}

pub struct Lane {
    pub config: PathBuf,
    pub capabilities: BTreeMap<String, bool>,
    pub configured: Vec<PathBuf>,
    pub root: PathBuf,
    worker: PathBuf,
    worker_config: Value,
    scratch_env: BTreeMap<String, Value>,
    scratch_gpu_admission: Option<PathBuf>,
    scratch_gpu_env: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Receipt {
    schema: String,
    jobs: BTreeSet<String>,
    pub formats: BTreeSet<String>,
    pub visual: bool,
    pub packed: bool,
    // Negotiated by the engine at invocation, not a permission to read sources.
    #[serde(default, rename = "proof_source_fds")]
    _proof_source_fds: bool,
}

impl Lanes {
    pub async fn prepare(data: &Path, dist: &Path, config: Option<&Value>) -> Result<Self, Error> {
        let mut result = Self::default();
        let Some(hosts) = config.and_then(|v| v["hosts"].as_object()) else {
            return Ok(result);
        };
        for (id, profile) in hosts {
            if profile["local"] == true {
                result.reasons.insert(id.clone(), if profile["config"].is_string() {
                    "The controller could not verify the local editor package. Check its dispatcher, worker, engine and editor helper."
                } else {
                    "No local worker configuration is defined for this host."
                });
            }
        }
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
        let state = dir.parent().ok_or("local editor has no state directory")?;
        let assets = dist.canonicalize().or_else(|_| std::path::absolute(dist))?;
        result.protected = vec![
            data.canonicalize()?,
            assets.clone(),
            dispatch.clone(),
            worker.clone(),
            engine.clone(),
            helper.clone(),
        ];
        result.protected.extend(tuner.iter().cloned());
        for key in [
            "BREWERY_DISPATCH_CONFIG",
            "BREWERY_ADMIN_CATALOG",
            "BREWERY_ADMIN_RUNTIME_HELPER",
            "XDG_CONFIG_HOME",
        ] {
            result.protected.extend(
                env::var_os(key)
                    .map(PathBuf::from)
                    .filter(|p| p.is_absolute()),
            );
        }
        for profile in hosts.values().filter(|profile| profile["local"] == true) {
            for key in ["worker", "config", "hf_token_file"] {
                result.protected.extend(
                    profile[key]
                        .as_str()
                        .map(PathBuf::from)
                        .filter(|p| p.is_absolute()),
                );
            }
            if let Some(original) = profile["config"]
                .as_str()
                .and_then(|p| trusted_config(Path::new(p)))
            {
                for value in [
                    &original["root"],
                    &original["engine"],
                    &original["admission"],
                    &original["pinned_publication"]["receipt"],
                    &original["managed_settings"]["catalog"],
                    &original["hf_download"]["token_file"],
                    &original["backend_weight_download_root"],
                ] {
                    result.protected.extend(
                        value
                            .as_str()
                            .map(PathBuf::from)
                            .filter(|p| p.is_absolute()),
                    );
                }
            }
        }
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
                result.reasons.insert(
                    id.clone(),
                    "The local worker configuration could not be read and verified.",
                );
                continue;
            };
            let Some(roots) = source_roots(&original, state) else {
                let reason = if original["source_roots"].is_null()
                    || original["source_roots"]
                        .as_array()
                        .is_some_and(Vec::is_empty)
                {
                    "The editor is installed, but this worker has no approved model directory."
                } else {
                    "The editor is installed, but the configured model directories did not pass validation."
                };
                result.reasons.insert(id.clone(), reason);
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
            // Dispatch-owned proof for this editor only. Never grant a source
            // root or change the original worker's qualification config here.
            if let Some(index) = profile.get("local_edit_chunk_index") {
                let pair = index["path"].as_str().zip(index["sha256"].as_str());
                let Some((path, digest)) = pair.filter(|(p, d)| {
                    index.as_object().is_some_and(|o| o.len() == 2)
                        && brewery_worker::absolute(Path::new(p)).is_ok()
                        && brewery_worker::is_digest(d)
                }) else {
                    result.reasons.insert(
                        id.clone(),
                        "The local editor chunk index needs an absolute path and a SHA-256 digest.",
                    );
                    continue;
                };
                engine_env["BREWERY_WEIGHT_CHUNK_INDEX"] = json!(path);
                engine_env["BREWERY_WEIGHT_CHUNK_INDEX_SHA256"] = json!(digest);
            }
            let worker_config = json!({
                "root":root,"engine":engine,"source_roots":roots,"engine_env":engine_env,
                "disk_reserve_bytes":original["disk_reserve_bytes"].as_u64().unwrap_or(1 << 30)
            });
            let scratch_env = [
                "BREWERY_GGUF_SAFETENSORS",
                "BREWERY_GGUF_QUANTIZER",
                "BREWERY_GGUF_HF_EXPORTER",
                "BREWERY_GGUF_PYTHON",
                "BREWERY_LLAMA_CPP",
                "BREWERY_CONVERTER_DISK_RESERVE_BYTES",
                scratch_eval::SCRIPT,
                scratch_eval::RUNNER,
                scratch_eval::RUNNER_SHA,
                scratch_eval::RUNNER_LIB_DIR,
                scratch_eval::CACHE,
                scratch_eval::CONTRAST_CACHE,
                trace::EXECUTABLE,
                trace::LIB_DIR,
                brewery_worker::ablation_search::EXECUTABLE,
                locate_edit::EXECUTABLE,
                locate_edit::LIB_DIR,
                "BREWERY_CPU_THREADS",
            ]
            .into_iter()
            .filter_map(|name| {
                original["engine_env"][name]
                    .as_str()
                    .map(|v| (name.into(), json!(v)))
            })
            .collect();
            let scratch_gpu_admission =
                original["admission"]
                    .as_str()
                    .map(PathBuf::from)
                    .filter(|path| {
                        path.is_absolute() && original["engine_env"][locate_edit::GPU_GATE] == "1"
                    });
            let scratch_gpu_env = [Backend::Cuda, Backend::Rocm]
                .into_iter()
                .flat_map(Backend::env_keys)
                .filter_map(|key| {
                    original["engine_env"][key]
                        .as_str()
                        .map(|value| (key.into(), json!(value)))
                })
                .collect();
            brewery_worker::atomic(&config, &worker_config)?;
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
                    configured: roots,
                    root,
                    worker: worker.clone(),
                    worker_config,
                    scratch_env,
                    scratch_gpu_admission,
                    scratch_gpu_env,
                },
            );
        }
        Ok(result)
    }

    pub fn get(&self, id: &str) -> Option<&Lane> {
        self.lanes.get(id)
    }

    pub fn unavailable_reason(&self, id: &str) -> &'static str {
        if id == crate::backend_hf::ID {
            return "The backend download worker could not verify its editor package.";
        }
        self.reasons.get(id).copied().unwrap_or(
            "This worker has no controller-managed local editor. Use the host that owns the model in Library.",
        )
    }
}

impl Lane {
    pub fn locate_edit_configured(&self, backend: Backend) -> bool {
        let pin = backend.pin();
        let env = if backend.is_gpu() {
            &self.scratch_gpu_env
        } else {
            &self.scratch_env
        };
        [pin.executable, pin.lib_dir].iter().all(|k| {
            env.get(*k)
                .and_then(Value::as_str)
                .is_some_and(|p| Path::new(p).is_absolute())
        })
    }

    pub fn scratch_gpu_allowed(&self) -> bool {
        self.scratch_gpu_admission.is_some()
    }

    pub fn scratch_disk_reserve(&self) -> u64 {
        self.worker_config["disk_reserve_bytes"]
            .as_u64()
            .unwrap_or(1 << 30)
    }

    pub fn job_config(&self, id: &str) -> PathBuf {
        self.config
            .with_file_name(format!("grant-{id}-dispatch.json"))
    }

    pub fn snapshot(
        &self,
        host: &str,
        id: &str,
        grants: &[brewery_worker::source_grants::SourceGrant],
        protected: &[PathBuf],
    ) -> Result<(), Error> {
        let config = self
            .config
            .with_file_name(format!("grant-{id}-worker.json"));
        let mut worker_config = self.worker_config.clone();
        worker_config["source_grant_host"] = json!(host);
        worker_config["source_grants"] = json!(grants);
        worker_config["source_grant_protected"] = json!(protected);
        brewery_worker::atomic(&config, &worker_config)?;
        brewery_worker::atomic(
            &self.job_config(id),
            &json!({
                "hosts":{host:{"local":true,"worker":self.worker,"config":config}}
            }),
        )?;
        Ok(())
    }

    pub fn scratch_config(&self, id: &str) -> PathBuf {
        self.config
            .with_file_name(format!("scratch-{id}-dispatch.json"))
    }

    pub fn snapshot_scratch(
        &self,
        host: &str,
        id: &str,
        pins: &BTreeMap<String, String>,
        backend: Backend,
    ) -> Result<(), Error> {
        use std::io::Write as _;
        let config = self
            .config
            .with_file_name(format!("scratch-{id}-worker.json"));
        let mut worker_config = self.worker_config.clone();
        let env = worker_config["engine_env"].as_object_mut().unwrap();
        env.extend(self.scratch_env.clone());
        if backend.is_gpu() {
            for key in backend.visibility_vars() {
                env.remove(*key);
            }
            env.extend(backend.env_keys().filter_map(|key| {
                self.scratch_gpu_env
                    .get(key)
                    .map(|value| (key.into(), value.clone()))
            }));
            worker_config["admission"] = json!(
                self.scratch_gpu_admission
                    .as_ref()
                    .ok_or("Missing scratch GPU admission")?
            );
        }
        worker_config["scratch_access"] = json!({"job_id":id,"paths":pins});
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&config)?;
        serde_json::to_writer(&mut file, &worker_config)?;
        file.flush()?;
        file.sync_all()?;
        brewery_worker::atomic(
            &self.scratch_config(id),
            &json!({"hosts":{host:{"local":true,"worker":self.worker,"config":config}}}),
        )?;
        Ok(())
    }

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
    if config["source_roots"].is_null() {
        return Some(Vec::new());
    }
    let values = config["source_roots"].as_array()?;
    if values.len() > 64 {
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
