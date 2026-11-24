mod cache;
mod progress;
mod publication;
mod stat;
mod suggestions;

use anyhow::{Context, Result, bail, ensure};
use brewery_worker::*;
use cache::Cache;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json, to_value};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::Write,
    os::unix::{fs::MetadataExt, process::CommandExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    root: PathBuf,
    #[serde(default)]
    source_roots: Vec<PathBuf>,
    #[serde(default)]
    source_grants: Vec<source_grants::SourceGrant>,
    source_grant_host: Option<String>,
    #[serde(default)]
    source_grant_protected: Vec<PathBuf>,
    scratch_access: Option<ScratchAccess>,
    engine: PathBuf,
    admission: Option<PathBuf>,
    exl3_gpu_queue: Option<PathBuf>,
    #[serde(default = "reserve")]
    disk_reserve_bytes: u64,
    #[serde(default)]
    engine_env: BTreeMap<String, String>,
    pinned_publication: Option<publication::Pin>,
    managed_settings: Option<operational::Policy>,
    hf_download: Option<hf_download::Config>,
    backend_weight_download_root: Option<PathBuf>,
    backend_weight_calibration_root: Option<PathBuf>,
    resource_bounds: Option<resources::Bounds>,
    hy4_plan: Option<Hy4Plan>,
    weight_editor_host: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScratchAccess {
    job_id: String,
    paths: BTreeMap<PathBuf, String>,
}

impl ScratchAccess {
    fn permits(&self, action: &str, job: &Job) -> bool {
        if action != "start" {
            return false;
        }
        let kinds = || job.inputs.iter().map(|i| i.kind.as_str());
        let pinned = match job.kind.as_str() {
            scratch_eval::GGUF
                if job.prompt.is_none()
                    && scratch_eval::gguf_options(&job.options)
                    && kinds().eq(["safetensors", "assets"]) =>
            {
                &job.inputs[..1]
            }
            scratch_eval::CHECKS
                if job.prompt.is_none()
                    && scratch_eval::check_options(&job.options).is_ok()
                    && kinds().eq(scratch_eval::CHECK_INPUTS) =>
            {
                &job.inputs[..]
            }
            trace::JOB if kinds().eq(["gguf"]) => &job.inputs[..],
            crate::circuit::JOB
                if job.prompt.is_none()
                    && kinds().eq(["gguf"])
                    && crate::circuit::Request::parse(&job.options).is_ok_and(|req| {
                        self.paths.get(&job.inputs[0].path) == Some(&req.model_sha256)
                    }) =>
            {
                &job.inputs[..]
            }
            sensitivity::JOB
                if job.prompt.is_none()
                    && job.options.get(sensitivity::REVISION_PIN).is_some()
                    && sensitivity::Request::resolved(&job.options).is_ok()
                    && kinds().eq(["gguf"]) =>
            {
                &job.inputs[..]
            }
            ablation_search::SCREEN
                if job.prompt.is_none()
                    && kinds().eq(["gguf"])
                    && ablation_search::ScreenRequest::parse(&job.options).is_ok() =>
            {
                &job.inputs[..]
            }
            locate_edit::JOB
                if job.prompt.is_none()
                    && locate_edit::Request::parse(&job.options).is_ok()
                    && kinds().eq(locate_edit::INPUTS) =>
            {
                &job.inputs[..]
            }
            kind if heretic::job(kind)
                && job.prompt.is_none()
                && (kinds().eq(["safetensors"]) || kinds().eq(["safetensors", "assets"]))
                && heretic::Options::parse(kind, &job.options).is_ok() =>
            {
                &job.inputs[..1]
            }
            _ => return false,
        };
        job.inputs.iter().all(|i| i.host == pinned[0].host)
            && self
                .paths
                .keys()
                .all(|p| pinned.iter().any(|i| i.path == *p))
            && pinned.iter().all(|i| self.paths.contains_key(&i.path))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Hy4Plan {
    host: String,
    source: PathBuf,
    halo_tools: PathBuf,
    encode: Option<Hy4Encode>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Hy4Encode {
    control: PathBuf,
    preflight: PathBuf,
    quantizer: PathBuf,
}

impl Hy4Plan {
    fn validate(&self, action: &str, job: &Job) -> bool {
        action == "start"
            && hy4::capability(&job.kind, &job.options)
                == Some(if self.encode.is_some() {
                    hy4::ENCODE
                } else {
                    hy4::CAPABILITY
                })
            && job.prompt.is_none()
            && matches!(job.inputs.as_slice(), [i] if i.kind == "hf" && i.host == self.host
                && i.path.as_os_str() == self.source.as_os_str())
    }
}
fn reserve() -> u64 {
    1024 * 1024 * 1024
}

fn standalone_gguf(job: &Job) -> bool {
    job.options["output_format"] == "gguf"
        && job
            .inputs
            .first()
            .is_some_and(|input| input.kind == "safetensors")
}

impl Config {
    fn configured_source_root(&self, path: &Path) -> Option<&Path> {
        self.source_roots
            .iter()
            .filter(|root| {
                root.is_absolute()
                    && !root.starts_with(&self.root)
                    && !self.root.starts_with(root)
                    && path.starts_with(root)
            })
            .max_by_key(|root| root.components().count())
            .map(PathBuf::as_path)
    }

    fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }

    fn cache_source_root(&self, path: &Path) -> Option<PathBuf> {
        let cache = self.cache_dir();
        let mut parts = path.strip_prefix(&cache).ok()?.components();
        let digest = parts.next()?.as_os_str().to_str()?;
        (self.backend_weight_download_root.is_none()
            && is_digest(digest)
            && parts.next()?.as_os_str() == "data")
            .then(|| cache.join(digest).join("data"))
    }

    fn load() -> Result<(Self, PathBuf)> {
        let path = env::var_os("BREWERY_WORKER_CONFIG")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("HOME")
                    .map(|h| PathBuf::from(h).join(".config/model-brewery/worker.json"))
            })
            .context("set BREWERY_WORKER_CONFIG")?
            .canonicalize()?;
        let cfg: Config = load(&path)?;
        if let Some(access) = &cfg.scratch_access {
            token(&access.job_id)?;
            let md = fs::symlink_metadata(&path)?;
            ensure!(
                md.is_file()
                    && md.uid() == unsafe { libc::geteuid() }
                    && md.mode() & 0o077 == 0
                    && (1..=4).contains(&access.paths.len())
                    && cfg.source_grants.is_empty()
                    && (cfg.admission.is_none()
                        || cfg
                            .engine_env
                            .get(locate_edit::GPU_GATE)
                            .is_some_and(|v| v == "1"))
                    && cfg.pinned_publication.is_none()
                    && cfg.managed_settings.is_none()
                    && cfg.hf_download.is_none()
                    && cfg.backend_weight_download_root.is_none()
                    && cfg.backend_weight_calibration_root.is_none()
                    && cfg.hy4_plan.is_none()
                    && cfg.weight_editor_host.is_none(),
                "Scratch access requires a private local worker config without other execution policies"
            );
            for (input, digest) in &access.paths {
                absolute(input)?;
                ensure!(
                    is_digest(digest)
                        && !input.starts_with(&cfg.root)
                        && input != &path
                        && input != &cfg.engine,
                    "Invalid scratch input pin"
                );
                let name = input
                    .file_name()
                    .and_then(|n| n.to_str())
                    .context("Invalid scratch input pin")?;
                ensure!(
                    cache::split_gguf(name).is_none(),
                    "Scratch inputs must be single files"
                );
            }
        }
        if !cfg.source_grants.is_empty() {
            let host = cfg
                .source_grant_host
                .as_deref()
                .context("Missing source grant host")?;
            token(host)?;
            let md = fs::symlink_metadata(&path)?;
            ensure!(
                md.is_file()
                    && md.uid() == unsafe { libc::geteuid() }
                    && md.mode() & 0o022 == 0
                    && cfg.source_grants.len() <= 128
                    && cfg.source_grants.iter().all(|grant| grant.host == host)
                    && cfg.admission.is_none()
                    && cfg.pinned_publication.is_none()
                    && cfg.managed_settings.is_none()
                    && cfg.hf_download.is_none()
                    && cfg.backend_weight_download_root.is_none()
                    && cfg.backend_weight_calibration_root.is_none()
                    && !cfg
                        .engine_env
                        .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
                    && env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_none(),
                "Source grants require a trusted local editor config without other execution policies"
            );
        }
        if let Some(bounds) = &cfg.resource_bounds {
            bounds.check_ranges()?;
        }
        absolute(&cfg.root)?;
        absolute(&cfg.engine)?;
        if let Some(host) = &cfg.weight_editor_host {
            token(host)?;
            ensure!(
                cfg.source_roots.len() == 1
                    && cfg.source_grants.is_empty()
                    && cfg.admission.is_none()
                    && cfg.hy4_plan.is_none()
                    && cfg.pinned_publication.is_none()
                    && cfg.managed_settings.is_none()
                    && cfg.hf_download.is_none()
                    && cfg.backend_weight_download_root.is_none()
                    && cfg.backend_weight_calibration_root.is_none(),
                "Remote weight editor requires a dedicated worker and one model directory"
            );
        }
        if let Some(p) = &cfg.admission {
            absolute(p)?;
        }
        if let Some(p) = &cfg.exl3_gpu_queue {
            absolute(p)?;
        }
        if let Some(plan) = &cfg.hy4_plan {
            token(&plan.host)?;
            absolute(&plan.source)?;
            absolute(&plan.halo_tools)?;
            if let Some(encode) = &plan.encode {
                for path in [
                    &encode.control,
                    &encode.preflight,
                    &encode.quantizer,
                    &cfg.engine,
                ] {
                    absolute(path)?;
                    ensure!(
                        !path.starts_with(&cfg.root),
                        "Hy4 encoder pins must be outside worker state"
                    );
                }
            }
            ensure!(
                cfg.source_roots.is_empty()
                    && cfg.source_grants.is_empty()
                    && cfg.admission.is_none()
                    && cfg.pinned_publication.is_none()
                    && cfg.managed_settings.is_none()
                    && cfg.hf_download.is_none()
                    && cfg.backend_weight_download_root.is_none()
                    && cfg.backend_weight_calibration_root.is_none()
                    && cfg.engine_env.is_empty()
                    && !cfg.root.starts_with(&plan.source)
                    && !plan.source.starts_with(&cfg.root),
                "Hy4 planning requires a dedicated worker and separate state root"
            );
        }
        if let Some(download) = &cfg.hf_download {
            download.validate()?;
            ensure!(
                download
                    .token_file
                    .as_ref()
                    .is_none_or(|p| !p.starts_with(&cfg.root)),
                "Hugging Face credentials must stay outside worker state"
            );
            ensure!(
                cfg.pinned_publication.is_none()
                    && cfg.admission.is_none()
                    && cfg.managed_settings.is_none()
                    && cfg.source_roots.is_empty()
                    && cfg.engine_env.is_empty(),
                "Hugging Face downloads require a dedicated worker"
            );
        }
        if let Some(root) = &cfg.backend_weight_download_root {
            operational::private_root(root)?;
            ensure!(
                cfg.hf_download.is_none()
                    && cfg.source_roots.is_empty()
                    && cfg.admission.is_none()
                    && cfg.pinned_publication.is_none()
                    && cfg.managed_settings.is_none()
                    && !root.starts_with(&cfg.root)
                    && !cfg.root.starts_with(root),
                "invalid dedicated backend editor configuration"
            );
        }
        if let Some(root) = &cfg.backend_weight_calibration_root {
            operational::private_root(root)?;
            ensure!(
                cfg.backend_weight_download_root
                    .as_ref()
                    .is_some_and(
                        |downloads| !root.starts_with(downloads) && !downloads.starts_with(root)
                    )
                    && !root.starts_with(&cfg.root)
                    && !cfg.root.starts_with(root),
                "invalid backend calibration root"
            );
        }
        Ok((cfg, path))
    }
}

struct Worker {
    cfg: Config,
    cfg_path: PathBuf,
    jobs: PathBuf,
    cache: Cache,
}

#[derive(Serialize, Deserialize)]
struct Record {
    job: Option<Job>,
    action: String,
    reply: Reply,
    cancel: bool,
    inputs: BTreeMap<usize, Input>,
    pins: BTreeSet<String>,
    #[serde(default)]
    boot: Option<String>,
    #[serde(default)]
    started: Option<u64>,
    #[serde(default)]
    started_ms: Option<u64>,
}

fn boot_id() -> Result<String> {
    Ok(fs::read_to_string("/proc/sys/kernel/random/boot_id")?)
}

impl Worker {
    fn resource_enabled(&self) -> bool {
        self.cfg.pinned_publication.is_none()
            && !self
                .cfg
                .engine_env
                .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
            && env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_none()
            && self.cfg.hf_download.is_none()
            && self.cfg.backend_weight_download_root.is_none()
    }

    fn resource_bounds(&self) -> resources::Bounds {
        self.cfg
            .resource_bounds
            .unwrap_or(resources::Bounds::standard(true))
    }

    fn resource_floor(&self, r: &Record) -> u64 {
        let requested = r.job.as_ref().and_then(|j| j.resources.disk_reserve_bytes);
        self.disk_floor(requested)
    }

    fn disk_floor(&self, requested: Option<u64>) -> u64 {
        let floor = if self.resource_enabled() {
            self.cfg
                .resource_bounds
                .and_then(|b| b.disk_reserve_bytes)
                .map_or(0, |b| b.min)
        } else {
            0
        };
        self.cfg
            .disk_reserve_bytes
            .max(floor)
            .max(requested.unwrap_or(0))
    }

    fn cache_for(&self, r: &Record) -> Cache {
        Cache {
            reserve: self.resource_floor(r),
            complete: self.cache.complete.clone(),
            partial: self.cache.partial.clone(),
        }
    }

    fn conversion_slot(&self, job: &Job) -> Result<bool> {
        if !matches!(job.kind.as_str(), "convert" | scratch_eval::GGUF) || !self.resource_enabled()
        {
            return Ok(true);
        }
        let ceiling = self.resource_bounds().concurrency.map_or(1, |r| r.max);
        let cap = job.resources.concurrency.unwrap_or(1).min(ceiling);
        let mut active = 0;
        for entry in fs::read_dir(&self.jobs)? {
            let entry = entry?;
            let id = entry.file_name();
            if let Some(r) = self.get(id.to_str().context("invalid worker job name")?)?
                && r.reply.state == "running"
            {
                active += 1;
                if active >= cap {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn active_input(r: &Record) -> Option<(usize, usize)> {
        let count = r.job.as_ref()?.inputs.len();
        (0..count)
            .find(|i| !r.inputs.contains_key(i))
            .map(|i| (i, count))
    }

    fn open() -> Result<Self> {
        let (cfg, path) = Config::load()?;
        fs::create_dir_all(&cfg.root)?;
        let md = fs::symlink_metadata(&cfg.root)?;
        ensure!(
            cfg.root.canonicalize()? == cfg.root
                && md.is_dir()
                && md.uid() == unsafe { libc::geteuid() }
                && md.mode() & 0o077 == 0,
            "worker root must be canonical, private and owned by the worker"
        );
        for r in &cfg.source_roots {
            ensure!(
                r.is_dir() && r.canonicalize()? == *r && r.is_absolute(),
                "invalid source root"
            );
            ensure!(
                !cfg.root.starts_with(r) && !r.starts_with(&cfg.root),
                "source roots must be disjoint from worker state"
            );
        }
        let jobs = cfg.root.join("jobs");
        let cache = Cache {
            complete: cfg.cache_dir(),
            partial: cfg.root.join("partial"),
            reserve: cfg.disk_reserve_bytes,
        };
        for p in [&jobs, &cache.complete, &cache.partial] {
            fs::create_dir_all(p)?;
            ensure!(
                p.canonicalize()? == *p && p.is_dir(),
                "unsafe state subdirectory"
            );
        }
        Ok(Self {
            cfg,
            cfg_path: path,
            jobs,
            cache,
        })
    }

    fn directory(&self, id: &str) -> Result<PathBuf> {
        token(id)?;
        Ok(self.jobs.join(id))
    }
    fn state_lock(&self) -> Result<File> {
        lock(&self.cfg.root.join("state.lock"))
    }
    fn progress_callback(&self, id: &str) -> Result<impl FnMut(&ProgressDetail) -> Result<()>> {
        let base = self.directory(id)?;
        Ok(move |detail: &ProgressDetail| {
            ensure!(!base.join("cancelled").exists(), "cancel requested");
            // An unavailable progress sidecar must not change job results.
            let _ = progress::publish(&base.join("worker-progress.json"), detail);
            Ok(())
        })
    }
    fn get(&self, id: &str) -> Result<Option<Record>> {
        let p = self.directory(id)?.join("state.json");
        if !p.exists() {
            return Ok(None);
        }
        let mut r: Record = load(&p)?;
        if r.reply.state == "running"
            && let Some(boot) = &r.boot
            && *boot != boot_id()?
        {
            r.reply = Reply::error("compute host rebooted; job will not be restarted");
        }
        Ok(Some(r))
    }
    fn require(&self, id: &str) -> Result<Record> {
        self.get(id)?.context("job is missing")
    }
    fn put(&self, id: &str, r: &Record) -> Result<()> {
        let p = self.directory(id)?;
        if !p.exists() {
            fs::create_dir(&p)?;
            File::open(&self.jobs)?.sync_all()?;
        }
        atomic(&p.join("state.json"), r)
    }

    fn source(&self, p: &Path) -> Result<(PathBuf, PathBuf)> {
        ensure!(
            self.cfg.backend_weight_download_root.is_none(),
            "backend editing requires an exact verified publication"
        );
        ensure!(
            !self.cfg.scratch_access.as_ref().is_some_and(|access| {
                access
                    .paths
                    .keys()
                    .any(|pin| pin.starts_with(p) || p.starts_with(pin))
            }),
            "scratch input requires a job-scoped grant"
        );
        if let Some(root) = self.cfg.configured_source_root(p) {
            cache::no_symlink_open(p, libc::O_PATH)?;
            return Ok((p.to_path_buf(), root.to_path_buf()));
        }
        if let Some(root) = self.cfg.cache_source_root(p) {
            cache::no_symlink_open(p, libc::O_PATH)?;
            return Ok((p.to_path_buf(), root));
        }
        // Stat only reports the exact artifact published by a prior job.
        // A job can also consume other files inside a successful job's work directory.
        if let Ok(rel) = p.strip_prefix(&self.jobs) {
            let id = rel
                .components()
                .next()
                .and_then(|p| p.as_os_str().to_str())
                .context("invalid source")?;
            let r = self.require(id)?;
            ensure!(
                r.reply.state == "succeeded"
                    && r.action == "start"
                    && r.job.as_ref().is_none_or(|job| !matches!(
                        job.kind.as_str(),
                        sensitivity::JOB | circuit::JOB | damage::JOB
                    ) && !scratch_eval::private_output(
                        &job.kind
                    ))
                    && !r.inputs.values().any(scratch_eval::granted),
                "job output is not published"
            );
            let work = self.jobs.join(id).join("work");
            let root = if r.job.is_some_and(|job| job.kind == "hf_download") {
                work.join("download")
            } else {
                work
            };
            return Ok((cache::authorized(p, std::slice::from_ref(&root))?, root));
        }
        bail!("input outside allowed source roots")
    }

    fn scratch_pin(&self, id: &str, job: &Job, index: usize) -> Option<&str> {
        let access = self
            .cfg
            .scratch_access
            .as_ref()
            .filter(|a| a.job_id == id)?;
        if job.kind == scratch_eval::GGUF && index != 0 {
            return None;
        }
        access
            .paths
            .get(&job.inputs.get(index)?.path)
            .map(String::as_str)
    }

    fn scratch_bundle(
        &self,
        path: &Path,
        pin: &str,
        progress: &mut progress::Notify<'_>,
    ) -> Result<Bundle> {
        let p = path.canonicalize().context("Scratch input unavailable")?;
        ensure!(
            p.as_os_str() == path.as_os_str(),
            "Scratch input must be canonical"
        );
        let d = cache::describe_exact_file(path, progress).context("Scratch input unavailable")?;
        ensure!(
            d.bundle.files[0].sha256 == pin,
            "scratch input SHA-256 mismatch"
        );
        Ok(d.bundle)
    }

    // Grants are separate from regular roots on purpose. Do not let Describe,
    // Read, suggestions, staging or secondary inputs consult this list.
    fn editor_grant(
        &self,
        action: &str,
        job: &Job,
        index: usize,
        input: &Input,
    ) -> Result<Option<(&Path, File)>> {
        let grant = self
            .cfg
            .source_grants
            .iter()
            .filter(|grant| input.path.starts_with(&grant.path))
            .max_by_key(|grant| grant.path.components().count());
        let Some(grant) = grant else {
            return Ok(None);
        };
        let host = self
            .cfg
            .source_grant_host
            .as_deref()
            .context("Missing source grant host")?;
        ensure!(
            action == "start" && index == 0 && weight_source::job(&job.kind) && input.host == host,
            "Source input unavailable"
        );
        let mut protected = self.cfg.source_grant_protected.clone();
        protected.extend([
            self.cfg.root.clone(),
            self.cfg.engine.clone(),
            self.cfg_path.clone(),
        ]);
        protected.extend(
            self.cfg
                .engine_env
                .values()
                .map(PathBuf::from)
                .filter(|p| p.is_absolute()),
        );
        let dir = source_grants::editor_input(host, grant, &input.path, &input.kind, &protected)
            .map_err(|_| {
                anyhow::anyhow!("Source input unavailable; check the model directory grant")
            })?;
        Ok(Some((&grant.path, dir)))
    }

    // The backend editor cannot read arbitrary paths through ordinary source
    // RPCs. An approved input names exactly one successful job publication.
    fn backend_weight_source(&self, input: &Input) -> Result<(PathBuf, String)> {
        let invalid = "backend input is not a verified publication";
        ensure!(input.host == "backend-hf", "{invalid}");
        if input.kind == "weight-calibration" {
            let root = self
                .cfg
                .backend_weight_calibration_root
                .as_ref()
                .context(invalid)?;
            let digest = input
                .path
                .file_stem()
                .and_then(|s| s.to_str())
                .filter(|s| is_digest(s))
                .context(invalid)?;
            ensure!(
                input.path.as_os_str() == root.join(format!("{digest}.json")).as_os_str(),
                "{invalid}"
            );
            let path = cache::authorized(&input.path, std::slice::from_ref(root))?;
            let meta = fs::symlink_metadata(&path)?;
            use std::os::unix::fs::MetadataExt;
            ensure!(
                meta.is_file()
                    && meta.uid() == unsafe { libc::geteuid() }
                    && meta.nlink() == 1
                    && meta.mode() & 0o7777 == 0o400,
                "{invalid}"
            );
            return Ok((path, digest.into()));
        }
        let download_jobs = self
            .cfg
            .backend_weight_download_root
            .as_ref()
            .context(invalid)?
            .join("jobs");
        let (record_root, expected_kind, rel) =
            if let Ok(rel) = input.path.strip_prefix(&download_jobs) {
                ensure!(
                    weight_source::supports(&input.kind) || input.kind == "gguf",
                    "{invalid}"
                );
                (&download_jobs, "hf_download", rel)
            } else {
                let rel = input.path.strip_prefix(&self.jobs).context(invalid)?;
                let expected = match input.kind.as_str() {
                    "hf" | "safetensors" | "gguf" => "weight_apply",
                    "weight-edit-proposal" => "proposal",
                    _ => bail!("{invalid}"),
                };
                (&self.jobs, expected, rel)
            };
        let mut components = rel.components();
        let producer = components
            .next()
            .and_then(|p| p.as_os_str().to_str())
            .context(invalid)?;
        token(producer)?;
        ensure!(
            components.next().is_some_and(|p| p.as_os_str() == "work"),
            "{invalid}"
        );
        let work = record_root.join(producer).join("work");
        let receipt: Record = decode(cache::regular(
            &record_root.join(producer).join("state.json"),
        )?)?;
        let producer_kind = receipt
            .job
            .as_ref()
            .map(|j| j.kind.as_str())
            .context(invalid)?;
        ensure!(
            receipt.action == "start"
                && receipt.reply.state == "succeeded"
                && !receipt.cancel
                && (producer_kind == expected_kind
                    || (expected_kind == "proposal"
                        && matches!(producer_kind, "weight_propose" | "weight_tune"))),
            "{invalid}"
        );
        let result = receipt.reply.result.as_ref().context(invalid)?;
        let artifact = &result["artifact"];
        ensure!(
            (expected_kind == "proposal" || result["verified"] == true)
                && result["kind"] == producer_kind
                && artifact["kind"] == input.kind
                && artifact["path"].as_str().map(Path::new) == Some(input.path.as_path()),
            "{invalid}"
        );
        let leaf = match (expected_kind, input.kind.as_str()) {
            ("hf_download", "hf") => work.join("download"),
            ("hf_download", "safetensors" | "gguf") => {
                let spec = hf_download::Spec::parse(&json!({
                    "repo_id":result["repo_id"],"revision":result["revision"],"filename":result["filename"]
                }))?;
                work.join("download").join(spec.filename.context(invalid)?)
            }
            ("weight_apply", "hf") => work.join("edited.hf"),
            ("weight_apply", "safetensors") => work.join("edited.safetensors"),
            ("weight_apply", "gguf") => work.join("edited.gguf"),
            ("proposal", "weight-edit-proposal") => work.join("weight-proposal.json"),
            _ => bail!("{invalid}"),
        };
        ensure!(leaf.as_os_str() == input.path.as_os_str(), "{invalid}");
        let digest = artifact["sha256"]
            .as_str()
            .filter(|s| is_digest(s))
            .context(invalid)?;
        Ok((cache::authorized(&input.path, &[leaf])?, digest.into()))
    }

    fn prepare(&self, id: &str, action: String, job: Job) -> Result<Reply> {
        token(&job.kind)?;
        ensure!(
            matches!(
                action.as_str(),
                "start" | "transfer" | "preview_transfer" | "cleanup"
            ),
            "invalid action"
        );
        ensure!(
            job.options.is_object() && job.inputs.len() <= 128,
            "invalid job envelope"
        );
        ensure!(
            self.cfg
                .scratch_access
                .as_ref()
                .is_none_or(|access| access.job_id == id),
            "scratch worker config is scoped to one job"
        );
        for input in &job.inputs {
            token(&input.host)?;
            token(&input.kind)?;
            absolute(&input.path)?;
        }
        if let Some(host) = &self.cfg.weight_editor_host {
            ensure!(
                action == "start"
                    && job.prompt.is_none()
                    && matches!(
                        job.kind.as_str(),
                        "weight_inspect" | "weight_propose" | "weight_tune" | "weight_apply"
                    )
                    && job.inputs.iter().all(|input| &input.host == host),
                "Remote editor allows only direct weight jobs on its own host"
            );
        }
        if self.cfg.hy4_plan.as_ref().map_or_else(
            || job.options.get(hy4::OPTION).is_some(),
            |plan| !plan.validate(&action, &job),
        ) {
            return Ok(Reply::error(
                "Only the configured direct Hy4 metadata plan is allowed",
            ));
        }
        if let Some(r) = self.get(id)? {
            if let Some(old) = r.job {
                ensure!(
                    r.action == action && to_value(old)? == to_value(job)?,
                    "job ID already has a different request"
                );
            }
            return Ok(r.reply);
        }
        let access = self
            .cfg
            .scratch_access
            .as_ref()
            .filter(|access| access.job_id == id);
        if scratch_eval::private_output(&job.kind) || access.is_some() {
            let valid = access.is_some_and(|access| access.permits(&action, &job))
                && (job.kind != scratch_eval::CHECKS
                    || [
                        scratch_eval::SCRIPT,
                        scratch_eval::RUNNER,
                        scratch_eval::CACHE,
                    ]
                    .iter()
                    .all(|key| {
                        self.cfg
                            .engine_env
                            .get(*key)
                            .is_some_and(|p| Path::new(p).is_absolute())
                    }) && self
                        .cfg
                        .engine_env
                        .get(scratch_eval::RUNNER_SHA)
                        .is_some_and(|sha| is_digest(sha)));
            if !valid {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error("Invalid scratch access, job, or worker check configuration"),
                );
            }
            if job.kind == scratch_eval::CHECKS
                && !self
                    .cfg
                    .engine_env
                    .get(scratch_eval::RUNNER_LIB_DIR)
                    .is_some_and(|p| Path::new(p).is_absolute())
            {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error("BREWERY_SCRATCH_CHECKS_RUNNER_LIB_DIR must be an absolute path"),
                );
            }
            if job.kind == scratch_eval::CHECKS
                && scratch_eval::check_options(&job.options)?.is_some()
                && !self
                    .cfg
                    .engine_env
                    .get(scratch_eval::CONTRAST_CACHE)
                    .is_some_and(|p| Path::new(p).is_absolute())
            {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error(
                        "BREWERY_CONTRAST_CACHE must be an absolute path for behavior checks",
                    ),
                );
            }
            if job.kind == locate_edit::JOB {
                let backend = locate_edit::Request::parse(&job.options)?.edit.backend;
                let pin = backend.pin();
                let error = if ![pin.executable, pin.lib_dir].iter().all(|key| {
                    self.cfg
                        .engine_env
                        .get(*key)
                        .is_some_and(|path| Path::new(path).is_absolute())
                }) {
                    Some(format!(
                        "Pinned Lanternfish {} editor is not configured",
                        backend.label()
                    ))
                } else if backend.is_gpu()
                    && (self.cfg.admission.is_none()
                        || self
                            .cfg
                            .engine_env
                            .get(locate_edit::GPU_GATE)
                            .is_none_or(|v| v != "1"))
                {
                    Some(format!(
                        "{} edit requires GPU load permission and a GPU admission wrapper",
                        backend.label()
                    ))
                } else {
                    None
                };
                if let Some(error) = error {
                    return self.record_prepare(id, action, job, Reply::error(error));
                }
            }
            if heretic::job(&job.kind)
                && (self.cfg.admission.is_none()
                    || self
                        .cfg
                        .engine_env
                        .get(locate_edit::GPU_GATE)
                        .is_none_or(|v| v != "1")
                    || self
                        .cfg
                        .engine_env
                        .get(heretic::RUNTIME)
                        .is_none_or(|p| !Path::new(p).is_absolute()))
            {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error("Heretic requires a pinned runtime and GPU admission"),
                );
            }
        }
        if !job.resources.is_empty() {
            let accepted = self.resource_enabled()
                && job.resources.for_job(&job.kind) == job.resources
                && matches!(action.as_str(), "start" | "transfer" | "preview_transfer")
                && resources::Bounds::MAX.validate(&job.resources).is_ok()
                && {
                    let supported = self.resource_bounds();
                    let req = job.resources;
                    req.concurrency
                        .is_none_or(|n| supported.concurrency.is_some_and(|bound| n <= bound.max))
                        && (req.cpu_threads.is_none() || supported.cpu_threads.is_some())
                        && (req.min_free_ram_bytes.is_none()
                            || supported.min_free_ram_bytes.is_some())
                        && (req.disk_reserve_bytes.is_none()
                            || supported.disk_reserve_bytes.is_some())
                };
            if !accepted {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error("Worker does not support requested resource controls"),
                );
            }
        }
        if self.cfg.backend_weight_download_root.is_some()
            && !(action == "start"
                && job.prompt.is_none()
                && job
                    .inputs
                    .first()
                    .is_some_and(|i| weight_source::supports(&i.kind) || i.kind == "gguf")
                && match job.kind.as_str() {
                    "weight_inspect" | "weight_propose" => job.inputs.len() == 1,
                    "weight_tune" => {
                        self.cfg.backend_weight_calibration_root.is_some()
                            && job.inputs.len() == 3
                            && job.inputs[1..]
                                .iter()
                                .all(|i| i.kind == "weight-calibration")
                            && job.inputs[1].path != job.inputs[2].path
                    }
                    "weight_apply" => {
                        job.inputs.len() == 2 && job.inputs[1].kind == "weight-edit-proposal"
                    }
                    _ => false,
                })
        {
            return self.record_prepare(
                id,
                action,
                job,
                Reply::error("unsupported backend weight job"),
            );
        }
        if job.kind == "hf_download" {
            let reply = if self.cfg.hf_download.is_none() {
                Reply::error("Hugging Face downloads are not configured on this worker")
            } else if action != "start" || !job.inputs.is_empty() || job.prompt.is_some() {
                Reply::error("Hugging Face downloads do not accept inputs or prompts")
            } else if hf_download::Spec::parse(&job.options).is_err() {
                Reply::error("Invalid Hugging Face download options")
            } else {
                Reply::new("queued", "waiting for download worker")
            };
            return self.record_prepare(id, action, job, reply);
        }
        if self.cfg.hf_download.is_some() {
            return self.record_prepare(
                id,
                action,
                job,
                Reply::error("This worker only runs Hugging Face download jobs"),
            );
        }
        if job.kind == behavior_direction::JOB
            && !(action == "start"
                && job.prompt.is_none()
                && job.inputs.len() == 1
                && job.inputs[0].kind == "gguf"
                && [trace::EXECUTABLE, trace::LIB_DIR, behavior_direction::CACHE]
                    .iter()
                    .all(|key| {
                        self.cfg
                            .engine_env
                            .get(*key)
                            .is_some_and(|p| Path::new(p).is_absolute())
                    })
                && behavior_direction::Request::parse(&job.options).is_ok())
        {
            return self.record_prepare(
                id,
                action,
                job,
                Reply::error("Invalid direction request or unavailable CPU trace runtime or cache"),
            );
        }
        if job.kind == ablation_search::SCREEN
            && ![
                ablation_search::EXECUTABLE,
                trace::PROJECTION_LIB_DIR,
                behavior_direction::CACHE,
            ]
            .iter()
            .all(|key| {
                self.cfg
                    .engine_env
                    .get(*key)
                    .is_some_and(|p| Path::new(p).is_absolute())
            })
        {
            return self.record_prepare(
                id,
                action,
                job,
                Reply::error("Configure the pinned CPU ablation runtime and contrast cache"),
            );
        }
        if job.kind == trace::JOB {
            let absolute = |var| {
                self.cfg
                    .engine_env
                    .get(var)
                    .is_some_and(|p| Path::new(p).is_absolute())
            };
            let request = trace::Request::worker(&job.options, job.prompt.as_deref());
            let allowed = action == "start"
                && job.inputs.len() == 1
                && request
                    .as_ref()
                    .is_ok_and(|request| match job.inputs[0].kind.as_str() {
                        "gguf" => request.lanternfish_runtime().is_ok_and(|runtime| {
                            let (executable, lib_dir, _, _) = runtime.pins();
                            absolute(executable) && absolute(lib_dir)
                        }),
                        kind if trace::is_hgn(kind) => {
                            Path::new(&job.inputs[0].path)
                                .extension()
                                .is_some_and(|e| e == "hgn")
                                && absolute(trace::HGN_PYTHON)
                                && absolute(trace::HGN_ASSETS)
                                && self
                                    .cfg
                                    .engine_env
                                    .get(trace::HGN_PYTHON_SHA256)
                                    .is_some_and(|s| is_digest(s))
                                && request.interventions.is_empty()
                        }
                        _ => false,
                    });
            if !allowed {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error("Invalid trace request or unavailable pinned CPU trace runtime"),
                );
            }
        }
        if job.kind == sensitivity::JOB {
            let allowed = action == "start"
                && job.prompt.is_none()
                && job.inputs.len() == 1
                && job.inputs[0].kind == "gguf"
                && (job.options.get(sensitivity::REVISION_PIN).is_some() == access.is_some())
                && sensitivity::Request::resolved(&job.options).is_ok_and(|(req, _)| {
                    sensitivity::configured(
                        &self.cfg.engine_env,
                        req.backend,
                        self.cfg.admission.as_deref(),
                    )
                });
            if !allowed {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error("Invalid sensitivity request or unavailable pinned backend"),
                );
            }
        }
        if job.kind == circuit::JOB {
            let allowed = action == "start"
                && job.prompt.is_none()
                && job.inputs.len() == 1
                && job.inputs[0].kind == "gguf"
                && circuit::Request::parse(&job.options).is_ok()
                && sensitivity::configured(&self.cfg.engine_env, locate_edit::Backend::Cpu, None);
            if !allowed {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error("Invalid circuit request or unavailable pinned CPU runtime"),
                );
            }
        }
        if job.kind == damage::JOB {
            let allowed = action == "start"
                && job.prompt.is_none()
                && access.is_none()
                && job.inputs.len() == 2
                && job.inputs.iter().all(|i| i.kind == "gguf")
                && damage::Request::parse(&job.options).is_ok_and(|req| {
                    damage::configured(&self.cfg.engine_env, &req, self.cfg.admission.as_deref())
                });
            if !allowed {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error("Invalid damage map request or unavailable pinned backend"),
                );
            }
        }
        if job.options["output_format"] == hgn_recipe::FORMAT {
            if self.cfg.pinned_publication.is_some()
                || self
                    .cfg
                    .engine_env
                    .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
                || env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_some()
            {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error(
                        "Native HT3/HT4 layer recipes cannot run on a source-pinned worker",
                    ),
                );
            }
            let configured = self.cfg.backend_weight_download_root.is_none()
                && self.cfg.weight_editor_host.is_none()
                && self
                    .cfg
                    .engine_env
                    .get(hgn_recipe::HELPER)
                    .is_some_and(|p| Path::new(p).is_absolute())
                && self
                    .cfg
                    .engine_env
                    .get(hgn_recipe::HELPER_SHA)
                    .is_some_and(|sha| is_digest(sha));
            if !configured
                || action != "start"
                || job.prompt.is_some()
                || !hgn_recipe::options(&job.kind, &job.options)
                || !hgn_recipe::layout(
                    job.kind == "verify",
                    job.inputs
                        .iter()
                        .map(|i| (i.kind.as_str(), i.host.as_str())),
                )
            {
                return self.record_prepare(
                    id, action, job,
                    Reply::error("Native HT3/HT4 layer recipes require a pinned helper and direct local inputs"),
                );
            }
        }
        if job.options.get("encoder_family").is_some()
            || job.options.get("encoder_revision").is_some()
        {
            let valid = action == "start"
                && matches!(job.kind.as_str(), "inspect" | "plan" | "convert" | "verify")
                && job.options["output_format"] == "gguf"
                && job.options["encoder_family"] == "rocmfpx"
                && job.options["encoder_revision"] == ROCMFPX_REVISION
                && job.prompt.is_none()
                && self.cfg.pinned_publication.is_none()
                && !self
                    .cfg
                    .engine_env
                    .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
                && self
                    .cfg
                    .engine_env
                    .get("BREWERY_GGUF_ROCMFPX_QUANTIZER")
                    .is_some_and(|path| {
                        Path::new(path).is_absolute()
                            && self.cfg.engine_env.get("BREWERY_GGUF_QUANTIZER") != Some(path)
                    })
                && self
                    .cfg
                    .engine_env
                    .get("BREWERY_GGUF_ROCMFPX_SHA256")
                    .is_some_and(|sha| is_digest(sha));
            if !valid {
                return self.record_prepare(
                    id,
                    action,
                    job,
                    Reply::error("Worker does not support the pinned ROCmFPX GGUF encoder"),
                );
            }
        }
        if standalone_gguf(&job) {
            let source = &job.inputs[0];
            let assets = if job.kind == "verify" { 2 } else { 1 };
            let matrix = assets + 1;
            let has_matrix = job.options.get("imatrix").is_some();
            let denial = if self.cfg.pinned_publication.is_some()
                || self
                    .cfg
                    .engine_env
                    .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
                || self
                    .cfg
                    .engine_env
                    .get("BREWERY_GGUF_SAFETENSORS")
                    .map(String::as_str)
                    != Some("1")
                || !["BREWERY_GGUF_QUANTIZER", "BREWERY_GGUF_HF_EXPORTER"]
                    .iter()
                    .all(|key| {
                        self.cfg
                            .engine_env
                            .get(*key)
                            .is_some_and(|path| Path::new(path).is_absolute())
                    }) {
                Some("standalone SafeTensors GGUF is not configured on this worker")
            } else if action != "start"
                || !matches!(
                    job.kind.as_str(),
                    "inspect" | "plan" | "convert" | "verify" | scratch_eval::GGUF
                )
                || job.prompt.is_some()
                || job.inputs.len() != matrix + usize::from(has_matrix)
                || job.inputs[assets].kind != "assets"
                || (assets == 2 && job.inputs[1].kind != "gguf")
                || (has_matrix
                    && (job.options["imatrix"].as_u64() != Some(matrix as u64)
                        || job.inputs[matrix].kind != "imatrix"))
                || job.inputs.iter().any(|input| input.host != source.host)
            {
                Some(
                    "standalone SafeTensors GGUF requires direct weights, registered assets and optional imatrix",
                )
            } else {
                None
            };
            if let Some(error) = denial {
                return self.record_prepare(id, action, job, Reply::error(error));
            }
        }
        if job.kind == brewery_worker::exl3_layer::JOB {
            use brewery_worker::exl3_layer::{Backend, HELPER, Spec};
            let reply = if action != "start"
                || job.prompt.is_some()
                || job.inputs.len() != 1
                || job.inputs[0].kind != "safetensors"
            {
                Some("EXL3 requires one direct standalone SafeTensors input and no prompt")
            } else if self.cfg.pinned_publication.is_some()
                || self
                    .cfg
                    .engine_env
                    .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
                || !self
                    .cfg
                    .engine_env
                    .get(weight_source::GATE)
                    .is_some_and(|v| v == "1")
                || !self
                    .cfg
                    .engine_env
                    .get(HELPER)
                    .is_some_and(|v| Path::new(v).is_absolute())
                || weight_source::verify_limit(
                    self.cfg
                        .engine_env
                        .get(weight_source::VERIFY_BYTES)
                        .map(String::as_str),
                )
                .is_err()
            {
                Some("bounded EXL3 layer encoding is not configured on this worker")
            } else {
                match Spec::parse(&job.options) {
                    Ok(spec)
                        if spec.backend != Backend::Cpu && self.cfg.exl3_gpu_queue.is_none() =>
                    {
                        Some("EXL3 GPU admission is not configured on this worker")
                    }
                    Ok(_) => None,
                    Err(_) => Some("invalid EXL3 layer options"),
                }
            };
            if let Some(error) = reply {
                return self.record_prepare(id, action, job, Reply::error(error));
            }
        }
        if matches!(
            job.kind.as_str(),
            "weight_inspect" | "weight_propose" | "weight_tune" | "weight_apply"
        ) {
            let key = if job.kind == "weight_tune" {
                "BREWERY_WEIGHT_TUNE"
            } else {
                "BREWERY_WEIGHT_EDIT"
            };
            ensure!(
                action == "start"
                    && self.cfg.pinned_publication.is_none()
                    && !self
                        .cfg
                        .engine_env
                        .contains_key("BREWERY_EXPECTED_SOURCE_SHA256")
                    && self
                        .cfg
                        .engine_env
                        .get(key)
                        .is_some_and(|path| Path::new(path).is_absolute()),
                "weight editing is not configured on this worker"
            );
            if job
                .inputs
                .first()
                .is_some_and(|i| weight_source::supports(&i.kind))
            {
                let denial = if !self
                    .cfg
                    .engine_env
                    .get(weight_source::GATE)
                    .is_some_and(|v| v == "1")
                {
                    Some("SafeTensors weight editing is not configured on this worker")
                } else if job.options.get("packed").is_some() {
                    Some("packed weight editing is not supported for SafeTensors sources")
                } else {
                    None
                };
                if let Some(error) = denial {
                    return self.record_prepare(id, action, job, Reply::error(error));
                }
            }
            if let Some(packed) = job.options.get("packed") {
                ensure!(packed == true, "packed must be true when supplied");
                if self
                    .cfg
                    .engine_env
                    .get("BREWERY_WEIGHT_PACKED")
                    .map(String::as_str)
                    != Some("1")
                {
                    // Persist this denial so status can distinguish it from
                    // a Prepare request that never reached the worker.
                    return self.record_prepare(
                        id,
                        action,
                        job,
                        Reply::error("packed weight editing is not configured on this worker"),
                    );
                }
            }
            if job.inputs.first().is_some_and(|input| input.kind == "hgn") {
                ensure!(
                    self.cfg
                        .engine_env
                        .get("BREWERY_WEIGHT_HGN")
                        .is_some_and(|v| v == "1"),
                    "HGN weight editing is not configured on this worker"
                );
            }
            if (job.kind == "weight_inspect" && job.options.get("mode").is_some())
                || (matches!(job.kind.as_str(), "weight_propose" | "weight_tune")
                    && job.options.get("source_sha256").is_some())
            {
                ensure!(
                    self.cfg
                        .engine_env
                        .get("BREWERY_WEIGHT_VISUAL")
                        .is_some_and(|v| v == "1"),
                    "visual weight inspection is not configured on this worker"
                );
            }
        }
        self.record_prepare(id, action, job, Reply::new("queued", "resolving inputs"))
    }

    fn record_prepare(
        &self,
        id: &str,
        action: String,
        job: Job,
        mut reply: Reply,
    ) -> Result<Reply> {
        reply.resources_enforced = reply.state == "queued" && !job.resources.is_empty();
        self.put(
            id,
            &Record {
                job: Some(job),
                action,
                reply: reply.clone(),
                cancel: false,
                inputs: BTreeMap::new(),
                pins: BTreeSet::new(),
                boot: None,
                started: None,
                started_ms: None,
            },
        )?;
        Ok(reply)
    }

    fn pending(&self, id: &str) -> Result<Record> {
        let r = self.require(id)?;
        ensure!(r.reply.state == "queued", "job is {}", r.reply.state);
        Ok(r)
    }

    fn cancel(&self, id: &str) -> Result<Reply> {
        let cancelled = Reply::new("cancelled", "cancelled");
        let mut r = self.get(id)?.unwrap_or(Record {
            job: None,
            action: String::new(),
            reply: cancelled.clone(),
            cancel: true,
            inputs: BTreeMap::new(),
            pins: BTreeSet::new(),
            boot: None,
            started: None,
            started_ms: None,
        });
        match r.reply.state.as_str() {
            "queued" => {
                r.cancel = true;
                r.reply = cancelled;
            }
            "running" => {
                r.cancel = true;
                r.reply.progress = "cancelling; waiting for process tree".into();
            }
            // EXL3 cancellation only changes queued/running records. A late
            // cancel returns the already durable outcome and its receipt.
            _ if r
                .job
                .as_ref()
                .is_some_and(|job| job.kind == brewery_worker::exl3_layer::JOB) =>
            {
                return Ok(r.reply);
            }
            _ => {}
        }
        self.put(id, &r)?;
        if r.cancel {
            File::create(self.directory(id)?.join("cancelled"))?;
        }
        Ok(r.reply)
    }

    fn resolve(
        &self,
        id: &str,
        index: usize,
        mut input: Input,
        provided: Option<Bundle>,
        guard: File,
    ) -> Result<Value> {
        let mut guard = Some(guard);
        ensure!(
            self.cfg
                .scratch_access
                .as_ref()
                .is_none_or(|access| access.job_id == id),
            "scratch worker config is scoped to one job"
        );
        let mut r = self.pending(id)?;
        let original = r
            .job
            .as_ref()
            .and_then(|j| j.inputs.get(index))
            .context("unknown input index")?
            .clone();
        ensure!(input.kind == original.kind, "resolved input kind changed");
        let job = r.job.as_ref().unwrap();
        let scratch_pin = self.scratch_pin(id, job, index);
        let private_job = scratch_eval::private_output(&job.kind) || scratch_pin.is_some();
        ensure!(
            !private_job || provided.is_none(),
            "Scratch jobs do not accept staged inputs"
        );
        ensure!(
            !private_job
                || self
                    .cfg
                    .scratch_access
                    .as_ref()
                    .is_some_and(|access| access.job_id == id && access.permits(&r.action, job)),
            "Scratch access does not match the job"
        );
        if let Some(plan) = &self.cfg.hy4_plan {
            ensure!(
                plan.validate(&r.action, job)
                    && index == 0
                    && provided.is_none()
                    && input.host == original.host
                    && input.path.as_os_str() == original.path.as_os_str(),
                "Hy4 planning does not accept staged or substituted inputs"
            );
            input.extra.clear();
            input
                .extra
                .insert("integrity_basis".into(), json!("hy4_metadata_only"));
            r.inputs.insert(index, input.clone());
            self.put(id, &r)?;
            return Ok(to_value(input)?);
        }
        let standalone = standalone_gguf(job);
        let native_hgn = job.options["output_format"] == hgn_recipe::FORMAT;
        ensure!(
            !standalone || provided.is_none(),
            "standalone SafeTensors GGUF requires direct inputs"
        );
        ensure!(
            !native_hgn || provided.is_none(),
            "native HGN layer recipes require direct inputs"
        );
        let exl3 = job.kind == brewery_worker::exl3_layer::JOB;
        let native_safe = index == 0
            && (weight_source::job(&job.kind) || exl3)
            && weight_source::supports(&input.kind);
        ensure!(
            !native_safe || provided.is_none(),
            "SafeTensors weight jobs require a direct local source"
        );
        ensure!(
            self.cfg.backend_weight_download_root.is_none() || provided.is_none(),
            "backend weight jobs do not accept staged inputs"
        );
        let mut pinned = false;
        ensure!(
            !matches!(
                job.kind.as_str(),
                sensitivity::JOB | circuit::JOB | damage::JOB
            ) || (provided.is_none() && input.host == original.host),
            "sensitivity, circuit and damage map jobs require direct sources"
        );
        let (bundle, cached) = if let Some(b) = provided {
            b.validate()?;
            ensure!(
                r.action == "preview_transfer" || r.pins.contains(&b.digest),
                "bundle not planned by job"
            );
            let ready = self.cache.ready(&b.digest);
            if ready.exists() {
                let existing: Bundle = load(&ready.join("manifest.json"))?;
                ensure!(b.files == existing.files, "resolved manifest mismatch");
            } else {
                ensure!(r.action == "preview_transfer", "cache not published");
            }
            input.path = ready.join("data");
            if b.entry != "." {
                input.path.push(&b.entry);
            }
            (b, ready.exists())
        } else {
            ensure!(
                input.path == original.path && input.host == original.host && r.action == "start",
                "invalid direct input"
            );
            let job = r.job.as_ref().unwrap();
            let weight_hgn = index == 0
                && input.kind == "hgn"
                && matches!(
                    job.kind.as_str(),
                    "weight_inspect" | "weight_propose" | "weight_tune" | "weight_apply"
                );
            let pin = self.cfg.pinned_publication.as_ref().filter(|pin| {
                index == 0
                    && input.kind == "hf"
                    && matches!(job.kind.as_str(), "inspect" | "plan" | "convert" | "verify")
                    && original.path.as_os_str() == pin.source.as_os_str()
            });
            let count = job.inputs.len();
            // Static roots retain their original policy, including where they
            // overlap a managed grant. Only consult grants as a fallback.
            let static_source = if self.cfg.backend_weight_download_root.is_none()
                && pin.is_none()
                && scratch_pin.is_none()
            {
                self.source(&input.path).ok()
            } else {
                None
            };
            let grant = if static_source.is_none()
                && self.cfg.backend_weight_download_root.is_none()
                && pin.is_none()
                && scratch_pin.is_none()
            {
                self.editor_grant(&r.action, job, index, &input)?
            } else {
                None
            };
            let bundle = if let Some(pin) = scratch_pin {
                drop(guard.take());
                let mut emit = self.progress_callback(id)?;
                let bundle = self.scratch_bundle(
                    &input.path,
                    pin,
                    &mut progress::Reporter::new(&mut emit, Some((index, count))),
                )?;
                ensure!(
                    input.path.extension().and_then(|ext| ext.to_str())
                        == Some(input.kind.as_str()),
                    "Scratch file extension does not match input kind"
                );
                bundle
            } else if let Some(pin) = pin {
                ensure!(
                    input.path.as_os_str() == pin.source.as_os_str(),
                    "invalid direct pinned input path"
                );
                drop(guard.take());
                let mut emit = self.progress_callback(id)?;
                let bundle = pin
                    .manifest(
                        &self.cfg.source_roots,
                        &mut progress::Reporter::new(&mut emit, Some((index, count))),
                    )
                    .context("Pinned publication manifest unavailable")?;
                pinned = true;
                bundle
            } else {
                let (p, boundary, expected) = if self.cfg.backend_weight_download_root.is_some() {
                    let (p, digest) = self.backend_weight_source(&input)?;
                    let boundary = if input.kind == "hf" {
                        p.clone()
                    } else {
                        p.parent().context("backend input has no parent")?.into()
                    };
                    (p, boundary, Some(digest))
                } else if let Some((p, boundary)) = static_source {
                    (p, boundary, None)
                } else if let Some((boundary, _)) = &grant {
                    (input.path.clone(), boundary.to_path_buf(), None)
                } else {
                    bail!("Source input unavailable");
                };
                if weight_hgn {
                    ensure!(
                        fs::symlink_metadata(&p)?.is_file()
                            && p.extension().is_some_and(|ext| ext == "hgn"),
                        "HGN weight jobs require a regular .hgn file, not a bundle"
                    );
                }
                if standalone {
                    let metadata = fs::symlink_metadata(&p)?;
                    ensure!(
                        if input.kind == "assets" {
                            metadata.is_dir()
                        } else {
                            metadata.is_file()
                                && (index != 0
                                    || p.extension().is_some_and(|ext| ext == "safetensors"))
                        },
                        "standalone SafeTensors GGUF requires regular files and an assets directory"
                    );
                }
                if native_hgn {
                    let metadata = fs::symlink_metadata(&p)?;
                    ensure!(
                        if input.kind == "safetensors" {
                            metadata.is_file()
                                && p.extension().is_some_and(|ext| ext == "safetensors")
                        } else {
                            metadata.is_dir()
                        },
                        "native HGN recipes require an indexed HF directory or standalone SafeTensors and an assets directory"
                    );
                }
                // A concurrent cache cleanup may not delete a direct source in use.
                if let Ok(rel) = p.strip_prefix(&self.cache.complete) {
                    r.pins.insert(
                        rel.components()
                            .next()
                            .unwrap()
                            .as_os_str()
                            .to_string_lossy()
                            .into_owned(),
                    );
                    self.put(id, &r)?;
                }
                drop(guard.take());
                let mut emit = self.progress_callback(id)?;
                let mut progress = progress::Reporter::new(
                    &mut emit as &mut dyn FnMut(&ProgressDetail) -> Result<()>,
                    Some((index, count)),
                );
                let proof = if native_safe && !exl3 {
                    weight_source::proof(&p, &input.kind, |key| {
                        self.cfg.engine_env.get(key).map(std::ffi::OsString::from)
                    })
                    .context("SafeTensors immutable publication invalid")?
                } else {
                    None
                };
                if let Some(proof) = proof {
                    let bundle = proof.bundle();
                    let digest = if bundle.entry == "." {
                        &bundle.digest
                    } else {
                        &bundle.files[0].sha256
                    };
                    ensure!(
                        expected.as_ref().is_none_or(|pin| pin == digest),
                        "SafeTensors proof does not match the registered publication digest"
                    );
                    bundle.clone()
                } else {
                    let max_bytes = if native_safe {
                        let limit = weight_source::verify_limit(
                            self.cfg
                                .engine_env
                                .get(weight_source::VERIFY_BYTES)
                                .map(String::as_str),
                        )?;
                        weight_source::bounded_inventory(&p, &input.kind, limit)?;
                        limit
                    } else if native_hgn {
                        if input.kind == hgn_recipe::ARCHIVE {
                            128 << 20
                        } else {
                            64 << 20
                        }
                    } else if expected.is_some() {
                        match input.kind.as_str() {
                            "weight-calibration" => 256 << 10,
                            "gguf" => 32 << 20,
                            _ => 4 << 20,
                        }
                    } else {
                        u64::MAX
                    };
                    // An exact backend download is one file, including if
                    // its name resembles a split GGUF; never open siblings.
                    let describe_kind = if expected.is_some() && input.kind == "gguf" {
                        "safetensors"
                    } else {
                        &input.kind
                    };
                    let d = cache::describe_bounded(
                        &p,
                        describe_kind,
                        &boundary,
                        "hash_input",
                        &mut progress,
                        max_bytes,
                    )?;
                    ensure!(
                        d.root.starts_with(&boundary),
                        "bundle outside the configured source root"
                    );
                    if let Some(expected) = expected {
                        let digest = if d.bundle.entry == "." {
                            &d.bundle.digest
                        } else {
                            &d.bundle
                                .files
                                .iter()
                                .find(|f| f.path == d.bundle.entry)
                                .context("backend publication member missing")?
                                .sha256
                        };
                        ensure!(*digest == expected, "backend publication hash changed");
                    }
                    d.bundle
                }
            };
            if grant.is_some() {
                // The hashing and optional immutable proof must not turn a
                // changed directory into a newly approved source.
                self.editor_grant(&r.action, job, index, &input)?;
            }
            guard = Some(self.state_lock()?);
            r = self.pending(id)?;
            (bundle, true)
        };
        if exl3 {
            let spec = brewery_worker::exl3_layer::Spec::parse(&r.job.as_ref().unwrap().options)?;
            ensure!(
                bundle.files.len() == 1
                    && bundle.files[0].path == bundle.entry
                    && bundle.files[0].sha256 == spec.source_sha256,
                "EXL3 source SHA-256 does not match the selected fingerprint"
            );
        }
        input.extra = original.extra.clone();
        for k in [
            "sha256",
            "bundle_sha256",
            "bytes",
            "files",
            "cached",
            "source_host",
            "integrity_basis",
            scratch_eval::GRANT,
        ] {
            input.extra.remove(k);
        }
        if let Some(f) = bundle.files.iter().find(|f| f.path == bundle.entry) {
            input.extra.insert("sha256".into(), json!(f.sha256));
        }
        let job = r.job.as_ref().unwrap();
        if job.kind == trace::JOB {
            trace::check_model_revision(
                &job.options,
                input.extra["sha256"].as_str().unwrap_or_default(),
            )?;
        }
        if job.kind == sensitivity::JOB {
            sensitivity::Request::resolved(&job.options)?
                .0
                .check_model(input.extra["sha256"].as_str().unwrap_or_default())?;
        }
        if job.kind == circuit::JOB {
            circuit::Request::parse(&job.options)?
                .check_model(input.extra["sha256"].as_str().unwrap_or_default())?;
        }
        if job.kind == damage::JOB {
            let req = damage::Request::parse(&job.options)?;
            ensure!(
                req.hashes().get(index).copied() == input.extra["sha256"].as_str(),
                "damage map source SHA-256 mismatch"
            );
        }
        if job.kind == behavior_direction::JOB {
            let request = behavior_direction::Request::parse(&job.options)?;
            ensure!(
                input.extra["sha256"] == request.model_sha256,
                "direction model SHA-256 mismatch"
            );
        }
        input.extra.extend([
            ("bundle_sha256".into(), json!(bundle.digest)),
            ("bytes".into(), json!(bundle.bytes()?)),
            ("files".into(), json!(bundle.files)),
            ("cached".into(), json!(cached)),
            ("source_host".into(), json!(original.host)),
        ]);
        if pinned {
            input
                .extra
                .insert("integrity_basis".into(), json!("pinned_publication"));
        }
        if scratch_pin.is_some() {
            input.extra.insert(scratch_eval::GRANT.into(), json!(true));
        }
        r.inputs.insert(index, input.clone());
        self.put(id, &r)?;
        drop(guard);
        Ok(to_value(input)?)
    }

    fn cleanup(&self, id: &str) -> Result<Reply> {
        let mut r = self.require(id)?;
        if r.reply.state != "queued" {
            return Ok(r.reply);
        }
        ensure!(r.action == "cleanup", "not a cleanup job");
        let options = r.job.as_ref().unwrap().options.as_object().unwrap();
        ensure!(
            options.keys().all(|k| k == "bundle_sha256"),
            "unknown cleanup option"
        );
        let selected = options
            .get("bundle_sha256")
            .map(|v| {
                v.as_str()
                    .filter(|s| is_digest(s))
                    .context("invalid cleanup digest")
            })
            .transpose()?;
        let mut pinned = BTreeSet::new();
        for f in fs::read_dir(&self.jobs)? {
            let p = f?.path().join("state.json");
            if !p.exists() {
                continue;
            }
            let r: Record = load(&p)?;
            if matches!(r.reply.state.as_str(), "queued" | "running") {
                pinned.extend(r.pins);
            }
        }
        let mut removed = BTreeSet::new();
        let mut skipped = BTreeSet::new();
        for dir in [&self.cache.complete, &self.cache.partial] {
            for f in fs::read_dir(dir)? {
                let f = f?;
                let digest = f.file_name().to_string_lossy().into_owned();
                if !is_digest(&digest)
                    || selected.is_some_and(|s| s != digest)
                    || !f.file_type()?.is_dir()
                {
                    continue;
                }
                if pinned.contains(&digest) {
                    skipped.insert(digest);
                } else {
                    fs::remove_dir_all(f.path())?;
                    removed.insert(digest);
                }
            }
            File::open(dir)?.sync_all()?;
        }
        r.reply = Reply::success(json!({"removed":removed,"skipped":skipped}));
        self.put(id, &r)?;
        Ok(r.reply)
    }

    fn handle(&self, request: Rpc) -> Result<Value> {
        ensure!(
            self.cfg.scratch_access.is_none()
                || matches!(
                    &request,
                    Rpc::Prepare { .. }
                        | Rpc::Resolve { .. }
                        | Rpc::Start { .. }
                        | Rpc::Status { .. }
                        | Rpc::Cancel { .. }
                        | Rpc::Finish { .. }
                        | Rpc::TraceRead { .. }
                ),
            "scratch worker config does not allow source or cache RPCs"
        );
        ensure!(
            self.cfg.weight_editor_host.is_none()
                || !matches!(
                    &request,
                    Rpc::Plan { .. }
                        | Rpc::Seal { .. }
                        | Rpc::Write { .. }
                        | Rpc::Cleanup { .. }
                        | Rpc::Read { .. }
                        | Rpc::Describe { .. }
                        | Rpc::TraceRead { .. }
                ),
            "Remote editor does not permit transfer or cache operations"
        );
        ensure!(
            self.cfg.hy4_plan.is_none()
                || matches!(
                    &request,
                    Rpc::Prepare { .. }
                        | Rpc::Resolve { .. }
                        | Rpc::Start { .. }
                        | Rpc::Status { .. }
                        | Rpc::Cancel { .. }
                        | Rpc::Finish { .. }
                ),
            "Hy4 metadata worker does not support this operation"
        );
        // Reads/description may hash large files; they never mutate durable state.
        let request = match request {
            Rpc::TraceRead { id, slice } => {
                let root = self.cfg.root.join("trace-results");
                let Some(result) = self.get(&id)?.and_then(|record| {
                    (record.reply.state == "succeeded"
                        && record.job.is_some_and(|job| job.kind == trace::JOB))
                    .then_some(record.reply.result)
                    .flatten()
                }) else {
                    return Ok(to_value(trace::ReadReply::Missing)?);
                };
                let _trace = lock(&self.cfg.root.join("trace.lock"))?;
                let dir = root.join(&id);
                if !root.exists() || !dir.exists() {
                    return Ok(to_value(trace::ReadReply::Missing)?);
                }
                ensure!(fs::symlink_metadata(root)?.is_dir(), "invalid trace store");
                return Ok(to_value(trace::read(&dir, &slice, &result)?)?);
            }
            Rpc::Describe { input } => {
                let (p, root) = self
                    .source(&input.path)
                    .context("Source input unavailable")?;
                let d = cache::describe(
                    &p,
                    &input.kind,
                    &root,
                    "hash_input",
                    &mut progress::Reporter::new(&mut |_: &ProgressDetail| Ok(()), None),
                )?;
                ensure!(
                    d.root.starts_with(root),
                    "bundle outside the configured source root"
                );
                return Ok(to_value(d)?);
            }
            Rpc::Read {
                root,
                entry,
                offset,
            } => {
                let (_, boundary) = self
                    .source(&root.join(&entry.path))
                    .context("Source input unavailable")?;
                return Ok(json!(cache::read(&root, &[boundary], &entry, offset)?));
            }
            request => request,
        };
        // Only cache mutations serialize against cache publication/cleanup.
        // Status and cancellation do not wait for full-file transfer hashing.
        if let Rpc::StageProgress { id, detail } = &request {
            let base = self.directory(id)?;
            ensure!(
                base.is_dir() && !base.join("cancelled").exists(),
                "job unavailable or cancelled"
            );
            ensure!(
                detail.valid()
                    && matches!(
                        (detail.phase.as_str(), detail.unit),
                        ("inventory_input", ProgressUnit::Files)
                            | ("hash_input", ProgressUnit::Bytes)
                            | ("copy_input", ProgressUnit::Bytes)
                    )
                    && detail.input_index.is_some_and(|i| i <= 128)
                    && detail.input_count.is_some_and(|n| n <= 128),
                "invalid staging progress"
            );
            let record = self.pending(id)?;
            ensure!(
                Self::active_input(&record)
                    .is_some_and(|(i, n)| detail.input_index == Some(i as u64 + 1)
                        && detail.input_count == Some(n as u64)),
                "obsolete staging progress"
            );
            let mut detail = detail.clone();
            detail.updated_at_ms = progress::now_ms();
            let file = if detail.phase == "copy_input" {
                "relay-progress.json"
            } else {
                "worker-progress.json"
            };
            let _ = progress::publish(&base.join(file), &detail);
            return Ok(Value::Null);
        }
        if let Rpc::Status { id } = &request {
            // State files are published by atomic rename. No lock is needed
            // for status, even while another command writes/syncs a chunk.
            let Some(mut r) = self.get(id)? else {
                return Ok(to_value(Reply::new("missing", "not found"))?);
            };
            let base = self.directory(id)?;
            if r.reply.state == "running"
                && !r.cancel
                && let (Some(started), Some(job)) = (r.started, r.job.as_ref())
            {
                let admission = if job.kind == exl3_layer::JOB {
                    exl3_layer::Spec::parse(&job.options)
                        .is_ok_and(|spec| spec.backend != exl3_layer::Backend::Cpu)
                } else if job.kind == locate_edit::JOB {
                    locate_edit::Request::parse(&job.options)
                        .is_ok_and(|request| request.edit.backend.is_gpu())
                } else if job.kind == sensitivity::JOB {
                    sensitivity::Request::resolved(&job.options)
                        .is_ok_and(|(request, _)| request.backend.is_gpu())
                } else if job.kind == damage::JOB {
                    damage::Request::parse(&job.options)
                        .is_ok_and(|request| request.backend.is_gpu())
                } else if heretic::job(&job.kind) {
                    true
                } else if matches!(
                    job.kind.as_str(),
                    trace::JOB | circuit::JOB | behavior_direction::JOB
                ) || scratch_eval::private_output(&job.kind)
                {
                    false
                } else {
                    self.cfg.admission.is_some()
                };
                r.reply.progress = report::live(&base, &job.kind, admission, started);
            }
            if !r.cancel {
                let detail = match r.reply.state.as_str() {
                    "queued" => {
                        let local = progress::read(&base.join("worker-progress.json"), 0);
                        let relay = progress::read(&base.join("relay-progress.json"), 0);
                        let detail = match (local, relay) {
                            (Some(mut local), Some(relay))
                                if local.input_index == relay.input_index
                                    && local.phase == "copy_input"
                                    && local.completed >= relay.completed =>
                            {
                                local.threads = relay.threads;
                                Some(bound_progress(local, 0))
                            }
                            (Some(local), Some(relay))
                                if relay.updated_at_ms >= local.updated_at_ms
                                    && matches!(
                                        local.phase.as_str(),
                                        "plan_transfer" | "copy_input"
                                    ) =>
                            {
                                Some(relay)
                            }
                            (Some(local), _) => Some(local),
                            (None, relay) => relay,
                        };
                        detail.filter(|p| {
                            p.input_count.is_none_or(|n| {
                                r.job.as_ref().is_some_and(|j| j.inputs.len() as u64 == n)
                            }) && p
                                .input_index
                                .is_none_or(|i| !r.inputs.contains_key(&(i as usize - 1)))
                        })
                    }
                    "running" => r.started_ms.and_then(|started| {
                        if self
                            .cfg
                            .hy4_plan
                            .as_ref()
                            .is_some_and(|p| p.encode.is_some())
                        {
                            progress::hy4_encode(&base, started)
                        } else {
                            progress::read(&base.join("worker-progress.json"), started)
                                .filter(|p| {
                                    matches!(p.phase.as_str(), "inventory_output" | "verify_output")
                                })
                                .or_else(|| progress::engine(&base, started))
                        }
                    }),
                    _ => None,
                };
                if let Some(detail) = detail {
                    progress::apply(&mut r.reply, detail);
                }
            }
            return Ok(to_value(r.reply)?);
        }
        let _cache = if matches!(
            &request,
            Rpc::Plan { .. } | Rpc::Seal { .. } | Rpc::Write { .. } | Rpc::Cleanup { .. }
        ) {
            Some(lock(&self.cfg.root.join("cache.lock"))?)
        } else {
            None
        };
        let state = self.state_lock()?;
        match request {
            Rpc::Prepare { id, action, job } => Ok(to_value(self.prepare(&id, action, job)?)?),
            Rpc::Status { .. } => unreachable!(),
            Rpc::Cancel { id } => Ok(to_value(self.cancel(&id)?)?),
            Rpc::Plan {
                id,
                bundle,
                preview,
            } => {
                let mut r = self.pending(&id)?;
                ensure!(
                    preview == (r.action == "preview_transfer") && r.action != "cleanup",
                    "plan action mismatch"
                );
                bundle.validate()?;
                if !preview {
                    r.pins.insert(bundle.digest.clone());
                    self.put(&id, &r)?;
                }
                drop(state);
                let active = Self::active_input(&r);
                let mut emit = self.progress_callback(&id)?;
                let plan = self.cache_for(&r).plan(
                    &bundle,
                    preview,
                    &mut progress::Reporter::new(&mut emit, active),
                )?;
                Ok(to_value(plan)?)
            }
            Rpc::Write {
                id,
                digest,
                entry,
                offset,
                data,
            } => {
                let r = self.pending(&id)?;
                ensure!(r.pins.contains(&digest), "unplanned digest");
                let active = Self::active_input(&r);
                drop(state);
                let base = self.directory(&id)?;
                let recent =
                    progress::read(&base.join("worker-progress.json"), 0).is_some_and(|p| {
                        p.phase == "copy_input"
                            && progress::now_ms().saturating_sub(p.updated_at_ms)
                                < progress::INTERVAL.as_millis() as u64
                    });
                let mut emit = self.progress_callback(&id)?;
                let mut reporter = progress::Reporter::new(
                    &mut emit as &mut dyn FnMut(&ProgressDetail) -> Result<()>,
                    active,
                );
                let (next, _) = self.cache_for(&r).write(
                    &digest,
                    &entry,
                    offset,
                    data,
                    (!recent).then_some(&mut reporter),
                )?;
                Ok(json!(next))
            }
            Rpc::Seal { id, digest } => {
                let r = self.pending(&id)?;
                ensure!(r.pins.contains(&digest), "unplanned digest");
                drop(state);
                let active = Self::active_input(&r);
                let mut emit = self.progress_callback(&id)?;
                self.cache
                    .seal(&digest, &mut progress::Reporter::new(&mut emit, active))?;
                Ok(Value::Null)
            }
            Rpc::Resolve {
                id,
                index,
                input,
                bundle,
            } => self.resolve(&id, index, input, bundle, state),
            Rpc::Finish {
                id,
                failure,
                result,
            } => {
                let mut r = self.require(&id)?;
                if r.reply.state == "queued" {
                    if let Some(err) = failure {
                        r.reply = Reply::error(staging_error(&err));
                    } else {
                        ensure!(
                            matches!(r.action.as_str(), "preview_transfer" | "transfer")
                                && r.inputs.len() == r.job.as_ref().unwrap().inputs.len(),
                            "transfer inputs unresolved"
                        );
                        let mut result = result
                            .filter(Value::is_object)
                            .context("invalid transfer result")?;
                        result["inputs"] = to_value(r.inputs.values().collect::<Vec<_>>())?;
                        r.reply = Reply::success(result);
                    }
                    self.put(&id, &r)?;
                }
                Ok(to_value(r.reply)?)
            }
            Rpc::Cleanup { id } => Ok(to_value(self.cleanup(&id)?)?),
            Rpc::Start { id } => {
                let r = self.require(&id)?;
                if r.reply.state != "queued" {
                    return Ok(to_value(r.reply)?);
                }
                ensure!(
                    r.action == "start" && r.inputs.len() == r.job.as_ref().unwrap().inputs.len(),
                    "job inputs unresolved"
                );
                process::detach(&self.cfg_path, &self.directory(&id)?, &id)?;
                Ok(to_value(Reply::new("queued", "launch submitted"))?)
            }
            Rpc::Describe { .. }
            | Rpc::Read { .. }
            | Rpc::SuggestPaths(_)
            | Rpc::TraceRead { .. }
            | Rpc::Stat(_)
            | Rpc::DescribeProgress { .. }
            | Rpc::StageProgress { .. } => unreachable!(),
        }
    }
}

mod process;
mod report;

fn main() {
    unsafe {
        libc::umask(0o077);
    }
    let result = (|| -> Result<Option<Value>> {
        let args: Vec<_> = env::args().skip(1).collect();
        match args.as_slice() {
            [op] if op == "rpc" => {
                let request = decode(std::io::stdin().lock())?;
                let value = match request {
                    // Do not initialize or lock worker state for this read-only RPC.
                    Rpc::SuggestPaths(SuggestRequest { prefix }) => {
                        to_value(suggestions::run(prefix)?)?
                    }
                    Rpc::Stat(StatRequest { path }) => {
                        let reply = suggestions::within_limit(move || stat::get(&path))?
                            .context("worker path check timed out")?;
                        to_value(reply)?
                    }
                    Rpc::DescribeProgress {
                        input,
                        index,
                        count,
                    } => {
                        ensure!(index < count && count <= 128, "invalid input index");
                        let w = Worker::open()?;
                        ensure!(
                            w.cfg.hy4_plan.is_none(),
                            "Hy4 metadata worker does not describe bundles"
                        );
                        ensure!(
                            w.cfg.scratch_access.is_none(),
                            "scratch worker config does not describe bundles"
                        );
                        let (p, root) =
                            w.source(&input.path).context("Source input unavailable")?;
                        let mut emit = |detail: &ProgressDetail| -> Result<()> {
                            let mut out = std::io::stdout().lock();
                            serde_json::to_writer(&mut out, &json!({"progress":detail}))?;
                            writeln!(out)?;
                            out.flush()?;
                            Ok(())
                        };
                        to_value(cache::describe(
                            &p,
                            &input.kind,
                            &root,
                            "hash_input",
                            &mut progress::Reporter::new(&mut emit, Some((index, count))),
                        )?)?
                    }
                    request => Worker::open()?.handle(request)?,
                };
                Ok(Some(value))
            }
            [op, id] if op == "supervise" => {
                process::supervise(&Worker::open()?, id)?;
                Ok(None)
            }
            [op, receipt, tree, output] if op == "build-weight-chunk-index" => {
                let sha256 =
                    weight_source::build_chunk_index(Path::new(receipt), tree, Path::new(output))?;
                Ok(Some(json!({"sha256": sha256})))
            }
            _ => bail!("usage: brewery-worker rpc"),
        }
    })();
    match result {
        Ok(Some(value)) => {
            println!("{}", json!({"ok":value}));
        }
        Ok(None) => {}
        Err(e) => {
            if env::args().nth(1).as_deref() == Some("rpc") {
                println!("{}", json!({"error":format!("{e:#}")}));
            } else {
                let _ = writeln!(std::io::stderr(), "{e:#}");
                std::process::exit(1);
            }
        }
    }
}
