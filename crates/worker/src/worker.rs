mod cache;
mod progress;
mod publication;
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
    engine: PathBuf,
    admission: Option<PathBuf>,
    #[serde(default = "reserve")]
    disk_reserve_bytes: u64,
    #[serde(default)]
    engine_env: BTreeMap<String, String>,
    pinned_publication: Option<publication::Pin>,
    managed_settings: Option<operational::Policy>,
    hf_download: Option<hf_download::Config>,
    backend_weight_download_root: Option<PathBuf>,
}
fn reserve() -> u64 {
    1024 * 1024 * 1024
}

impl Config {
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
        absolute(&cfg.root)?;
        absolute(&cfg.engine)?;
        if let Some(p) = &cfg.admission {
            absolute(p)?;
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
            complete: cfg.root.join("cache"),
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
        if let Ok(p) = cache::authorized(p, &self.cfg.source_roots) {
            let root = self
                .cfg
                .source_roots
                .iter()
                .filter(|r| p.starts_with(r))
                .max_by_key(|r| r.components().count())
                .unwrap();
            return Ok((p, root.clone()));
        }
        // Only published cache data or output of succeeded jobs is a source.
        for (base, is_job) in [(&self.cache.complete, false), (&self.jobs, true)] {
            if let Ok(rel) = p.strip_prefix(base) {
                let id = rel
                    .components()
                    .next()
                    .and_then(|p| p.as_os_str().to_str())
                    .context("invalid source")?;
                let root = if is_job {
                    let r = self.require(id)?;
                    ensure!(
                        r.reply.state == "succeeded" && r.action == "start",
                        "job output is not published"
                    );
                    let work = base.join(id).join("work");
                    if r.job.is_some_and(|job| job.kind == "hf_download") {
                        work.join("download")
                    } else {
                        work
                    }
                } else {
                    ensure!(is_digest(id), "invalid cache source");
                    base.join(id).join("data")
                };
                return Ok((cache::authorized(p, std::slice::from_ref(&root))?, root));
            }
        }
        bail!("input outside allowed source roots")
    }

    // The backend editor cannot read arbitrary paths through ordinary source
    // RPCs. An approved input names exactly one successful job publication.
    fn backend_weight_source(&self, input: &Input) -> Result<(PathBuf, String)> {
        let invalid = "backend input is not a verified publication";
        ensure!(input.host == "backend-hf", "{invalid}");
        let download_jobs = self
            .cfg
            .backend_weight_download_root
            .as_ref()
            .context(invalid)?
            .join("jobs");
        let (record_root, expected_kind, rel) =
            if let Ok(rel) = input.path.strip_prefix(&download_jobs) {
                ensure!(weight_source::supports(&input.kind), "{invalid}");
                (&download_jobs, "hf_download", rel)
            } else {
                let rel = input.path.strip_prefix(&self.jobs).context(invalid)?;
                let expected = match input.kind.as_str() {
                    "hf" | "safetensors" => "weight_apply",
                    "weight-edit-proposal" => "weight_propose",
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
        ensure!(
            receipt.action == "start"
                && receipt.reply.state == "succeeded"
                && !receipt.cancel
                && receipt.job.is_some_and(|j| j.kind == expected_kind),
            "{invalid}"
        );
        let result = receipt.reply.result.as_ref().context(invalid)?;
        let artifact = &result["artifact"];
        ensure!(
            (expected_kind == "weight_propose" || result["verified"] == true)
                && result["kind"] == expected_kind
                && artifact["kind"] == input.kind
                && artifact["path"].as_str().map(Path::new) == Some(input.path.as_path()),
            "{invalid}"
        );
        let leaf = match (expected_kind, input.kind.as_str()) {
            ("hf_download", "hf") => work.join("download"),
            ("hf_download", "safetensors") => {
                let spec = hf_download::Spec::parse(&json!({
                    "repo_id":result["repo_id"],"revision":result["revision"],"filename":result["filename"]
                }))?;
                work.join("download").join(spec.filename.context(invalid)?)
            }
            ("weight_apply", "hf") => work.join("edited.hf"),
            ("weight_apply", "safetensors") => work.join("edited.safetensors"),
            ("weight_propose", "weight-edit-proposal") => work.join("weight-proposal.json"),
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
        for input in &job.inputs {
            token(&input.host)?;
            token(&input.kind)?;
            absolute(&input.path)?;
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
        if self.cfg.backend_weight_download_root.is_some()
            && !(action == "start"
                && job.prompt.is_none()
                && job.options.get("packed").is_none()
                && job
                    .inputs
                    .first()
                    .is_some_and(|i| weight_source::supports(&i.kind))
                && match job.kind.as_str() {
                    "weight_inspect" | "weight_propose" => job.inputs.len() == 1,
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

    fn record_prepare(&self, id: &str, action: String, job: Job, reply: Reply) -> Result<Reply> {
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
        let mut r = self.pending(id)?;
        let original = r
            .job
            .as_ref()
            .and_then(|j| j.inputs.get(index))
            .context("unknown input index")?
            .clone();
        ensure!(input.kind == original.kind, "resolved input kind changed");
        let job = r.job.as_ref().unwrap();
        let native_safe =
            index == 0 && weight_source::job(&job.kind) && weight_source::supports(&input.kind);
        ensure!(
            !native_safe || provided.is_none(),
            "SafeTensors weight jobs require a direct local source"
        );
        ensure!(
            self.cfg.backend_weight_download_root.is_none() || provided.is_none(),
            "backend weight jobs do not accept staged inputs"
        );
        let mut pinned = false;
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
            let bundle = if let Some(pin) = pin {
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
                } else {
                    let (p, boundary) = self
                        .source(&input.path)
                        .context("Source input unavailable")?;
                    (p, boundary, None)
                };
                if weight_hgn {
                    ensure!(
                        fs::symlink_metadata(&p)?.is_file()
                            && p.extension().is_some_and(|ext| ext == "hgn"),
                        "HGN weight jobs require a regular .hgn file, not a bundle"
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
                let proof = if native_safe {
                    self.cfg
                        .engine_env
                        .get(weight_source::PUBLICATION)
                        .map(|receipt| {
                            weight_source::publication(&p, &input.kind, Path::new(receipt))
                        })
                        .transpose()
                        .context("SafeTensors immutable publication invalid")?
                        .flatten()
                } else {
                    None
                };
                if let Some(proof) = proof {
                    proof.bundle
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
                    } else if expected.is_some() {
                        4 << 20
                    } else {
                        u64::MAX
                    };
                    let d = cache::describe_bounded(
                        &p,
                        &input.kind,
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
            guard = Some(self.state_lock()?);
            r = self.pending(id)?;
            (bundle, true)
        };
        input.extra = original.extra.clone();
        for k in [
            "sha256",
            "bundle_sha256",
            "bytes",
            "files",
            "cached",
            "source_host",
            "integrity_basis",
        ] {
            input.extra.remove(k);
        }
        if let Some(f) = bundle.files.iter().find(|f| f.path == bundle.entry) {
            input.extra.insert("sha256".into(), json!(f.sha256));
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
        // Reads/description may hash large files; they never mutate durable state.
        let request = match request {
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
                r.reply.progress =
                    report::live(&base, &job.kind, self.cfg.admission.is_some(), started);
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
                        progress::read(&base.join("worker-progress.json"), started)
                            .filter(|p| {
                                matches!(p.phase.as_str(), "inventory_output" | "verify_output")
                            })
                            .or_else(|| progress::engine(&base, started))
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
                let plan = self.cache.plan(
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
                let (next, _) = self.cache.write(
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
                    Rpc::DescribeProgress {
                        input,
                        index,
                        count,
                    } => {
                        ensure!(index < count && count <= 128, "invalid input index");
                        let w = Worker::open()?;
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
