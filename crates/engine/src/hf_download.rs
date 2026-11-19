//! Fetch public model snapshots with a provisioned, pinned hfr executable.
use crate::{progress::Progress, Job};
use anyhow::{ensure, Context, Result};
use brewery_worker::{
    absolute, available, decode_limited,
    hf_download::{is_commit, Config, Spec},
    is_digest, tree_digest, Entry,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File, OpenOptions},
    io::Read,
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt},
        process::CommandExt,
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const MAX_ENTRIES: usize = 4096;
const METADATA_LIMIT: u64 = 4 * 1024 * 1024;
const SIDECAR_FLOOR: u64 = 65536;

#[derive(Deserialize)]
struct ModelInfo {
    id: String,
    sha: String,
    private: bool,
    gated: Value,
}

#[derive(Deserialize)]
struct Row {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: u64,
    oid: String,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    oid: String,
    size: u64,
}

#[derive(Deserialize, Serialize, PartialEq, Eq)]
struct Expected {
    bytes: u64,
    blob: String,
    lfs: Option<String>,
}

#[derive(Deserialize, Serialize, PartialEq, Eq)]
struct Inventory {
    origin: String,
    repo: String,
    commit: String,
    files: BTreeMap<String, Expected>,
}

fn page<T: DeserializeOwned>(client: &ureq::Agent, url: &str) -> Result<(T, Option<String>)> {
    let mut response = client
        .get(url)
        .call()
        .context("Hub metadata request failed")?;
    ensure!(response.status() == 200, "unexpected Hub metadata status");
    let next = response
        .headers()
        .get("link")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.split(',').find(|link| link.contains("rel=\"next\"")))
        .map(|link| -> Result<String> {
            let (_, address) = link.split_once('<').context("invalid Hub pagination")?;
            let (url, _) = address.split_once('>').context("invalid Hub pagination")?;
            Ok(url.into())
        })
        .transpose()?;
    let body = decode_limited(response.body_mut().as_reader(), METADATA_LIMIT)?;
    Ok((body, next))
}

fn safe_tree_path(path: &str) -> bool {
    path.len() <= 512
        && path.split('/').count() <= 16
        && path.split('/').all(|p| {
            let lower = p.to_ascii_lowercase();
            !p.is_empty()
                && p.len() <= 255
                && !matches!(p, "." | "..")
                && !p.starts_with('-')
                && !matches!(lower.as_str(), ".git" | ".cache")
                && !p.ends_with(".incomplete")
                && !p.ends_with(".incomplete.chunks")
                && p.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._- @+".contains(&c))
        })
}

fn inventory(cfg: &Config, spec: &Spec) -> Result<Inventory> {
    let client: ureq::Agent = ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .into();
    let origin = cfg.endpoint();
    let api = format!("{origin}/api/models/{}", spec.repo_id);
    let revision = spec.revision.replace('/', "%2F");
    let (info, _): (ModelInfo, _) = page(&client, &format!("{api}/revision/{revision}"))?;
    ensure!(
        info.id.eq_ignore_ascii_case(&spec.repo_id)
            && is_commit(&info.sha)
            && !info.private
            && info.gated == false
            && (!is_commit(&spec.revision) || spec.revision == info.sha),
        "Hub did not identify an immutable public, ungated model"
    );
    let prefix = format!("{api}/tree/{}?", info.sha);
    let mut url = format!("{prefix}recursive=true&expand=false");
    let mut inventory = Inventory {
        origin,
        repo: spec.repo_id.clone(),
        commit: info.sha,
        files: BTreeMap::new(),
    };
    let mut listed = BTreeSet::new();
    for _ in 0..32 {
        let (rows, next): (Vec<Row>, _) = page(&client, &url)?;
        for row in rows {
            if let Some(name) = &spec.filename {
                if &row.path != name {
                    continue;
                }
            }
            ensure!(
                listed.len() < MAX_ENTRIES
                    && safe_tree_path(&row.path)
                    && listed.insert(row.path.clone()),
                "unsafe, duplicate or oversized Hub tree"
            );
            if row.kind == "directory" {
                continue;
            }
            ensure!(
                row.kind == "file" && is_commit(&row.oid),
                "Hub tree lacks a Git blob identity"
            );
            let lfs = row
                .lfs
                .map(|lfs| -> Result<String> {
                    ensure!(
                        is_digest(&lfs.oid) && lfs.size == row.size,
                        "invalid Hub LFS identity or size"
                    );
                    Ok(lfs.oid)
                })
                .transpose()?;
            inventory.files.insert(
                row.path,
                Expected {
                    bytes: row.size,
                    blob: row.oid,
                    lfs,
                },
            );
        }
        match next {
            None => {
                ensure!(
                    !inventory.files.is_empty(),
                    "requested model file or snapshot was not found"
                );
                ensure!(
                    inventory
                        .dirs()
                        .iter()
                        .all(|dir| !inventory.files.contains_key(dir)),
                    "conflicting file and directory paths in Hub tree"
                );
                return Ok(inventory);
            }
            Some(next) => {
                ensure!(
                    next.starts_with(&prefix) && next.len() <= 4096 && next != url,
                    "invalid Hub pagination"
                );
                url = next;
            }
        }
    }
    anyhow::bail!("Hub tree pagination limit exceeded")
}

impl Inventory {
    fn dirs(&self) -> BTreeSet<String> {
        self.files
            .keys()
            .flat_map(|name| name.match_indices('/').map(|(i, _)| name[..i].to_owned()))
            .collect()
    }

    fn bytes(&self) -> Result<u64> {
        self.files.values().try_fold(0u64, |sum, f| {
            sum.checked_add(f.bytes).context("model size overflow")
        })
    }
}

fn scan(root: &Path, inv: &Inventory, partial: bool) -> Result<(u64, u64)> {
    ensure!(
        fs::symlink_metadata(root)?.is_dir(),
        "unsafe download stage"
    );
    let dirs = inv.dirs();
    let mut stack = vec![root.to_path_buf()];
    let (mut bytes, mut files) = (0u64, 0u64);
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            let name = path
                .strip_prefix(root)?
                .to_str()
                .context("unsafe stage name")?;
            let md = match fs::symlink_metadata(&path) {
                Ok(md) => md,
                // hfr renames the partial and removes its journal on finish.
                Err(e) if partial && e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            if md.is_dir() {
                ensure!(dirs.contains(name), "unexpected directory in download");
                stack.push(path);
            } else {
                ensure!(
                    md.is_file() && md.nlink() == 1,
                    "untrusted file type in download"
                );
                if let Some(expected) = inv.files.get(name) {
                    ensure!(
                        md.len() == expected.bytes,
                        "download file size differs from metadata"
                    );
                    bytes += expected.bytes;
                    files += 1;
                } else if partial {
                    let (name, journal) =
                        if let Some(name) = name.strip_suffix(".incomplete.chunks") {
                            (name, true)
                        } else {
                            (
                                name.strip_suffix(".incomplete")
                                    .context("unexpected stage file")?,
                                false,
                            )
                        };
                    let expected = inv.files.get(name).context("unexpected stage file")?;
                    let cap = if journal {
                        expected.bytes.max(SIDECAR_FLOOR)
                    } else {
                        expected.bytes
                    };
                    ensure!(md.len() <= cap, "download staging byte limit exceeded");
                } else {
                    anyhow::bail!("unexpected file in download");
                }
            }
        }
    }
    if !partial {
        ensure!(
            files == inv.files.len() as u64,
            "download inventory incomplete"
        );
    }
    Ok((bytes, files))
}

fn prepare_stage(root: &Path, inv: &Inventory) -> Result<PathBuf> {
    let manifest = root.join("inventory.json");
    let data = root.join("data");
    if root.exists() {
        let reuse = brewery_worker::load::<Inventory>(&manifest).is_ok_and(|old| old == *inv)
            && scan(&data, inv, true).is_ok();
        if reuse {
            return Ok(data);
        }
        fs::remove_dir_all(root)?;
    }
    fs::create_dir(root)?;
    fs::create_dir(&data)?;
    brewery_worker::atomic(&manifest, inv)?;
    Ok(data)
}

fn transfer(
    cfg: &Config,
    inv: &Inventory,
    data: &Path,
    home: &Path,
    pg: &Progress,
    reserve: u64,
) -> Result<()> {
    // Group only equal-size files: RLIMIT_FSIZE then caps total payload growth
    // even if a repeated tree request unexpectedly returns different sizes.
    let mut groups: BTreeMap<u64, Vec<&str>> = BTreeMap::new();
    let mut budget = 0u64;
    for (name, f) in &inv.files {
        let cap = f.bytes.max(SIDECAR_FLOOR);
        groups.entry(cap).or_default().push(name);
        budget = budget
            .checked_add(cap.checked_mul(3).context("stage size overflow")?)
            .context("stage size overflow")?;
    }
    ensure!(
        available(data)?.saturating_sub(reserve) >= budget,
        "insufficient target disk for download"
    );
    let mut phase = pg.start_for(
        "hf_download_transfer",
        "bytes",
        Some(inv.bytes()?),
        Some(inv.files.len() as u64),
        None,
    );
    let mut files_done = 0;
    let deadline = Instant::now() + Duration::from_secs(24 * 3600);
    for (cap, names) in groups.into_iter().rev() {
        for batch in names.chunks(64) {
            let mut command = Command::new(&cfg.hfr);
            command
                .env_clear()
                .env("HOME", home)
                .env("HF_HOME", home)
                .env("XDG_CACHE_HOME", home)
                .current_dir(home)
                .args([
                    "--endpoint",
                    &inv.origin,
                    "-j",
                    "4",
                    "download",
                    "--repo-type",
                    "model",
                    "--revision",
                    &inv.commit,
                    "--transfer",
                    "range",
                    "--verify",
                    "--local-dir",
                ])
                .arg(data)
                .arg("--")
                .arg(&inv.repo)
                .args(batch)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            unsafe {
                command.pre_exec(move || {
                    let limit = libc::rlimit {
                        rlim_cur: cap,
                        rlim_max: cap,
                    };
                    if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            let mut child = command.spawn().context("cannot run pinned hfr")?;
            let result = loop {
                let poll = (|| -> Result<bool> {
                    let (bytes, files) = scan(data, inv, true)?;
                    phase.set_completed(bytes);
                    for _ in files_done..files {
                        phase.file_done();
                    }
                    files_done = files;
                    ensure!(
                        Instant::now() < deadline,
                        "Hugging Face download deadline exceeded"
                    );
                    if let Some(status) = child.try_wait()? {
                        ensure!(
                            status.success(),
                            "hfr range transfer or verification failed"
                        );
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                })();
                match poll {
                    Ok(true) => break Ok(()),
                    Ok(false) => thread::sleep(Duration::from_millis(300)),
                    Err(error) => break Err(error),
                }
            };
            if result.is_err() {
                let _ = child.kill();
                let _ = child.wait();
            }
            result?;
        }
    }
    phase.finish();
    Ok(())
}

fn verify(inv: &Inventory, data: &Path, pg: &Progress) -> Result<Vec<Entry>> {
    scan(data, inv, false)?;
    let mut phase = pg.start_for(
        "hf_download_verify",
        "bytes",
        Some(inv.bytes()?),
        Some(inv.files.len() as u64),
        None,
    );
    let mut files = Vec::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    for (name, expected) in &inv.files {
        let path = data.join(name);
        phase.file(&path);
        let mut f = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&path)?;
        let meta = f.metadata()?;
        ensure!(
            meta.is_file() && meta.nlink() == 1 && meta.len() == expected.bytes,
            "unexpected download file metadata"
        );
        let mut git = expected.lfs.is_none().then(|| {
            let mut git = Sha1::new();
            git.update(format!("blob {}\0", expected.bytes));
            git
        });
        let mut sha = Sha256::new();
        let mut read = 0u64;
        loop {
            let n = f.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            read += n as u64;
            ensure!(
                read <= expected.bytes,
                "download file grew during verification"
            );
            sha.update(&buffer[..n]);
            if let Some(git) = &mut git {
                git.update(&buffer[..n]);
            }
            phase.advance(n as u64);
        }
        let sha256 = format!("{:x}", sha.finalize());
        let valid = match (git, &expected.lfs) {
            (Some(git), None) => format!("{:x}", git.finalize()) == expected.blob,
            (None, Some(lfs)) => lfs == &sha256,
            _ => false,
        };
        if !valid || read != expected.bytes {
            // hfr skips an existing same-size file. Allow a clean retry of
            // this item rather than retaining a permanently poisoned stage.
            fs::remove_file(&path)?;
            anyhow::bail!("download checksum or size did not match the pinned Hub inventory");
        }
        f.sync_all()?;
        files.push(Entry {
            path: name.clone(),
            sha256,
            bytes: read,
        });
        phase.file_done();
    }
    for dir in inv.dirs().iter().rev() {
        File::open(data.join(dir))?.sync_all()?;
    }
    File::open(data)?.sync_all()?;
    phase.finish();
    Ok(files)
}

pub(crate) fn run(job: &Job, cwd: &Path, pg: &Progress) -> Result<Value> {
    ensure!(
        job.inputs.is_empty() && job.prompt.is_none(),
        "download accepts no models or prompts as inputs"
    );
    let spec = Spec::parse(&job.options)?;
    let config = env::var_os("BREWERY_HF_DOWNLOAD_CONFIG").context("download is not configured")?;
    let (cfg, root, reserve): (Config, PathBuf, u64) = brewery_worker::load(Path::new(&config))?;
    cfg.validate()?;
    absolute(&root)?;
    let published = cwd.join("download");
    ensure!(!published.exists(), "download already published");
    let step = pg.start("hf_download_resolve", "items", None);
    let inv = inventory(&cfg, &spec)?;
    step.finish();
    let bytes = inv.bytes()?;
    ensure!(
        bytes > 0 && bytes <= cfg.max_bytes,
        "model exceeds configured download size"
    );
    let data = prepare_stage(&root, &inv)?;
    let home = tempfile::Builder::new()
        .prefix(".hfr-home-")
        .tempdir_in(cwd)?;
    transfer(&cfg, &inv, &data, home.path(), pg, reserve)?;
    let files = verify(&inv, &data, pg)?;
    let mut artifact = published.clone();
    let sha256 = if let Some(filename) = &spec.filename {
        artifact.push(filename);
        files[0].sha256.clone()
    } else {
        tree_digest(&files)
    };
    let kind = spec.artifact_kind();
    let mut receipt = json!({
        "kind": "hf_download", "repo_id":spec.repo_id, "revision": spec.revision,
        "commit": inv.commit, "verified": true,
        "artifact": {"kind":kind,"path":artifact,"bytes":bytes,"sha256":sha256}
    });
    if let Some(filename) = spec.filename {
        receipt["filename"] = json!(filename);
    }
    // The worker exposes this directory only after the supervisor records a
    // successful job. The private stage and work are on the same filesystem.
    let _ = pg.start("publish_output", "items", None);
    fs::rename(&data, &published)?;
    File::open(cwd)?.sync_all()?;
    File::open(root)?.sync_all()?;
    Ok(receipt)
}
