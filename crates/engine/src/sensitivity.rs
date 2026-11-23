use crate::{
    bounded, child_threads,
    gguf_conversion::{
        inventory::{Model, Namespace},
        Tool,
    },
    integrity,
    progress::Progress,
    w0_libraries, write_json, Job,
};
use anyhow::{ensure, Context, Result};
use brewery_worker::{
    is_digest,
    locate_edit::{self, Backend},
    sensitivity::{self as wire, Artifact, Request, Tensor},
    weight_source::open_regular,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::Read,
    os::unix::fs::PermissionsExt,
    path::{Component, Path, PathBuf},
    time::{Duration, Instant},
};

const MANIFEST: &str = "SHA256SUMS";
const CORPUS: &str = include_str!("../../eval/data/edit-checks/heldout.txt");
const MAX_CONTEXT: usize = 128;
const TOOL_TIMEOUT: Duration = Duration::from_secs(120);
const JOB_TIMEOUT: Duration = Duration::from_secs(1800);

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) struct Pin {
    root: PathBuf,
    pub(crate) sha: String,
    members: BTreeMap<String, String>,
}

impl Pin {
    pub(crate) fn load() -> Result<Self> {
        let root =
            PathBuf::from(env::var_os(wire::PIN).context("missing sensitivity runtime pin")?);
        ensure!(
            root.is_absolute() && integrity::canonical(&root)?.is_dir(),
            "invalid sensitivity pin directory"
        );
        let expected =
            env::var(wire::MANIFEST_SHA).context("missing sensitivity manifest SHA-256 pin")?;
        let contents = bounded(open_regular(&root.join(MANIFEST))?, 256 << 10)?;
        ensure!(
            is_digest(&expected) && expected == digest(&contents),
            "sensitivity runtime manifest SHA-256 mismatch"
        );
        let mut members = BTreeMap::new();
        for line in std::str::from_utf8(&contents)?.lines() {
            let (hash, name) = line
                .split_once("  ")
                .context("invalid sensitivity pin record")?;
            ensure!(
                is_digest(hash)
                    && !name.is_empty()
                    && name.bytes().all(|b| b.is_ascii_alphanumeric()
                        || matches!(b, b'/' | b'.' | b'_' | b'-' | b'+'))
                    && Path::new(name)
                        .components()
                        .all(|c| matches!(c, Component::Normal(_)))
                    && members.insert(name.to_owned(), hash.to_owned()).is_none(),
                "invalid or duplicate sensitivity pin member"
            );
        }
        Ok(Self {
            root,
            sha: expected,
            members,
        })
    }

    pub(crate) fn stage(&self, backend: Backend, work: &Path) -> Result<(Tool, PathBuf)> {
        let work = work.join(backend.key());
        fs::create_dir(&work)?;
        let binary = "lanternfish";
        let configured = Tool::from_path(self.root.join(binary))?;
        let sha = self
            .members
            .get(binary)
            .context("missing Lanternfish executable pin")?;
        let tool = Tool::stage_digest(&work, configured, sha, binary, "Lanternfish sensitivity")?;
        let prefix = format!("lib/{}/", backend.key());
        let libs: Vec<_> = self
            .members
            .iter()
            .filter_map(|(path, sha)| path.strip_prefix(&prefix).map(|name| (name, sha.as_str())))
            .collect();
        ensure!(
            !libs.is_empty(),
            "no pinned Lanternfish libraries for {}",
            backend.label()
        );
        let libs = w0_libraries::stage_from(
            &work,
            &self.root.join(prefix.trim_end_matches('/')),
            wire::PIN,
            "Lanternfish sensitivity",
            &libs,
        )?;
        Ok((tool, libs))
    }
}

#[derive(Deserialize)]
struct Tokenized {
    tokens: Vec<i32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Site {
    kind: String,
    name: String,
}

#[derive(Serialize)]
struct Observation<'a> {
    tokens: &'a [i32],
    target_id: i32,
}

#[derive(Deserialize)]
struct Gradient {
    schema_version: u32,
    backend: Backend,
    placement: Placement,
    shape: Vec<u64>,
    tokens: Vec<i32>,
    target_id: i32,
    site: Site,
    loss: f64,
    sha256: String,
}

#[derive(Deserialize)]
struct Placement {
    strict: bool,
    device: String,
}

fn observations(
    request: &Request,
    input: &Path,
    id: &integrity::Identity,
    runtime: &(Tool, PathBuf),
    work: &Path,
    threads: usize,
) -> Result<Vec<(Vec<i32>, i32)>> {
    let (tool, libs) = runtime;
    let tokenize = |text: &str| -> Result<Vec<i32>> {
        let mut cmd = tool.command(work, threads);
        cmd.env("LD_LIBRARY_PATH", libs)
            .arg("tokenize")
            .arg("--model")
            .arg(input)
            .arg("--text")
            .arg(text);
        id.check()?;
        let (out, _) = tool
            .capture_for(&mut cmd, work, TOOL_TIMEOUT)
            .context("Lanternfish tokenization failed")?;
        id.check()?;
        let tokens: Tokenized = serde_json::from_slice(&out)?;
        ensure!(
            (1..=MAX_CONTEXT).contains(&tokens.tokens.len())
                && tokens.tokens.iter().all(|&t| t >= 0),
            "invalid held-out token IDs or context length"
        );
        Ok(tokens.tokens)
    };
    let mut out = Vec::new();
    for line in CORPUS.lines() {
        let first_word = line
            .split_once(' ')
            .context("missing W4 held-out suffix")?
            .0;
        let full = tokenize(line)?;
        let prefix = tokenize(first_word)?;
        ensure!(
            full.starts_with(&prefix) && full.len() > prefix.len(),
            "held-out prefix does not match the full token sequence"
        );
        for position in prefix.len()..full.len() {
            out.push((full[..position].to_vec(), full[position]));
            if out.len() == request.observations {
                return Ok(out);
            }
        }
    }
    anyhow::bail!("not enough held-out observations for the requested budget")
}

fn selection(model: &Path, req: &Request) -> Result<Vec<Tensor>> {
    let model = Model::open(model, Namespace::Stock)?;
    ensure!(
        matches!(model.architecture.as_str(), "qwen2" | "llama"),
        "sensitivity requires a dense Qwen2 or Llama GGUF"
    );
    ensure!(
        model.tensors.values().all(|t| matches!(t.ty, 0 | 30)),
        "sensitivity requires F32 or BF16 GGUF weights"
    );
    let named: BTreeSet<_> = req.tensors.iter().collect();
    let layers: BTreeSet<_> = req.layers.iter().copied().collect();
    let out: Vec<_> = model
        .tensors
        .into_iter()
        .filter_map(|(name, t)| {
            let layer = wire::layer(&name);
            if named.contains(&name)
                || (t.shape.len() == 2
                    && name.ends_with(".weight")
                    && layer.is_some_and(|l| layers.contains(&l)))
            {
                Some(Tensor {
                    name,
                    layer,
                    shape: t.shape,
                    fisher: 0.,
                })
            } else {
                None
            }
        })
        .collect();
    ensure!(
        if named.is_empty() {
            out.iter().filter_map(|t| t.layer).collect::<BTreeSet<_>>() == layers
        } else {
            out.len() == named.len()
        },
        "selected layers or tensors are not in the GGUF inventory"
    );
    req.calls(out.len())?;
    let bytes = out
        .iter()
        .try_fold(0, |n, t| Ok::<_, anyhow::Error>(n + t.bytes()?))?;
    ensure!(
        bytes * req.observations as u64 <= wire::MAX_TOTAL_BYTES,
        "sensitivity exceeds the total gradient byte budget"
    );
    Ok(out)
}

fn matches_device(backend: Backend, device: &str) -> bool {
    let prefix = match backend {
        Backend::Cpu => return device == "CPU",
        Backend::Cuda => "CUDA",
        Backend::Rocm => "ROCm",
    };
    device
        .strip_prefix(prefix)
        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
}

fn squares(
    out: &Path,
    backend: Backend,
    observation: &Observation<'_>,
    site: &Site,
    t: &Tensor,
) -> Result<f64> {
    let metadata = bounded(open_regular(&out.join("gradient.json"))?, 16 << 10)?;
    let meta: Gradient = serde_json::from_slice(&metadata)?;
    ensure!(
        meta.schema_version == 1
            && meta.placement.strict
            && meta.backend == backend
            && matches_device(backend, &meta.placement.device)
            && meta.shape == t.shape
            && meta.tokens == observation.tokens
            && meta.target_id == observation.target_id
            && &meta.site == site
            && meta.loss.is_finite()
            && meta.loss >= 0.
            && is_digest(&meta.sha256),
        "invalid gradient metadata, site or strict backend placement"
    );
    let mut file = open_regular(&out.join("gradient.f32"))?;
    let mut left = t.bytes()?;
    ensure!(
        file.metadata()?.len() == left,
        "gradient byte count does not match GGUF shape"
    );
    let mut hash = Sha256::new();
    let mut sum = 0.;
    let mut chunk = vec![0u8; 256 << 10];
    while left != 0 {
        let n = left.min(chunk.len() as u64) as usize;
        let data = &mut chunk[..n];
        file.read_exact(data)?;
        hash.update(&*data);
        for word in data.chunks_exact(4) {
            let value = f32::from_le_bytes(word.try_into().expect("four bytes"));
            ensure!(value.is_finite(), "non-finite gradient");
            sum += f64::from(value).powi(2);
        }
        left -= n as u64;
    }
    ensure!(
        file.read(&mut chunk[..1])? == 0 && format!("{:x}", hash.finalize()) == meta.sha256,
        "gradient size or SHA-256 mismatch"
    );
    Ok(sum)
}

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let (req, revision_sha256) = Request::resolved(&job.options)?;
    let [input] = job.inputs.as_slice() else {
        anyhow::bail!("sensitivity needs one GGUF input")
    };
    let model_id = input
        .id
        .as_deref()
        .filter(|id| !id.is_empty())
        .context("sensitivity needs a model ID")?;
    ensure!(
        job.prompt.is_none() && input.kind == "gguf",
        "sensitivity needs a GGUF input and no prompt"
    );
    req.check_model(input.sha256.as_deref().unwrap_or_default())?;
    ensure!(
        digest(CORPUS.as_bytes()) == req.corpus_sha256,
        "held-out corpus SHA-256 mismatch"
    );
    let backend = req.backend;
    ensure!(
        !backend.is_gpu() || env::var(locate_edit::GPU_GATE).as_deref() == Ok("1"),
        "sensitivity GPU loads are not enabled"
    );
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let id = &ids[0];
    req.check_model(&id.sha256)?;
    ensure!(
        id.files.len() == 1 && input.path.is_file(),
        "sensitivity needs a single-file GGUF"
    );
    let mut tensors = selection(&input.path, &req)?;
    id.check()?;
    let work = tempfile::Builder::new()
        .prefix(".sensitivity-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(cwd)?;
    let work = work.path();
    let threads = child_threads(None)?.min(16);
    let pin = Pin::load()?;
    let cpu = pin.stage(Backend::Cpu, work)?;
    let obs = observations(&req, &input.path, id, &cpu, work, threads)?;
    let (tool, libs) = if backend == Backend::Cpu {
        cpu
    } else {
        pin.stage(backend, work)?
    };
    let mut phase = progress.start(
        "sensitivity_gradients",
        "items",
        Some(req.calls(tensors.len())? as u64),
    );
    let started = Instant::now();
    for t in &mut tensors {
        let site = Site {
            kind: "weight".into(),
            name: t.name.clone(),
        };
        let mut sum = 0.;
        for (tokens, target_id) in &obs {
            let timeout = JOB_TIMEOUT
                .checked_sub(started.elapsed())
                .context("sensitivity exceeded time budget")?;
            let attempt = tempfile::tempdir_in(work)?;
            let spec = attempt.path().join("spec.json");
            let out = attempt.path().join("gradient");
            let observation = Observation {
                tokens,
                target_id: *target_id,
            };
            write_json(
                &spec,
                &serde_json::json!({"tokens":tokens,"target_id":target_id,"site":site}),
            )?;
            let mut cmd = tool.command(attempt.path(), threads);
            cmd.env("LD_LIBRARY_PATH", &libs)
                .env("LANTERNFISH_THREADS", threads.to_string())
                .arg("grad-select")
                .arg("--model")
                .arg(&input.path)
                .arg("--spec")
                .arg(spec)
                .arg("--out")
                .arg(&out)
                .args(["--backend", backend.key()]);
            for &key in backend.visibility_vars() {
                if let Some(value) = env::var_os(key) {
                    cmd.env(key, value);
                } else {
                    cmd.env_remove(key);
                }
            }
            match backend {
                Backend::Cpu => {}
                Backend::Cuda => {
                    cmd.env("NVIDIA_TF32_OVERRIDE", "0")
                        .env("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32");
                }
                Backend::Rocm => {
                    cmd.env_remove(locate_edit::CUDA_DEVICES);
                }
            }
            id.check()?;
            tool.capture_for(&mut cmd, attempt.path(), timeout.min(TOOL_TIMEOUT))
                .context("Lanternfish grad-select failed")?;
            id.check()?;
            sum += squares(&out, backend, &observation, &site, t)?;
            phase.advance(1);
        }
        t.fisher = sum / ((t.bytes()? / 4) as f64 * obs.len() as f64);
    }
    phase.finish();
    let layers = wire::aggregate(&tensors)?;
    let artifact = Artifact {
        schema_version: 1,
        kind: wire::JOB.into(),
        qualification: wire::QUALIFICATION.into(),
        model_id: model_id.into(),
        model_sha256: id.sha256.clone(),
        revision_sha256,
        corpus_sha256: wire::CORPUS_SHA256.into(),
        backend,
        observation_count: obs.len(),
        observations_sha256: digest(&serde_json::to_vec(
            &obs.iter()
                .map(|(tokens, target_id)| Observation {
                    tokens,
                    target_id: *target_id,
                })
                .collect::<Vec<_>>(),
        )?),
        executable_sha256: tool.sha,
        manifest_sha256: pin.sha,
        tensors,
        layers,
    };
    artifact.check(&req, model_id, &artifact.revision_sha256)?;
    Ok(serde_json::to_value(artifact)?)
}
