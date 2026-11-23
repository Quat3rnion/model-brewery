use crate::{
    Job, bounded, child_threads,
    gguf_conversion::{
        Tool,
        inventory::{Model, Namespace},
    },
    integrity::{self, Identity},
    progress::Progress,
    sensitivity::Pin,
    write_json,
};
use anyhow::{Context, Result, ensure};
use brewery_worker::{
    circuit::{self, Artifact, Request, Site},
    locate_edit::Backend,
    trace::{self, Capture, CircuitTrace, Key, Operation, Rule},
    weight_source::open_regular,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

const CALL_TIMEOUT: Duration = Duration::from_secs(120);
const JOB_TIMEOUT: Duration = Duration::from_secs(1800);

#[derive(Serialize, Deserialize, PartialEq)]
struct Activation {
    kind: String,
    #[serde(flatten)]
    key: Key,
}

#[derive(Deserialize)]
struct Gradient {
    schema_version: u32,
    objective: String,
    dtype: String,
    backend: Backend,
    placement: Placement,
    tokens: Vec<i32>,
    target_id: i32,
    site: Activation,
    shape: Vec<usize>,
    logit: f64,
    sha256: String,
}

#[derive(Deserialize)]
struct Placement {
    strict: bool,
    device: String,
}

struct Runtime<'a> {
    tool: Tool,
    libs: PathBuf,
    model: &'a Path,
    identity: &'a Identity,
    work: &'a Path,
    threads: usize,
    started: Instant,
    width: usize,
    targets: (i32, Option<i32>),
}

impl Runtime<'_> {
    fn command(&self, cwd: &Path, subcommand: &str) -> Command {
        let mut cmd = self.tool.command(cwd, self.threads);
        cmd.env("LD_LIBRARY_PATH", &self.libs)
            .env("LANTERNFISH_THREADS", self.threads.to_string())
            .arg(subcommand)
            .arg("--model")
            .arg(self.model);
        cmd
    }

    fn call(&self, cwd: &Path, cmd: &mut Command) -> Result<Vec<u8>> {
        let timeout = JOB_TIMEOUT
            .checked_sub(self.started.elapsed())
            .context("circuit exceeded time budget")?;
        self.identity.check()?;
        let (out, _) = self.tool.capture_for(cmd, cwd, timeout.min(CALL_TIMEOUT))?;
        self.identity.check()?;
        Ok(out)
    }

    fn tokens(&self, prompt: &str) -> Result<Vec<i32>> {
        #[derive(Deserialize)]
        struct Tokens {
            tokens: Vec<i32>,
        }
        let mut cmd = self.command(self.work, "tokenize");
        cmd.arg(format!("--text={prompt}"));
        let tokens: Tokens = serde_json::from_slice(
            &self
                .call(self.work, &mut cmd)
                .context("Lanternfish circuit tokenization failed")?,
        )?;
        Ok(tokens.tokens)
    }

    fn trace(
        &self,
        prompt: &str,
        tokens: &[i32],
        spec: &Capture,
        patch: Option<(Key, Vec<f32>)>,
    ) -> Result<CircuitTrace> {
        let attempt = tempfile::tempdir_in(self.work)?;
        let cwd = attempt.path();
        let input = cwd.join("spec.json");
        let out = cwd.join("trace");
        write_json(&input, &serde_json::to_value(spec)?)?;
        let mut cmd = self.command(cwd, "trace");
        cmd.arg(format!("--prompt={prompt}"))
            .arg("--spec")
            .arg(input)
            .arg("--out")
            .arg(&out)
            .arg("--threads")
            .arg(self.threads.to_string());
        let key = patch.as_ref().map(|(key, _)| *key);
        if let Some((key, vector)) = patch {
            let rules = [Rule {
                layer: key.layer,
                role: key.role,
                positions: [key.position].into(),
                op: Operation::Replace { vector },
            }];
            Rule::check_all(&rules)?;
            let file = cwd.join("intervention.json");
            write_json(&file, &serde_json::to_value(rules)?)?;
            cmd.arg("--intervention").arg(file);
        }
        self.call(cwd, &mut cmd)
            .context("Lanternfish circuit trace failed")?;
        trace::circuit_trace(&out, spec, tokens, self.width, self.targets, key)
    }

    fn gradient(&self, tokens: &[i32], key: Key, target_id: i32) -> Result<Vec<f32>> {
        let attempt = tempfile::tempdir_in(self.work)?;
        let cwd = attempt.path();
        let spec = cwd.join("spec.json");
        let out = cwd.join("gradient");
        let site = Activation {
            kind: "activation".into(),
            key,
        };
        write_json(
            &spec,
            &json!({
                "tokens":tokens, "target_id":target_id, "objective":"target_logit", "site":site,
            }),
        )?;
        let mut cmd = self.command(cwd, "grad-select");
        cmd.arg("--spec")
            .arg(spec)
            .arg("--out")
            .arg(&out)
            .args(["--backend", "cpu"]);
        self.call(cwd, &mut cmd)
            .context("Lanternfish activation gradient failed")?;
        ensure!(
            fs::symlink_metadata(&out)?.is_dir(),
            "invalid circuit gradient directory"
        );
        let meta: Gradient = serde_json::from_slice(&bounded(
            open_regular(&out.join("gradient.json"))?,
            16 << 10,
        )?)?;
        ensure!(
            meta.schema_version == 1
                && meta.objective == "target_logit"
                && meta.dtype == "f32_le"
                && meta.backend == Backend::Cpu
                && meta.placement.strict
                && meta.placement.device == "CPU"
                && meta.tokens == tokens
                && meta.target_id == target_id
                && meta.site == site
                && meta.shape == [self.width]
                && meta.logit.is_finite(),
            "invalid circuit gradient metadata"
        );
        let size = self.width * 4;
        let data = bounded(open_regular(&out.join("gradient.f32"))?, size as u64)?;
        ensure!(
            data.len() == size && format!("{:x}", Sha256::digest(&data)) == meta.sha256,
            "invalid circuit gradient size or SHA-256"
        );
        let values: Vec<_> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite circuit gradient"
        );
        Ok(values)
    }
}

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let req = Request::parse(&job.options)?;
    let [input] = job.inputs.as_slice() else {
        anyhow::bail!("circuit needs exactly one Library GGUF");
    };
    ensure!(
        job.prompt.is_none() && input.kind == "gguf",
        "circuit needs a GGUF and options prompts"
    );
    let model_id = input
        .id
        .as_deref()
        .filter(|id| !id.is_empty())
        .context("missing circuit model ID")?;
    req.check_model(input.sha256.as_deref().unwrap_or_default())?;
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let identity = &ids[0];
    req.check_model(&identity.sha256)?;
    ensure!(
        identity.files.len() == 1 && input.path.is_file(),
        "circuit needs a single-file GGUF"
    );
    let model = Model::open(&input.path, Namespace::Stock)?;
    ensure!(
        matches!(model.architecture.as_str(), "qwen2" | "llama")
            && model.tensors.values().all(|t| matches!(t.ty, 0 | 30)),
        "circuit requires a dense Qwen2 or Llama F32 or BF16 GGUF"
    );
    let Some([width, vocab]) = model
        .tensors
        .get("token_embd.weight")
        .map(|t| t.shape.as_slice())
    else {
        anyhow::bail!("missing circuit token embedding dimensions");
    };
    ensure!(
        (1..=16384).contains(width)
            && (1..=262144).contains(vocab)
            && std::iter::once(req.target_id)
                .chain(req.contrast_id)
                .all(|id| (id as u64) < *vocab),
        "circuit hidden width or token IDs are outside the supported vocabulary"
    );
    identity.check()?;
    let private = tempfile::Builder::new()
        .prefix(".circuit-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(cwd)?;
    let pin = Pin::load()?;
    let (tool, libs) = pin.stage(Backend::Cpu, private.path())?;
    let runtime = Runtime {
        tool,
        libs,
        model: &input.path,
        identity,
        work: private.path(),
        threads: child_threads(None)?.min(16),
        started: Instant::now(),
        width: *width as usize,
        targets: (req.target_id, req.contrast_id),
    };
    let clean_tokens = runtime.tokens(&req.clean_prompt)?;
    let corrupt_tokens = runtime.tokens(&req.corrupt_prompt)?;
    req.check_tokens(&clean_tokens, &corrupt_tokens)?;
    ensure!(
        clean_tokens
            .iter()
            .chain(&corrupt_tokens)
            .all(|&id| (id as u64) < *vocab),
        "circuit input token ID outside vocabulary"
    );
    let activation = Capture::circuit(&req.layers, &req.roles, &req.positions, 0);
    let logits = Capture::circuit(&[], &[], &[clean_tokens.len() - 1], *vocab as usize);
    let mut clean = runtime
        .trace(&req.clean_prompt, &clean_tokens, &activation, None)?
        .activations;
    let corrupt = runtime
        .trace(&req.corrupt_prompt, &corrupt_tokens, &activation, None)?
        .activations;
    let clean_metric = runtime
        .trace(&req.clean_prompt, &clean_tokens, &logits, None)?
        .metric
        .context("missing clean circuit metric")?;
    let corrupt_metric = runtime
        .trace(&req.corrupt_prompt, &corrupt_tokens, &logits, None)?
        .metric
        .context("missing corrupt circuit metric")?;
    let mut phase = progress.start(
        "circuit_attribution",
        "items",
        Some(req.keys().count() as u64),
    );
    let mut sites = Vec::new();
    for key in req.keys() {
        let target = runtime.gradient(&corrupt_tokens, key, req.target_id)?;
        let contrast = req
            .contrast_id
            .map(|id| runtime.gradient(&corrupt_tokens, key, id))
            .transpose()?;
        let a = clean.get(&key).context("missing clean activation")?;
        let b = corrupt.get(&key).context("missing corrupt activation")?;
        let attribution = a
            .iter()
            .zip(b)
            .zip(target)
            .enumerate()
            .map(|(i, ((a, b), g))| {
                (f64::from(*a) - f64::from(*b))
                    * (f64::from(g) - contrast.as_ref().map_or(0., |v| f64::from(v[i])))
            })
            .sum();
        sites.push(Site {
            key,
            attribution,
            patched_delta: None,
        });
        phase.advance(1);
    }
    phase.finish();
    sites.sort_by(Site::rank);
    let patches = req.patch_ranks(sites.len());
    let mut phase = progress.start("circuit_patches", "items", Some(patches.len() as u64));
    for index in patches {
        let site = &mut sites[index];
        let vector = clean
            .remove(&site.key)
            .context("missing clean replacement activation")?;
        let patched = runtime
            .trace(
                &req.corrupt_prompt,
                &corrupt_tokens,
                &logits,
                Some((site.key, vector)),
            )?
            .metric
            .context("missing patched circuit metric")?;
        site.patched_delta = Some(patched - corrupt_metric);
        phase.advance(1);
    }
    phase.finish();
    let (pearson_r, spearman_rho) = circuit::correlations(&sites);
    let artifact = Artifact {
        schema_version: 1,
        kind: circuit::JOB.into(),
        qualification: circuit::QUALIFICATION.into(),
        model_id: model_id.into(),
        model_sha256: identity.sha256.clone(),
        backend: Backend::Cpu,
        clean_prompt: req.clean_prompt.clone(),
        corrupt_prompt: req.corrupt_prompt.clone(),
        clean_tokens,
        corrupt_tokens,
        target_id: req.target_id,
        contrast_id: req.contrast_id,
        clean_metric,
        corrupt_metric,
        top_k: req.top_k,
        sample_lower: req.sample_lower,
        pearson_r,
        spearman_rho,
        sites,
        executable_sha256: runtime.tool.sha,
        manifest_sha256: pin.sha,
    };
    artifact.check(&req, model_id)?;
    Ok(serde_json::to_value(artifact)?)
}
