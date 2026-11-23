mod corpus;
mod gguf;

use crate::{
    Job, bounded, child_threads,
    converter::disk_preflight,
    gguf_conversion::Tool,
    gguf_conversion::inventory::{Model, Tensor, align},
    integrity,
    progress::Progress,
    sensitivity::{Pin, digest, matches_device},
    write_json,
};
use anyhow::{Context, Result, ensure};
use brewery_worker::{
    damage::{self as wire, MAX_GRADIENT_BYTES, Objective},
    locate_edit::{Backend, GPU_GATE},
    sensitivity, trace,
    weight_source::open_regular,
};
use gguf::{CopyTensor, EMBEDDING, HEAD, STOCK};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const PROMPTS: &str = include_str!("../../eval/data/edit-checks/heldout.txt");
const TOOL_TIMEOUT: Duration = Duration::from_secs(150);
const JOB_TIMEOUT: Duration = Duration::from_secs(3600);
const TOP_K: usize = 100;

struct Runtime {
    tool: Tool,
    libs: PathBuf,
    threads: usize,
    backend: Backend,
    started: Instant,
}

impl Runtime {
    fn run(&self, dir: &Path, method: &str, model: &Path, args: &[&str]) -> Result<Vec<u8>> {
        let mut command = self.tool.command(dir, self.threads);
        command
            .env("LD_LIBRARY_PATH", &self.libs)
            .arg(method)
            .arg("--model")
            .arg(model)
            .args(args);
        if matches!(method, "grad-select" | "grad-corpus") {
            command.args([
                "--backend",
                self.backend.key(),
                "--threads",
                &self.threads.to_string(),
            ]);
            for &key in self.backend.visibility_vars() {
                if let Some(v) = env::var_os(key) {
                    command.env(key, v);
                }
            }
            if self.backend == Backend::Cuda {
                command
                    .env("NVIDIA_TF32_OVERRIDE", "0")
                    .env("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32");
            } else if self.backend == Backend::Rocm {
                command.env_remove(brewery_worker::locate_edit::CUDA_DEVICES);
            }
        }
        let timeout = JOB_TIMEOUT
            .checked_sub(self.started.elapsed())
            .context("damage map exceeded time budget")?;
        let timeout = if method == "grad-corpus" {
            timeout
        } else {
            TOOL_TIMEOUT.min(timeout)
        };
        Ok(self
            .tool
            .capture_for(&mut command, dir, timeout)
            .with_context(|| format!("Lanternfish {method} failed"))?
            .0)
    }

    fn tokenize(&self, dir: &Path, model: &Path, prompt: &str) -> Result<Vec<i32>> {
        let value: Value =
            serde_json::from_slice(&self.run(dir, "tokenize", model, &["--text", prompt])?)?;
        let tokens: Vec<i32> = serde_json::from_value(value["tokens"].clone())?;
        ensure!(
            (1..=128).contains(&tokens.len()) && tokens.iter().all(|t| *t >= 0),
            "invalid trace input tokens"
        );
        Ok(tokens)
    }

    fn trace(&self, dir: &Path, model: &Path, prompt: &str, tokens: &[i32]) -> Result<Value> {
        let spec = json!({"layers":[],"roles":[],"positions":[tokens.len()-1],"top_k":TOP_K,"byte_cap":65536});
        let path = dir.join("spec.json");
        write_json(&path, &spec)?;
        let out = dir.join("trace");
        self.run(
            dir,
            "trace",
            model,
            &[
                "--prompt",
                prompt,
                "--spec",
                path.to_str().context("non-UTF8 work path")?,
                "--out",
                out.to_str().context("non-UTF8 work path")?,
                "--threads",
                &self.threads.to_string(),
            ],
        )?;
        let value = trace::verified_diff_input(&out, &serde_json::from_value(spec)?)?;
        ensure!(
            value["tokens"]
                .as_array()
                .is_some_and(|seen| seen.len() == tokens.len()
                    && seen
                        .iter()
                        .zip(tokens)
                        .all(|(seen, token)| seen["token_id"] == *token)),
            "quantization damage traces must tokenize identically"
        );
        Ok(value)
    }
}

#[derive(Serialize)]
struct Observation {
    prompt: &'static str,
    tokens: Vec<i32>,
    target_id: i32,
    reference_logit: f64,
    quantized_logit: f64,
    measured_drift: f64,
    predicted_recovery_sum: f64,
    w5: Value,
}

#[derive(Serialize)]
struct Score<'a> {
    name: &'a str,
    layer: Option<u32>,
    shape: &'a [u64],
    reference_type: &'static str,
    quantized_type: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    predicted_recovery: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    predicted_nll_increase: Option<f64>,
    mean_absolute_damage: f64,
    measured_recovery: Option<Vec<f64>>,
}

fn logit(trace: &Value, target: i32) -> Result<f64> {
    trace["model_logits"][0]["top"]
        .as_array()
        .and_then(|top| top.iter().find(|row| row["token_id"] == target))
        .and_then(|row| row["logit"].as_f64())
        .with_context(|| format!("target {target} is missing from the compared top-{TOP_K} trace"))
}

fn mean_abs(values: &[f64]) -> f64 {
    values.iter().map(|x| x.abs()).sum::<f64>() / values.len() as f64
}

fn dot(
    out: &Path,
    backend: Backend,
    spec: &Value,
    shape: &[u64],
    obs: &Observation,
    files: &mut [File; 2],
    reference: &Model,
    quantized: &Model,
    a: &Tensor,
    b: &Tensor,
) -> Result<f64> {
    let meta: Value = serde_json::from_slice(&bounded(
        open_regular(&out.join("gradient.json"))?,
        16 << 10,
    )?)?;
    let sha = meta["sha256"]
        .as_str()
        .filter(|s| brewery_worker::is_digest(s))
        .context("missing gradient payload digest")?;
    ensure!(
        meta["schema_version"] == 1
            && meta["dtype"] == "f32_le"
            && ["site", "tokens", "target_id", "objective"]
                .iter()
                .all(|k| meta[k] == spec[k])
            && meta["shape"] == json!(shape)
            && meta["backend"] == backend.key()
            && meta["placement"]["strict"] == true
            && meta["placement"]["device"]
                .as_str()
                .is_some_and(|d| matches_device(backend, d))
            && meta["logit"]
                .as_f64()
                .is_some_and(|x| (x - obs.reference_logit).abs() <= 0.02),
        "invalid target logit gradient metadata, placement, or F32 reference logit (trace={}, grad={})",
        obs.reference_logit,
        meta["logit"]
    );
    dot_payload(
        &out.join("gradient.f32"),
        sha,
        files,
        reference,
        quantized,
        a,
        b,
    )
}

fn dot_payload(
    path: &Path,
    sha: &str,
    files: &mut [File; 2],
    reference: &Model,
    quantized: &Model,
    a: &Tensor,
    b: &Tensor,
) -> Result<f64> {
    let bytes = a.bytes(0, STOCK)?;
    ensure!(
        bytes <= MAX_GRADIENT_BYTES,
        "gradient exceeds damage map budget"
    );
    let file = open_regular(path)?;
    ensure!(
        file.metadata()?.len() == bytes,
        "gradient payload size mismatch"
    );
    let mut reader = BufReader::with_capacity(256 << 10, file);
    let mut hash = Sha256::new();
    let mut buf = [0u8; 1024];
    let (mut sum, mut count) = (0., 0);
    let [a_file, b_file] = files;
    gguf::delta(
        gguf::region(a_file, reference, a)?,
        gguf::region(b_file, quantized, b)?,
        a.ty,
        b.ty,
        bytes / 4,
        |delta| {
            let payload = &mut buf[..delta.len() * 4];
            reader.read_exact(payload)?;
            hash.update(&*payload);
            for (weight, word) in delta.iter().zip(payload.chunks_exact(4)) {
                let gradient = gguf::f32_at(word);
                ensure!(
                    weight.is_finite() && gradient.is_finite(),
                    "non-finite weight delta or gradient"
                );
                sum += f64::from(*weight) * f64::from(gradient);
            }
            count += payload.len() as u64;
            Ok(())
        },
    )?;
    ensure!(
        count == bytes
            && reader.read(&mut buf[..1])? == 0
            && format!("{:x}", hash.finalize()) == sha,
        "gradient payload size or SHA-256 mismatch"
    );
    Ok(sum)
}

fn proposal(
    scores: &[Score<'_>],
    reference: &Model,
    quant: &Model,
    req: &wire::Request,
) -> Result<Value> {
    // Exchange exact stored types among equally sized matrices, keeping the
    // aligned tensor byte sum fixed for the measured quantized inventory.
    let mut types = BTreeMap::new();
    let mut pairs = Vec::new();
    let bytes = |t: &Tensor, ty| align(t.bytes(ty, STOCK)?, quant.align);
    for top in scores {
        if pairs.len() >= req.swaps || top.layer.is_none() || top.mean_absolute_damage == 0. {
            continue;
        }
        let a = &quant.tensors[top.name];
        if types.contains_key(top.name) || a.shape.len() != 2 {
            continue;
        }
        for bottom in scores.iter().rev() {
            if bottom.layer.is_none()
                || types.contains_key(bottom.name)
                || bottom.mean_absolute_damage * 2. >= top.mean_absolute_damage
            {
                continue;
            }
            let b = &quant.tensors[bottom.name];
            if !matches!(a.ty, 2 | 6 | 8 | 12 | 13 | 14)
                || !matches!(b.ty, 2 | 6 | 8 | 12 | 13 | 14)
                || a.shape != b.shape
                || bytes(a, a.ty)? >= bytes(a, b.ty)?
            {
                continue;
            }
            if bytes(a, a.ty)? + bytes(b, b.ty)? != bytes(a, b.ty)? + bytes(b, a.ty)? {
                continue;
            }
            types.insert(top.name, gguf::name(b.ty));
            types.insert(bottom.name, gguf::name(a.ty));
            pairs.push(json!({"raise":top.name,"raise_layer":top.layer,
                "lower":bottom.name,"lower_layer":bottom.layer,
                "bytes_moved":bytes(a,b.ty)?-bytes(a,a.ty)?}));
            break;
        }
    }
    // State each observed quantized type, not just the changes. Re-running a
    // preset without these options can trigger rejected GGML row-width fallbacks.
    let baseline = quant
        .tensors
        .iter()
        .filter(|(n, t)| {
            matches!(t.ty, 2 | 6 | 8 | 12 | 13 | 14) && reference.tensors.contains_key(*n)
        })
        .map(|(n, t)| (n.as_str(), gguf::name(t.ty)))
        .collect::<BTreeMap<_, _>>();
    ensure!(
        baseline.len() <= 256,
        "damage map recipe exceeds GGUF override limit"
    );
    let mut proposed = baseline.clone();
    proposed.extend(types);
    let profile = json!({"format":"gguf","name":"Quantization damage proposal",
        "preset":req.gguf_type,"overrides":proposed});
    let uniform_profile = json!({"format":"gguf","name":"Quantization damage baseline",
        "preset":req.gguf_type,"overrides":baseline});
    Ok(
        json!({"applied":false,"reference_sha256":req.reference_sha256,"baseline_sha256":req.quantized_sha256,
        "size_basis":"aligned payload bytes on the original reference inventory; verify actual base and output sizes",
        "payload_bytes_delta":0,"layer_pairs":pairs,
        "profile":profile,"uniform_profile":uniform_profile,
        "uniform_options":{"output_format":"gguf","gguf_type":req.gguf_type,"tensor_types":baseline},
        "options":{"output_format":"gguf","gguf_type":req.gguf_type,"tensor_types":proposed}}),
    )
}

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let started = Instant::now();
    let req = wire::Request::parse(&job.options)?;
    let [a, b] = job.inputs.as_slice() else {
        anyhow::bail!("damage map needs exactly two GGUFs");
    };
    let model_id = |i: &crate::Input| {
        i.id.clone()
            .filter(|id| !id.is_empty())
            .context("missing GGUF model ID")
    };
    let model_ids = [model_id(a)?, model_id(b)?];
    ensure!(
        job.prompt.is_none() && a.kind == "gguf" && b.kind == "gguf" && a.host == b.host,
        "damage map needs colocated GGUFs and no prompt"
    );
    ensure!(
        !req.backend.is_gpu() || env::var(GPU_GATE).as_deref() == Ok("1"),
        "damage map GPU not enabled"
    );
    ensure!(
        digest(PROMPTS.as_bytes()) == sensitivity::CORPUS_SHA256,
        "damage map corpus SHA-256 mismatch"
    );
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    for (id, hash) in ids.iter().zip(req.hashes()) {
        ensure!(
            id.files.len() == 1 && id.sha256 == hash && id.bytes <= wire::MAX_MODEL_BYTES,
            "damage map input digest or single-file size mismatch"
        );
    }
    let check_ids = || -> Result<()> {
        for id in &ids {
            id.check()?;
        }
        Ok(())
    };
    let (reference, quantized) = (Model::open(&a.path, STOCK)?, Model::open(&b.path, STOCK)?);
    ensure!(
        reference.architecture == quantized.architecture
            && matches!(reference.architecture.as_str(), "llama" | "qwen2")
            && reference.tensors.values().all(|t| matches!(t.ty, 0 | 30)),
        "damage map requires same-architecture dense BF16/F32 reference"
    );
    let untie = !reference.tensors.contains_key(HEAD) && quantized.tensors.contains_key(HEAD);
    let reference_tensor = |n: &str| {
        reference
            .tensors
            .get(if untie && n == HEAD { EMBEDDING } else { n })
    };
    ensure!(
        reference.tensors.len() + usize::from(untie) == quantized.tensors.len()
            && quantized.tensors.iter().all(|(n, t)| reference_tensor(n)
                .is_some_and(|r| r.shape == t.shape)
                && gguf::supported(t.ty)),
        "damage map tensor inventory, shape, or dtype mismatch"
    );
    let work = tempfile::tempdir_in(cwd)?;
    let work = work.path();
    let mut files = [open_regular(&a.path)?, open_regular(&b.path)?];
    let (mut names, mut largest, mut total) = (Vec::new(), 0, 0u64);
    for (n, t) in &quantized.tensors {
        if req.tensors.is_empty() && matches!(t.ty, 0 | 1 | 30) {
            continue;
        }
        if !req.tensors.is_empty() && !req.tensors.contains(n) {
            continue;
        }
        let size = t.bytes(0, STOCK)?;
        ensure!(
            size <= MAX_GRADIENT_BYTES,
            "gradient exceeds damage map budget for {n}"
        );
        largest = size.max(largest);
        total = total.checked_add(size).context("gradient size overflow")?;
        names.push(n.as_str());
    }
    ensure!(
        (1..=wire::MAX_TENSORS).contains(&names.len())
            && (req.tensors.is_empty() || req.tensors.len() == names.len()),
        "invalid damage map tensor selection"
    );
    let entries = quantized
        .tensors
        .keys()
        .map(|n| {
            let t = reference_tensor(n).expect("checked inventory");
            (
                n.as_str(),
                CopyTensor {
                    tensor: t,
                    model: &reference,
                    file: 0,
                    ty: 0,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let expanded_bytes = entries
        .values()
        .try_fold(reference.data, |n, entry| -> Result<u64> {
            Ok(
                n.checked_add(align(entry.tensor.bytes(0, STOCK)?, reference.align)?)
                    .context("F32 GGUF size overflow")?,
            )
        })?;
    let is_corpus = req.objective == Objective::CorpusNll;
    let gradient_bytes = if is_corpus { total } else { largest };
    ensure!(
        disk_preflight(
            cwd,
            (expanded_bytes + gradient_bytes).max(ids[1].bytes + largest / 2)
                + (128 << 20)
                + req.backend.runtime_stage_reserve()
        )?["fits"]
            == true,
        "insufficient disk for private F32 expansion and damage map gradients"
    );
    let expanded = work.join("reference.f32.gguf");
    let gradient_model = if untie || reference.tensors.values().any(|t| t.ty == 30) {
        gguf::repack(&expanded, &reference, 0, &mut files, &entries)?;
        &expanded
    } else {
        &a.path
    };
    check_ids()?;
    let pin = Pin::load()?;
    let threads = child_threads(None)?.min(16);
    let report = |grad: &Runtime| {
        json!({"kind":wire::JOB,"schema_version":1,"qualification":sensitivity::QUALIFICATION,
            "reference_id":model_ids[0],"quantized_id":model_ids[1],"reference_sha256":ids[0].sha256,"quantized_sha256":ids[1].sha256,
            "backend":req.backend,"corpus_sha256":sensitivity::CORPUS_SHA256,"untied_reference_head":untie,
            "manifest_sha256":pin.sha,"executable_sha256":grad.tool.sha})
    };
    if is_corpus {
        let (tool, libs) = pin.stage(req.backend, work)?;
        let runtime = Runtime {
            tool,
            libs,
            threads,
            backend: req.backend,
            started,
        };
        let mut report = report(&runtime);
        let gradients = corpus::run(&runtime, work, gradient_model, &names)?;
        check_ids()?;
        let mut phase = progress.start("damage_gradients", "items", Some(names.len() as u64));
        let mut scores = Vec::new();
        let mut layers = BTreeMap::<u32, f64>::new();
        for name in names {
            let (a, b) = (reference_tensor(name).unwrap(), &quantized.tensors[name]);
            let loss = gradients.attribute(name, &mut files, &reference, &quantized, a, b)?;
            let layer = sensitivity::layer(name);
            if let Some(layer) = layer {
                *layers.entry(layer).or_default() += loss;
            }
            scores.push(Score {
                name,
                layer,
                shape: &a.shape,
                reference_type: gguf::name(a.ty),
                quantized_type: gguf::name(b.ty),
                mean_absolute_damage: loss.abs(),
                predicted_recovery: Vec::new(),
                predicted_nll_increase: Some(loss),
                measured_recovery: None,
            });
            phase.advance(1);
        }
        phase.finish();
        check_ids()?;
        rank(&mut scores);
        let mut layers = layers.into_iter().collect::<Vec<_>>();
        layers.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then_with(|| a.0.cmp(&b.0)));
        report["objective"] = json!("corpus_nll");
        report["corpus"] = json!({
            "line_count": wire::CORPUS_LINES,
            "observation_count": gradients.observation_count,
            "reference_mean_nll": gradients.loss,
            "predicted_nll_increase": scores.iter().filter_map(|s| s.predicted_nll_increase).sum::<f64>()
        });
        report["tensors"] = json!(scores);
        report["layers"] = json!(layers.into_iter().map(|(layer, loss)|
            json!({"layer":layer, "mean_absolute_damage":loss.abs(), "predicted_nll_increase":loss})
        ).collect::<Vec<_>>());
        report["observations"] = json!([]);
        report["recipe_proposal"] = proposal(&scores, &reference, &quantized, &req)?;
        wire::check_result(&req, &report, [&model_ids[0], &model_ids[1]])?;
        return Ok(report);
    }
    let (tool, libs) = pin.stage(Backend::Cpu, work)?;
    let cpu = Runtime {
        tool,
        libs,
        threads,
        backend: Backend::Cpu,
        started,
    };
    let gpu = if req.backend.is_gpu() {
        let (tool, libs) = pin.stage(req.backend, work)?;
        Some(Runtime {
            tool,
            libs,
            threads,
            backend: req.backend,
            started,
        })
    } else {
        None
    };
    let grad = gpu.as_ref().unwrap_or(&cpu);
    let mut observations = Vec::new();
    for prompt in PROMPTS.lines().take(req.observations) {
        let tokens = cpu.tokenize(work, &a.path, prompt)?;
        let left = cpu.trace(tempfile::tempdir_in(work)?.path(), &a.path, prompt, &tokens)?;
        let right = cpu.trace(tempfile::tempdir_in(work)?.path(), &b.path, prompt, &tokens)?;
        let target_id = left["model_logits"][0]["top"][0]["token_id"]
            .as_i64()
            .context("missing reference top logit")? as i32;
        let reference_logit = logit(&left, target_id)?;
        let quantized_logit = logit(&right, target_id)?;
        observations.push(Observation {
            prompt,
            tokens,
            target_id,
            reference_logit,
            quantized_logit,
            measured_drift: reference_logit - quantized_logit,
            predicted_recovery_sum: 0.,
            w5: serde_json::to_value(trace::diff::compare(left, right)?)?,
        });
        check_ids()?;
    }
    let mut phase = progress.start(
        "damage_gradients",
        "items",
        Some((names.len() * observations.len()) as u64),
    );
    let mut scores = Vec::new();
    for name in names {
        let (a, b) = (reference_tensor(name).unwrap(), &quantized.tensors[name]);
        let mut predictions = Vec::new();
        for obs in &observations {
            let attempt = tempfile::tempdir_in(work)?;
            let (spec, out) = (
                attempt.path().join("spec.json"),
                attempt.path().join("gradient"),
            );
            let value = json!({"tokens":obs.tokens,"target_id":obs.target_id,"objective":"target_logit","site":{"kind":"weight","name":name}});
            write_json(&spec, &value)?;
            grad.run(
                attempt.path(),
                "grad-select",
                gradient_model,
                &[
                    "--spec",
                    spec.to_str().context("non-UTF8 work path")?,
                    "--out",
                    out.to_str().context("non-UTF8 work path")?,
                ],
            )?;
            predictions.push(
                dot(
                    &out,
                    req.backend,
                    &value,
                    &a.shape,
                    obs,
                    &mut files,
                    &reference,
                    &quantized,
                    a,
                    b,
                )
                .with_context(|| format!("attributing {name}"))?,
            );
            check_ids()?;
            phase.advance(1);
        }
        scores.push(Score {
            name,
            layer: sensitivity::layer(name),
            shape: &a.shape,
            reference_type: gguf::name(a.ty),
            quantized_type: gguf::name(b.ty),
            mean_absolute_damage: mean_abs(&predictions),
            predicted_recovery: predictions,
            predicted_nll_increase: None,
            measured_recovery: None,
        });
    }
    phase.finish();
    if expanded.exists() {
        fs::remove_file(&expanded)?;
    }
    rank(&mut scores);
    let mut phase = progress.start(
        "damage_swaps",
        "items",
        Some(scores.len().min(req.swaps) as u64),
    );
    for score in scores.iter_mut().take(req.swaps) {
        let attempt = tempfile::tempdir_in(work)?;
        let path = attempt.path().join("swap.gguf");
        let swapped = quantized
            .tensors
            .iter()
            .map(|(name, tensor)| {
                let entry = if name == score.name {
                    let tensor = reference_tensor(name).unwrap();
                    CopyTensor {
                        tensor,
                        model: &reference,
                        file: 0,
                        ty: tensor.ty,
                    }
                } else {
                    CopyTensor {
                        tensor,
                        model: &quantized,
                        file: 1,
                        ty: tensor.ty,
                    }
                };
                (name.as_str(), entry)
            })
            .collect();
        gguf::repack(&path, &quantized, 1, &mut files, &swapped)?;
        let out = observations
            .iter()
            .map(|obs| {
                let value = cpu.trace(
                    tempfile::tempdir_in(work)?.path(),
                    &path,
                    obs.prompt,
                    &obs.tokens,
                )?;
                Ok(logit(&value, obs.target_id)? - obs.quantized_logit)
            })
            .collect::<Result<Vec<_>>>()?;
        score.measured_recovery = Some(out);
        check_ids()?;
        phase.advance(1);
    }
    phase.finish();
    let mut totals = BTreeMap::new();
    for score in &scores {
        for (obs, &prediction) in observations.iter_mut().zip(&score.predicted_recovery) {
            obs.predicted_recovery_sum += prediction;
        }
        if let Some(layer) = score.layer {
            for (t, s) in totals
                .entry(layer)
                .or_insert_with(|| vec![0.; observations.len()])
                .iter_mut()
                .zip(&score.predicted_recovery)
            {
                *t += s;
            }
        }
    }
    let mut layers = totals
        .into_iter()
        .map(|(l, v)| (l, mean_abs(&v), v))
        .collect::<Vec<_>>();
    layers.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let layers = layers.into_iter().map(|(layer,score,signed)| json!({"layer":layer,"mean_absolute_damage":score,"predicted_recovery":signed})).collect::<Vec<_>>();
    let mut report = report(grad);
    report["objective"] = json!("target_logit");
    report["trace_backend"] = json!("cpu");
    report["observations"] = json!(observations);
    report["tensors"] = json!(scores);
    report["layers"] = json!(layers);
    report["recipe_proposal"] = proposal(&scores, &reference, &quantized, &req)?;
    wire::check_result(&req, &report, [&model_ids[0], &model_ids[1]])?;
    Ok(report)
}

fn rank(scores: &mut [Score<'_>]) {
    scores.sort_by(|a, b| {
        b.mean_absolute_damage
            .total_cmp(&a.mean_absolute_damage)
            .then_with(|| a.name.cmp(b.name))
    });
}
