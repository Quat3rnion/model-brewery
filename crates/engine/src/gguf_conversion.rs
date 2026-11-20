mod inventory;

use crate::{
    bounded, child_budget, child_threads, converter, integrity, progress::Progress, write_json, Job,
};
use anyhow::{bail, ensure, Context, Result};
use brewery_worker::ROCMFPX_REVISION;
use inventory::{Model, Namespace, Types};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    env, fs,
    io::Seek,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const PRESETS: &[(&str, u64)] = &[
    ("F16", 1),
    ("Q4_0", 2),
    ("Q4_1", 3),
    ("Q8_0", 7),
    ("Q5_0", 8),
    ("Q5_1", 9),
    ("Q2_K", 10),
    ("Q3_K_S", 11),
    ("Q3_K_M", 12),
    ("Q3_K_L", 13),
    ("Q4_K_S", 14),
    ("Q4_K_M", 15),
    ("Q5_K_S", 16),
    ("Q5_K_M", 17),
    ("Q6_K", 18),
    ("IQ2_XXS", 19),
    ("IQ2_XS", 20),
    ("Q2_K_S", 21),
    ("IQ3_XS", 22),
    ("IQ3_XXS", 23),
    ("IQ1_S", 24),
    ("IQ4_NL", 25),
    ("IQ3_S", 26),
    ("IQ3_M", 27),
    ("IQ2_S", 28),
    ("IQ2_M", 29),
    ("IQ4_XS", 30),
    ("IQ1_M", 31),
    ("BF16", 32),
    ("Q1_0", 40),
    ("Q2_0", 41),
    ("MXFP4_MOE", 38),
    ("TQ1_0", 36),
    ("TQ2_0", 37),
    ("Q3_K", 12),
    ("Q4_K", 15),
    ("Q5_K", 17),
];
const IMATRIX_REQUIRED: &[&str] = &[
    "IQ1_M", "IQ1_S", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_XS", "IQ3_XXS", "Q2_K_S",
];
// llama_ftype (general.file_type), NOT the GGML tensor type number.
const ROCMFPX_PRESETS: &[(&str, u64)] = &[
    ("Q4_0_ROCMFP4", 100),
    ("Q4_0_ROCMFP4_FAST", 103),
    ("Q6_0_ROCMFPX", 110),
    ("Q8_0_ROCMFPX", 111),
    ("Q3_0_ROCMFPX", 112),
];
const LOG_LIMIT: u64 = 8 << 20;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    output_format: String,
    gguf_type: String,
    #[serde(default)]
    tensor_types: BTreeMap<String, String>,
    imatrix: Option<usize>,
    threads: Option<u64>,
    encoder_family: Option<String>,
    encoder_revision: Option<String>,
}

struct Tool {
    path: PathBuf,
    sha: String,
    stamp: integrity::Stamp,
}

impl Tool {
    fn open(var: &str) -> Result<Self> {
        let path = env::var_os(var)
            .map(PathBuf::from)
            .with_context(|| format!("{var} is required"))?;
        ensure!(path.is_absolute(), "{var} must be absolute");
        let metadata = integrity::canonical(&path)?;
        ensure!(
            metadata.is_file() && metadata.permissions().mode() & 0o111 != 0,
            "{var} must name a regular executable"
        );
        let (member, stamp) = integrity::file_digest(&path, "binary".into())?;
        Ok(Self {
            path,
            sha: member.sha256,
            stamp,
        })
    }

    fn rocmfpx(work: &Path) -> Result<Self> {
        const VAR: &str = "BREWERY_GGUF_ROCMFPX_QUANTIZER";
        const SHA: &str = "BREWERY_GGUF_ROCMFPX_SHA256";
        let expected = env::var(SHA).with_context(|| format!("{SHA} is required"))?;
        ensure!(
            expected.len() == 64
                && expected
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "{SHA} must be lowercase hexadecimal SHA-256"
        );
        let configured = Tool::open(VAR)?;
        ensure!(
            configured.sha == expected,
            "ROCmFPX quantizer SHA-256 mismatch"
        );
        // Run private verified bytes, not the mutable configured pathname.
        // The operator must supply a self-contained CPU quantizer (no adjacent libraries).
        let path = work.join("rocmfpx-quantize");
        fs::copy(&configured.path, &path)?;
        configured.stamp.check(&configured.path)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o500))?;
        let (member, stamp) = integrity::file_digest(&path, "binary".into())?;
        ensure!(
            member.sha256 == expected,
            "staged ROCmFPX quantizer SHA-256 mismatch"
        );
        Ok(Self {
            path,
            sha: member.sha256,
            stamp,
        })
    }

    fn command(&self, work: &Path, threads: usize) -> Command {
        let mut cmd = Command::new(&self.path);
        cmd.current_dir(work)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", work)
            .env("TMPDIR", work)
            .env("LC_ALL", "C")
            .env("CUDA_VISIBLE_DEVICES", "")
            .env("HIP_VISIBLE_DEVICES", "")
            .env("ROCR_VISIBLE_DEVICES", "")
            .env("GGML_VK_VISIBLE_DEVICES", "")
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1");
        child_budget(&mut cmd, threads);
        cmd
    }

    fn capture(&self, cmd: &mut Command, work: &Path) -> Result<(Vec<u8>, Vec<u8>)> {
        let mut out = tempfile::tempfile_in(work)?;
        let mut err = tempfile::tempfile_in(work)?;
        self.stamp.check(&self.path)?;
        let mut child = cmd
            .stdin(Stdio::null())
            .stdout(out.try_clone()?)
            .stderr(err.try_clone()?)
            .spawn()?;
        let started = Instant::now();
        loop {
            let sampled = out
                .metadata()
                .and_then(|o| err.metadata().map(|e| o.len().saturating_add(e.len())));
            if sampled.is_err()
                || sampled? > LOG_LIMIT
                || started.elapsed() >= Duration::from_secs(12 * 3600)
            {
                let _ = child.kill();
                let _ = child.wait();
                bail!("offline GGUF tool exceeded log or time limit");
            }
            match child.try_wait()? {
                Some(exit) => {
                    ensure!(exit.success(), "offline GGUF tool failed ({exit})");
                    break;
                }
                None => thread::sleep(Duration::from_millis(40)),
            }
        }
        self.stamp.check(&self.path)?;
        out.rewind()?;
        err.rewind()?;
        let stdout = bounded(out, LOG_LIMIT)?;
        let stderr = bounded(err, LOG_LIMIT - stdout.len() as u64)?;
        Ok((stdout, stderr))
    }
}

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let options: Options =
        serde_json::from_value(job.options.clone()).context("invalid GGUF options")?;
    let namespace = match (
        options.encoder_family.as_deref(),
        options.encoder_revision.as_deref(),
    ) {
        (None, None) => Namespace::Stock,
        (Some("rocmfpx"), Some(ROCMFPX_REVISION)) => Namespace::RocmFpx,
        _ => bail!("unsupported GGUF encoder family or revision"),
    };
    let fork = namespace == Namespace::RocmFpx;
    let lookup = |presets: &[(&str, u64)]| {
        presets
            .iter()
            .find(|(name, _)| *name == options.gguf_type)
            .map(|(_, id)| *id)
    };
    let preset = if fork {
        lookup(ROCMFPX_PRESETS).or_else(|| {
            (options.gguf_type != "Q2_0"
                && options
                    .tensor_types
                    .values()
                    .any(|ty| namespace.exact_dtype(ty).is_some_and(|id| id >= 100)))
            .then(|| lookup(PRESETS))
            .flatten()
        })
    } else {
        lookup(PRESETS)
    }
    .context("unsupported GGUF preset")?;
    ensure!(
        options.output_format == "gguf" && job.prompt.is_none(),
        "invalid GGUF job"
    );
    ensure!(
        env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_none(),
        "GGUF output is disabled on source-pinned qualification workers"
    );
    let verify = job.kind == "verify";
    let source = job.inputs.first().context("GGUF source is required")?;
    let standalone = source.kind == "safetensors";
    let assets_index = if verify { 2 } else { 1 };
    let base = assets_index + usize::from(standalone);
    ensure!(
        job.inputs.len() == base + usize::from(options.imatrix.is_some())
            && options.imatrix.is_none_or(|index| index == base),
        "GGUF expects the source, verify output, standalone assets and optional indexed imatrix in order"
    );
    ensure!(
        matches!(source.kind.as_str(), "hf" | "gguf" | "safetensors"),
        "GGUF source must be HF, unquantized GGUF or standalone SafeTensors"
    );
    ensure!(
        source.integrity_basis.is_none(),
        "GGUF output is disabled for pinned qualification inputs"
    );
    ensure!(
        !job.inputs.iter().any(|input| cwd.starts_with(&input.path)),
        "GGUF work directory overlaps an input"
    );
    if verify {
        ensure!(
            job.inputs[1].kind == "gguf" && integrity::canonical(&job.inputs[1].path)?.is_file(),
            "verify output must be a registered GGUF file"
        );
    }
    let mut roles = vec![("source", 0)];
    let assets = if standalone {
        let input = &job.inputs[assets_index];
        ensure!(
            job.inputs.iter().all(|item| item
                .source_host
                .as_deref()
                .is_none_or(|original| original == source.host)),
            "standalone SafeTensors GGUF requires direct inputs on one worker"
        );
        ensure!(
            source
                .path
                .extension()
                .is_some_and(|ext| ext == "safetensors")
                && integrity::canonical(&source.path)?.is_file(),
            "expected a registered standalone .safetensors file"
        );
        ensure!(
            input.kind == "assets" && integrity::canonical(&input.path)?.is_dir(),
            "standalone SafeTensors requires a registered assets directory"
        );
        roles.push(("assets", assets_index));
        Some(input.path.as_path())
    } else {
        None
    };
    let imatrix = if let Some(index) = options.imatrix {
        let input = &job.inputs[index];
        let meta = integrity::canonical(&input.path)?;
        ensure!(
            input.kind == "imatrix" && meta.is_file() && (1..=256 << 20).contains(&meta.len()),
            "expected a bounded registered imatrix file"
        );
        roles.push(("imatrix", index));
        Some((index, input.path.as_path()))
    } else {
        None
    };
    ensure!(
        options.tensor_types.len() <= 256,
        "too many GGUF tensor overrides"
    );
    for (name, ty) in &options.tensor_types {
        ensure!(
            inventory::valid_name(name) && namespace.exact_dtype(ty).is_some(),
            "invalid or unsupported GGUF tensor override"
        );
    }
    ensure!(
        imatrix.is_some()
            || !std::iter::once(&options.gguf_type)
                .chain(options.tensor_types.values())
                .any(|ty| IMATRIX_REQUIRED.contains(&ty.as_str())),
        "this GGUF selection requires a registered imatrix"
    );
    ensure!(
        !matches!(preset, 0 | 1 | 32) || options.tensor_types.is_empty(),
        "floating-point presets do not support tensor overrides"
    );
    let scratch = tempfile::Builder::new().prefix(".gguf-").tempdir_in(cwd)?;
    let work = scratch.path();
    let quantizer = if fork {
        Tool::rocmfpx(work)?
    } else {
        Tool::open("BREWERY_GGUF_QUANTIZER")?
    };
    let threads = child_threads(options.threads)?;
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    ensure!(
        job.inputs
            .iter()
            .zip(&ids)
            .all(|(input, id)| input.kind != "gguf" || id.files.len() == 1),
        "split GGUF is not supported"
    );
    let source_sha = integrity::source_hash(&job.inputs, &ids, &roles);
    let _phase = progress.start(
        match job.kind.as_str() {
            "inspect" => "inspect",
            "plan" => "plan",
            "convert" => "convert",
            _ => "verify",
        },
        "items",
        None,
    );
    let mut exporter_sha = None;
    let mut hf = Value::Null;
    let mut fp_path = source.path.clone();
    let needs_exporter = source.kind == "hf" || standalone;
    if needs_exporter {
        ensure!(
            standalone || integrity::canonical(&source.path)?.is_dir(),
            "HF source must be an indexed snapshot directory"
        );
        let exporter = Tool::open("BREWERY_GGUF_HF_EXPORTER")?;
        let inspect = export(&exporter, work, threads, &source.path, assets, None)?;
        let supported = inspect["export_supported"]
            .as_bool()
            .context("invalid HF GGUF inspection receipt")?;
        hf = inspect;
        exporter_sha = Some(exporter.sha.clone());
        if !supported && job.kind != "inspect" {
            bail!("HF architecture cannot be exported to a single supported GGUF");
        }
        if supported && (verify || job.kind == "convert") {
            let estimate = ids[0]
                .bytes
                .checked_mul(4)
                .context("HF disk estimate overflow")?;
            ensure!(
                converter::disk_preflight(cwd, estimate)?["fits"] == true,
                "insufficient disk for transient HF export and GGUF output"
            );
            let export_dir = work.join("hf-export");
            ids.iter().try_for_each(integrity::Identity::check)?;
            export(
                &exporter,
                work,
                threads,
                &source.path,
                assets,
                Some(&export_dir),
            )?;
            fp_path = export_dir.join("model.gguf");
            ensure!(
                integrity::canonical(&fp_path)?.is_file()
                    && fs::read_dir(&export_dir)?.count() == 1,
                "HF exporter must emit exactly one regular model.gguf"
            );
        }
    }
    let estimate_only = needs_exporter && matches!(job.kind.as_str(), "inspect" | "plan");
    let temp_output = work.join("model.gguf");
    let mut report = json!({"kind":job.kind,"source_format":source.kind,"gguf_type":options.gguf_type,
        "requested_tensor_types":options.tensor_types,"hf_export":hf,"tensor_payloads_read":false});
    let mut file_verification = json!({"status":"not_performed"});
    let artifact;
    let mut disk;
    if estimate_only {
        let supported = hf["export_supported"] == true;
        if supported {
            let tensors = inventory::hf_inventory(&hf["expected_gguf_tensors"])?;
            for name in options.tensor_types.keys() {
                ensure!(
                    tensors.contains_key(name),
                    "unknown exact GGUF tensor override {name}"
                );
            }
            report["gguf_tensors"] = json!(tensors
                .iter()
                .map(|(name, shape)| json!({"name":name,"shape":shape,"current_type":null}))
                .collect::<Vec<_>>());
        }
        report["planning_scope"] = json!("GGUF tensor names and dimensions only; type layout confirmed after transient FP export");
        disk = converter::disk_preflight(
            cwd,
            ids[0]
                .bytes
                .checked_mul(4)
                .context("HF disk estimate overflow")?,
        )?;
        if !supported {
            disk["fits"] = json!(false);
        }
        artifact = report_artifact(cwd, progress, &report, &job.kind)?;
    } else {
        let fp = Model::open(&fp_path, Namespace::Stock)?;
        fp.check_stock_source()?;
        ensure!(
            fp.tensors
                .values()
                .all(|tensor| matches!(tensor.ty, 0 | 1 | 24..=27 | 30)),
            "source GGUF must contain only F32/F16/BF16 weights and integral tensors"
        );
        if needs_exporter {
            let names = inventory::hf_inventory(&hf["expected_gguf_tensors"])?;
            ensure!(
                fp.tensors
                    .iter()
                    .all(|(name, t)| names.get(name) == Some(&t.shape))
                    && names.len() == fp.tensors.len(),
                "HF-export inventory differs from metadata inspection"
            );
        }
        for (name, ty) in &options.tensor_types {
            let tensor = fp
                .tensors
                .get(name)
                .with_context(|| format!("unknown exact GGUF tensor override {name}"))?;
            tensor.bytes(namespace.exact_dtype(ty).unwrap(), namespace)?;
        }
        let prepared = if let Some((index, path)) = imatrix {
            // Do not embed a registered source path into GGUF imatrix metadata.
            let dest = work.join("imatrix.data");
            fs::copy(path, &dest)?;
            let (copy, _) = integrity::file_digest(&dest, "imatrix".into())?;
            ensure!(
                copy.sha256 == ids[index].sha256 && copy.bytes == ids[index].bytes,
                "staged imatrix does not match the verified input"
            );
            ids[index].check()?;
            Some(Path::new("imatrix.data"))
        } else {
            None
        };
        let quant = |dry: bool| -> Result<Types> {
            let mut cmd = quantizer.command(work, threads);
            if dry {
                cmd.arg("--dry-run");
            }
            if let Some(path) = prepared {
                cmd.arg("--imatrix").arg(path);
            }
            for (name, ty) in &options.tensor_types {
                let ty = if fork {
                    ty.to_ascii_lowercase()
                } else {
                    ty.clone()
                };
                cmd.arg("--tensor-type")
                    .arg(format!("^{}$={ty}", escape_exact(name)));
            }
            cmd.arg(&fp_path)
                .arg(&temp_output)
                .arg(&options.gguf_type)
                .arg(threads.to_string());
            let (stdout, stderr) = quantizer.capture(&mut cmd, work)?;
            let log = String::from_utf8([stdout, stderr].concat())
                .context("invalid GGUF tool log encoding")?;
            let warnings = log.to_ascii_lowercase();
            ensure!(
                ![
                    "falling back to",
                    "required fallback quantization",
                    "will require an imatrix",
                    "missing importance matrix"
                ]
                .iter()
                .any(|s| warnings.contains(s)),
                "GGUF quantizer reported a fallback or missing importance matrix"
            );
            if dry {
                fp.parse_plan(&log, &options.tensor_types, namespace)
            } else {
                Ok(BTreeMap::new())
            }
        };
        let expected = quant(true)?;
        if fork {
            ensure!(
                expected.values().any(|ty| (100..=104).contains(ty)),
                "ROCmFPX dry run selected no ROCmFPX tensor types"
            );
        }
        ensure!(
            !temp_output.exists(),
            "GGUF dry run unexpectedly produced an output"
        );
        let output_bytes = fp.predicted_bytes(&expected, namespace)?;
        disk = converter::disk_preflight(cwd, output_bytes)?;
        report["architecture"] = json!(fp.architecture);
        report["gguf_tensors"] = fp.summary(&expected, namespace);
        report["output_gguf_bytes_estimate"] = json!(output_bytes);
        if job.kind == "convert" || verify {
            if !verify {
                ensure!(disk["fits"] == true, "insufficient disk for GGUF output");
                quant(false)?;
            }
            let path = if verify {
                &job.inputs[1].path
            } else {
                &temp_output
            };
            // Pin freshly emitted bytes before parsing their structure. Verify
            // uses the already pinned registered output (input index 1).
            let output = if verify {
                None
            } else {
                ensure!(
                    fs::symlink_metadata(path)?.is_file(),
                    "quantizer did not emit a regular GGUF"
                );
                Some(integrity::identify_output(path, progress)?)
            };
            let actual = Model::open(path, namespace)?;
            fp.verify(&actual, &expected, preset)?;
            report["gguf_tensors"] = actual.summary(&BTreeMap::new(), namespace);
            report["passed"] = json!(true);
            report["tensor_payloads_read"] = json!(true);
            file_verification = json!({"status":"passed","scope":"physical SHA-256, GGUF header, tensor inventory, shapes, payload bounds and exact planned types; no numerical or runtime check"});
            ids.iter().try_for_each(integrity::Identity::check)?;
            if let Some(output) = output {
                let destination = cwd.join("converted.gguf");
                output.check()?;
                let mut publish = progress.start("publish_output", "items", Some(1));
                converter::publish_directory(&temp_output, &destination)?;
                publish.advance(1);
                publish.finish();
                artifact = output.artifact(&destination, "gguf");
            } else {
                artifact = report_artifact(cwd, progress, &report, &job.kind)?;
            }
        } else {
            artifact = report_artifact(cwd, progress, &report, &job.kind)?;
        }
    }
    disk["scope"] = json!("GGUF payload estimate plus 5% and 64 MiB; not a disk reservation");
    ids.iter().try_for_each(integrity::Identity::check)?;
    let mut backend_receipt = json!({"tool":"llama-quantize","executable_sha256":quantizer.sha,
        "hf_exporter_sha256":exporter_sha,"threads":threads,"requested_options":job.options});
    if fork {
        backend_receipt["encoder_family"] = json!("rocmfpx");
        backend_receipt["encoder_revision"] = json!(ROCMFPX_REVISION);
        backend_receipt["stock_runtime_compatible"] = json!(false);
    }
    Ok(
        json!({"kind":job.kind,"host":source.host,"output_format":"gguf","gguf_type":options.gguf_type,
        "source_sha256":source_sha,"source_digest_scope":"role-qualified physical source bundles",
        "inputs":ids.iter().zip(&job.inputs).map(|(id,input)| id.input(input)).collect::<Vec<_>>(),
        "artifact":artifact,"report":report,"disk_preflight":disk,"file_verification":file_verification,
        "backend_receipt":backend_receipt,
        "release_qualified":false}),
    )
}

fn export(
    tool: &Tool,
    work: &Path,
    threads: usize,
    source: &Path,
    assets: Option<&Path>,
    output: Option<&Path>,
) -> Result<Value> {
    let mut cmd = tool.command(work, threads);
    for name in ["BREWERY_GGUF_PYTHON", "BREWERY_LLAMA_CPP"] {
        if let Some(value) = env::var_os(name) {
            ensure!(Path::new(&value).is_absolute(), "{name} must be absolute");
            cmd.env(name, value);
        }
    }
    cmd.arg(if output.is_some() {
        "export"
    } else {
        "inspect"
    })
    .arg("--source")
    .arg(source)
    .args(["--outtype", "bf16"]);
    if let Some(assets) = assets {
        cmd.arg("--assets").arg(assets);
    }
    if let Some(output) = output {
        cmd.arg("--output").arg(output);
    }
    let (stdout, _) = tool.capture(&mut cmd, work)?;
    let receipt: Value = serde_json::from_slice(&stdout).context("invalid HF exporter receipt")?;
    ensure!(
        assets.is_none() || receipt["source_format"] == "safetensors",
        "HF exporter did not attest standalone SafeTensors"
    );
    Ok(receipt)
}

fn report_artifact(cwd: &Path, progress: &Progress, report: &Value, kind: &str) -> Result<Value> {
    let path = cwd.join(format!("gguf-{kind}.json"));
    write_json(&path, report)?;
    Ok(integrity::identify_output(&path, progress)?.artifact(&path, "report"))
}

fn escape_exact(name: &str) -> String {
    let mut out = String::new();
    for b in name.bytes() {
        // llama-quantize lowercases the pattern, so hex-escape uppercase.
        if b.is_ascii_uppercase() {
            out.push_str(&format!("\\x{b:02x}"));
        } else if b == b'.' {
            out.push_str("\\.");
        } else {
            out.push(b as char);
        }
    }
    out
}
