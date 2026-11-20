use super::{
    Model, Namespace, Tool, child_threads, converter, integrity, report_artifact, write_json,
};
use crate::{Job, progress::Progress};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, path::Path, process::Command};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    output_format: String,
    layer_recipe: Value,
    layer_plan_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct StoredTensor {
    shape: Vec<u64>,
    ggml_type: u64,
}

#[derive(Deserialize)]
struct Compiled {
    kind: String,
    architecture: String,
    source_format: String,
    source_sha256: String,
    assets_sha256: Option<String>,
    layer_recipe: Value,
    expected_gguf_tensors: BTreeMap<String, Vec<u64>>,
    expected_ggml_types: BTreeMap<String, u64>,
}

fn expected(ty: &str) -> Option<u64> {
    match ty {
        "F32" => Some(0),
        "F16" => Some(1),
        "Q8_0" => Some(8),
        "BF16" => Some(30),
        _ => None,
    }
}

impl Compiled {
    fn tensors(&self) -> Result<BTreeMap<String, StoredTensor>> {
        ensure!(
            !self.expected_gguf_tensors.is_empty()
                && self.expected_gguf_tensors.len() == self.expected_ggml_types.len(),
            "GGUF layer compiler returned an incomplete tensor plan"
        );
        let layers = self.layer_recipe["layers"]
            .as_object()
            .context("layer recipe requires layer assignments")?;
        let default = self.layer_recipe["default_type"]
            .as_str()
            .and_then(expected)
            .context("layer recipe requires a supported default type")?;
        let mut used = std::collections::BTreeSet::new();
        let tensors = self
            .expected_gguf_tensors
            .iter()
            .map(|(name, shape)| {
                ensure!(
                    super::inventory::valid_name(name)
                        && matches!(shape.len(), 1 | 2)
                        && shape.iter().all(|dim| *dim > 0),
                    "invalid GGUF layer tensor plan"
                );
                let ty = if shape.len() == 1
                    || (self.architecture == "GPT2LMHeadModel" && name == "position_embd.weight")
                {
                    0
                } else {
                    ensure!(name.ends_with(".weight"), "unclassified GGUF layer matrix");
                    let layer = name
                        .strip_prefix("blk.")
                        .and_then(|tail| tail.split_once('.'))
                        .map(|(i, _)| i);
                    if let Some((layer, ty)) = layer.and_then(|id| layers.get_key_value(id)) {
                        used.insert(layer);
                        ty.as_str()
                            .and_then(expected)
                            .context("unsupported GGUF layer assignment")?
                    } else {
                        default
                    }
                };
                ensure!(
                    self.expected_ggml_types.get(name) == Some(&ty),
                    "GGUF compiler did not apply the reviewed policy to {name}"
                );
                Ok((
                    name.clone(),
                    StoredTensor {
                        shape: shape.clone(),
                        ggml_type: ty,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        ensure!(
            used.len() == layers.len(),
            "GGUF recipe selected an unknown decoder layer"
        );
        Ok(tensors)
    }
}

pub(super) fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let options: Options =
        serde_json::from_value(job.options.clone()).context("invalid GGUF layer recipe options")?;
    let verify = job.kind == "verify";
    ensure!(
        env::var("BREWERY_GGUF_LAYER_RECIPE").as_deref() == Ok("1")
            && env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_none(),
        "GGUF layer recipes are not configured on this worker"
    );
    ensure!(
        matches!(job.kind.as_str(), "inspect" | "plan" | "convert" | "verify")
            && job.prompt.is_none()
            && options.output_format == "gguf"
            && options.layer_plan_sha256.is_none() == (job.kind == "inspect"),
        "invalid GGUF layer recipe stage"
    );
    let source = job.inputs.first().context("missing SafeTensors source")?;
    let standalone = source.kind == "safetensors";
    let assets_index = 1 + usize::from(verify);
    ensure!(
        (standalone || source.kind == "hf")
            && job.inputs.len() == assets_index + usize::from(standalone)
            && (!standalone || job.inputs[assets_index].kind == "assets")
            && (!verify || job.inputs[1].kind == "gguf")
            && job.inputs.iter().all(|input| input.host == source.host
                && input
                    .source_host
                    .as_ref()
                    .is_none_or(|host| *host == source.host)
                && input.integrity_basis.is_none()
                && !cwd.starts_with(&input.path)),
        "GGUF layer recipe requires direct SafeTensors inputs in order"
    );
    let roles = if standalone {
        vec![("source", 0), ("assets", assets_index)]
    } else {
        vec![("source", 0)]
    };
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let original_sha = &ids[0].sha256;
    let assets_sha = standalone.then(|| &ids[assets_index].bundle_sha256);
    ensure!(
        options.layer_recipe["source_sha256"].as_str() == Some(original_sha),
        "layer recipe source SHA-256 mismatch"
    );
    let source_sha = integrity::source_hash(&job.inputs, &ids, &roles);
    let scratch = tempfile::Builder::new()
        .prefix(".gguf-layers-")
        .tempdir_in(cwd)?;
    let work = scratch.path();
    let recipe = work.join("recipe.json");
    write_json(&recipe, &options.layer_recipe)?;
    let tool = Tool::open("BREWERY_GGUF_HF_EXPORTER")?;
    let command = |op: &str| -> Result<Command> {
        let mut cmd = tool.command(work, child_threads(None)?);
        for name in ["BREWERY_GGUF_PYTHON", "BREWERY_LLAMA_CPP"] {
            if let Some(value) = env::var_os(name) {
                ensure!(Path::new(&value).is_absolute(), "{name} must be absolute");
                cmd.env(name, value);
            }
        }
        cmd.arg(op)
            .arg("--source")
            .arg(&source.path)
            .arg("--recipe")
            .arg(&recipe);
        if standalone {
            cmd.arg("--assets").arg(&job.inputs[assets_index].path);
        }
        Ok(cmd)
    };
    let (receipt, _) = tool.capture(&mut command("compile-layer-recipe")?, work)?;
    let compiled: Compiled =
        serde_json::from_slice(&receipt).context("invalid GGUF layer compile receipt")?;
    ensure!(
        compiled.kind == "compile-layer-recipe"
            && compiled.source_format == source.kind
            && compiled.source_sha256 == *original_sha
            && compiled.assets_sha256.as_ref() == assets_sha
            && compiled.layer_recipe == options.layer_recipe,
        "GGUF layer compiler did not attest pinned source and recipe"
    );
    let architecture = match compiled.architecture.as_str() {
        "GPT2LMHeadModel" => "gpt2",
        "Qwen2ForCausalLM" => "qwen2",
        _ => anyhow::bail!("unsupported GGUF layer recipe architecture"),
    };
    let tensors = compiled.tensors()?;
    let estimated = tensors.values().try_fold(0u64, |total, tensor| {
        let t = super::inventory::Tensor {
            shape: tensor.shape.clone(),
            ty: tensor.ggml_type,
        };
        total
            .checked_add(t.bytes(t.ty, Namespace::Stock)?)
            .context("GGUF plan extent overflow")
    })?;
    let source_bytes = roles.iter().try_fold(0u64, |total, (_, i)| {
        total
            .checked_add(ids[*i].bytes)
            .context("GGUF source size overflow")
    })?;
    let scratch_bytes = estimated
        .checked_mul(2)
        .and_then(|n| n.checked_add(source_bytes))
        .and_then(|n| n.checked_add(128 << 20))
        .context("GGUF disk estimate overflow")?;
    let disk = converter::disk_preflight(cwd, scratch_bytes)?;
    let plan = json!({"recipe":options.layer_recipe,"assets_sha256":assets_sha,
        "source_bundle_sha256":source_sha,"architecture":architecture,"tensors":tensors,
        "exporter_sha256":tool.sha});
    let plan_sha = format!("{:x}", Sha256::digest(serde_json::to_vec(&plan)?));
    ensure!(
        options
            .layer_plan_sha256
            .as_ref()
            .is_none_or(|pinned| *pinned == plan_sha),
        "GGUF layer plan SHA-256 changed since inspection"
    );
    let mut report = json!({"architecture":architecture,"output_gguf_bytes_estimate":estimated,
        "planned_tensors":tensors});
    let mut file_verification = Value::Null;
    let artifact = if job.kind == "convert" || verify {
        let staging = work.join("staged");
        let path = if verify {
            job.inputs[1].path.clone()
        } else {
            staging.join("model.gguf")
        };
        let identified = if verify {
            None
        } else {
            ensure!(
                disk["fits"] == true,
                "insufficient disk for GGUF layer recipe"
            );
            let mut export = command("export-layer-recipe")?;
            export.arg("--output").arg(&staging);
            let (receipt, _) = tool.capture(&mut export, work)?;
            let emitted: Value =
                serde_json::from_slice(&receipt).context("invalid GGUF layer export receipt")?;
            ensure!(
                emitted["kind"] == "export-layer-recipe"
                    && emitted["source_sha256"].as_str() == Some(original_sha)
                    && emitted["assets_sha256"] == json!(assets_sha)
                    && emitted["layer_recipe"] == plan["recipe"]
                    && emitted["gguf_inventory"] == plan["tensors"]
                    && emitted["artifact"]["path"] == json!(path),
                "exporter did not attest the complete GGUF layer recipe"
            );
            let identified = integrity::identify_output(&path, progress)?;
            ensure!(
                emitted["artifact"]["sha256"] == identified.sha256
                    && emitted["artifact"]["bytes"] == identified.bytes,
                "GGUF layer exporter output identity mismatch"
            );
            Some(identified)
        };
        let pinned_output = identified.as_ref().unwrap_or_else(|| &ids[1]);
        pinned_output.check()?;
        let actual = Model::open(&path, Namespace::Stock)?;
        let stored = actual
            .tensors
            .iter()
            .map(|(name, t)| {
                (
                    name.clone(),
                    StoredTensor {
                        shape: t.shape.clone(),
                        ggml_type: t.ty,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        ensure!(
            actual.architecture == architecture && stored == tensors,
            "independent GGUF inventory differs from the reviewed layer plan"
        );
        report["stored_tensors"] = json!(stored);
        report["gguf_tensors"] = actual.summary(&BTreeMap::new(), Namespace::Stock);
        report["passed"] = json!(true);
        report["tensor_payloads_read"] = json!(true);
        file_verification = json!({"status":"passed","scope":"physical SHA-256, GGUF architecture, complete tensor inventory, shapes, payload bounds and exact planned types; no numerical or runtime check"});
        ids.iter().try_for_each(integrity::Identity::check)?;
        pinned_output.check()?;
        if let Some(output) = identified {
            let destination = cwd.join("converted.gguf");
            converter::publish_directory(&path, &destination)?;
            output.artifact(&destination, "gguf")
        } else {
            report_artifact(cwd, progress, &report, &job.kind)?
        }
    } else {
        report_artifact(cwd, progress, &report, &job.kind)?
    };
    ids.iter().try_for_each(integrity::Identity::check)?;
    Ok(
        json!({"kind":job.kind,"host":source.host,"output_format":"gguf","source_sha256":source_sha,
        "source_digest_scope":"role-qualified physical source bundles",
        "inputs":ids.iter().zip(&job.inputs).map(|(id,input)| id.input(input)).collect::<Vec<_>>(),
        "layer_plan_sha256":plan_sha,"layer_recipe_plan":plan,"artifact":artifact,"report":report,
        "disk_preflight":disk,"file_verification":file_verification,
        "backend_receipt":{"tool":"gguf-export-hf","executable_sha256":tool.sha},"release_qualified":false}),
    )
}
