use crate::{
    Job, child_threads, converter, gguf_conversion::Tool, integrity, progress::Progress, write_json,
};
use anyhow::{Context, Result, ensure};
use brewery_worker::hgn_recipe as contract;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};

const EXPORTER: &[u8] = include_bytes!("../../../tools/layer_recipe_hgn.py");
const SAFETENSORS: &[u8] = include_bytes!("../../../tools/weight_safetensors.py");

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    ensure!(
        env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_none(),
        "native HGN layer recipes cannot run on a source-pinned worker"
    );
    let verify = job.kind == "verify";
    ensure!(
        job.prompt.is_none()
            && contract::options(&job.kind, &job.options)
            && contract::layout(
                verify,
                job.inputs
                    .iter()
                    .map(|input| { (input.kind.as_str(), input.host.as_str()) })
            )
            && job.inputs.iter().all(|input| {
                input.bundle_sha256.is_some()
                    && input.source_host.as_ref() == Some(&input.host)
                    && input.integrity_basis.is_none()
                    && !cwd.starts_with(&input.path)
            }),
        "invalid native HGN layer recipe or direct input roles"
    );
    let source = &job.inputs[0];
    let assets = (source.kind == "safetensors").then_some(1 + usize::from(verify));
    let mut roles = vec![("source", 0)];
    if let Some(index) = assets {
        roles.push(("assets", index));
    }
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let recipe = &job.options["layer_recipe"];
    ensure!(
        recipe["source_sha256"] == ids[0].sha256,
        "native HGN recipe source SHA-256 mismatch"
    );
    let source_sha = integrity::source_hash(&job.inputs, &ids, &roles);
    let scratch = tempfile::Builder::new()
        .prefix(".hgn-recipe-")
        .tempdir_in(cwd)?;
    let work = scratch.path();
    let helper = Tool::pinned(
        work,
        contract::HELPER,
        contract::HELPER_SHA,
        "native-hgn-helper",
        "native HGN helper",
    )?;
    let (identity, _) = helper
        .capture(
            helper.command(work, child_threads(None)?).arg("identity"),
            work,
        )
        .context("native HGN helper not compatible; rebuild the locked HT3/HT4 helper")?;
    let identity: Value = serde_json::from_slice(&identity)?;
    // The interpreter is from the worker OS, not the job, PATH, a venv, or
    // a source directory. -I -S excludes Python environment and site hooks.
    let python = Tool::from_path(Path::new("/usr/bin/python3").canonicalize()?)?;
    fs::write(work.join("layer_recipe_hgn.py"), EXPORTER)?;
    fs::write(work.join("weight_safetensors.py"), SAFETENSORS)?;
    let recipe_path = work.join("recipe.json");
    write_json(&recipe_path, recipe)?;
    let run = |op: &str, output: &Path| -> Result<Value> {
        let mut cmd = python.command(work, child_threads(None)?);
        cmd.args(["-I", "-S", "-B", "-c",
            "import sys; sys.path.insert(0, sys.argv.pop(1)); from layer_recipe_hgn import main; sys.exit(main())"])
            .arg(work)
            .arg(&source.path)
            .arg(&recipe_path)
            .arg(output)
            .arg("--helper").arg(helper.path())
            .arg("--managed").arg(op);
        if let Some(index) = assets {
            cmd.arg("--assets").arg(&job.inputs[index].path);
        }
        let (out, _) = python.capture(&mut cmd, work)?;
        serde_json::from_slice(&out).context("invalid native HGN exporter response")
    };
    let output = cwd.join(contract::OUTPUT);
    let compiled = run("compile", &output)?;
    ensure!(
        identity["protocol"] == "brewery-layer-hgn.v1"
            && identity["layer_types"] == json!(["HT3", "HT4"])
            && identity["spec_commit"] == compiled["trellis"]["spec_commit"]
            && identity["encoder_sha256"] == compiled["trellis"]["encoder_sha256"],
        "native HGN helper did not attest the pinned HT3/HT4 ABI"
    );
    let plan = json!({
        "recipe":recipe, "source_bundle_sha256":source_sha, "compiled":compiled,
        "helper_sha256":helper.sha, "python_sha256":python.sha,
        "exporter_sha256":hash(EXPORTER), "safetensors_reader_sha256":hash(SAFETENSORS),
        "calibration":"none",
    });
    let plan_sha = hash(&serde_json::to_vec(&plan)?);
    ensure!(
        job.kind == "inspect" || job.options["layer_plan_sha256"] == plan_sha,
        "native HGN layer plan changed after inspection"
    );
    let sizes = &compiled["sizes"];
    let size = |name: &str| {
        sizes[name]
            .as_u64()
            .filter(|&n| (1..=128 << 20).contains(&n))
            .with_context(|| format!("invalid native HGN plan size: {name}"))
    };
    let model_bytes = size("model_bytes")?;
    let source_stream_bytes = size("source_stream_bytes")?;
    let asset_bytes = size("asset_bytes")?;
    let receipt_max_bytes = size("receipt_max_bytes")?;
    let copy_bytes = model_bytes + asset_bytes;
    let archive_max_bytes = copy_bytes + receipt_max_bytes;
    let convert_peak = copy_bytes + source_stream_bytes.max(receipt_max_bytes);
    let (phase, retained, verify_copy, extra) = match job.kind.as_str() {
        "verify" => {
            // The producer's archive is already on disk. Verify only writes
            // a private copy of model.hgn and the checked source assets.
            let model = ids[1]
                .files
                .iter()
                .find(|file| file.path == "model.hgn")
                .context("native HGN archive is missing model.hgn")?;
            ensure!(
                model.bytes == model_bytes,
                "native HGN model differs from planned size"
            );
            let actual_copy = model.bytes + asset_bytes;
            ("verify", ids[1].bytes, actual_copy, actual_copy)
        }
        "convert" => ("convert", 0, 0, convert_peak),
        // Later verification must fit with the archive still present.
        _ => (
            "preflight",
            archive_max_bytes,
            copy_bytes,
            convert_peak.max(archive_max_bytes + copy_bytes),
        ),
    };
    let mut disk = converter::disk_preflight(cwd, extra)?;
    disk["phase"] = phase.into();
    disk["retained_archive_bytes"] = retained.into();
    disk["verify_copy_bytes"] = verify_copy.into();
    disk["source_stream_bytes"] = source_stream_bytes.into();
    disk["scope"] = "peak additional native HGN bytes for this phase, plus 5%, 64 MiB and configured reserve; not a reservation".into();
    let mut result = json!({
        "kind":job.kind, "host":source.host, "output_format":"hgn",
        "inputs":ids.iter().zip(&job.inputs).map(|(id, input)| id.input(input)).collect::<Vec<_>>(),
        "source_sha256":source_sha, "layer_recipe_plan":plan, "layer_plan_sha256":plan_sha,
        "report":{"passed":!verify,"disk":disk},
        "runtime_load_verified":false, "release_qualified":false,
    });
    if matches!(job.kind.as_str(), "convert" | "verify") {
        ensure!(
            disk["fits"] == true,
            "insufficient disk for native HGN layer recipe"
        );
        let output = if verify { &job.inputs[1].path } else { &output };
        let op = if verify { "verify" } else { "convert" };
        let receipt = run(op, output)?;
        ensure!(
            receipt["structural_reader_verified"] == true
                && receipt["runtime_load_verified"] == false
                && receipt["source_sha256"] == recipe["source_sha256"]
                && receipt["architecture"] == compiled["architecture"]
                && receipt["asset_sha256"] == compiled["asset_sha256"]
                && receipt["recipe_sha256"] == compiled["recipe_sha256"]
                && receipt["model"]["bytes"] == model_bytes
                && receipt["trellis"] == compiled["trellis"],
            "native HGN reader did not attest the exact structural archive"
        );
        result["structural_reader_verified"] = true.into();
        if verify {
            result["file_verification"] = json!({"status":"passed", "scope":"structure_only",
                "archive_sha256":ids[1].sha256,"model_sha256":receipt["model"]["sha256"]});
            result["report"]["passed"] = true.into();
        } else {
            let published = integrity::identify_output(output, progress)?;
            ensure!(
                published.bytes <= archive_max_bytes,
                "native HGN archive exceeds planned size"
            );
            result["artifact"] = published.artifact(output, contract::ARCHIVE);
        }
        result["layer_recipe_receipt"] = receipt;
    }
    for id in &ids {
        id.check()?;
    }
    Ok(result)
}
