use crate::{
    check_inputs, integrity, native_json_admitted, progress::Progress, read_json, recipe::Recipe,
    tool, write_json, Job,
};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    ffi::CString,
    fs::{self, File},
    io::Read,
    os::unix::{ffi::OsStrExt, fs::MetadataExt},
    path::{Path, PathBuf},
    process::Command,
};

const SIDE_INPUTS: &[&str] = &["assets", "ngram", "vision", "mtp"];

#[path = "encoder_activity.rs"]
mod encoder_activity;

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let kind = job.kind.as_str();
    let verify = kind == "verify";
    let convert = kind == "convert";
    let base_inputs = if verify { 2 } else { 1 };
    ensure!(
        (base_inputs..=base_inputs + SIDE_INPUTS.len()).contains(&job.inputs.len()),
        "wrong number of converter inputs"
    );
    let source = &job.inputs[0];
    ensure!(
        matches!(source.kind.as_str(), "hf" | "hgn" | "gguf"),
        "converter source kind must be hf, hgn, or gguf"
    );
    ensure!(
        job.prompt.is_none(),
        "converter jobs do not accept a prompt"
    );
    let destination = cwd.join("converted");
    let evidence = cwd.join("converter-evidence");
    let publish = evidence.join("publish");
    let output = if verify {
        let input = &job.inputs[1];
        ensure!(
            matches!(input.kind.as_str(), "hgn" | "hgn-v2" | "ht43")
                && integrity::canonical(&input.path)?.is_dir(),
            "verify requires an HGN output directory as its second input"
        );
        Some(native_subtree(&input.path))
    } else if convert {
        ensure_missing(&destination)?;
        Some(publish.join("native"))
    } else {
        None
    };

    let source_meta = integrity::canonical(&source.path)?;
    let source_path = if source.kind == "hgn" && source_meta.is_dir() {
        native_subtree(&source.path)
    } else {
        source.path.clone()
    };
    let mut args = vec![
        "--source".into(),
        source_path.clone(),
        "--source-format".into(),
        source.kind.clone().into(),
    ];
    let composite = source.kind == "hf" && job.inputs.len() > base_inputs;
    let expected_source = std::env::var_os("BREWERY_EXPECTED_SOURCE_SHA256");
    let recipe = job
        .options
        .get("profile_recipe")
        .map(Recipe::parse)
        .transpose()?;
    if let Some(recipe) = &recipe {
        ensure!(
            expected_source.is_none() && source.integrity_basis.is_none(),
            "custom profiles are disabled on source-pinned qualification workers"
        );
        ensure!(
            verify || job.options["profile"] == recipe.base(),
            "custom profile selection does not match the stage profile"
        );
    }
    ensure!(
        !composite || expected_source.is_none(),
        "HF component overrides are disabled on source-pinned qualification workers"
    );
    let roles = options(job, base_inputs, &mut args, composite)?;
    if source.kind == "hf" {
        ensure!(
            source_meta.is_dir(),
            "HF requires an indexed snapshot directory"
        );
    }
    let cpu_only = match std::env::var("BREWERY_COMPONENT_CPU_ONLY") {
        Ok(v) if v == "1" => true,
        Ok(v) if v == "0" => false,
        Err(std::env::VarError::NotPresent) => false,
        _ => bail!("BREWERY_COMPONENT_CPU_ONLY must be 0 or 1"),
    };
    let selected_backends: &[&str] = if verify {
        &["decode_backend"]
    } else {
        &["ht_backend", "i4r_backend", "q4cp_backend"]
    };
    let accelerator = selected_backends
        .iter()
        .any(|key| job.options.get(*key).is_some_and(|value| value != "cpu"));
    ensure!(
        !(cpu_only || recipe.is_some()) || !accelerator,
        "CPU-only worker rejects accelerator backends"
    );
    if composite || cpu_only || recipe.is_some() {
        for key in selected_backends {
            if job.options.get(*key).is_none() {
                args.push(format!("--{}", key.replace('_', "-")).into());
                args.push("cpu".into());
            }
        }
    }
    let cuda_convert = convert
        && ["ht_backend", "i4r_backend", "q4cp_backend"]
            .iter()
            .any(|key| job.options[*key] == "cuda");
    let admission = if cuda_convert || composite && accelerator {
        std::env::var_os("BREWERY_CONVERTER_ADMISSION").map(PathBuf::from)
    } else {
        None
    };
    ensure!(
        !composite || !accelerator || admission.as_ref().is_some_and(|p| p.is_absolute()),
        "HF component accelerator stages require an absolute BREWERY_CONVERTER_ADMISSION helper"
    );
    let component_exe = composite
        .then(|| dedicated_converter("BREWERY_COMPONENT_CONVERTER"))
        .transpose()?;
    let custom_exe = recipe
        .as_ref()
        .map(|_| dedicated_converter("BREWERY_CUSTOM_CONVERTER"))
        .transpose()?;
    if source.kind == "hgn" {
        if source_meta.is_file() {
            // Do not let the converter load sibling assets or sidecars outside
            // the set that was fingerprinted by the worker and engine.
            for &role in SIDE_INPUTS {
                if roles.iter().any(|(r, _)| *r == role) {
                    continue;
                }
                ensure!(
                    role != "assets",
                    "bare HGN input requires a registered assets directory"
                );
                let implicit = source
                    .path
                    .with_file_name(format!("qwen38-flash-next-{role}.hgn"));
                ensure_missing(&implicit).context(
                    "register implicit HGN sidecars explicitly or register the source directory",
                )?;
            }
        } else if !roles.iter().any(|(r, _)| *r == "assets") {
            // Converter output keeps the flat assets inside tokenizer/.
            // This is already covered by the source directory identity.
            let assets = source_path.join("tokenizer");
            if assets.is_dir() {
                args.extend(["--assets".into(), assets]);
            }
        }
    }
    ensure!(
        !job.inputs
            .iter()
            .any(|input| evidence.starts_with(&input.path)
                || recipe.is_some() && cwd.starts_with(&input.path)
                || convert && destination.starts_with(&input.path)),
        "job outputs must be outside every source input"
    );
    let ids = check_inputs(&job.inputs, progress)?;
    let source_sha256 = integrity::source_hash(&job.inputs, &ids, &roles);
    if let Some(expected) = expected_source {
        ensure!(
            expected == source_sha256.as_str(),
            "input does not match the approved checkpoint digest"
        );
    }
    let exe = match custom_exe.or(component_exe) {
        Some(exe) => exe,
        None => tool("BREWERY_CONVERTER", "bf16-to-hgn")?,
    };
    fs::create_dir(&evidence).context("cannot create fresh converter evidence directory")?;
    if convert {
        fs::create_dir(&publish)?;
    }
    let mut cmd = Command::new(&exe);
    let threads = crate::child_threads(job.options.get("threads").and_then(Value::as_u64))?;
    crate::child_budget(&mut cmd, threads);
    cmd.arg(kind).args(args);
    if matches!(kind, "plan" | "convert") && job.options.get("threads").is_none() {
        cmd.arg("--threads").arg(threads.to_string());
    }
    if let Some(path) = &output {
        cmd.arg("--output").arg(path);
    }
    let recipe_file = recipe
        .as_ref()
        .map(|recipe| recipe.snapshot(cwd))
        .transpose()?;
    if let Some(file) = &recipe_file {
        cmd.arg("--recipe").arg(file.path());
    }
    if let (true, Some(recipe), Some(output)) = (verify, &recipe, &output) {
        check_recipe_manifest(recipe, output)?;
    }
    let capture = encoder_activity::Capture::configure(cuda_convert, &mut cmd, &exe, &evidence);
    ids.iter().try_for_each(integrity::Identity::check)?;
    let phase = match kind {
        "convert" => "convert",
        "verify" => "verify",
        "inspect" => "inspect",
        _ => "plan",
    };
    let (mut report, receipt) = native_json_admitted(
        &mut cmd,
        &evidence.join("stderr.log"),
        admission.as_deref(),
        Some((progress, phase)),
    )?;
    ids.iter().try_for_each(integrity::Identity::check)?;
    let mut file_verification =
        json!({"status":"not_performed","scope":"converter source/output validation"});
    let mut disk = Value::Null;
    let mut inference_bundle = Value::Null;
    ensure!(report.is_object(), "converter returned a non-object report");
    if let Some(recipe) = &recipe {
        let selected_report = if kind == "plan" {
            &report["summary"]
        } else {
            &report
        };
        recipe.check(selected_report)?;
        ensure!(
            verify || selected_report["profile"] == recipe.base(),
            "native report did not bind the selected base profile"
        );
        if kind == "plan" {
            recipe.check_plan(&report["plan"])?;
        }
    }
    let components = component_evidence(job, &roles, &ids, &report, composite)?;
    let artifact = match kind {
        "inspect" => {
            let payload = check_summary(&report, &source.kind)?;
            disk = disk_preflight(cwd, payload)?;
            report_artifact(&evidence, kind, &report, progress)?
        }
        "plan" => {
            let payload = check_summary(&report["summary"], &source.kind)?;
            ensure!(
                report["plan"].is_object() && report["layouts"].is_array(),
                "incomplete native plan"
            );
            disk = disk_preflight(cwd, payload)?;
            report_artifact(&evidence, kind, &report, progress)?
        }
        "convert" => {
            let output = output.as_deref().unwrap();
            ensure!(
                report["published"] == true
                    && report["manifest"] == "conversion-manifest.json"
                    && report["source_format"] == source.kind,
                "converter did not attest fresh publication"
            );
            let manifest = read_json(&output.join("conversion-manifest.json"))?;
            ensure!(
                manifest["schema"] == 2
                    && manifest["source_kind"] == source.kind
                    && manifest["source_provenance"].is_object()
                    && manifest["options"]["profile"] == report["profile"]
                    && digest_string(&manifest["encoder_sha256"]),
                "incomplete or inconsistent native conversion manifest"
            );
            ensure!(
                !composite || manifest["source_provenance"]["format"] == "huggingface-composite-v1",
                "native manifest did not bind the composite source"
            );
            // Keep converter inventory exact. Adding any file inside native/
            // would make the upstream verifier reject its own output.
            let identity = integrity::identify_output(&publish, progress)?;
            inference_bundle = if let Some(recipe) = &recipe {
                validate_recipe_manifest(recipe, output, &manifest)?;
                ensure_missing(&publish.join("manifest.json"))
                    .context("custom artifacts must not include a standard runtime manifest")?;
                unqualified_runtime()
            } else {
                runtime_manifest(&publish, &identity, &manifest, &source_sha256)?
            };
            let identity = if inference_bundle["status"] == "manifest_created" {
                inference_bundle["manifest"] = json!(destination.join("manifest.json"));
                identity.with_added_file(&publish, "manifest.json")?
            } else {
                identity
            };
            identity.check()?;
            let mut phase = progress.start("publish_output", "items", Some(1));
            publish_directory(&publish, &destination)?;
            phase.advance(1);
            phase.finish();
            report["manifest"] = json!("native/conversion-manifest.json");
            report["encoder_source_sha256"] = manifest["encoder_sha256"].clone();
            report["native_directory"] = json!(destination.join("native"));
            identity.artifact(&destination, "hgn")
        }
        "verify" => {
            ensure!(
                report["manifest_schema"] == 2
                    && report["provenance_attested"] == true
                    && positive(&report["source_tensors"])
                    && positive(&report["output_tensors"])
                    && positive(&report["payload_bytes"])
                    && report["errors"].is_object(),
                "verification did not attest schema-2 source provenance and output payloads"
            );
            // The native VerifyReport omits `passed`; a successful exit and
            // the receipt above imply success only if no explicit field disagrees.
            ensure!(
                report
                    .get("passed")
                    .is_none_or(|value| value.as_bool() == Some(true)),
                "native verification report has conflicting or invalid passed value"
            );
            report["passed"] = Value::Bool(true);
            file_verification = json!({"status":"passed","scope":"converter verified source provenance, output payloads and decoded values"});
            if recipe.is_some() {
                file_verification["scope"] = json!(
                    "custom recipe source provenance, output payloads and decoded values only"
                );
                inference_bundle = unqualified_runtime();
            }
            report_artifact(&evidence, kind, &report, progress)?
        }
        _ => unreachable!(),
    };
    let observed_activity = capture.result(&receipt, &artifact);
    let mut result = json!({
        "kind":kind,"host":source.host,
        "inputs":ids.iter().zip(&job.inputs).map(|(id,input)| id.input(input)).collect::<Vec<_>>(),
        "source_sha256":source_sha256,"source_digest_scope":"role-qualified physical source bundles",
        "artifact":artifact,"report":report,"disk_preflight":disk,"inference_bundle":inference_bundle,
        "backend_receipt":{"tool":"bf16-to-hgn","processes":[receipt],"requested_options":job.options},
        "file_verification":file_verification,
        "hardware_proof":{"status":"not_proved","reason":"Process logs record native progress; no independent device or runtime load attestation"}
    });
    result["components"] = components;
    if let Some(recipe) = recipe {
        result["custom_profile"] = recipe.result();
        result["release_qualified"] = json!(false);
    }
    if let Some(activity) = observed_activity {
        result["encoder_activity"] = activity;
    }
    write_json(&evidence.join("result.json"), &result)?;
    Ok(result)
}

fn dedicated_converter(var: &str) -> Result<PathBuf> {
    let configured = std::env::var_os(var)
        .map(PathBuf::from)
        .with_context(|| format!("converter stage requires {var}"))?;
    ensure!(configured.is_absolute(), "{var} must be absolute");
    let custom = integrity::canonical(&configured)?;
    ensure!(
        custom.is_file() && custom.mode() & 0o111 != 0,
        "{var} must be an executable regular file"
    );
    if let Ok(standard) = tool("BREWERY_CONVERTER", "bf16-to-hgn") {
        let meta = fs::metadata(standard)?;
        ensure!(
            (custom.dev(), custom.ino()) != (meta.dev(), meta.ino()),
            "{var} must be distinct from BREWERY_CONVERTER"
        );
    }
    Ok(configured)
}

fn unqualified_runtime() -> Value {
    json!({"status":"not_created","reason":"Custom container runtime compatibility was not tested","release_qualified":false})
}

fn check_recipe_manifest(recipe: &Recipe, output: &Path) -> Result<()> {
    let manifest = read_json(&output.join("conversion-manifest.json"))?;
    validate_recipe_manifest(recipe, output, &manifest)
}

fn validate_recipe_manifest(recipe: &Recipe, output: &Path, manifest: &Value) -> Result<()> {
    recipe.check(manifest)?;
    recipe.check_plan(&manifest["plan"])?;
    ensure!(
        manifest["schema"] == 2 && manifest["options"]["profile"] == recipe.base(),
        "custom manifest does not bind the selected base profile"
    );
    let main = output.join("model.hgn");
    let meta = integrity::canonical(&main)?;
    ensure!(
        meta.is_file() && hgn_header(&main, meta.len(), &recipe.main_identity())?,
        "custom manifest has no matching custom container"
    );
    Ok(())
}

fn ensure_missing(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
        Ok(_) => bail!("path already exists: {}", path.display()),
    }
}

fn native_subtree(root: &Path) -> PathBuf {
    if root.join("native/conversion-manifest.json").is_file() {
        root.join("native")
    } else {
        root.to_owned()
    }
}

pub(crate) fn publish_directory(src: &Path, dest: &Path) -> Result<()> {
    let src = CString::new(src.as_os_str().as_bytes())?;
    let dest = CString::new(dest.as_os_str().as_bytes())?;
    // SAFETY: Both path arguments are valid C strings for this call.
    if unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            src.as_ptr(),
            libc::AT_FDCWD,
            dest.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error())
            .context("cannot publish fresh conversion bundle");
    }
    Ok(())
}

fn component_format(role: &str, path: &Path, composite: bool) -> Result<&'static str> {
    let meta = integrity::canonical(path)?;
    if role == "assets" {
        ensure!(meta.is_dir(), "assets input must be a directory");
        return Ok("assets-directory");
    }
    if meta.is_dir() && composite {
        let index = integrity::canonical(&path.join("model.safetensors.index.json"))
            .with_context(|| format!("{role} requires an indexed safetensors directory"))?;
        ensure!(index.is_file(), "{role} safetensors index is not a file");
        return Ok("safetensors-index");
    }
    ensure!(meta.is_file(), "{role} input must be a regular file");
    let mut prefix = [0; 4];
    File::open(path)?.read_exact(&mut prefix)?;
    if &prefix == b"HGN1" {
        return Ok("hgn");
    }
    ensure!(
        composite && path.extension().is_some_and(|ext| ext == "safetensors"),
        "{role} requires a compatible HGN sidecar or an explicitly supported safetensors input"
    );
    // The native role reader validates its header, tensor set and shapes.
    Ok("safetensors")
}

fn component_evidence(
    job: &Job,
    roles: &[(&str, usize)],
    ids: &[integrity::Identity],
    report: &Value,
    composite: bool,
) -> Result<Value> {
    let selection = if job.kind == "plan" {
        &report["summary"]["source_selection"]
    } else {
        &report["source_selection"]
    };
    if composite {
        ensure!(
            selection["mode"] == "hf-role-overlays-v1",
            "native converter did not attest HF role selection"
        );
    }
    let mut evidence = serde_json::Map::new();
    for &role in SIDE_INPUTS {
        let selected = roles
            .iter()
            .find_map(|&(name, index)| (name == role).then_some(index));
        let native = selection["roles"][role].as_str();
        let data = if let Some(index) = selected {
            ensure!(
                !composite || native == Some("selected"),
                "native converter did not confirm selected {role} input"
            );
            let input = &job.inputs[index];
            let identity = &ids[index];
            json!({"source":"input","input_id":input.id,
                "format":component_format(role, &input.path, composite)?,
                "sha256":identity.sha256,"bundle_sha256":identity.bundle_sha256,"bytes":identity.bytes})
        } else {
            ensure!(
                !composite || matches!(native, Some("embedded" | "none")),
                "native converter did not identify the {role} source"
            );
            json!({"source":match native {
                Some("embedded") if composite => "embedded",
                Some("none") if composite => "none",
                _ => "unknown",
            }})
        };
        evidence.insert(role.into(), data);
    }
    Ok(Value::Object(evidence))
}

fn options(
    job: &Job,
    base: usize,
    args: &mut Vec<PathBuf>,
    composite: bool,
) -> Result<Vec<(&'static str, usize)>> {
    let mut roles = vec![("source", 0)];
    for &role in SIDE_INPUTS {
        if let Some(value) = job.options.get(role) {
            let index = value
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .context("auxiliary inputs are selected by index")?;
            ensure!(
                index >= base && index < job.inputs.len(),
                "invalid {role} input index"
            );
            ensure!(
                job.inputs[index].kind == role,
                "{role} input must have registered kind {role}"
            );
            component_format(role, &job.inputs[index].path, composite)?;
            roles.push((role, index));
            args.push(format!("--{role}").into());
            args.push(job.inputs[index].path.clone());
        }
    }
    ensure!(
        (base..job.inputs.len())
            .all(|i| roles.iter().filter(|(_, index)| *index == i).count() == 1),
        "each auxiliary input must be used once"
    );
    for (opt, value) in job.options.as_object().unwrap() {
        if SIDE_INPUTS.contains(&opt.as_str()) || opt == "profile_recipe" {
            continue;
        }
        let allowed_values: &[&str] = match (job.kind.as_str(), opt.as_str()) {
            ("verify", "decode_backend") => &["cpu", "hip", "cuda"],
            ("inspect" | "plan" | "convert", "profile") => &["v2", "ht43"],
            ("inspect" | "plan" | "convert", "ht_backend" | "i4r_backend" | "q4cp_backend") => {
                &["cpu", "hip", "cuda"]
            }
            ("plan" | "convert", "seed" | "threads" | "ht_beam") => &[],
            _ => bail!("unsupported {} option: {opt}", job.kind),
        };
        let parameter = if allowed_values.is_empty() {
            let number = value
                .as_u64()
                .with_context(|| format!("{opt} must be a nonnegative integer"))?;
            ensure!(
                opt == "seed"
                    || opt == "threads" && (1..=64).contains(&number)
                    || opt == "ht_beam" && (1..=256).contains(&number),
                "out-of-range {opt}"
            );
            if opt == "threads" {
                crate::child_threads(Some(number))?.to_string()
            } else {
                number.to_string()
            }
        } else {
            value
                .as_str()
                .filter(|v| allowed_values.contains(v))
                .with_context(|| format!("invalid {opt}"))?
                .into()
        };
        args.push(format!("--{}", opt.replace('_', "-")).into());
        args.push(parameter.into());
    }
    Ok(roles)
}

fn positive(value: &Value) -> bool {
    value.as_u64().is_some_and(|n| n > 0)
}

fn digest_string(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

fn check_summary(report: &Value, source_kind: &str) -> Result<u64> {
    ensure!(
        report["source_format"] == source_kind
            && report["tensor_payloads_read"] == false
            && report["outputs"].is_array()
            && positive(&report["output_hgn_bytes"]),
        "incomplete converter summary"
    );
    Ok(report["output_hgn_bytes"].as_u64().unwrap())
}

pub(crate) fn disk_preflight(path: &Path, payload_bytes: u64) -> Result<Value> {
    let path = CString::new(path.as_os_str().as_bytes())?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: statvfs initializes the structure on success.
    if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error()).context("cannot inspect output free space");
    }
    let stat = unsafe { stat.assume_init() };
    let avail = (stat.f_bavail as u128) * (stat.f_frsize as u128);
    let payload = u128::from(payload_bytes);
    let reserve = std::env::var("BREWERY_CONVERTER_DISK_RESERVE_BYTES")
        .ok()
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(0);
    let required = payload + payload / 20 + (64 << 20) + u128::from(reserve);
    Ok(json!({
        "payload_bytes":payload_bytes,"available_bytes":u64::try_from(avail)?,
        "required_bytes":u64::try_from(required)?,"fits":avail >= required,
        "scope":"converter payloads plus 5%, 64 MiB, and any configured reserve; assets and concurrent disk usage excluded; not a reservation"
    }))
}

fn report_artifact(root: &Path, kind: &str, report: &Value, progress: &Progress) -> Result<Value> {
    let path = root.join(format!("{kind}.json"));
    let mut phase = progress.start("publish_output", "items", Some(1));
    write_json(&path, report)?;
    phase.advance(1);
    phase.finish();
    Ok(integrity::identify_output(&path, progress)?.artifact(&path, kind))
}

fn runtime_manifest(
    output: &Path,
    bundle: &integrity::Identity,
    manifest: &Value,
    source: &str,
) -> Result<Value> {
    let pending = |reason| json!({"status":"not_created","reason":reason});
    let format = match manifest["options"]["profile"].as_str() {
        Some("v2") => "hgn-v2",
        Some("ht43") => "ht43",
        _ => return Ok(pending("Unrecognized output profile")),
    };
    if manifest["plan"]["architecture"] != "qwen4_exp" {
        return Ok(pending(
            "No inference manifest mapping for this architecture",
        ));
    }
    let Some(plans) = manifest["plan"]["files"]
        .as_array()
        .filter(|files| files.len() == 3)
    else {
        return Ok(pending(
            "Expected checkpoint, lookup, and vision containers",
        ));
    };
    let mut files = Vec::new();
    for (role, path, expected) in [
        (
            "checkpoint",
            "model.hgn",
            if format == "ht43" {
                "qwen3.8-flash-next-ht43"
            } else {
                "qwen3.8-flash-next-v2"
            },
        ),
        (
            "lookup",
            "qwen38-flash-next-ngram.hgn",
            "qwen3.8-flash-next-ngram",
        ),
        (
            "vision",
            "qwen38-flash-next-vision.hgn",
            "qwen3.8-flash-next-vision",
        ),
    ] {
        let relative = format!("native/{path}");
        let Some(member) = bundle.files.iter().find(|f| f.path == relative) else {
            return Ok(pending("Required runtime file missing"));
        };
        if !plans
            .iter()
            .any(|plan| plan["filename"] == path && plan["identity"] == expected)
            || !hgn_header(&output.join(&relative), member.bytes, expected)?
        {
            return Ok(pending("Unexpected runtime container header or identity"));
        }
        files.push(json!({"role":role,"path":relative,"sha256":member.sha256}));
    }
    let mut tokenizer = Vec::new();
    let mut has_tokenizer = false;
    for file in &bundle.files {
        let Some(relative) = file.path.strip_prefix("native/tokenizer/") else {
            continue;
        };
        if relative.contains('/') || file.bytes == 0 {
            return Ok(pending(
                "Tokenizer directory must contain only flat nonempty files",
            ));
        }
        has_tokenizer |= relative == "tokenizer.json";
        tokenizer.push(json!({"path":file.path,"sha256":file.sha256}));
    }
    if !has_tokenizer {
        return Ok(pending("Tokenizer JSON missing"));
    }
    let path = output.join("manifest.json");
    write_json(
        &path,
        &json!({"format":format,"files":files,"tokenizer":tokenizer,"source_sha256":source}),
    )?;
    Ok(
        json!({"status":"manifest_created","manifest":path,"format":format,
              "scope":"file identities only; runtime qualification and GPU load not performed"}),
    )
}

fn hgn_header(path: &Path, bytes: u64, expected: &str) -> Result<bool> {
    let mut header = [0; 104];
    if bytes < header.len() as u64 {
        return Ok(false);
    }
    File::open(path)?.read_exact(&mut header)?;
    let le = |start| u64::from_le_bytes(header[start..start + 8].try_into().unwrap());
    let count = le(8);
    let dir = le(16);
    let data = le(24);
    Ok(&header[..8] == b"HGN1\x02\0\0\0"
        && (1..=1 << 20).contains(&count)
        && dir == 104
        && data >= dir + count * 160
        && data < bytes
        && le(32) == bytes
        && header[40..].split(|b| *b == 0).next() == Some(expected.as_bytes()))
}
