use crate::{
    domain::{Envelope, Job, Model, NewModel, id},
    profiles::valid_revision,
};
use brewery_worker::weight_source;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub fn is_kind(kind: &str) -> bool {
    matches!(
        kind,
        "weight_inspect" | "weight_propose" | "weight_tune" | "weight_apply"
    )
}

pub fn is_safetensors(kind: &str) -> bool {
    matches!(kind, "safetensors" | "hf")
}

fn tensor(value: &Value, native_safe: bool) -> bool {
    value
        .as_str()
        .is_some_and(|s| weight_source::tensor_name(s, native_safe))
}

fn index(value: &Value, rank: Option<usize>) -> bool {
    value.as_array().is_some_and(|coords| {
        rank.map_or_else(|| (1..=4).contains(&coords.len()), |n| coords.len() == n)
            && coords
                .iter()
                .all(|v| v.as_u64().is_some_and(|n| n <= u32::MAX.into()))
    })
}

fn number(value: &Value) -> bool {
    value.as_f64().is_some_and(f64::is_finite)
}

pub fn needs_visual(kind: &str, options: &Map<String, Value>) -> bool {
    is_kind(kind) && (options.contains_key("mode") || options.contains_key("source_sha256"))
}

pub fn is_packed(options: &Map<String, Value>) -> bool {
    options.get("packed").and_then(Value::as_bool) == Some(true)
}

fn inspect_options(value: &Value, count: usize, has_pin: bool, native_safe: bool) -> bool {
    match value["mode"].as_str() {
        Some("inventory") => {
            count == 3 + usize::from(has_pin)
                && value["offset"]
                    .as_u64()
                    .is_some_and(|n| n <= u32::MAX.into())
                && value["limit"]
                    .as_u64()
                    .is_some_and(|n| (1..=128).contains(&n))
        }
        Some("tile") => {
            let Some(shape) = value["shape"].as_array() else {
                return false;
            };
            let rank = shape.len();
            count == 5
                && has_pin
                && tensor(&value["tensor"], native_safe)
                && (1..=4).contains(&rank)
                && index(&value["origin"], Some(rank))
                && shape.iter().enumerate().all(|(axis, n)| {
                    let limit = if rank == 1 {
                        1024
                    } else if axis + 2 < rank {
                        1
                    } else {
                        32
                    };
                    n.as_u64().is_some_and(|n| (1..=limit).contains(&n))
                })
        }
        _ => count == 2 && tensor(&value["tensor"], native_safe) && index(&value["index"], None),
    }
}

pub fn options(kind: &str, source_kind: &str, value: &Value) -> Result<(), String> {
    let native_safe = weight_source::supports(source_kind);
    let packed = value.get("packed");
    if packed.is_some_and(|v| v != true || kind == "weight_apply") {
        return Err(format!("invalid {kind} packed option"));
    }
    let count = value.as_object().map_or(0, Map::len) - usize::from(packed.is_some());
    let has_pin = value.get("source_sha256").is_some();
    if has_pin && !value["source_sha256"].as_str().is_some_and(valid_revision) {
        return Err("invalid source_sha256".into());
    }
    let valid = match kind {
        "weight_inspect" => inspect_options(value, count, has_pin, native_safe),
        "weight_propose" => {
            count == 1 + usize::from(has_pin)
                && value["edits"].as_array().is_some_and(|edits| {
                    let mut coords = BTreeSet::new();
                    (1..=256).contains(&edits.len())
                        && edits.iter().all(|e| {
                            e.as_object().is_some_and(|e| e.len() == 4)
                                && tensor(&e["tensor"], native_safe)
                                && index(&e["index"], None)
                                && matches!(e["op"].as_str(), Some("set" | "delta"))
                                && number(&e["value"])
                                && coords.insert((e["tensor"].as_str(), e["index"].to_string()))
                        })
                })
        }
        "weight_tune" => {
            count == 3 + usize::from(has_pin)
                && tensor(&value["tensor"], native_safe)
                && value["max_update"]
                    .as_f64()
                    .is_some_and(|n| n.is_finite() && n > 0.)
                && value["indices"].as_array().is_some_and(|indices| {
                    let mut coords = BTreeSet::new();
                    (1..=128).contains(&indices.len())
                        && indices
                            .iter()
                            .all(|i| index(i, Some(2)) && coords.insert(i.to_string()))
                })
        }
        "weight_apply" => {
            count == 2
                && value["reviewed"] == true
                && value["proposal_job_id"]
                    .as_str()
                    .is_some_and(|s| uuid::Uuid::parse_str(s).is_ok())
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!("invalid {kind} options"))
    }
}

pub fn check_packed_receipt(
    job: &Job,
    envelope: &Envelope,
    result: &Value,
) -> Result<(), &'static str> {
    let invalid = "worker did not verify the requested packed weight operation";
    if !is_packed(&envelope.options) {
        return if result.get("packed").is_none() {
            Ok(())
        } else {
            Err(invalid)
        };
    }
    let source = envelope.inputs.first().ok_or(invalid)?;
    let digest = &result["source_sha256"];
    if result["packed"] != true
        || result["kind"] != job.kind
        || result["host"] != job.host
        || !source_format_matches(source, result)
        || !digest.as_str().is_some_and(valid_revision)
        || envelope
            .options
            .get("source_sha256")
            .is_some_and(|expected| expected != digest)
    {
        return Err(invalid);
    }
    let report = &result["report"];
    let valid = match job.kind.as_str() {
        "weight_inspect" if report["source_sha256"] != *digest => false,
        "weight_inspect"
            if envelope
                .options
                .get("mode")
                .is_some_and(|mode| mode == "inventory") =>
        {
            report["tensors"].as_array().is_some_and(|rows| {
                rows.iter()
                    .all(|row| row["editable"] == false || checked_dtype(source, &row["dtype"]))
            })
        }
        "weight_inspect" => checked_dtype(source, &report["dtype"]),
        "weight_propose" | "weight_tune" => {
            let proposal = &report["proposal"];
            proposal["schema"] == "brewery.weight-edit-proposal.v1"
                && proposal["source_sha256"] == *digest
                && proposal["edits"].as_array().is_some_and(|edits| {
                    (1..=256).contains(&edits.len())
                        && edits.iter().all(|edit| {
                            checked_dtype(source, &edit["dtype"])
                                && number(&edit["before"])
                                && number(&edit["after"])
                                && edit["before"].as_f64() != edit["after"].as_f64()
                        })
                })
        }
        "weight_apply" => envelope
            .options
            .get("proposal_sha256")
            .is_some_and(|approved| {
                approved.as_str().is_some_and(valid_revision)
                    && result["proposal_sha256"] == *approved
            }),
        _ => false,
    };
    valid.then_some(()).ok_or(invalid)
}

fn checked_dtype(source: &Model, dtype: &Value) -> bool {
    matches!(
        (source.kind.as_str(), dtype.as_str()),
        ("gguf" | "hgn", Some("F32" | "F16" | "BF16"))
            | ("gguf", Some("Q4_0" | "Q8_0"))
            | ("hgn", Some("Q8G64"))
    )
}

pub fn check_source_receipt(job: &Job, result: &Value) -> Result<(), &'static str> {
    if !needs_visual(&job.kind, &job.options) {
        return Ok(());
    }
    let observed = &result["source_sha256"];
    if result["kind"] != job.kind
        || result["host"] != job.host
        || !observed.as_str().is_some_and(valid_revision)
        || job
            .options
            .get("source_sha256")
            .is_some_and(|expected| expected != observed)
    {
        return Err("worker did not confirm the selected weight source fingerprint");
    }
    if job.kind == "weight_inspect" {
        let report = &result["report"];
        if job.options.get("mode") != report.get("mode") || report["source_sha256"] != *observed {
            return Err("worker returned a mismatched visual weight report");
        }
        let valid = match report["mode"].as_str() {
            Some("inventory") => {
                let offset = job.options["offset"].as_u64().unwrap_or_default();
                let limit = job.options["limit"].as_u64().unwrap_or_default();
                report["offset"] == offset
                    && report["total"].as_u64().is_some_and(|total| {
                        report["tensors"].as_array().is_some_and(|page| {
                            offset <= total
                                && page.len() as u64 == (total - offset).min(limit)
                                && report.get("next_offset")
                                    == Some(&json!(
                                        (offset + (page.len() as u64) < total)
                                            .then_some(offset + page.len() as u64)
                                    ))
                        })
                    })
            }
            Some("tile") => {
                let shape = job.options["shape"]
                    .as_array()
                    .map_or(&[][..], Vec::as_slice);
                let rows = shape
                    .iter()
                    .rev()
                    .nth(1)
                    .and_then(Value::as_u64)
                    .unwrap_or(1);
                let cols = shape.last().and_then(Value::as_u64).unwrap_or_default();
                report["tensor"] == job.options["tensor"]
                    && report["origin"] == job.options["origin"]
                    && report["tile_shape"] == job.options["shape"]
                    && report["values"].as_array().is_some_and(|grid| {
                        grid.len() as u64 == rows
                            && grid.iter().all(|row| {
                                row.as_array().is_some_and(|values| {
                                    values.len() as u64 == cols && values.iter().all(number)
                                })
                            })
                    })
            }
            _ => false,
        };
        if !valid {
            return Err("worker returned invalid visual weight data");
        }
    }
    Ok(())
}

pub fn check_safetensors_receipt(
    job: &Job,
    envelope: &Envelope,
    result: &Value,
) -> Result<(), &'static str> {
    let Some(source) = envelope.inputs.first().filter(|m| is_safetensors(&m.kind)) else {
        return Ok(());
    };
    if result["kind"] != job.kind
        || result["host"] != job.host
        || !source_format_matches(source, result)
        || !result["source_sha256"].as_str().is_some_and(valid_revision)
    {
        return Err("worker did not confirm the registered SafeTensors source");
    }
    Ok(())
}

pub fn apply_inputs(
    source: &Model,
    previous: &Job,
    saved: &Envelope,
    result: &Value,
) -> Result<(Model, Value), String> {
    let invalid =
        || "apply requires a successful proposal for the exact source and worker".to_string();
    if !matches!(previous.kind.as_str(), "weight_propose" | "weight_tune")
        || previous.state != crate::domain::State::Succeeded
        || previous.cancel_requested
        || previous.host != source.host
        || !saved.inputs.first().is_some_and(|prior| {
            prior.id == source.id
                && prior.path == source.path
                && prior.host == source.host
                && prior.kind == source.kind
        })
    {
        return Err(invalid());
    }
    let proposal = &result["report"]["proposal"];
    let artifact = &result["artifact"];
    let source_sha = proposal["source_sha256"].as_str().ok_or_else(invalid)?;
    if result["kind"] != previous.kind
        || result["host"] != previous.host
        || !source_format_matches(source, result)
        || check_packed_receipt(previous, saved, result).is_err()
        || proposal["schema"] != "brewery.weight-edit-proposal.v1"
        || !valid_revision(source_sha)
        || result["source_sha256"] != source_sha
        || (is_safetensors(&source.kind) && source_manifest(source, result).is_none())
        || !proposal["edits"]
            .as_array()
            .is_some_and(|edits| (1..=256).contains(&edits.len()))
    {
        return Err(invalid());
    }
    let path = artifact_path(
        previous,
        artifact,
        "weight-edit-proposal",
        "weight-proposal.json",
    )
    .ok_or_else(invalid)?;
    let input = Model {
        id: id(),
        host: source.host.clone(),
        kind: "weight-edit-proposal".into(),
        path: path.into(),
        name: "Reviewed weight proposal".into(),
    };
    let mut checks = json!({
        "source_sha256":source_sha,"proposal_sha256":artifact["sha256"],
    });
    if is_packed(&saved.options) {
        checks["packed"] = json!(true);
    }
    Ok((input, checks))
}

pub fn verified_artifact(job: &Job, envelope: &Envelope, result: &Value) -> Result<Model, String> {
    let invalid =
        || "worker did not verify the weight edit output for the reviewed source".to_string();
    // The durable input, not the worker receipt, determines the only acceptable output.
    let source = envelope.inputs.first().ok_or_else(invalid)?;
    let (label, leaf) = match source.kind.as_str() {
        "gguf" => ("GGUF", "edited.gguf"),
        "hgn" => ("HGN", "edited.hgn"),
        "safetensors" => ("SafeTensors", "edited.safetensors"),
        "hf" => ("HF", "edited.hf"),
        _ => return Err(invalid()),
    };
    let report = &result["report"];
    let artifact = &result["artifact"];
    let source_sha = envelope.options.get("source_sha256").ok_or_else(invalid)?;
    if result["kind"] != job.kind
        || result["host"] != job.host
        || !source_format_matches(source, result)
        || check_packed_receipt(job, envelope, result).is_err()
        || result["verified"] != true
        || report["verified"] != true
        || report["schema"] != "brewery.weight-edit-result.v1"
        || result["source_sha256"] != *source_sha
        || report["source_sha256"] != *source_sha
        || report["output_sha256"] != artifact["sha256"]
        || artifact["sha256"] == *source_sha
        || !report["edit_count"]
            .as_u64()
            .is_some_and(|n| (1..=256).contains(&n))
    {
        return Err(invalid());
    }
    let path = artifact_path(job, artifact, &source.kind, leaf).ok_or_else(invalid)?;
    if is_safetensors(&source.kind) {
        let approved = envelope
            .options
            .get("proposal_sha256")
            .ok_or_else(invalid)?;
        if !approved.as_str().is_some_and(valid_revision)
            || result["proposal_sha256"] != *approved
            || !verified_safetensors_output(source, result)
        {
            return Err(invalid());
        }
    }
    NewModel {
        host: job.host.clone(),
        name: Some(format!("Edited {label} ({})", job.id)),
        kind: source.kind.clone(),
        path: path.into(),
    }
    .validate()
    .map_err(|_| invalid())
}

type Members<'a> = BTreeMap<&'a str, (u64, &'a str)>;

fn manifest(receipt: &Value) -> Option<Members<'_>> {
    let files = receipt["files"].as_array()?;
    if !(1..=16_384).contains(&files.len()) {
        return None;
    }
    let mut members = BTreeMap::new();
    let mut total = 0u64;
    for file in files {
        let path = file["path"].as_str()?;
        let bytes = file["bytes"].as_u64()?;
        let sha = file["sha256"].as_str()?;
        if path.len() > 4096
            || path.contains('\\')
            || path.chars().any(char::is_control)
            || path.split('/').any(|p| matches!(p, "" | "." | ".."))
            || !valid_revision(sha)
            || members.insert(path, (bytes, sha)).is_some()
        {
            return None;
        }
        total = total.checked_add(bytes)?;
    }
    let mut hash = Sha256::new();
    for (path, (bytes, sha)) in &members {
        hash.update(format!("{path}\0{bytes}\0{sha}\n"));
    }
    (receipt["bytes"] == total && receipt["bundle_sha256"] == format!("{:x}", hash.finalize()))
        .then_some(members)
}

fn source_manifest<'a>(source: &Model, result: &'a Value) -> Option<Members<'a>> {
    let input = &result["inputs"][0];
    if input["id"] != source.id
        || input["kind"] != source.kind
        || input["host"] != source.host
        || input["sha256"] != result["source_sha256"]
    {
        return None;
    }
    let members = manifest(input)?;
    match source.kind.as_str() {
        "hf" if input["sha256"] == input["bundle_sha256"] => Some(members),
        "safetensors"
            if members.len() == 1
                && members
                    .values()
                    .all(|(bytes, sha)| *bytes > 0 && input["sha256"] == *sha) =>
        {
            Some(members)
        }
        _ => None,
    }
}

fn verified_safetensors_output(source: &Model, result: &Value) -> bool {
    let Some(input) = source_manifest(source, result) else {
        return false;
    };
    let output = &result["artifact"];
    if source.kind == "safetensors" {
        return output.get("files").is_none()
            && output.get("bundle_sha256").is_none()
            && input.values().all(|(bytes, _)| output["bytes"] == *bytes);
    }
    let Some(output_members) = manifest(output) else {
        return false;
    };
    output["sha256"] == output["bundle_sha256"]
        && output_members.contains_key("model.safetensors.index.json")
        && output_members
            .keys()
            .any(|path| path.ends_with(".safetensors"))
        && output_members.len() == input.len()
        && input.iter().all(|(path, (bytes, sha))| {
            output_members
                .get(path)
                .is_some_and(|(new_bytes, new_sha)| {
                    bytes == new_bytes && (path.ends_with(".safetensors") || sha == new_sha)
                })
        })
}

fn source_format_matches(source: &Model, result: &Value) -> bool {
    match result.get("source_format") {
        Some(format) => format == &source.kind,
        None => source.kind == "gguf", // Legacy workers omit the marker.
    }
}

fn artifact_path<'a>(job: &Job, artifact: &'a Value, kind: &str, leaf: &str) -> Option<&'a str> {
    let path = artifact["path"].as_str()?;
    (artifact["kind"] == kind
        && artifact["sha256"].as_str().is_some_and(valid_revision)
        && artifact["bytes"].as_u64().is_some_and(|bytes| bytes > 0)
        && path.len() <= 4096
        && path.starts_with('/')
        && !path.contains('\\')
        && !path.chars().any(char::is_control)
        && path.ends_with(&format!("/jobs/{}/work/{leaf}", job.id))
        && path[1..]
            .split('/')
            .all(|part| !matches!(part, "" | "." | "..")))
    .then_some(path)
}
