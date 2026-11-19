use crate::{
    domain::{Job, Model, NewModel},
    profiles::valid_revision,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, sync::LazyLock};

pub const CATALOG_JSON: &str = include_str!("gguf_catalog.json");
pub const STANDALONE_CAPABILITY: &str = "gguf_safetensors_output";

pub fn input_layout(inputs: &[Model], verify: bool) -> Result<Option<usize>, String> {
    let (source, mut remaining) = inputs.split_first().ok_or("GGUF source is required")?;
    if verify {
        let (output, rest) = remaining
            .split_first()
            .ok_or("GGUF verify output is required")?;
        if output.kind != "gguf" {
            return Err("GGUF verify output must be a GGUF file".into());
        }
        remaining = rest;
    }
    let mut next = if verify { 2 } else { 1 };
    match source.kind.as_str() {
        "gguf" | "hf" => {}
        "safetensors" => {
            let (assets, rest) = remaining
                .split_first()
                .filter(|(assets, _)| assets.kind == "assets")
                .ok_or("standalone SafeTensors requires a registered assets directory")?;
            if assets.host != source.host {
                return Err("SafeTensors weights and assets must be on the same worker".into());
            }
            remaining = rest;
            next += 1;
        }
        _ => return Err("GGUF source must be GGUF, HF or standalone SafeTensors".into()),
    }
    let matrix = match remaining {
        [] => None,
        [matrix] if matrix.kind == "imatrix" => Some(next),
        _ => return Err("only one registered imatrix may follow the GGUF source inputs".into()),
    };
    if source.kind == "safetensors" && inputs.iter().any(|input| input.host != source.host) {
        return Err("all standalone SafeTensors GGUF inputs must be on the source worker".into());
    }
    Ok(matrix)
}

static CATALOG: LazyLock<Result<Catalog, serde_json::Error>> =
    LazyLock::new(|| serde_json::from_str(CATALOG_JSON));

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    #[default]
    Hgn,
    Gguf,
}

impl Format {
    pub fn is_hgn(&self) -> bool {
        *self == Self::Hgn
    }
}

#[derive(Deserialize)]
struct Catalog {
    encoder_revision: String,
    default_preset: String,
    presets: Vec<Entry>,
    tensor_types: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    id: String,
    requires_imatrix: bool,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Selection {
    pub preset: String,
    pub overrides: BTreeMap<String, String>,
    pub encoder_revision: String,
    pub requires_imatrix: bool,
}

impl Selection {
    pub fn new(preset: Option<&str>, overrides: BTreeMap<String, String>) -> Result<Self, String> {
        let catalog = CATALOG.as_ref().map_err(|_| "GGUF catalog unavailable")?;
        let preset = preset.unwrap_or(&catalog.default_preset);
        let preset = catalog
            .presets
            .iter()
            .find(|p| p.id == preset)
            .ok_or("unknown GGUF preset")?;
        if overrides.len() > 256 {
            return Err("too many GGUF tensor overrides".into());
        }
        if matches!(preset.id.as_str(), "F16" | "F32" | "BF16") && !overrides.is_empty() {
            return Err("floating-point GGUF presets do not accept tensor overrides".into());
        }
        let mut requires_imatrix = preset.requires_imatrix;
        for (name, choice) in &overrides {
            if name.is_empty()
                || name.len() > 63
                || name.starts_with('.')
                || name.contains("..")
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            {
                return Err("GGUF overrides must use exact tensor names".into());
            }
            let entry = catalog
                .tensor_types
                .iter()
                .find(|t| t.id == *choice)
                .ok_or("unknown GGUF tensor type")?;
            requires_imatrix |= entry.requires_imatrix;
        }
        Ok(Self {
            preset: preset.id.clone(),
            overrides,
            encoder_revision: catalog.encoder_revision.clone(),
            requires_imatrix,
        })
    }

    pub fn inputs(&self, inputs: &[Model], verify: bool) -> Result<Option<usize>, String> {
        let matrix = input_layout(inputs, verify)?;
        if self.requires_imatrix && matrix.is_none() {
            return Err("this GGUF preset or tensor override requires a registered imatrix".into());
        }
        Ok(matrix)
    }

    pub fn options(&self, threads: Option<&Value>, imatrix: Option<usize>) -> Map<String, Value> {
        let mut options = Map::from_iter([
            ("output_format".into(), json!("gguf")),
            ("gguf_type".into(), json!(self.preset)),
            ("tensor_types".into(), json!(self.overrides)),
        ]);
        if let Some(index) = imatrix {
            options.insert("imatrix".into(), json!(index));
        }
        if let Some(threads) = threads {
            options.insert("threads".into(), threads.clone());
        }
        options
    }

    pub fn receipt(&self, job: &Job, result: &Value) -> Result<(), String> {
        if result["kind"] != job.kind
            || result["output_format"] != "gguf"
            || result["gguf_type"] != self.preset
            || result["host"] != job.host
            || result["release_qualified"] != false
            || !result["source_sha256"].as_str().is_some_and(valid_revision)
        {
            return Err("worker did not attest the selected GGUF stage and source".into());
        }
        Ok(())
    }
}

pub fn artifact(job: &Job, result: &Value) -> Result<Model, String> {
    let artifact = &result["artifact"];
    let path = artifact["path"].as_str().unwrap_or_default();
    let suffix = format!("/jobs/{}/work/converted.gguf", job.id);
    if result["report"]["passed"] != true
        || result["file_verification"]["status"] != "passed"
        || artifact["kind"] != "gguf"
        || !path.starts_with('/')
        || !path.ends_with(&suffix)
        || path.contains('\\')
        || path[1..]
            .split('/')
            .any(|part| matches!(part, "" | "." | ".."))
        || !artifact["sha256"].as_str().is_some_and(valid_revision)
        || !artifact["bytes"].as_u64().is_some_and(|bytes| bytes > 0)
    {
        return Err("worker did not return a valid GGUF artifact on its producer".into());
    }
    NewModel {
        name: Some(format!("Converted GGUF ({})", job.id)),
        host: job.host.clone(),
        path: path.into(),
        kind: "gguf".into(),
    }
    .validate()
}

pub fn verification(result: &Value, convert: &Value, host: &str) -> Result<(), String> {
    let output = &result["inputs"][1];
    let artifact = &convert["artifact"];
    if result["report"]["passed"] != true
        || result["file_verification"]["status"] != "passed"
        || result["source_sha256"] != convert["source_sha256"]
        || output["host"] != host
        || output["kind"] != "gguf"
        || output["path"] != artifact["path"]
        || output["sha256"] != artifact["sha256"]
        || output["bytes"] != artifact["bytes"]
    {
        return Err("independent GGUF verification did not attest the published artifact".into());
    }
    Ok(())
}
