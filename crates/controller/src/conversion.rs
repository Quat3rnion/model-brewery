use crate::domain::{Envelope, Host, Job, Model, NewJob, NewModel, State, candidates, id, now};
use crate::gguf::{Format, Selection};
use crate::profiles::Profile;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

const STAGES: &[&str] = &["inspect", "plan", "convert", "verify"];
const AUXILIARIES: &[&str] = &["assets", "ngram", "vision", "mtp"];

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NewConversion {
    #[serde(default, skip_serializing_if = "Format::is_hgn")]
    pub format: Format,
    pub host: Option<String>,
    pub input_ids: Vec<String>,
    #[serde(default)]
    pub options: Map<String, Value>,
    pub decode_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_revision: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub struct Conversion {
    #[serde(default, skip_serializing_if = "Format::is_hgn")]
    pub format: Format,
    pub id: String,
    pub host: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    requested_host: Option<String>,
    pub state: State,
    pub stage: String,
    pub job_ids: BTreeMap<String, String>,
    pub input_ids: Vec<String>,
    options: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_backend: Option<String>,
    pub cancel_requested: bool,
    pub created_at: u64,
    pub updated_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    finished_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_name: Option<String>,
}

impl Conversion {
    pub fn start(
        request: NewConversion,
        inputs: Vec<Model>,
        hosts: &[Host],
        active: &[Job],
        profile: Option<&Profile>,
    ) -> Result<Option<(Self, Job, Envelope)>, String> {
        if request
            .decode_backend
            .as_deref()
            .is_some_and(|v| !matches!(v, "cpu" | "hip" | "cuda"))
        {
            return Err("invalid decode_backend".into());
        }
        let source = inputs.first().ok_or("a source model is required")?;
        if profile.is_some_and(|p| p.format() != request.format) {
            return Err("saved profile format does not match conversion format".into());
        }
        let gguf = if request.format == Format::Gguf {
            if request.decode_backend.is_some()
                || request
                    .options
                    .keys()
                    .any(|key| !matches!(key.as_str(), "preset" | "threads"))
                || request
                    .options
                    .get("threads")
                    .is_some_and(|value| !value.as_u64().is_some_and(|n| (1..=64).contains(&n)))
            {
                return Err("GGUF conversion accepts only preset and threads options".into());
            }
            let selection = if let Some(profile) = profile {
                if request.options.contains_key("preset") {
                    return Err("profile_revision cannot be combined with options.preset".into());
                }
                profile.gguf().ok_or("saved GGUF profile is missing")?
            } else {
                let preset = request
                    .options
                    .get("preset")
                    .map(|v| v.as_str().ok_or("GGUF preset must be a string"))
                    .transpose()?;
                Selection::new(preset, BTreeMap::new())?
            };
            selection.inputs(&inputs)?;
            Some(selection)
        } else {
            None
        };
        if profile.is_some() && request.format.is_hgn() {
            if request.options.contains_key("profile") {
                return Err("profile_revision cannot be combined with options.profile".into());
            }
            if source.kind != "hf"
                || request
                    .decode_backend
                    .as_deref()
                    .is_some_and(|b| b != "cpu")
                || ["ht_backend", "i4r_backend", "q4cp_backend"]
                    .iter()
                    .any(|key| request.options.get(*key).is_some_and(|b| b != "cpu"))
            {
                return Err("custom profiles require an HF source and CPU backends".into());
            }
        }
        let host = request.host.unwrap_or_else(|| source.host.clone());
        let requested_host = (host == "auto").then(|| host.clone());
        let created_at = now();
        let mut row = Self {
            format: request.format,
            id: id(),
            host,
            requested_host,
            state: State::Running,
            stage: "inspect".into(),
            job_ids: BTreeMap::new(),
            input_ids: request.input_ids,
            options: request.options,
            decode_backend: request.decode_backend,
            cancel_requested: false,
            created_at,
            updated_at: created_at,
            finished_at: None,
            error: None,
            artifact_model_id: None,
            profile_revision: profile.map(|profile| profile.revision().into()),
            profile_name: profile.map(|profile| profile.name().into()),
        };
        // Validate the complete conversion options and role layout before
        // choosing a worker or enqueuing the first stage.
        let recipe = if let Some(selection) = gguf {
            Some(serde_json::to_value(selection).map_err(|_| "invalid GGUF policy")?)
        } else {
            profile.and_then(Profile::recipe)
        };
        let (convert, envelope) = row.build("convert", inputs, None, recipe.as_ref())?;
        let eligible = candidates(&convert, &envelope, hosts)
            .into_iter()
            .filter(|host| {
                !host.component_cpu_only || row.decode_backend.as_deref().is_none_or(|b| b == "cpu")
            })
            .filter(|host| STAGES.iter().all(|stage| host.capabilities[*stage]))
            .collect::<Vec<_>>();
        let Some(host) = eligible
            .iter()
            .find(|host| active.iter().all(|job| job.host != host.id))
            .or_else(|| eligible.first())
        else {
            return Ok(None);
        };
        row.host = host.id.clone();
        let (job, envelope) = row.build("inspect", envelope.inputs, None, recipe.as_ref())?;
        row.job_ids.insert(row.stage.clone(), job.id.clone());
        Ok(Some((row, job, envelope)))
    }

    pub fn build(
        &self,
        stage: &str,
        mut inputs: Vec<Model>,
        output: Option<Model>,
        recipe: Option<&Value>,
    ) -> Result<(Job, Envelope), String> {
        if self.format == Format::Gguf {
            let selection: Selection =
                serde_json::from_value(recipe.ok_or("GGUF policy missing")?.clone())
                    .map_err(|_| "invalid GGUF policy")?;
            selection.inputs(&inputs)?;
            let has_matrix = inputs.len() == 2;
            if stage == "verify" {
                inputs.insert(1, output.ok_or("GGUF output missing")?);
            }
            let options = selection.options(
                self.options.get("threads"),
                has_matrix.then_some(if stage == "verify" { 2 } else { 1 }),
            );
            let (mut job, mut envelope) = NewJob {
                kind: stage.into(),
                host: Some(self.host.clone()),
                input_ids: inputs.iter().map(|m| m.id.clone()).collect(),
                options: Map::new(),
                prompt_id: None,
                prompt: None,
            }
            .build_for(self.host.clone(), inputs, None, self.format)?;
            job.profile_revision.clone_from(&self.profile_revision);
            job.options =
                Map::from_iter([("preset".into(), Value::from(selection.preset.clone()))]);
            envelope.options = options;
            return Ok((job, envelope));
        }
        let mut options = self.options.clone();
        match stage {
            "inspect" => {
                options.retain(|key, _| !matches!(key.as_str(), "seed" | "threads" | "ht_beam"))
            }
            "verify" => {
                options.retain(|key, _| AUXILIARIES.contains(&key.as_str()));
                for index in options.values_mut() {
                    *index =
                        Value::from(index.as_u64().ok_or("invalid stored auxiliary index")? + 1);
                }
                if let Some(backend) = &self.decode_backend {
                    options.insert("decode_backend".into(), backend.as_str().into());
                }
                inputs.insert(1, output.ok_or("conversion artifact is missing")?);
            }
            _ => {}
        }
        if let Some(recipe) = recipe
            && stage != "verify"
        {
            options.insert("profile".into(), recipe["base"].clone());
        }
        let (mut job, mut envelope) = NewJob {
            kind: stage.into(),
            host: Some(self.host.clone()),
            input_ids: inputs.iter().map(|input| input.id.clone()).collect(),
            options,
            prompt_id: None,
            prompt: None,
        }
        .build(self.host.clone(), inputs, None)?;
        if let Some(revision) = &self.profile_revision {
            let recipe = recipe.ok_or("saved profile policy is missing")?;
            if recipe["revision"] != *revision {
                return Err("saved profile identity changed".into());
            }
            job.profile_revision = Some(revision.clone());
            envelope
                .options
                .insert("profile_recipe".into(), recipe.clone());
        }
        Ok((job, envelope))
    }

    pub fn current_job(&self) -> &str {
        &self.job_ids[&self.stage]
    }

    pub fn gguf_selection(
        &self,
        inspect: &Envelope,
        profile: Option<&Profile>,
    ) -> Result<Selection, String> {
        let preset = inspect
            .options
            .get("gguf_type")
            .and_then(Value::as_str)
            .ok_or("stored GGUF preset missing")?;
        let overrides: BTreeMap<String, String> =
            serde_json::from_value(inspect.options["tensor_types"].clone())
                .map_err(|_| "invalid stored GGUF overrides")?;
        if let Some(profile) = profile {
            let selection = profile.gguf().ok_or("stored GGUF profile missing")?;
            if selection.preset != preset || selection.overrides != overrides {
                return Err("saved GGUF profile does not match workflow snapshot".into());
            }
            Ok(selection)
        } else {
            Selection::new(Some(preset), overrides)
        }
    }

    pub fn next(&mut self, job: &Job) {
        self.stage.clone_from(&job.kind);
        self.job_ids.insert(self.stage.clone(), job.id.clone());
        self.updated_at = now();
    }

    pub fn finish(&mut self, state: State, error: Option<String>) {
        self.state = state;
        self.error = error;
        self.updated_at = now();
        self.finished_at = Some(self.updated_at);
    }
}

pub fn preflight(result: &Value, stage: &str) -> Result<(), String> {
    if result["kind"] != stage || result["disk_preflight"]["fits"].as_bool() != Some(true) {
        return Err(format!(
            "{stage} did not confirm that the conversion fits on disk"
        ));
    }
    Ok(())
}

pub fn artifact(result: &Value, job: &Job) -> Result<Model, String> {
    let invalid =
        || "convert did not return a valid published HGN artifact on its producer".to_string();
    let artifact = &result["artifact"];
    let path = artifact["path"].as_str().ok_or_else(invalid)?;
    let suffix = format!("/jobs/{}/work/converted", job.id);
    if result["kind"] != "convert"
        || result["host"].as_str() != Some(job.host.as_str())
        || result["report"]["published"].as_bool() != Some(true)
        || artifact["kind"] != "hgn"
        || !digest(&artifact["sha256"])
        || !positive(&artifact["bytes"])
        || !path.starts_with('/')
        || !path.ends_with(&suffix)
        || path.contains('\\')
        || path[1..]
            .split('/')
            .any(|part| matches!(part, "" | "." | ".."))
    {
        return Err(invalid());
    }
    NewModel {
        name: Some(format!("Converted HGN ({})", job.id)),
        host: job.host.clone(),
        path: path.into(),
        kind: "hgn".into(),
    }
    .validate()
    .map_err(|_| invalid())
}

pub fn verification(result: &Value, convert: &Value, host: &str) -> Result<(), String> {
    let report = &result["report"];
    let artifact = &convert["artifact"];
    let output = &result["inputs"][1];
    if result["kind"] != "verify"
        || result["host"].as_str() != Some(host)
        || report["passed"].as_bool() != Some(true)
        || report["manifest_schema"].as_u64() != Some(2)
        || report["provenance_attested"].as_bool() != Some(true)
        || !positive(&report["source_tensors"])
        || !positive(&report["output_tensors"])
        || !positive(&report["payload_bytes"])
        || !report["errors"].is_object()
        || result["file_verification"]["status"] != "passed"
        || !digest(&convert["source_sha256"])
        || result["source_sha256"] != convert["source_sha256"]
        || output["kind"] != "hgn"
        || output["host"].as_str() != Some(host)
        || output["path"] != artifact["path"]
        || output["sha256"] != artifact["sha256"]
        || output["bytes"] != artifact["bytes"]
    {
        return Err(
            "verification did not return a passing native result for this artifact and source"
                .into(),
        );
    }
    Ok(())
}

fn positive(value: &Value) -> bool {
    value.as_u64().is_some_and(|n| n > 0)
}

fn digest(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

pub fn same_component_source(inspect: &Value, current: &Value) -> Result<(), String> {
    if !digest(&inspect["source_sha256"]) || current["source_sha256"] != inspect["source_sha256"] {
        return Err("base checkpoint or selected component bytes changed after inspection".into());
    }
    Ok(())
}
