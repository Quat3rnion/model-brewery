use super::{Result, artifact};
use crate::{
    App,
    domain::{Envelope, Job, State},
    store::Store,
    web::ApiError,
};
use axum::http::StatusCode;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use brewery_worker::{
    is_digest,
    trace::{self, Operation, ReadReply, Role, Rule, Slice, View},
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

const MAX_PATCHES: usize = 8;
// Bound the resolved job envelope before it crosses the worker RPC limit,
// including when short source selectors expand to many vectors.
const MAX_VALUES: usize = 1 << 18;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactRef {
    id: String,
    version: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Patch {
    source_job_id: String,
    #[serde(default)]
    source_prompt: usize,
    layer: usize,
    role: Role,
    positions: BTreeSet<usize>,
}

pub struct TracePlan {
    rules: Vec<Rule>,
    sources: Vec<(Job, Patch)>,
}

impl TracePlan {
    pub fn prepare(
        store: &Store,
        job: &mut Job,
        envelope: &mut Envelope,
        policy: trace::Policy,
    ) -> Result<Option<Self>> {
        let options = &mut envelope.options;
        let saved: Option<ArtifactRef> = take(options, "intervention_artifact")?;
        let patches: Vec<Patch> = take(options, "activation_patches")?.unwrap_or_default();
        let sae: Option<crate::sae::Selection> = take(options, "sae")?;
        let steering: Option<crate::sae::Steering> = take(options, "feature_steering")?;
        let request = trace::Request::controller(
            &Value::Object(options.clone()),
            envelope.prompt.as_deref(),
            policy,
        )
        .map_err(|e| ApiError::bad(e.to_string()))?;
        let model = &envelope.inputs[0];
        if model.kind == trace::HGN_KIND
            && (saved.is_some()
                || !patches.is_empty()
                || sae.is_some()
                || steering.is_some()
                || !request.interventions.is_empty())
        {
            return Err(ApiError::bad(
                "HGN traces do not support interventions or SAE",
            ));
        }
        let mut rules = request.interventions;
        job.options
            .insert("byte_cap".into(), json!(request.capture.byte_cap));
        options.insert("byte_cap".into(), json!(request.capture.byte_cap));
        options.insert("retain_jobs".into(), json!(request.retain_jobs));
        if patches.len() > MAX_PATCHES
            || patches
                .iter()
                .any(|patch| patch.positions.is_empty() || patch.positions.len() > 64)
        {
            return Err(ApiError::bad(
                "Expected at most 8 patch selectors and 1 to 64 positions per selector",
            ));
        }
        let mut digest = envelope.scratch_pins.get(&model.path).cloned();
        if let Some(saved) = saved {
            let saved = artifact(store, &saved.id, saved.version)?;
            if saved.binding.model_id != model.id {
                return Err(ApiError::bad(
                    "Intervention artifact is bound to a different model",
                ));
            }
            if digest.as_ref().is_some_and(|sha| sha != &saved.binding.model_sha256) {
                return Err(ApiError::bad("Intervention artifact is bound to a different GGUF hash"));
            }
            digest = Some(saved.binding.model_sha256);
            rules.extend(saved.interventions);
        }
        if let Some(sae) = sae {
            sae.prepare(store, &model.id, &mut digest, options)?;
        }
        if let Some(steering) = steering {
            rules.push(steering.rule(store, &model.id, &mut digest)?);
        }
        let fixed = rules.len();
        let fixed_values = value_count(&rules);
        let mut sources = Vec::new();
        for patch in patches {
            let prior = store
                .job(&patch.source_job_id)
                .map_err(ApiError::storage)?
                .filter(|job| job.kind == trace::JOB && job.state == State::Succeeded)
                .ok_or_else(|| ApiError::not_found("patch source trace"))?;
            let receipt = store
                .job_result(&prior.id)
                .map_err(ApiError::storage)?
                .filter(|receipt| receipt["kind"] == trace::JOB && receipt["schema_version"] == 1)
                .ok_or_else(|| ApiError::not_found("patch source receipt"))?;
            if receipt["model_id"].as_str() != Some(model.id.as_str()) {
                return Err(ApiError::bad("Patch source is bound to a different model"));
            }
            let source_digest = receipt["model_sha256"]
                .as_str()
                .filter(|s| is_digest(s))
                .ok_or_else(|| ApiError::bad("Patch source has no verified model SHA-256"))?;
            if digest
                .as_deref()
                .is_some_and(|expected| expected != source_digest)
            {
                return Err(ApiError::bad(
                    "Interventions require the same model SHA-256",
                ));
            }
            digest = Some(source_digest.to_owned());
            patch
                .slice()?
                .check()
                .map_err(|e| ApiError::bad(e.to_string()))?;
            if receipt["prompts"].get(patch.source_prompt).is_none() {
                return Err(ApiError::not_found("patch source prompt"));
            }
            // Check overlap and the expanded rule count before reading any payload.
            rules.extend(
                patch
                    .positions
                    .iter()
                    .map(|&p| patch.rule(p, Operation::Zero)),
            );
            sources.push((prior, patch));
        }
        Rule::check_all(&rules).map_err(|e| ApiError::bad(e.to_string()))?;
        if fixed_values > MAX_VALUES {
            return Err(ApiError::bad("Trace intervention vectors are too large"));
        }
        rules.truncate(fixed);
        if let Some(digest) = digest {
            options.insert("expected_model_sha256".into(), json!(digest));
        }
        if sources.is_empty() {
            set_rules(options, rules);
            Ok(None)
        } else {
            Ok(Some(Self { rules, sources }))
        }
    }

    pub async fn resolve(mut self, app: &App, envelope: &mut Envelope) -> Result<()> {
        let mut values = value_count(&self.rules);
        tokio::time::timeout(Duration::from_secs(30), async {
            for (prior, patch) in self.sources {
                let reply = app
                    .dispatcher
                    .trace_read(&prior, &patch.slice()?)
                    .await
                    .map_err(|e| {
                        eprintln!("Patch source trace read failed for {}: {e}", prior.id);
                        invalid_payload()
                    })?;
                let data = match reply {
                    ReadReply::Ok(data) => data,
                    ReadReply::Missing => {
                        return Err(ApiError::not_found("patch source trace slice"));
                    }
                    ReadReply::TooLarge => {
                        return Err(ApiError(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "Select a smaller patch source range".into(),
                        ));
                    }
                };
                let Rows {
                    prompt,
                    layer,
                    role,
                    dtype,
                    rows,
                    ..
                } = serde_json::from_value(data).map_err(|_| invalid_payload())?;
                if (prompt, layer, role, dtype.as_str())
                    != (patch.source_prompt, patch.layer, patch.role, "f32_le")
                {
                    return Err(invalid_payload());
                }
                let mut vectors = BTreeMap::new();
                for row in rows
                    .into_iter()
                    .filter(|r| patch.positions.contains(&r.position))
                {
                    let vector = row.vector()?;
                    values += vector.len();
                    if values > MAX_VALUES {
                        return Err(ApiError::bad("Trace intervention vectors are too large"));
                    }
                    if vectors.insert(row.position, vector).is_some() {
                        return Err(invalid_payload());
                    }
                }
                if vectors.len() != patch.positions.len() {
                    return Err(ApiError::not_found("patch source activation position"));
                }
                self.rules.extend(
                    vectors
                        .into_iter()
                        .map(|(p, vector)| patch.rule(p, Operation::Replace { vector })),
                );
            }
            Ok(())
        })
        .await
        .map_err(|_| {
            ApiError(
                StatusCode::GATEWAY_TIMEOUT,
                "Patch source reads timed out".into(),
            )
        })??;
        set_rules(&mut envelope.options, self.rules);
        Ok(())
    }
}

impl Patch {
    fn slice(&self) -> Result<Slice> {
        let start = self.positions.first().copied();
        let end = self.positions.last().and_then(|p| p.checked_add(1));
        if start.is_none() || end.is_none() {
            return Err(ApiError::bad("Invalid patch positions"));
        }
        Ok(Slice {
            view: View::Tensor,
            prompt: self.source_prompt,
            layer: Some(self.layer),
            role: Some(self.role),
            start,
            end,
        })
    }

    fn rule(&self, p: usize, op: Operation) -> Rule {
        Rule {
            layer: self.layer,
            role: self.role,
            positions: BTreeSet::from([p]),
            op,
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct Rows {
    pub prompt: usize,
    pub layer: usize,
    pub role: Role,
    pub dtype: String,
    pub rows: Vec<Row>,
    #[serde(default)]
    pub missing_positions: Vec<usize>,
}

#[derive(Deserialize)]
pub(crate) struct Row {
    pub position: usize,
    shape: Vec<usize>,
    sha256: String,
    data_base64: String,
}

impl Row {
    pub fn vector(&self) -> Result<Vec<f32>> {
        let [width] = self.shape.as_slice() else {
            return Err(invalid_payload());
        };
        if !(1..=16384).contains(width) || self.data_base64.len() != (width * 4).div_ceil(3) * 4 {
            return Err(invalid_payload());
        }
        let bytes = STANDARD
            .decode(&self.data_base64)
            .map_err(|_| invalid_payload())?;
        if bytes.len() != width * 4 || format!("{:x}", Sha256::digest(&bytes)) != self.sha256 {
            return Err(invalid_payload());
        }
        let vector: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect();
        if !vector.iter().all(|v| v.is_finite()) {
            return Err(invalid_payload());
        }
        Ok(vector)
    }
}

fn take<T: DeserializeOwned>(options: &mut Map<String, Value>, name: &str) -> Result<Option<T>> {
    options
        .remove(name)
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| ApiError::bad(format!("Invalid {name}")))
}

fn set_rules(options: &mut Map<String, Value>, rules: Vec<Rule>) {
    if !rules.is_empty() {
        options.insert("interventions".into(), json!(rules));
    }
}

fn value_count(rules: &[Rule]) -> usize {
    rules
        .iter()
        .map(|rule| match &rule.op {
            Operation::Add { vector } | Operation::Replace { vector } => vector.len(),
            _ => 0,
        })
        .sum()
}

fn invalid_payload() -> ApiError {
    ApiError(
        StatusCode::BAD_GATEWAY,
        "Cannot verify patch source activation".into(),
    )
}
