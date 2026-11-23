//! Admission and source-bound W1 observations for dense scratch pruning.
use super::*;
use crate::domain::State as JobState;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use brewery_worker::trace::{self, ReadReply, Role, Slice, View};
use serde_json::Value;

#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub(super) struct Params {
    pub(super) plan: scratch::prune::Plan,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Evidence {
    gguf_sha256: String,
    manifest_sha256: String,
    lanternfish_sha256: String,
}

pub(super) struct Calibration {
    pub(super) rows: Vec<Vec<f32>>,
    pub(super) evidence: Evidence,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    sha256: String,
    pruning: Params,
}

pub(super) async fn model(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<Request>,
) -> Mutation {
    mutate(
        app,
        headers,
        "scratch-prune",
        Some(id),
        read(payload)?,
        |app, target, request, key| {
            let id = request_id(target)?;
            {
                let store = app.store.lock().map_err(ApiError::storage)?;
                visible(&store, &id)?;
            }
            build_special(
                app,
                id,
                Some(request.sha256),
                Vec::new(),
                None,
                Some(NativeEdit::Prune(request.pruning)),
                key,
            )
        },
    )
    .await
}

impl Params {
    pub(super) fn calibration(&self, app: &App, parent_id: &str) -> Result<Option<Calibration>> {
        let scratch::prune::Plan::WandaMlp {
            layer,
            trace: chosen,
            ..
        } = &self.plan
        else {
            return Ok(None);
        };
        let slice = Slice {
            view: View::Tensor,
            prompt: chosen.prompt,
            layer: Some(*layer),
            role: Some(Role::FfnInp),
            start: Some(chosen.start),
            end: Some(chosen.end),
        };
        slice.check().map_err(|e| ApiError::bad(e.to_string()))?;
        if !(1..=64).contains(&(chosen.end - chosen.start)) {
            return Err(ApiError::bad("select 1 to 64 trace positions"));
        }
        let (job, sha, manifest_sha256) = {
            let store = app.store.lock().map_err(ApiError::storage)?;
            let (input, sha) = jobs::trace_input(&store, parent_id)?;
            let job = store
                .job(&chosen.id)
                .map_err(ApiError::storage)?
                .filter(|job| {
                    job.kind == trace::JOB
                        && job.state == JobState::Succeeded
                        && job.host == input.host
                        && job.input_ids == [parent_id]
                })
                .ok_or_else(|| conflict("use a completed trace of this private scratch GGUF"))?;
            let receipt = store
                .job_result(&job.id)
                .map_err(ApiError::storage)?
                .filter(|r| {
                    r["model_id"] == parent_id
                        && r["model_sha256"] == sha
                        && r["executable_sha256"] == trace::SHA256
                        && r["lanternfish_commit"] == trace::COMMIT
                })
                .ok_or_else(|| conflict("trace receipt does not match the parent and runtime"))?;
            let manifest_sha256 = receipt["prompts"][chosen.prompt]["manifest_sha256"]
                .as_str()
                .filter(|hash| brewery_worker::is_digest(hash))
                .ok_or_else(|| conflict("trace has no pinned prompt manifest"))?
                .to_owned();
            (job, sha, manifest_sha256)
        };
        // mutate runs on the blocking pool after the idempotency lookup. Do
        // not hold the Store lock while reading the worker's pinned trace.
        let handle = tokio::runtime::Handle::current();
        let read_trace =
            |slice: &Slice| match handle.block_on(app.dispatcher.trace_read(&job, slice)) {
                Ok(ReadReply::Ok(data)) => Ok(data),
                _ => Err(conflict("the selected trace slice is not available")),
            };
        let manifest = read_trace(&Slice {
            view: View::Manifest,
            prompt: chosen.prompt,
            layer: None,
            role: None,
            start: None,
            end: None,
        })?;
        if manifest["model_id"] != parent_id
            || manifest["model_sha256"] != sha
            || manifest["model"]["arch"] != "qwen2"
            || manifest["model"]["n_experts"] != 0
            || manifest["intervention_count"] != 0
            || manifest["lanternfish_commit"] != trace::COMMIT
            || manifest["manifest_sha256"] != manifest_sha256
        {
            return Err(conflict(
                "use an unmodified Qwen2 trace of this private GGUF hash",
            ));
        }
        let data = read_trace(&slice)?;
        let rows = data["rows"]
            .as_array()
            .filter(|rows| {
                rows.len() == chosen.end - chosen.start
                    && data["missing_positions"]
                        .as_array()
                        .is_some_and(Vec::is_empty)
                    && data["dtype"] == "f32_le"
                    && data["layer"] == *layer
                    && data["prompt"] == chosen.prompt
                    && data["role"] == "ffn_inp"
            })
            .ok_or_else(|| conflict("trace does not cover all selected ffn_inp positions"))?;
        let hidden = manifest["model"]["hidden"]
            .as_u64()
            .filter(|n| (1..=16384).contains(n))
            .ok_or_else(|| conflict("invalid trace model width"))? as usize;
        let mut activations = Vec::with_capacity(rows.len());
        for (position, row) in (chosen.start..chosen.end).zip(rows) {
            if row["position"] != position
                || row["shape"]
                    .as_array()
                    .is_none_or(|shape| shape.len() != 1 || shape[0] != hidden)
            {
                return Err(conflict("unexpected ffn_inp trace position or shape"));
            }
            activations.push(decode_row(row, hidden)?);
        }
        Ok(Some(Calibration {
            rows: activations,
            evidence: Evidence {
                gguf_sha256: sha,
                manifest_sha256,
                lanternfish_sha256: trace::SHA256.into(),
            },
        }))
    }
}

fn decode_row(row: &Value, hidden: usize) -> Result<Vec<f32>> {
    let invalid = || conflict("invalid ffn_inp trace bytes");
    let data = STANDARD
        .decode(row["data_base64"].as_str().ok_or_else(invalid)?)
        .map_err(|_| invalid())?;
    if data.len() != hidden * 4 {
        return Err(invalid());
    }
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect())
}
