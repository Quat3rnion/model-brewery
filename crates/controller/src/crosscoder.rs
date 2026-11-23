//! Immutable weight-space reports imported from the offline paired TopK trainer.
use crate::{App, domain::State as JobState, store::Store, web::ApiError};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, Path, State, multipart::MultipartRejection},
    http::{StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use brewery_worker::{is_digest, trace, weight_source::Header};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc, time::Duration};
use tokio::sync::Semaphore;

type Result<T> = std::result::Result<T, ApiError>;
type Upload = std::result::Result<Multipart, MultipartRejection>;
const MAX_WEIGHTS: usize = 128 << 20;
const MAX_METADATA: usize = 32 << 10;
const UNIQUE: f64 = 0.10;
const CHANGED: f64 = 0.995;
const SUPPORT: f64 = 1e-8;
const METHOD: &str = "paired-cpu-top-k";
static SLOT: Semaphore = Semaphore::const_new(1);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    id: String,
    sha256: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Split {
    Train,
    Heldout,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    parent_job_id: String,
    child_job_id: String,
    split: Split,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metrics {
    mse: f64,
    explained_variance: f64,
    l0: f64,
    dead_features: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sides<T> {
    parent: T,
    child: T,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    torch: String,
    steps: usize,
    difference_weight: f64,
    train_rows: usize,
    heldout_rows: usize,
    dataset_sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    schema_version: u32,
    parent: Binding,
    child: Binding,
    layer: usize,
    role: trace::Role,
    pairs: Vec<Pair>,
    width: u64,
    k: usize,
    scale: f64,
    seen: Vec<bool>,
    metrics: Sides<Metrics>,
    training: Training,
}

#[derive(Serialize)]
struct Feature {
    feature: usize,
    classification: &'static str,
    child_norm_ratio: Option<f64>,
    decoder_cosine: Option<f64>,
    parent_norm: f64,
    child_norm: f64,
}

pub fn router() -> Router<Arc<App>> {
    Router::new()
        .route(
            "/crosscoders",
            post(import).layer(DefaultBodyLimit::max(MAX_WEIGHTS + MAX_METADATA + 8192)),
        )
        .route("/crosscoders/{id}", get(report))
        .route("/crosscoders/{id}/safetensors", get(export))
}

fn uuid(id: &str) -> bool {
    uuid::Uuid::try_parse(id).is_ok_and(|value| value.to_string() == id)
}

fn slot() -> Result<tokio::sync::SemaphorePermit<'static>> {
    SLOT.try_acquire().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "Another crosscoder transfer is active".into(),
        )
    })
}

async fn read_parts(body: Upload) -> Result<(Metadata, Vec<u8>)> {
    let multipart_error =
        |err: axum::extract::multipart::MultipartError| ApiError(err.status(), err.body_text());
    let mut parts = body.map_err(|e| ApiError::bad(e.to_string()))?;
    let mut metadata = None;
    let mut weights = None;
    while let Some(mut field) = parts.next_field().await.map_err(multipart_error)? {
        let (target, limit) = match field.name() {
            Some("metadata") if metadata.is_none() => (&mut metadata, MAX_METADATA),
            Some("weights") if weights.is_none() => (&mut weights, MAX_WEIGHTS),
            _ => {
                return Err(ApiError::bad(
                    "Expected one metadata part and one weights part",
                ));
            }
        };
        let mut bytes = Vec::new();
        while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
            if bytes.len() + chunk.len() > limit {
                return Err(ApiError(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "Crosscoder part exceeds size limit".into(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        *target = Some(bytes);
    }
    let metadata = serde_json::from_slice(
        &metadata.ok_or_else(|| ApiError::bad("Missing crosscoder metadata"))?,
    )
    .map_err(|e| ApiError::bad(format!("Invalid crosscoder metadata: {e}")))?;
    Ok((
        metadata,
        weights.ok_or_else(|| ApiError::bad("Missing crosscoder weights"))?,
    ))
}

async fn import(State(app): State<Arc<App>>, body: Upload) -> Result<(StatusCode, Json<Value>)> {
    let permit = slot()?;
    let (meta, weights) = tokio::time::timeout(Duration::from_secs(120), read_parts(body))
        .await
        .map_err(|_| {
            ApiError(
                StatusCode::REQUEST_TIMEOUT,
                "Crosscoder upload timed out".into(),
            )
        })??;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        save(&app, meta, weights)
    })
    .await
    .map_err(ApiError::storage)?
}

fn save(app: &App, meta: Metadata, weights: Vec<u8>) -> Result<(StatusCode, Json<Value>)> {
    meta.check()?;
    let features = meta.features(&weights)?;
    let weights_hash = format!("{:x}", Sha256::digest(&weights));
    let mut hash = Sha256::new();
    hash.update(b"brewery-crosscoder.v1\0");
    hash.update(weights_hash.as_bytes());
    hash.update(serde_json::to_vec(&meta).map_err(ApiError::storage)?);
    let id = format!("{:x}", hash.finalize());
    let store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(report) = store.crosscoder(&id).map_err(ApiError::storage)? {
        return Ok((StatusCode::OK, Json(report)));
    }
    let manifests = meta.trace_manifests(&store)?;
    let scratch_hashes = Sides {
        parent: scratch_hash(&store, &meta.parent.id)?,
        child: scratch_hash(&store, &meta.child.id)?,
    };
    let mut value = serde_json::to_value(meta).map_err(ApiError::storage)?;
    value
        .as_object_mut()
        .unwrap()
        .extend(serde_json::Map::from_iter([
            ("id".into(), json!(id)),
            ("version".into(), json!(1)),
            ("qualification".into(), json!("not release-qualified")),
            ("method".into(), json!(METHOD)),
            ("executable_sha256".into(), json!(trace::SHA256)),
            ("manifests".into(), json!(manifests)),
            ("scratch_revision_sha256".into(), json!(scratch_hashes)),
            ("weights_sha256".into(), json!(weights_hash)),
            ("weights_bytes".into(), json!(weights.len())),
            ("unique_norm_ratio".into(), json!(UNIQUE)),
            ("changed_cosine_below".into(), json!(CHANGED)),
            ("features".into(), json!(features)),
        ]));
    store
        .save_crosscoder(&id, &value, &weights)
        .map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(value)))
}

fn scratch_hash(store: &Store, id: &str) -> Result<Option<String>> {
    Ok(store
        .scratch(id)
        .map_err(ApiError::storage)?
        .map(|r| r.sha256))
}

async fn report(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<Json<Value>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(lookup(&store, &id)?))
}

fn lookup(store: &Store, id: &str) -> Result<Value> {
    if !is_digest(id) {
        return Err(ApiError::bad("Expected a crosscoder SHA-256 ID"));
    }
    store
        .crosscoder(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("crosscoder artifact"))
}

async fn export(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<impl IntoResponse> {
    let permit = slot()?;
    let (weights, permit) = tokio::task::spawn_blocking(move || -> Result<_> {
        let store = app.store.lock().map_err(ApiError::storage)?;
        lookup(&store, &id)?;
        let weights = store.crosscoder_weights(&id).map_err(ApiError::storage)?;
        Ok((axum::body::Bytes::from(weights), permit))
    })
    .await
    .map_err(ApiError::storage)??;
    let length = weights.len().to_string();
    let stream =
        futures_util::stream::unfold((weights, permit), |(mut bytes, permit)| async move {
            if bytes.is_empty() {
                return None;
            }
            let chunk = bytes.split_to(bytes.len().min(65536));
            Some((Ok::<_, std::convert::Infallible>(chunk), (bytes, permit)))
        });
    Ok((
        [
            (header::CONTENT_TYPE, "application/octet-stream".into()),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"crosscoder.safetensors\"".into(),
            ),
            (header::CONTENT_LENGTH, length),
        ],
        axum::body::Body::from_stream(stream),
    ))
}

impl Metadata {
    fn check(&self) -> Result<()> {
        let n = self.seen.len();
        let training = &self.training;
        if self.schema_version != 1
            || self.layer >= 4096
            || self.role == trace::Role::KqSoftMax
            || !(1..=4096).contains(&self.width)
            || !(1..=4096).contains(&n)
            || !(1..=n).contains(&self.k)
            || !self.scale.is_finite()
            || self.scale <= 0.
            || !(2..=64).contains(&self.pairs.len())
            || self.parent.sha256 == self.child.sha256
            || [&self.parent, &self.child]
                .iter()
                .any(|b| !uuid(&b.id) || !is_digest(&b.sha256))
            || !is_digest(&training.dataset_sha256)
            || training.torch != "2.7.1+cpu"
            || !(1..=10000).contains(&training.steps)
            || !(1.0..=100.0).contains(&training.difference_weight)
            || [training.train_rows, training.heldout_rows]
                .iter()
                .any(|r| !(2..=16384).contains(r))
        {
            return Err(ApiError::bad(
                "Invalid crosscoder binding or training parameters",
            ));
        }
        for m in [&self.metrics.parent, &self.metrics.child] {
            if !m.mse.is_finite()
                || m.mse < 0.
                || !m.explained_variance.is_finite()
                || m.explained_variance > 1.
                || !(0.0..=self.k as f64).contains(&m.l0)
                || m.dead_features > n
            {
                return Err(ApiError::bad("Invalid crosscoder metrics"));
            }
        }
        Ok(())
    }

    fn trace_manifests(&self, store: &Store) -> Result<Vec<Sides<Vec<String>>>> {
        let mut jobs = BTreeSet::new();
        let mut train = BTreeSet::new();
        let mut heldout = BTreeSet::new();
        let mut manifests = Vec::new();
        for pair in &self.pairs {
            let mut read = |id: &str, binding: &Binding| -> Result<_> {
                if !uuid(id) || !jobs.insert(id.to_owned()) {
                    return Err(ApiError::bad("Invalid or repeated crosscoder trace job"));
                }
                let job = store
                    .job(id)
                    .map_err(ApiError::storage)?
                    .filter(|j| j.kind == trace::JOB && j.state == JobState::Succeeded)
                    .ok_or_else(|| ApiError::not_found("crosscoder trace job"))?;
                let receipt = store
                    .job_result(id)
                    .map_err(ApiError::storage)?
                    .ok_or_else(|| ApiError::not_found("crosscoder trace receipt"))?;
                let env = store.envelope(id).map_err(ApiError::storage)?;
                let options = Value::Object(env.options);
                let nonempty = |key: &str| {
                    options
                        .get(key)
                        .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
                };
                if job.input_ids.as_slice() != [binding.id.as_str()]
                    || receipt["model_id"] != binding.id
                    || receipt["model_sha256"] != binding.sha256
                    || receipt["kind"] != trace::JOB
                    || receipt["schema_version"] != 1
                    || receipt["executable_sha256"] != trace::SHA256
                    || nonempty("interventions")
                    || !options["layers"]
                        .as_array()
                        .is_some_and(|v| v.contains(&json!(self.layer)))
                    || !options["roles"]
                        .as_array()
                        .is_some_and(|v| v.contains(&json!(self.role)))
                {
                    return Err(ApiError::bad(
                        "Trace receipt or capture does not match the crosscoder binding",
                    ));
                }
                let prompts: Vec<String> = match env.prompt {
                    Some(text) => vec![text],
                    None => serde_json::from_value(options["prompts"].clone())
                        .map_err(ApiError::storage)?,
                };
                let hashes: Vec<String> = receipt["prompts"]
                    .as_array()
                    .filter(|v| v.len() == prompts.len())
                    .and_then(|values| {
                        values
                            .iter()
                            .map(|v| {
                                v["manifest_sha256"]
                                    .as_str()
                                    .filter(|s| is_digest(s))
                                    .map(str::to_owned)
                            })
                            .collect()
                    })
                    .ok_or_else(|| ApiError::bad("Trace receipt is missing manifest hashes"))?;
                Ok((prompts, options["positions"].clone(), hashes))
            };
            let (parent_text, parent_positions, parent) = read(&pair.parent_job_id, &self.parent)?;
            let (child_text, child_positions, child) = read(&pair.child_job_id, &self.child)?;
            if parent_text != child_text || parent_positions != child_positions {
                return Err(ApiError::bad(
                    "Crosscoder trace pairs require matching prompts and positions",
                ));
            }
            match pair.split {
                Split::Train => train.extend(parent_text),
                Split::Heldout => heldout.extend(parent_text),
            }
            manifests.push(Sides { parent, child });
        }
        if train.is_empty() || heldout.is_empty() || !train.is_disjoint(&heldout) {
            return Err(ApiError::bad("Expected disjoint train and holdout prompts"));
        }
        Ok(manifests)
    }

    fn features(&self, weights: &[u8]) -> Result<Vec<Feature>> {
        let header = Header::read(weights, weights.len() as u64, MAX_METADATA as u64)
            .map_err(|err| ApiError::bad(format!("Invalid crosscoder SafeTensors: {err}")))?;
        let width = self.width;
        let count = self.seen.len() as u64;
        let expected = [
            ("W_enc", vec![width * 2, count]),
            ("W_dec", vec![2, count, width]),
            ("b_enc", vec![count]),
            ("b_dec", vec![2, width]),
        ];
        if header.entries.len() != expected.len() {
            return Err(ApiError::bad("Unexpected crosscoder tensor names"));
        }
        let mut decoder = &[][..];
        for (name, shape) in expected {
            let t = header
                .entries
                .get(name)
                .ok_or_else(|| ApiError::bad(format!("Missing crosscoder tensor {name}")))?;
            if t.dtype != "F32" || t.shape != shape {
                return Err(ApiError::bad(format!(
                    "Wrong dtype or shape for crosscoder tensor {name}"
                )));
            }
            let [start, end] = t.data_offsets.map(|n| (n + header.payload_start) as usize);
            let bytes = &weights[start..end];
            if floats(bytes).any(|v| !v.is_finite()) {
                return Err(ApiError::bad(format!(
                    "Invalid value in crosscoder tensor {name}"
                )));
            }
            if name == "W_dec" {
                decoder = bytes;
            }
        }
        let stride = width as usize * 4;
        let (parent, child) = decoder.split_at(decoder.len() / 2);
        let mut dead = [0; 2];
        let mut features = Vec::with_capacity(count as usize);
        for (i, (p, c)) in parent
            .chunks_exact(stride)
            .zip(child.chunks_exact(stride))
            .enumerate()
        {
            let (p2, c2, dot) = floats(p)
                .zip(floats(c))
                .fold((0., 0., 0.), |(p2, c2, dot), (p, c)| {
                    (p2 + p * p, c2 + c * c, dot + p * c)
                });
            let (pn, cn) = (p2.sqrt(), c2.sqrt());
            dead[0] += usize::from(!self.seen[i] || pn <= SUPPORT);
            dead[1] += usize::from(!self.seen[i] || cn <= SUPPORT);
            let ratio = (pn + cn > SUPPORT).then(|| cn / (pn + cn));
            let cosine = (pn > SUPPORT && cn > SUPPORT).then(|| (dot / (pn * cn)).clamp(-1., 1.));
            let classification = match (self.seen[i], ratio, cosine) {
                (false, _, _) | (_, None, _) => "dead",
                (_, Some(r), _) if r <= UNIQUE => "parent_unique",
                (_, Some(r), _) if r >= 1. - UNIQUE => "child_unique",
                (_, _, Some(c)) if c < CHANGED => "changed",
                _ => "shared",
            };
            features.push(Feature {
                feature: i,
                classification,
                child_norm_ratio: ratio,
                decoder_cosine: cosine,
                parent_norm: pn,
                child_norm: cn,
            });
        }
        if dead
            != [
                self.metrics.parent.dead_features,
                self.metrics.child.dead_features,
            ]
        {
            return Err(ApiError::bad(
                "Crosscoder dead counts do not match decoder support and training activity",
            ));
        }
        Ok(features)
    }
}

fn floats(bytes: &[u8]) -> impl Iterator<Item = f64> + '_ {
    bytes
        .chunks_exact(4)
        .map(|c| f64::from(f32::from_le_bytes(c.try_into().expect("four bytes"))))
}
