use crate::{
    App,
    domain::{self, State as JobState},
    idempotency::Replay,
    interventions::trace_plan::Rows,
    store::Store,
    web::{ApiError, read_submission},
};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Multipart, Path, Query, State, multipart::MultipartRejection},
    http::{HeaderMap, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use brewery_worker::{
    is_digest,
    trace::{self, Operation, ReadReply, Role, Rule, Slice, View},
    weight_source::Header,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeSet, io::Read, sync::Arc, time::Duration};
use tokio::sync::Semaphore;

type Result<T> = std::result::Result<T, ApiError>;
const MAX_WEIGHTS: usize = 128 << 20;
const MAX_METADATA: usize = 32 << 10;
const MAX_ROWS: usize = 16;
static IO_SLOTS: Semaphore = Semaphore::const_new(1);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    schema_version: u32,
    name: String,
    model_id: String,
    model_sha256: String,
    layer: usize,
    role: Role,
    d_in: usize,
    d_sae: usize,
    activation: Activation,
    normalization: Normalization,
    provenance: Map<String, Value>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Activation {
    Relu,
    TopK { k: usize },
    JumpRelu,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Normalization {
    scale: f32,
    subtract_decoder_bias: bool,
}

#[derive(Serialize, Deserialize)]
pub struct Artifact {
    pub id: String,
    pub version: u32,
    created_at_ms: u64,
    weights_sha256: String,
    weights_bytes: usize,
    #[serde(flatten)]
    pub metadata: Metadata,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    id: String,
    version: u32,
    top_k: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Steering {
    id: String,
    version: u32,
    feature: usize,
    alpha: f32,
    positions: BTreeSet<usize>,
}

pub fn router() -> Router<Arc<App>> {
    Router::new()
        .route("/saes", get(list).post(create).layer(upload_limit()))
        .route(
            "/saes/{id}/versions",
            post(new_version).layer(upload_limit()),
        )
        .route("/saes/{id}/versions/{version}", get(get_version))
        .route("/saes/{id}/versions/{version}/safetensors", get(export))
        .route("/jobs/{id}/trace/features", get(features))
}

fn upload_limit() -> DefaultBodyLimit {
    DefaultBodyLimit::max(MAX_WEIGHTS + MAX_METADATA + 8192)
}

fn slot() -> Result<tokio::sync::SemaphorePermit<'static>> {
    IO_SLOTS.try_acquire().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "Another SAE request is active".into(),
        )
    })
}

async fn list(State(app): State<Arc<App>>) -> Result<Json<Vec<Artifact>>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(store.saes().map_err(ApiError::storage)?))
}

fn artifact(store: &Store, id: &str, version: u32) -> Result<Artifact> {
    store
        .sae(id, version)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("SAE version"))
}

async fn get_version(
    State(app): State<Arc<App>>,
    Path((id, version)): Path<(String, u32)>,
) -> Result<Json<Artifact>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(artifact(&store, &id, version)?))
}

async fn export(
    State(app): State<Arc<App>>,
    Path((id, version)): Path<(String, u32)>,
) -> Result<impl IntoResponse> {
    let permit = slot()?;
    let (weights, permit) = tokio::task::spawn_blocking(move || -> Result<_> {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let value = artifact(&store, &id, version)?;
        let mut blob = store.sae_blob(&id, version).map_err(ApiError::storage)?;
        let mut weights = vec![0; value.weights_bytes];
        blob.read_exact(&mut weights).map_err(ApiError::storage)?;
        Ok((Bytes::from(weights), permit))
    })
    .await
    .map_err(ApiError::storage)??;
    let size = weights.len().to_string();
    let stream =
        futures_util::stream::unfold((weights, permit), |(mut bytes, permit)| async move {
            if bytes.is_empty() {
                return None;
            }
            let chunk = bytes.split_to(bytes.len().min(65536));
            Some((Ok::<_, std::convert::Infallible>(chunk), (bytes, permit)))
        });
    let headers = [
        (header::CONTENT_TYPE, "application/octet-stream".to_owned()),
        (
            header::CONTENT_DISPOSITION,
            "attachment; filename=\"sae.safetensors\"".to_owned(),
        ),
        (header::CONTENT_LENGTH, size),
    ];
    Ok((headers, Body::from_stream(stream)))
}

type Upload = std::result::Result<Multipart, MultipartRejection>;
type Mutation = Result<(StatusCode, Json<Artifact>)>;

async fn create(State(app): State<Arc<App>>, headers: HeaderMap, body: Upload) -> Mutation {
    upload(app, None, headers, body).await
}

async fn new_version(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Upload,
) -> Mutation {
    upload(app, Some(id), headers, body).await
}

async fn upload(app: Arc<App>, id: Option<String>, headers: HeaderMap, body: Upload) -> Mutation {
    let permit = slot()?;
    let (metadata, weights) = tokio::time::timeout(Duration::from_secs(120), upload_parts(body))
        .await
        .map_err(|_| ApiError(StatusCode::REQUEST_TIMEOUT, "SAE upload timed out".into()))??;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        save(&app, id, &headers, metadata, weights)
    })
    .await
    .map_err(ApiError::storage)?
}

async fn upload_parts(body: Upload) -> Result<(Metadata, Vec<u8>)> {
    let mut parts = body.map_err(|err| ApiError::bad(err.to_string()))?;
    let mut metadata = None;
    let mut weights = None;
    while let Some(mut field) = parts.next_field().await.map_err(upload_error)? {
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
        while let Some(chunk) = field.chunk().await.map_err(upload_error)? {
            if bytes.len() + chunk.len() > limit {
                return Err(ApiError(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "SAE part exceeds size limit".into(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        *target = Some(bytes);
    }
    let metadata: Metadata =
        serde_json::from_slice(&metadata.ok_or_else(|| ApiError::bad("Missing SAE metadata"))?)
            .map_err(|err| ApiError::bad(format!("Invalid SAE metadata: {err}")))?;
    let weights = weights.ok_or_else(|| ApiError::bad("Missing SAE weights"))?;
    Ok((metadata, weights))
}

fn upload_error(error: axum::extract::multipart::MultipartError) -> ApiError {
    ApiError(error.status(), error.body_text())
}

fn save(
    app: &App,
    id: Option<String>,
    headers: &HeaderMap,
    metadata: Metadata,
    weights: Vec<u8>,
) -> Mutation {
    metadata.check()?;
    let header = Header::read(
        weights.as_slice(),
        weights.len() as u64,
        MAX_METADATA as u64,
    )
    .map_err(|err| ApiError::bad(format!("Invalid SAE SafeTensors: {err}")))?;
    metadata.check_weights(&header, &weights)?;
    let digest = format!("{:x}", Sha256::digest(&weights));
    let scope = if id.is_some() {
        "sae-version"
    } else {
        "sae-create"
    };
    let (_, submission) = read_submission(headers, scope, Ok(Json((&id, &metadata, &digest))))?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(submission) = submission.as_ref() {
        match store.replay(submission).map_err(ApiError::storage)? {
            Replay::New => {}
            Replay::Matched(key) => {
                let (id, version) = key
                    .split_once('/')
                    .and_then(|(id, version)| Some((id, version.parse().ok()?)))
                    .ok_or_else(|| ApiError::storage("invalid SAE version replay"))?;
                return Ok((StatusCode::OK, Json(artifact(&store, id, version)?)));
            }
            Replay::Conflict => {
                return Err(ApiError(
                    StatusCode::CONFLICT,
                    "Idempotency-Key already used for a different request".into(),
                ));
            }
        }
    }
    if !store
        .model(&metadata.model_id)
        .map_err(ApiError::storage)?
        .is_some_and(|model| model.kind == "gguf")
    {
        let (_, sha) = crate::scratch::jobs::trace_input(&store, &metadata.model_id)
            .map_err(|_| ApiError::bad("SAE needs a Library or private scratch GGUF"))?;
        if sha != metadata.model_sha256 {
            return Err(ApiError::bad("SAE must match the private GGUF hash"));
        }
    }
    let (id, version) = if let Some(id) = id {
        let prior = store
            .latest_sae(&id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("SAE artifact"))?;
        if prior.metadata.binding() != metadata.binding() {
            return Err(ApiError::bad(
                "An SAE version cannot change the model, layer, role, or input width",
            ));
        }
        (
            id,
            prior
                .version
                .checked_add(1)
                .ok_or_else(|| ApiError::storage("SAE version overflow"))?,
        )
    } else {
        (domain::id(), 1)
    };
    let value = Artifact {
        id,
        version,
        created_at_ms: domain::now(),
        weights_sha256: digest,
        weights_bytes: weights.len(),
        metadata,
    };
    store
        .insert_sae(&value, &weights, submission.as_ref())
        .map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(value)))
}

impl Metadata {
    fn binding(&self) -> (&str, &str, usize, Role, usize) {
        (
            &self.model_id,
            &self.model_sha256,
            self.layer,
            self.role,
            self.d_in,
        )
    }

    fn check(&self) -> Result<()> {
        if self.schema_version != 1
            || self.name.trim().is_empty()
            || self.name.len() > 160
            || self.name.chars().any(char::is_control)
            || !is_digest(&self.model_sha256)
            || self.layer >= 4096
            || self.role.is_read_only()
            || !(1..=16384).contains(&self.d_in)
            || !(1..=65536).contains(&self.d_sae)
            || !self.normalization.scale.is_finite()
            || self.normalization.scale <= 0.
            || !self
                .provenance
                .get("method")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.trim().is_empty())
            || matches!(self.activation, Activation::TopK { k } if k == 0 || k > self.d_sae)
        {
            return Err(ApiError::bad(
                "Invalid SAE binding, dimensions, activation, normalization or provenance",
            ));
        }
        Ok(())
    }

    fn check_weights(&self, header: &Header, weights: &[u8]) -> Result<()> {
        let mut expected = vec![
            ("W_enc", vec![self.d_in as u64, self.d_sae as u64]),
            ("W_dec", vec![self.d_sae as u64, self.d_in as u64]),
            ("b_enc", vec![self.d_sae as u64]),
            ("b_dec", vec![self.d_in as u64]),
        ];
        if matches!(self.activation, Activation::JumpRelu) {
            expected.push(("threshold", vec![self.d_sae as u64]));
        }
        if header.entries.len() != expected.len() {
            return Err(ApiError::bad("Unexpected SAE tensor names"));
        }
        for (name, shape) in expected {
            let tensor = header
                .entries
                .get(name)
                .ok_or_else(|| ApiError::bad(format!("Missing SAE tensor {name}")))?;
            if tensor.dtype != "F32" || tensor.shape != shape {
                return Err(ApiError::bad(format!(
                    "Wrong dtype or shape for SAE tensor {name}"
                )));
            }
            let [start, end] = tensor
                .data_offsets
                .map(|n| (n + header.payload_start) as usize);
            if floats(&weights[start..end])
                .any(|v| !v.is_finite() || (name == "threshold" && v < 0.))
            {
                return Err(ApiError::bad(format!("Invalid value in SAE tensor {name}")));
            }
        }
        Ok(())
    }
}

fn floats(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("four bytes")))
}

fn pin(a: &Artifact, model: &str, digest: &mut Option<String>) -> Result<()> {
    let m = &a.metadata;
    if m.model_id != model || digest.as_ref().is_some_and(|sha| sha != &m.model_sha256) {
        return Err(ApiError::bad(
            "SAE version is bound to a different model or GGUF SHA-256",
        ));
    }
    *digest = Some(m.model_sha256.clone());
    Ok(())
}

impl Selection {
    pub fn prepare(
        &self,
        store: &Store,
        model: &str,
        digest: &mut Option<String>,
        options: &Map<String, Value>,
    ) -> Result<()> {
        let a = artifact(store, &self.id, self.version)?;
        pin(&a, model, digest)?;
        if !(1..=100).contains(&self.top_k)
            || !options["layers"]
                .as_array()
                .is_some_and(|x| x.contains(&json!(a.metadata.layer)))
            || !options["roles"]
                .as_array()
                .is_some_and(|x| x.contains(&json!(a.metadata.role)))
        {
            return Err(ApiError::bad(
                "SAE trace needs top_k 1 to 100 and must capture the bound layer and role",
            ));
        }
        Ok(())
    }
}

impl Steering {
    pub fn rule(self, store: &Store, model: &str, digest: &mut Option<String>) -> Result<Rule> {
        let a = artifact(store, &self.id, self.version)?;
        pin(&a, model, digest)?;
        let m = &a.metadata;
        if self.feature >= m.d_sae || !self.alpha.is_finite() {
            return Err(ApiError::bad("Invalid SAE feature or steering alpha"));
        }
        let (blob, header) = open_weights(store, &a)?;
        let offset = header.payload_start + header.entries["W_dec"].data_offsets[0];
        let mut bytes = vec![0; 4 * m.d_in];
        let start = offset as usize + self.feature * bytes.len();
        blob.read_at_exact(&mut bytes, start)
            .map_err(ApiError::storage)?;
        let vector = floats(&bytes).map(|v| self.alpha * v).collect();
        Ok(Rule {
            layer: m.layer,
            role: m.role,
            positions: self.positions,
            op: Operation::Add { vector },
        })
    }
}

fn open_weights<'s>(store: &'s Store, a: &Artifact) -> Result<(rusqlite::blob::Blob<'s>, Header)> {
    let mut blob = store
        .sae_blob(&a.id, a.version)
        .map_err(ApiError::storage)?;
    let len = blob.len() as u64;
    let header = Header::read(&mut blob, len, MAX_METADATA as u64).map_err(ApiError::storage)?;
    Ok((blob, header))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureQuery {
    #[serde(default)]
    prompt: usize,
    start: usize,
    end: usize,
}

async fn features(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    query: std::result::Result<Query<FeatureQuery>, axum::extract::rejection::QueryRejection>,
) -> Result<Json<Value>> {
    let Query(q) = query.map_err(|_| ApiError::bad("Invalid SAE feature trace query"))?;
    if !q
        .end
        .checked_sub(q.start)
        .is_some_and(|n| (1..=MAX_ROWS).contains(&n))
    {
        return Err(ApiError::bad("Select 1 to 16 trace positions"));
    }
    let (job, selection, a) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let job = store
            .job(&id)
            .map_err(ApiError::storage)?
            .filter(|job| job.kind == trace::JOB && job.state == JobState::Succeeded)
            .ok_or_else(|| ApiError::not_found("trace result"))?;
        let selection: Selection = serde_json::from_value(
            job.options
                .get("sae")
                .cloned()
                .ok_or_else(|| ApiError::not_found("SAE trace selection"))?,
        )
        .map_err(ApiError::storage)?;
        let a = artifact(&store, &selection.id, selection.version)?;
        let receipt = store
            .job_result(&id)
            .map_err(ApiError::storage)?
            .filter(|r| {
                r["kind"] == trace::JOB
                    && r["schema_version"] == 1
                    && r["model_id"] == a.metadata.model_id
                    && r["model_sha256"] == a.metadata.model_sha256
            })
            .ok_or_else(|| ApiError::not_found("trace receipt for SAE model revision"))?;
        if receipt["prompts"].get(q.prompt).is_none() {
            return Err(ApiError::not_found("trace prompt"));
        }
        (job, selection, a)
    };
    let permit = slot()?;
    let slice = Slice {
        view: View::Tensor,
        prompt: q.prompt,
        layer: Some(a.metadata.layer),
        role: Some(a.metadata.role),
        start: Some(q.start),
        end: Some(q.end),
    };
    slice.check().map_err(|e| ApiError::bad(e.to_string()))?;
    let reply = tokio::time::timeout(
        Duration::from_secs(30),
        app.dispatcher.trace_read(&job, &slice),
    )
    .await
    .map_err(|_| {
        ApiError(
            StatusCode::GATEWAY_TIMEOUT,
            "SAE trace read timed out".into(),
        )
    })?;
    let data = match reply.map_err(|e| {
        eprintln!("SAE trace read failed for {id}: {e}");
        bad_trace()
    })? {
        ReadReply::Ok(data) => data,
        ReadReply::Missing => return Err(ApiError::not_found("activation slice")),
        ReadReply::TooLarge => {
            return Err(ApiError(
                StatusCode::PAYLOAD_TOO_LARGE,
                "Select a smaller SAE trace range".into(),
            ));
        }
    };
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let rows: Rows = serde_json::from_value(data).map_err(|_| bad_trace())?;
        let m = &a.metadata;
        if (rows.prompt, rows.layer, rows.role, rows.dtype.as_str())
            != (q.prompt, m.layer, m.role, "f32_le")
        {
            return Err(bad_trace());
        }
        let encoder = {
            let store = app.store.lock().map_err(ApiError::storage)?;
            Encoder::read(&store, &a)?
        };
        let mut seen = BTreeSet::new();
        let mut top = Vec::new();
        for row in rows.rows {
            if !(q.start..q.end).contains(&row.position) || !seen.insert(row.position) {
                return Err(bad_trace());
            }
            let input = row.vector().map_err(|_| bad_trace())?;
            if input.len() != m.d_in {
                return Err(ApiError::bad(
                    "Captured activation width does not match the SAE",
                ));
            }
            top.push(
                json!({"position":row.position,"top":encoder.encode(m, &input, selection.top_k)?}),
            );
        }
        if rows
            .missing_positions
            .iter()
            .any(|&p| !(q.start..q.end).contains(&p) || !seen.insert(p))
        {
            return Err(bad_trace());
        }
        Ok(Json(
            json!({"schema_version":1, "sae_id":a.id, "version":a.version,
            "prompt":q.prompt, "layer":m.layer, "role":m.role, "rows":top,
            "missing_positions":rows.missing_positions}),
        ))
    })
    .await
    .map_err(ApiError::storage)?
}

fn bad_trace() -> ApiError {
    ApiError(
        StatusCode::BAD_GATEWAY,
        "Cannot verify SAE trace activations".into(),
    )
}

struct Encoder {
    weight: Vec<f32>,
    bias: Vec<f32>,
    decoder_bias: Vec<f32>,
    threshold: Vec<f32>,
}

#[derive(Serialize)]
struct ActiveFeature {
    feature: usize,
    activation: f32,
}

impl Encoder {
    fn read(store: &Store, a: &Artifact) -> Result<Self> {
        let (blob, header) = open_weights(store, a)?;
        let tensor = |name: &str| -> Result<Vec<f32>> {
            let [start, end] = header.entries[name]
                .data_offsets
                .map(|n| (header.payload_start + n) as usize);
            let mut bytes = vec![0; end - start];
            blob.read_at_exact(&mut bytes, start)
                .map_err(ApiError::storage)?;
            Ok(floats(&bytes).collect())
        };
        Ok(Self {
            weight: tensor("W_enc")?,
            bias: tensor("b_enc")?,
            decoder_bias: if a.metadata.normalization.subtract_decoder_bias {
                tensor("b_dec")?
            } else {
                Vec::new()
            },
            threshold: if matches!(a.metadata.activation, Activation::JumpRelu) {
                tensor("threshold")?
            } else {
                Vec::new()
            },
        })
    }

    fn encode(&self, m: &Metadata, input: &[f32], limit: usize) -> Result<Vec<ActiveFeature>> {
        let mut sums = vec![0.; m.d_sae];
        for (i, (x, weights)) in input
            .iter()
            .zip(self.weight.chunks_exact(m.d_sae))
            .enumerate()
        {
            let x = x * m.normalization.scale - self.decoder_bias.get(i).copied().unwrap_or(0.);
            for (sum, w) in sums.iter_mut().zip(weights) {
                *sum += x * w;
            }
        }
        let mut active = Vec::new();
        for (feature, sum) in sums.into_iter().enumerate() {
            let activation = sum + self.bias[feature];
            if !activation.is_finite() {
                return Err(ApiError::bad("SAE activation is not finite"));
            }
            if activation > self.threshold.get(feature).copied().unwrap_or(0.) {
                active.push(ActiveFeature {
                    feature,
                    activation,
                });
            }
        }
        active.sort_unstable_by(|a, b| {
            b.activation
                .total_cmp(&a.activation)
                .then(a.feature.cmp(&b.feature))
        });
        let limit = match m.activation {
            Activation::TopK { k } => k.min(limit),
            _ => limit,
        };
        active.truncate(limit);
        Ok(active)
    }
}
