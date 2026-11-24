//! Operator-only ingestion. No worker or agent can submit the imported job kind.
use crate::{
    App,
    domain::{NewJob, State as JobState},
    web::ApiError,
};
use axum::{
    Json,
    extract::{Multipart, State, multipart::MultipartRejection},
    http::{HeaderMap, StatusCode},
};
use brewery_worker::{behavior_direction as direction, direction_model, is_digest, trace};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub(crate) const JOB: &str = "imported_direction";
const MAX_WIDTH: usize = 16_384;
const MAX_HEADER: usize = 4096;
const MAX_MANIFEST: usize = 16 * 1024;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    layer: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    position: Option<i32>,
    pub(crate) model_sha256: String,
    npy_sha256: String,
    provenance: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Provenance {
    operator: String,
    imported_at_ms: u64,
    pub(crate) manifest: Manifest,
}

impl Provenance {
    pub(crate) fn from_artifact(row: &Value) -> Result<Option<Self>, ApiError> {
        if row["origin"] != "imported" {
            return Ok(None);
        }
        serde_json::from_value(row["import"].clone()).map_err(ApiError::storage)
    }
}

pub(super) async fn upload(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Result<Multipart, MultipartRejection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let operator = app
        .direction_operator
        .as_ref()
        .and_then(|op| op.authenticate(&headers))
        .ok_or_else(|| {
            ApiError(
                StatusCode::FORBIDDEN,
                "direction import requires operator credentials".into(),
            )
        })?;
    let mut body = body.map_err(|_| ApiError::bad("expected a multipart direction upload"))?;
    let (mut model_id, mut manifest, mut vector) = (None, None, None);
    while let Some(field) = body
        .next_field()
        .await
        .map_err(|_| ApiError::bad("invalid direction upload"))?
    {
        let (slot, limit) = match field.name() {
            Some("model_id") => (&mut model_id, 128),
            Some("manifest") => (&mut manifest, MAX_MANIFEST),
            Some("vector") => (&mut vector, MAX_WIDTH * 4 + MAX_HEADER + 12),
            _ => {
                return Err(ApiError::bad(
                    "expected model_id, manifest and vector fields",
                ));
            }
        };
        if slot.is_some() {
            return Err(ApiError::bad("duplicate multipart field"));
        }
        let bytes = field
            .bytes()
            .await
            .map_err(|_| ApiError::bad("invalid direction upload field"))?;
        if bytes.len() > limit {
            return Err(ApiError::bad(
                "direction upload field exceeds its size limit",
            ));
        }
        *slot = Some(bytes);
    }
    let (Some(model_id), Some(manifest), Some(npy)) = (model_id, manifest, vector) else {
        return Err(ApiError::bad(
            "expected model_id, manifest and vector fields",
        ));
    };
    let id = std::str::from_utf8(&model_id).map_err(|_| ApiError::bad("invalid model ID"))?;
    let manifest: Manifest = serde_json::from_slice(&manifest)
        .map_err(|_| ApiError::bad("invalid direction manifest"))?;
    if !is_digest(&manifest.model_sha256)
        || !is_digest(&manifest.npy_sha256)
        || manifest.layer >= 4096
        || manifest.provenance.trim().is_empty()
        || manifest.provenance.len() > 2048
        || manifest.provenance.contains('\0')
    {
        return Err(ApiError::bad("invalid direction manifest values"));
    }
    if format!("{:x}", Sha256::digest(&npy)) != manifest.npy_sha256 {
        return Err(ApiError::bad(".npy SHA-256 does not match manifest"));
    }
    let vector = parse_npy(&npy)
        .ok_or_else(|| ApiError::bad("expected a one-dimensional float32 .npy vector"))?;
    let norm = direction::residual_norm(&vector);
    if vector.iter().any(|x| !x.is_finite()) || (norm - 1.0).abs() > 1e-4 {
        return Err(ApiError::bad(
            "vector must be finite with unit norm within 1e-4",
        ));
    }
    let _permit = app.direction_import_slot.try_acquire().map_err(|_| {
        ApiError(
            StatusCode::CONFLICT,
            "another direction import is checking a checkpoint".into(),
        )
    })?;
    let load_model = |store: &crate::store::Store| {
        store
            .model(id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("Library model"))
    };
    let model = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let model = load_model(&store)?;
        if store.is_scratch_path(std::path::Path::new(&model.path)) {
            return Err(ApiError::bad("select a standalone Library checkpoint"));
        }
        model
    };
    let request = direction_model::Request {
        path: model.path.clone().into(),
        kind: model.kind.clone(),
        sha256: manifest.model_sha256.clone(),
    };
    request.check().map_err(|_| {
        ApiError::bad("select a standalone HGN v2, SafeTensors or GGUF Library model")
    })?;
    if model
        .sha256
        .as_ref()
        .is_some_and(|pin| pin != &manifest.model_sha256)
    {
        return Err(ApiError::bad(
            "model SHA-256 does not match the Library pin",
        ));
    }
    let (mut job, envelope) = NewJob {
        kind: JOB.into(),
        host: None,
        input_ids: vec![model.id.clone()],
        options: Default::default(),
        prompt_id: None,
        prompt: None,
    }
    .records(
        model.host.clone(),
        vec![model.clone()],
        None,
        crate::gguf::Format::Gguf,
    );
    let dimensions = app
        .dispatcher
        .direction_model(&job, &request)
        .await
        .map_err(|err| {
            eprintln!("direction model inspection failed: {err}");
            ApiError(
                StatusCode::BAD_GATEWAY,
                "source worker could not inspect the pinned model".into(),
            )
        })?
        .ok_or_else(|| {
            ApiError::bad("cannot verify the model SHA-256 or read its checkpoint dimensions")
        })?;
    if vector.len() != dimensions.hidden_size || manifest.layer >= dimensions.layer_count {
        return Err(ApiError::bad(
            "vector width or layer does not match the target checkpoint",
        ));
    }

    let store = app.store.lock().map_err(ApiError::storage)?;
    let current = load_model(&store)?;
    if (current.host, current.path, current.kind, current.sha256)
        != (model.host, model.path, model.kind.clone(), model.sha256)
    {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "Library model changed during import".into(),
        ));
    }
    job.created_at = crate::domain::now();
    job.updated_at = job.created_at;
    job.state = JobState::Succeeded;
    job.has_result = true;
    job.progress = "Operator imported a verified direction".into();
    let runtime = if trace::is_hgn(&model.kind) {
        trace::HGN_RUNTIME
    } else {
        &model.kind
    };
    let fit = direction::Fit {
        role: trace::Role::LOut,
        train_pairs: 0,
        held_out_pairs: 0,
        selected_layer: Some(manifest.layer),
        layers: vec![direction::Layer {
            layer: manifest.layer,
            norm,
            unit_vector: Some(vector),
            held_out_margin: None,
            held_out_cohens_d: None,
            mean_residual_norm: None,
            negative_mean: None,
        }],
    };
    let result = json!({
        "kind":direction::JOB, "schema_version":direction::VERSION, "release_qualified":false,
        "origin":"imported", "model_id":model.id, "model_sha256":manifest.model_sha256,
        "runtime":runtime, "position":manifest.position, "fit":fit,
        "import":Provenance {operator:operator.into(),imported_at_ms:job.created_at,manifest},
    });
    store
        .record_import(&job, &envelope, &result)
        .map_err(ApiError::storage)?;
    Ok((
        StatusCode::CREATED,
        Json(json!({"id":job.id, "version":direction::VERSION, "direction":result})),
    ))
}

// Parse only the literal subset emitted by NumPy for 1-D primitive arrays.
// Do not evaluate a Python literal or accept object, record, or array dtypes.
fn parse_npy(data: &[u8]) -> Option<Vec<f32>> {
    let rest = data.strip_prefix(b"\x93NUMPY")?;
    let (n, payload) = match rest {
        [1, 0, a, b, tail @ ..] => (usize::from(u16::from_le_bytes([*a, *b])), tail),
        [2 | 3, 0, a, b, c, d, tail @ ..] => (u32::from_le_bytes([*a, *b, *c, *d]) as usize, tail),
        _ => return None,
    };
    if n > MAX_HEADER {
        return None;
    }
    let (header, values) = payload.split_at_checked(n)?;
    let mut fields = std::str::from_utf8(header.strip_suffix(b"\n")?)
        .ok()?
        .trim_end()
        .strip_prefix('{')?
        .strip_suffix('}')?
        .trim();
    let (mut endian, mut width, mut order) = (None, None, None);
    while !fields.is_empty() {
        let (name, contents) = fields.split_once(':')?;
        let contents = contents.trim_start();
        let tail = match name.trim() {
            "'descr'" | "\"descr\"" if endian.is_none() => {
                let (value, tail) = contents.split_once(',').unwrap_or((contents, ""));
                endian = Some(match value.trim() {
                    "'<f4'" | "\"<f4\"" => false,
                    "'>f4'" | "\">f4\"" => true,
                    _ => return None,
                });
                tail
            }
            "'shape'" | "\"shape\"" if width.is_none() => {
                let (shape, tail) = contents.strip_prefix('(')?.split_once(')')?;
                let count = shape.trim_end().strip_suffix(',')?.trim();
                if count.is_empty() || !count.bytes().all(|byte| byte.is_ascii_digit()) {
                    return None;
                }
                width = Some(count.parse::<usize>().ok()?);
                if tail.trim().is_empty() {
                    ""
                } else {
                    tail.trim_start().strip_prefix(',')?
                }
            }
            "'fortran_order'" | "\"fortran_order\"" if order.is_none() => {
                let (value, tail) = contents.split_once(',').unwrap_or((contents, ""));
                order = Some(match value.trim() {
                    "False" | "True" => (),
                    _ => return None,
                });
                tail
            }
            _ => return None,
        };
        fields = tail.trim_start();
    }
    let (width, big_endian, ()) = (width?, endian?, order?);
    if !(1..=MAX_WIDTH).contains(&width) || values.len() != width * 4 {
        return None;
    }
    Some(
        values
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap();
                if big_endian {
                    f32::from_be_bytes(bytes)
                } else {
                    f32::from_le_bytes(bytes)
                }
            })
            .collect(),
    )
}
