//! An immutable direction artifact is stored atomically as its job result.
use crate::{App, domain::State as JobState, store::Store, web::ApiError};
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::{get, post},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use brewery_worker::{behavior_direction as direction, trace};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub(crate) mod import;

pub fn router() -> Router<Arc<App>> {
    Router::new()
        .route("/directions", get(list))
        .route("/directions/export", get(export))
        .route("/directions/{id}/versions/{version}", get(get_version))
        .route("/admin/directions/import", post(import::upload))
}

async fn list(State(app): State<Arc<App>>) -> Result<Json<Value>, ApiError> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(
        json!({"directions": store.direction_summaries().map_err(ApiError::storage)?}),
    ))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    layer: Option<usize>,
    reference_id: Option<String>,
    reference_layer: Option<usize>,
}

async fn get_version(
    State(app): State<Arc<App>>,
    Path((id, version)): Path<(String, u32)>,
    Query(selection): Query<Selection>,
) -> Result<Json<Value>, ApiError> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    let result = artifact(&store, &id, version, selection.layer)?;
    let reference = match (selection.reference_id, selection.reference_layer) {
        (Some(id), Some(layer)) if !id.is_empty() && id.len() <= 128 && layer < 4096 => {
            Some(compare_reference(&store, &result, &id, layer)?)
        }
        (None, None) => None,
        _ => {
            return Err(ApiError::bad(
                "provide both a valid reference_id and reference_layer",
            ));
        }
    };
    let mut response = json!({"id":id, "version":version, "direction":result});
    if let Some(reference) = reference {
        response["reference"] = reference;
    }
    Ok(Json(response))
}

fn compare_reference(
    store: &Store,
    result: &Value,
    id: &str,
    layer: usize,
) -> Result<Value, ApiError> {
    let reference = artifact(store, id, direction::VERSION, Some(layer))?;
    if result["origin"] == "imported" || reference["origin"] == "imported" {
        if result["schema_version"] != direction::VERSION
            || reference["model_sha256"] != result["model_sha256"]
        {
            return Err(ApiError::bad(
                "compare version 2 directions for the same checkpoint hash",
            ));
        }
    } else if result["runtime"] != trace::HGN_RUNTIME
        || reference["runtime"] != trace::HGN_RUNTIME
        || reference["behavior"] != result["behavior"]
        || reference["model_sha256"] != result["model_sha256"]
        || reference["assets_sha256"] != result["assets_sha256"]
    {
        return Err(ApiError::bad(
            "reference must use the same HGN checkpoint and reviewed behavior",
        ));
    }
    let row = &reference["fit"]["layers"][0];
    let target: direction::Layer =
        serde_json::from_value(row.clone()).map_err(ApiError::storage)?;
    let target = target
        .unit_vector
        .ok_or_else(|| ApiError::bad("reference layer has no direction"))?;
    let fit: direction::Fit =
        serde_json::from_value(result["fit"].clone()).map_err(ApiError::storage)?;
    let target_norm = direction::residual_norm(&target);
    let mut scores = Vec::with_capacity(fit.layers.len());
    for row in &fit.layers {
        let cosine = if let Some(v) = &row.unit_vector {
            if v.len() != target.len() {
                return Err(ApiError::bad("reference direction width mismatch"));
            }
            let dot = v
                .iter()
                .zip(&target)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>();
            Some((dot / (direction::residual_norm(v) * target_norm)).clamp(-1.0, 1.0))
        } else {
            None
        };
        scores.push(json!({"layer":row.layer,"cosine":cosine}));
    }
    Ok(json!({"id":id,"version":direction::VERSION,"layer":layer,"cosine_by_layer":scores}))
}

pub(crate) fn artifact(
    store: &Store,
    id: &str,
    version: u32,
    layer: Option<usize>,
) -> Result<Value, ApiError> {
    let mut result = load(store, id, version)?;
    if let Some(layer) = layer {
        let row = result["fit"]["layers"]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["layer"] == layer))
            .ok_or_else(|| ApiError::not_found("direction layer"))?;
        result["fit"]["layers"] = json!([row]);
    }
    Ok(result)
}

pub(crate) fn load(store: &Store, id: &str, version: u32) -> Result<Value, ApiError> {
    if !matches!(version, 1 | direction::VERSION) {
        return Err(ApiError::not_found("direction version"));
    }
    let job = store
        .job(id)
        .map_err(ApiError::storage)?
        .filter(|job| {
            matches!(job.kind.as_str(), direction::JOB | import::JOB)
                && job.state == JobState::Succeeded
        })
        .ok_or_else(|| ApiError::not_found("direction"))?;
    let result = store
        .job_result(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("direction result"))?;
    if result["schema_version"] != version {
        return Err(ApiError::not_found("direction version"));
    }
    if job.kind == import::JOB {
        // Only the authenticated controller writes this terminal kind; no
        // generic job submission or worker result can create it.
        return Ok(result);
    }
    let request = direction::Request::parse(&json!(job.options)).map_err(ApiError::storage)?;
    let envelope = store.envelope(id).map_err(ApiError::storage)?;
    let input = envelope
        .inputs
        .first()
        .ok_or_else(|| ApiError::storage("Direction input is missing"))?;
    request
        .verify_version(
            &result,
            &input.id,
            &request.model_sha256,
            &input.kind,
            version,
        )
        .map_err(ApiError::storage)?;
    Ok(result)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportQuery {
    id: String,
    layer: Option<usize>,
}

/// Export a stored fitted direction as the v2 operator-import bundle that
/// `/admin/directions/import` consumes: model_id, manifest, and the base64 .npy.
async fn export(
    State(app): State<Arc<App>>,
    Query(query): Query<ExportQuery>,
) -> Result<Json<Value>, ApiError> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    let result = load(&store, &query.id, direction::VERSION)?;
    if result["origin"] == "imported" {
        return Err(ApiError::bad(
            "only Brewery-fitted directions can be exported as a v2 bundle",
        ));
    }
    let model_id = result["model_id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let model_sha256 = result["model_sha256"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let rows = result["fit"]["layers"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let row = if let Some(layer) = query.layer {
        rows
            .into_iter()
            .find(|row| row["layer"] == layer)
            .ok_or_else(|| ApiError::bad("direction has no such fitted layer"))?
    } else {
        let selected = result["fit"]["selected_layer"].clone();
        rows.into_iter()
            .find(|row| row["layer"] == selected)
            .ok_or_else(|| ApiError::bad("direction has no selected layer"))?
    };
    let vector: Option<Vec<f32>> =
        serde_json::from_value(row["unit_vector"].clone()).map_err(ApiError::storage)?;
    let vector = vector.ok_or_else(|| ApiError::bad("direction layer has no direction"))?;
    let layer = row["layer"].as_u64().unwrap_or(0) as usize;
    let mut npy = Vec::with_capacity(12 + vector.len() * 4);
    npy.extend_from_slice(b"\x93NUMPY\x01\x00");
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}", vector.len());
    header.push_str(&" ".repeat(64 - (11 + header.len()) % 64));
    header.push('\n');
    npy.extend_from_slice(&(header.len() as u16).to_le_bytes());
    npy.extend_from_slice(header.as_bytes());
    for value in &vector {
        npy.extend_from_slice(&value.to_le_bytes());
    }
    let npy_sha256 = format!("{:x}", Sha256::digest(&npy));
    let position = result["position"].as_i64().map(|v| v as i32);
    let manifest = json!({
        "layer": layer,
        "position": position,
        "model_sha256": model_sha256,
        "npy_sha256": npy_sha256,
        "provenance": format!("brewery:directions/{}:v{}", query.id, direction::VERSION),
    });
    Ok(Json(json!({
        "id": query.id,
        "version": direction::VERSION,
        "model_id": model_id,
        "manifest": manifest,
        "npy_base64": STANDARD.encode(&npy),
        "bytes": npy.len(),
    })))
}

pub(crate) struct HgnLayer {
    pub sha256: String,
    pub behavior: Option<direction::Behavior>,
    pub registered: Option<direction::Registered>,
    pub vector: Vec<f32>,
    pub assets: trace::hgn_projection::AssetPins,
    pub held_out_pairs: usize,
    pub last_k: Option<usize>,
}

pub(crate) fn hgn_layer(
    store: &Store,
    model: &crate::domain::Model,
    direction_id: &str,
    source_layer: usize,
) -> Result<HgnLayer, ApiError> {
    let artifact = artifact(store, direction_id, direction::VERSION, Some(source_layer))?;
    if artifact["origin"] == "imported" {
        return Err(ApiError::bad(
            "HGN screening and trace require a Brewery-fitted direction",
        ));
    }
    let sha = artifact["model_sha256"]
        .as_str()
        .expect("verified artifact");
    if !trace::is_hgn(&model.kind)
        || artifact["runtime"] != trace::HGN_RUNTIME
        || artifact["model_id"] != model.id
        || model.sha256.as_deref().is_some_and(|pin| pin != sha)
    {
        return Err(ApiError::bad(
            "fit the version 2 direction on the same pinned HGN Library model",
        ));
    }
    let fit: direction::Fit =
        serde_json::from_value(artifact["fit"].clone()).map_err(ApiError::storage)?;
    let vector = fit
        .layers
        .into_iter()
        .next()
        .and_then(|layer| layer.unit_vector)
        .ok_or_else(|| ApiError::bad("the source layer has no nonzero fitted direction"))?;
    Ok(HgnLayer {
        sha256: sha.into(),
        behavior: serde_json::from_value(artifact["behavior"].clone())
            .map_err(ApiError::storage)?,
        registered: serde_json::from_value(
            artifact.get("registered").cloned().unwrap_or(Value::Null),
        )
        .map_err(ApiError::storage)?,
        vector,
        assets: serde_json::from_value(artifact["assets_sha256"].clone())
            .map_err(ApiError::storage)?,
        held_out_pairs: fit.held_out_pairs,
        last_k: artifact["last_k"].as_u64().map(|v| v as usize),
    })
}
