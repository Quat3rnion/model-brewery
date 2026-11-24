//! An immutable direction artifact is stored atomically as its job result.
use crate::{App, domain::State as JobState, store::Store, web::ApiError};
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::get,
};
use brewery_worker::{behavior_direction as direction, trace};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

pub fn router() -> Router<Arc<App>> {
    Router::new().route("/directions/{id}/versions/{version}", get(get_version))
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
    if result["runtime"] != trace::HGN_RUNTIME {
        return Err(ApiError::bad(
            "reference comparison requires HGN directions",
        ));
    }
    let reference = artifact(store, id, direction::VERSION, Some(layer))?;
    if reference["runtime"] != trace::HGN_RUNTIME
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
        .filter(|job| job.kind == direction::JOB && job.state == JobState::Succeeded)
        .ok_or_else(|| ApiError::not_found("direction"))?;
    let result = store
        .job_result(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("direction result"))?;
    if result["schema_version"] != version {
        return Err(ApiError::not_found("direction version"));
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

pub(crate) struct HgnLayer {
    pub sha256: String,
    pub behavior: direction::Behavior,
    pub vector: Vec<f32>,
    pub assets: trace::hgn_projection::AssetPins,
    pub held_out_pairs: usize,
}

pub(crate) fn hgn_layer(
    store: &Store,
    model: &crate::domain::Model,
    direction_id: &str,
    source_layer: usize,
) -> Result<HgnLayer, ApiError> {
    let artifact = artifact(store, direction_id, direction::VERSION, Some(source_layer))?;
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
        vector,
        assets: serde_json::from_value(artifact["assets_sha256"].clone())
            .map_err(ApiError::storage)?,
        held_out_pairs: fit.held_out_pairs,
    })
}
