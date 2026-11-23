//! Bind native ablation to an immutable fitted direction and a scratch parent.
use super::*;
use brewery_worker::{behavior_direction, weight_source::scratch::ablation};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    id: String,
    version: u32,
    layer: usize,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Evidence {
    direction: Source,
    source_gguf_sha256: String,
    #[serde(flatten)]
    settings: ablation::Settings,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    sha256: String,
    direction: Source,
    first_layer: usize,
    last_layer: usize,
    kernel: ablation::Kernel,
    variant: ablation::Variant,
    #[serde(default)]
    include_embeddings: bool,
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
        "scratch-ablate-direction",
        Some(id),
        read(payload)?,
        |app, target, req, key| {
            if req.include_embeddings {
                return Err(ApiError::bad(
                    "embedding ablation is not supported; the 32 MiB per-tensor limit applies",
                ));
            }
            if req.direction.version != behavior_direction::VERSION {
                return Err(ApiError::bad("select a version 2 behavior direction"));
            }
            let id = request_id(target)?;
            let result = {
                let store = app.store.lock().map_err(ApiError::storage)?;
                visible(&store, &id)?;
                crate::directions::load(&store, &req.direction.id, req.direction.version)?
            };
            let fit: behavior_direction::Fit =
                serde_json::from_value(result["fit"].clone()).map_err(ApiError::storage)?;
            let layer = fit
                .layers
                .iter()
                .find(|row| row.layer == req.direction.layer)
                .ok_or_else(|| ApiError::not_found("direction layer"))?;
            let direction = layer
                .unit_vector
                .as_deref()
                .ok_or_else(|| ApiError::bad("the source layer has no nonzero fitted direction"))?;
            let settings = ablation::Settings {
                first_layer: req.first_layer,
                last_layer: req.last_layer,
                kernel: req.kernel,
                variant: req.variant,
            };
            let plan =
                ablation::Plan::new(&settings, direction, layer.negative_mean.as_deref(), |at| {
                    fit.layers
                        .iter()
                        .find(|row| row.layer == at)?
                        .negative_mean
                        .as_deref()
                })
                .map_err(|e| ApiError::bad(e.to_string()))?;
            let evidence = Evidence {
                direction: req.direction,
                source_gguf_sha256: result["model_sha256"]
                    .as_str()
                    .ok_or_else(|| ApiError::storage("Direction source pin is missing"))?
                    .to_owned(),
                settings,
            };
            build_special(
                app,
                id,
                Some(req.sha256),
                Vec::new(),
                None,
                Some(NativeEdit::Ablate(plan, evidence)),
                key,
            )
        },
    )
    .await
}
