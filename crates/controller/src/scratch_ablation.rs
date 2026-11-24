//! Bind native ablation to an immutable fitted direction and a scratch parent.
use super::*;
use brewery_worker::{behavior_direction, weight_source::scratch::ablation};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub(super) id: String,
    pub(super) version: u32,
    pub(super) layer: usize,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Evidence {
    direction: Source,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_gguf_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) hgn: Option<super::hgn_edit::Evidence>,
    #[serde(flatten)]
    settings: ablation::Settings,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub(super) sha256: String,
    pub(super) direction: Source,
    pub(super) first_layer: usize,
    pub(super) last_layer: usize,
    pub(super) kernel: ablation::Kernel,
    pub(super) variant: ablation::Variant,
    #[serde(
        default = "ablation::unit_scale",
        skip_serializing_if = "ablation::is_unit_scale"
    )]
    pub(super) alpha: f64,
    #[serde(default)]
    pub(super) include_embeddings: bool,
}

pub(super) async fn model(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<Request>,
) -> Result<axum::response::Response> {
    use axum::response::IntoResponse;
    let req = read(payload)?;
    if super::hgn_edit::is_parent(&app, &id)? {
        return super::hgn_edit::submit(&app, id, req, &headers);
    }
    Ok(mutate(
        app,
        headers,
        "scratch-ablate-direction",
        Some(id),
        req,
        |app, target, req, key| apply(app, request_id(target)?, req, key),
    )
    .await?
    .into_response())
}

impl Request {
    pub(super) fn settings(&self) -> ablation::Settings {
        ablation::Settings {
            first_layer: self.first_layer,
            last_layer: self.last_layer,
            kernel: self.kernel,
            variant: self.variant,
            alpha: self.alpha,
        }
    }

    pub(super) fn evidence(self, hgn: super::hgn_edit::Evidence) -> Evidence {
        let settings = self.settings();
        Evidence {
            direction: self.direction,
            source_gguf_sha256: None,
            hgn: Some(hgn),
            settings,
        }
    }
}

pub(super) fn apply(app: &App, id: String, req: Request, key: Option<&Submission<'_>>) -> Mutation {
    if req.include_embeddings {
        return Err(ApiError::bad(
            "embedding ablation is not supported; the 32 MiB per-tensor limit applies",
        ));
    }
    if req.direction.version != behavior_direction::VERSION {
        return Err(ApiError::bad("select a version 2 behavior direction"));
    }
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
    let settings = req.settings();
    let plan = ablation::Plan::new(&settings, direction, layer.negative_mean.as_deref(), |at| {
        fit.layers
            .iter()
            .find(|row| row.layer == at)?
            .negative_mean
            .as_deref()
    })
    .map_err(|e| ApiError::bad(e.to_string()))?;
    let evidence = Evidence {
        direction: req.direction,
        source_gguf_sha256: Some(
            result["model_sha256"]
                .as_str()
                .ok_or_else(|| ApiError::storage("Direction source pin is missing"))?
                .to_owned(),
        ),
        hgn: None,
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
}
