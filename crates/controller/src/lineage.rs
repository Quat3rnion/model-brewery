use crate::{App, domain::Model, scratch::RevisionState, store::Store, web::ApiError};
use axum::{
    Json, Router,
    extract::{Path, Query, State, rejection::QueryRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use brewery_worker::{
    weight_source::scratch::MODEL,
    weight_spectra::{Input, MAX_CANDIDATES, lineage},
};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    path::Path as FsPath,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;

// Do not queue an unbounded amount of I/O on the controller.
static SLOT: Semaphore = Semaphore::const_new(1);
const DEADLINE: Duration = Duration::from_secs(90);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidates {
    candidate_1: Option<String>,
    candidate_2: Option<String>,
    candidate_3: Option<String>,
}

pub(crate) fn router() -> Router<Arc<App>> {
    Router::new().route("/models/{id}/lineage", get(read))
}

async fn read(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    query: Result<Query<Candidates>, QueryRejection>,
) -> Result<Response, ApiError> {
    let Query(q) = query.map_err(|_| ApiError::bad("Invalid lineage query"))?;
    let ids: Vec<_> = [q.candidate_1, q.candidate_2, q.candidate_3]
        .into_iter()
        .flatten()
        .collect();
    let valid =
        |s: &String| uuid::Uuid::try_parse(s).is_ok_and(|id| id.hyphenated().to_string() == *s);
    if !valid(&id)
        || ids.iter().any(|s| !valid(s) || s == &id)
        || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
    {
        return Err(ApiError::bad("Select distinct Library model IDs"));
    }
    let permit = SLOT.try_acquire().map_err(|_| {
        ApiError(
            StatusCode::TOO_MANY_REQUESTS,
            "A lineage read is already in progress".into(),
        )
    })?;
    let stopped = Arc::new(AtomicBool::new(false));
    let _cancel = CancelOnDrop(Arc::clone(&stopped));
    let deadline = Instant::now() + DEADLINE;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let (target, candidates) = collect(&app, &id, &ids)?;
        let inputs = candidates.iter().map(input).collect::<Vec<_>>();
        lineage(input(&target), &inputs, || {
            if stopped.load(Ordering::Relaxed) || Instant::now() >= deadline {
                Err(std::io::Error::other("Lineage read exceeded its time budget").into())
            } else {
                Ok(())
            }
        })
        .map_err(|e| ApiError(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))
    })
    .await
    .map_err(ApiError::storage)??;
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(result)).into_response())
}

fn input(model: &Model) -> Input<'_> {
    let pin = (model.kind != "hf")
        .then_some(model.sha256.as_deref())
        .flatten();
    (&model.id, FsPath::new(&model.path), &model.kind, pin)
}

fn collect(app: &App, id: &str, ids: &[String]) -> Result<(Model, Vec<Model>), ApiError> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    let lookup = |id: &str| {
        store
            .model(id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("Library model"))
    };
    let mut target = lookup(id)?;
    allowed(app, &store, &mut target)?;
    let compatible =
        |m: &Model| m.host == target.host && (m.kind == "gguf") == (target.kind == "gguf");
    let mut candidates = Vec::new();
    if ids.is_empty() {
        for mut model in store.models().map_err(ApiError::storage)? {
            if model.id != id && compatible(&model) && allowed(app, &store, &mut model).is_ok() {
                candidates.push(model);
                if candidates.len() > MAX_CANDIDATES {
                    return Err(ApiError::bad(
                        "Select up to three candidate IDs on this host",
                    ));
                }
            }
        }
    } else {
        for id in ids {
            let mut model = lookup(id)?;
            if !compatible(&model) {
                return Err(ApiError::bad(
                    "Select candidates on the same host and in the same format",
                ));
            }
            allowed(app, &store, &mut model)?;
            candidates.push(model);
        }
    }
    Ok((target, candidates))
}

fn allowed(app: &App, store: &Store, model: &mut Model) -> Result<(), ApiError> {
    if !matches!(model.kind.as_str(), "safetensors" | "hf" | "gguf") {
        return Err(ApiError::bad(
            "Lineage requires SafeTensors, HF weights or floating-point GGUF",
        ));
    }
    // Approval registers exactly this private file as a Library model. Do not
    // treat a manual registration of a scratch path as proof of approval.
    if let Ok(rel) = FsPath::new(&model.path).strip_prefix(store.scratch_root())
        && model.kind == "safetensors"
        && let Some(id) = rel.parent().and_then(FsPath::to_str)
        && uuid::Uuid::try_parse(id).is_ok()
        && rel == FsPath::new(id).join(MODEL)
        && let Some(row) = store.scratch(id).map_err(ApiError::storage)?
        && row.state == RevisionState::Approved
        && row.library_model_id.as_deref() == Some(model.id.as_str())
    {
        model.sha256 = Some(row.sha256);
        return Ok(());
    }
    crate::spectra::allow_library(app, store, model)
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
