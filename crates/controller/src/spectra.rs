use crate::{
    App, Error,
    domain::{Envelope, Job, Model, NewJob},
    scratch::{RevisionState, jobs},
    store::Store,
    web::{ApiError, read_submission, replay},
};
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use brewery_worker::{
    weight_source::{native_tensor_name, scratch},
    weight_spectra::{self, Filter, Input, JOB},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{path::Path, sync::Arc};

type Result<T> = std::result::Result<T, ApiError>;

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Format {
    Safetensors,
    Gguf,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: String,
    compare_id: Option<String>,
    representation: Option<Format>,
    compare_representation: Option<Format>,
    layer: Option<u32>,
    tensor: Option<String>,
}

pub fn router() -> Router<Arc<App>> {
    Router::new().route("/weight-spectra", post(queue))
}

async fn queue(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    payload: std::result::Result<Json<Request>, JsonRejection>,
) -> Result<Response> {
    let (q, submission) = read_submission(&headers, JOB, payload)?;
    if q.layer.is_some_and(|n| n > 4095)
        || q.tensor.as_deref().is_some_and(|s| !native_tensor_name(s))
        || uuid::Uuid::parse_str(&q.id).is_err()
        || q.compare_id
            .as_deref()
            .is_some_and(|id| uuid::Uuid::parse_str(id).is_err())
    {
        return Err(ApiError::bad(
            "Invalid spectral input ID, layer, or tensor name",
        ));
    }
    let store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(response) = replay(&store, submission.as_ref())? {
        return Ok(response);
    }
    let (a, pin, parent) = resolve(&app, &store, &q.id, q.representation)?;
    let host = a.host.clone();
    let mut pins = std::collections::BTreeMap::new();
    if let Some(pin) = pin {
        pins.insert(a.path.clone(), pin);
    }
    let mut inputs = vec![a];
    // In private GGUF mode the default reference remains the parent's SafeTensors.
    // The caller can select GGUF explicitly on either side.
    if let Some(id) = q
        .compare_id
        .as_deref()
        .or(parent.as_ref().map(|p| p.0.as_str()))
    {
        let (b, mut pin, _) = resolve(&app, &store, id, q.compare_representation)?;
        if b.host != host || (b.id == q.id && inputs[0].kind == b.kind) {
            return Err(ApiError::bad(
                "Select different inputs on the same local editor",
            ));
        }
        if let Some((_, expected)) = parent.as_ref().filter(|p| p.0 == id)
            && b.kind == "safetensors"
        {
            if pin.as_ref().is_some_and(|pin| pin != expected) {
                return Err(conflict("Parent hash does not match the scratch revision"));
            }
            pin = Some(expected.clone());
        }
        if let Some(pin) = pin {
            pins.insert(b.path.clone(), pin);
        }
        inputs.push(b);
    } else if q.compare_representation.is_some() {
        return Err(ApiError::bad(
            "Select a reference for the comparison representation",
        ));
    }
    let filter = Filter {
        layer: q.layer,
        tensor: q.tensor,
    };
    let request = NewJob {
        kind: JOB.into(),
        host: None,
        input_ids: inputs.iter().map(|m| m.id.clone()).collect(),
        options: serde_json::to_value(filter)
            .map_err(ApiError::storage)?
            .as_object()
            .unwrap()
            .clone(),
        prompt_id: None,
        prompt: None,
    };
    let (job, mut envelope) = request.records(host, inputs, None, Default::default());
    envelope.scratch_pins = pins;
    if crate::domain::candidates(
        &job,
        &envelope,
        &app.effective_hosts().map_err(ApiError::storage)?,
    )
    .is_empty()
    {
        return Err(conflict(
            "Enable the local SafeTensors editor and resume scheduling",
        ));
    }
    let tx = store.submission_transaction().map_err(ApiError::storage)?;
    Store::enqueue(&tx, &job, &envelope, submission.as_ref()).map_err(ApiError::storage)?;
    tx.commit().map_err(ApiError::storage)?;
    Ok((StatusCode::ACCEPTED, Json(job)).into_response())
}

type Source = (Model, Option<String>, Option<(String, String)>);

fn resolve(app: &App, store: &Store, id: &str, format: Option<Format>) -> Result<Source> {
    if let Some(row) = store.scratch(id).map_err(ApiError::storage)? {
        if row.state == RevisionState::Discarded {
            return Err(ApiError::not_found("scratch revision"));
        }
        let parent = (!row.is_compiled_program()).then(|| {
            (
                row.provenance.parent_id.clone(),
                row.provenance.parent_sha256.clone(),
            )
        });
        return if format == Some(Format::Gguf) {
            let (model, pin) = jobs::trace_input(store, id)?;
            Ok((model, Some(pin), parent))
        } else {
            let model = jobs::input(store, &row, scratch::MODEL, "safetensors");
            Ok((model, Some(row.sha256), parent))
        };
    }
    let model = store
        .model(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("Library model or scratch revision"))?;
    let actual = match model.kind.as_str() {
        "gguf" => Format::Gguf,
        "hf" | "safetensors" => Format::Safetensors,
        _ => return Err(ApiError::bad("Spectral input must be SafeTensors or GGUF")),
    };
    if format.is_some_and(|f| f != actual) {
        return Err(ApiError::bad(
            "Requested representation does not match the Library model",
        ));
    }
    allow_library(app, store, &model)?;
    // HF directory hashes in the Library bind the full publication, not
    // the subset of sorted weights this job reads and identifies.
    let pin = (model.kind != "hf").then(|| model.sha256.clone()).flatten();
    Ok((model, pin, None))
}

pub(crate) fn allow_library(app: &App, store: &Store, model: &Model) -> Result<()> {
    let path = Path::new(&model.path);
    if store.is_scratch_path(path)
        || store.scratch_root().starts_with(path)
        || store
            .is_private_editor_path(path)
            .map_err(ApiError::storage)?
        || !app
            .scratch_source_allowed(&model.host, path)
            .unwrap_or(false)
    {
        return Err(conflict(
            "Library source is not in an approved local model directory",
        ));
    }
    Ok(())
}

fn conflict(text: &str) -> ApiError {
    ApiError(StatusCode::CONFLICT, text.into())
}

pub(crate) async fn run(
    app: &Arc<App>,
    job: &Job,
    envelope: Envelope,
) -> std::result::Result<Value, Error> {
    // All I/O and SVD are off the async event loop. The scheduler retains its
    // slot until the blocking closure exits, including cooperative cancellation.
    let app = Arc::clone(app);
    let id = job.id.clone();
    tokio::task::spawn_blocking(move || {
        let filter: Filter = serde_json::from_value(Value::Object(envelope.options))?;
        {
            let store = app.store.lock().map_err(|_| "Storage unavailable")?;
            for m in &envelope.inputs {
                if store.scratch(&m.id)?.is_none() && allow_library(&app, &store, m).is_err() {
                    return Err("Library source is no longer in an approved model directory".into());
                }
            }
        }
        let source = |i: usize| -> Input<'_> {
            let m = &envelope.inputs[i];
            (
                &m.id,
                Path::new(&m.path),
                &m.kind,
                envelope.scratch_pins.get(&m.path).map(String::as_str),
            )
        };
        let mut last = None::<std::time::Instant>;
        let mut check = || {
            use std::io::Error;
            use std::time::{Duration, Instant};
            if last.is_some_and(|last| last.elapsed() < Duration::from_millis(100)) {
                return Ok(());
            }
            last = Some(Instant::now());
            let store = app
                .store
                .lock()
                .map_err(|_| Error::other("Storage unavailable"))?;
            if store
                .job(&id)
                .map_err(|_| Error::other("Storage unavailable"))?
                .is_none_or(|job| job.cancel_requested)
            {
                return Err(Error::other("Spectral job cancelled").into());
            }
            Ok(())
        };
        weight_spectra::analyze(
            source(0),
            (envelope.inputs.len() == 2).then(|| source(1)),
            &filter,
            &mut check,
        )
        .map_err(|error| error.to_string().into())
    })
    .await?
}
