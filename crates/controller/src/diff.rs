//! Read-only pair reports. Open artifacts before a cache lookup.
mod behavior;
mod tensors;

use crate::{
    App,
    domain::{Envelope, Job, State as JobState},
    web::ApiError,
};
use axum::{
    Router,
    body::Bytes,
    extract::{Query, State, rejection::QueryRejection},
    http::{StatusCode, header},
    response::IntoResponse,
    routing::get,
};
use brewery_worker::trace;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::VecDeque,
    fs::{File, Metadata},
    os::unix::fs::MetadataExt,
    sync::{Arc, Mutex},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

type Result<T> = std::result::Result<T, ApiError>;
type Params<T> = std::result::Result<Query<T>, QueryRejection>;
const CACHE_ENTRIES: usize = 8;

pub struct Service {
    entries: Mutex<VecDeque<Entry>>,
    slot: Arc<Semaphore>,
}

type Entry = (Key, Vec<Stamp>, Bytes);

impl Default for Service {
    fn default() -> Self {
        Self {
            entries: Mutex::default(),
            slot: Arc::new(Semaphore::new(1)),
        }
    }
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Clone)]
#[serde(deny_unknown_fields)]
struct Pair {
    a: String,
    b: String,
}

impl Pair {
    fn check(&self) -> Result<()> {
        for id in [&self.a, &self.b] {
            uuid::Uuid::parse_str(id)
                .ok()
                .filter(|uuid| uuid.to_string() == *id)
                .ok_or_else(|| ApiError::bad("expected canonical IDs in a and b"))?;
        }
        Ok(())
    }

    fn report<T>(self, data: T) -> Report<T> {
        Report {
            schema_version: 1,
            qualification: "not release-qualified",
            pair: self,
            data,
        }
    }
}

#[derive(Serialize)]
struct Report<T> {
    schema_version: u8,
    qualification: &'static str,
    #[serde(flatten)]
    pair: Pair,
    #[serde(flatten)]
    data: T,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Traces {
    a: String,
    b: String,
    #[serde(default)]
    prompt: usize,
    layer: Option<usize>,
    start: Option<usize>,
    end: Option<usize>,
}

#[derive(PartialEq, Eq)]
enum Key {
    Tensors(Pair),
    Behavior(Pair),
    Traces(Traces),
}

// The Library source can change outside the controller. ctime detects changes
// even if a writer restores mtime; device/inode detect replacement.
#[derive(PartialEq, Eq)]
struct Stamp(u64, u64, u64, i64, i64, i64, i64);

impl From<Metadata> for Stamp {
    fn from(m: Metadata) -> Self {
        Self(
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        )
    }
}

fn stamp(file: &File) -> Result<Stamp> {
    Ok(file.metadata().map_err(ApiError::storage)?.into())
}

impl Service {
    pub(crate) fn acquire(&self) -> Result<OwnedSemaphorePermit> {
        self.slot.clone().try_acquire_owned().map_err(|_| {
            ApiError(
                StatusCode::TOO_MANY_REQUESTS,
                "a diff is already in progress".into(),
            )
        })
    }

    fn get(&self, key: &Key, stamps: &[Stamp]) -> Result<Option<Bytes>> {
        let mut entries = self.entries.lock().map_err(ApiError::storage)?;
        let Some(i) = entries.iter().position(|(k, s, _)| k == key && s == stamps) else {
            return Ok(None);
        };
        let entry = entries.remove(i).expect("cache entry");
        let result = entry.2.clone();
        entries.push_front(entry);
        Ok(Some(result))
    }

    fn save(&self, key: Key, stamps: Vec<Stamp>, report: &impl Serialize) -> Result<Bytes> {
        let json = serde_json::to_vec(report).map_err(ApiError::storage)?;
        if json.len() > trace::RESPONSE_CAP {
            return Err(too_large());
        }
        let bytes = Bytes::from(json);
        let mut entries = self.entries.lock().map_err(ApiError::storage)?;
        entries.retain(|(k, _, _)| k != &key);
        entries.push_front((key, stamps, bytes.clone()));
        entries.truncate(CACHE_ENTRIES);
        Ok(bytes)
    }
}

fn response(bytes: Bytes) -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        bytes,
    )
}

fn parameters<T>(query: Params<T>) -> Result<T> {
    query
        .map(|Query(q)| q)
        .map_err(|_| ApiError::bad("invalid diff query"))
}

fn invalid(err: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::UNPROCESSABLE_ENTITY, err.to_string())
}

fn too_large() -> ApiError {
    ApiError(
        StatusCode::PAYLOAD_TOO_LARGE,
        "diff exceeds the size limit".into(),
    )
}

pub fn router() -> Router<Arc<App>> {
    Router::new()
        .route("/diff/tensors", get(tensors))
        .route("/diff/traces", get(traces))
        .route("/diff/behavior", get(behavior))
}

async fn tensors(State(app): State<Arc<App>>, query: Params<Pair>) -> Result<impl IntoResponse> {
    let q = parameters(query)?;
    q.check()?;
    let permit = app.diffs.acquire()?;
    let bytes = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        tensors::run(&app, q)
    })
    .await
    .map_err(ApiError::storage)??;
    Ok(response(bytes))
}

async fn behavior(State(app): State<Arc<App>>, query: Params<Pair>) -> Result<impl IntoResponse> {
    let q = parameters(query)?;
    q.check()?;
    let permit = app.diffs.acquire()?;
    let bytes = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        behavior::run(&app, q)
    })
    .await
    .map_err(ApiError::storage)??;
    Ok(response(bytes))
}

async fn traces(State(app): State<Arc<App>>, query: Params<Traces>) -> Result<impl IntoResponse> {
    let q = parameters(query)?;
    let pair = Pair {
        a: q.a.clone(),
        b: q.b.clone(),
    };
    pair.check()?;
    let slice = trace::Slice {
        view: trace::View::DiffInput,
        prompt: q.prompt,
        layer: q.layer,
        role: None,
        start: q.start,
        end: q.end,
    };
    slice.check().map_err(|e| ApiError::bad(e.to_string()))?;
    let permit = app.diffs.acquire()?;
    let (a, b, a_env, b_env) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let job = |id| {
            store
                .job(id)
                .map_err(ApiError::storage)?
                .filter(|j| j.kind == trace::JOB && j.state == JobState::Succeeded)
                .ok_or_else(|| ApiError::not_found("trace result"))
        };
        let a = job(&q.a)?;
        let b = job(&q.b)?;
        let a_env = store.envelope(&q.a).map_err(ApiError::storage)?;
        let b_env = store.envelope(&q.b).map_err(ApiError::storage)?;
        (a, b, a_env, b_env)
    };
    let prompts = |env: Envelope| -> Result<Vec<String>> {
        if let Some(prompt) = env.prompt {
            Ok(vec![prompt])
        } else {
            serde_json::from_value(env.options.get("prompts").cloned().unwrap_or(Value::Null))
                .map_err(ApiError::storage)
        }
    };
    let ordered = prompts(a_env)?;
    if ordered != prompts(b_env)? {
        return Err(invalid("trace prompts differ"));
    }
    if q.prompt >= ordered.len() {
        return Err(ApiError::not_found("trace prompt"));
    }
    let key = Key::Traces(q);
    let cached = app.diffs.get(&key, &[])?;
    let slice = if cached.is_some() {
        trace::Slice {
            view: trace::View::Manifest,
            prompt: slice.prompt,
            layer: None,
            role: None,
            start: None,
            end: None,
        }
    } else {
        slice
    };
    // Even on a cache hit, check retention on both workers.
    let (left, right) =
        tokio::try_join!(trace_read(&app, &a, &slice), trace_read(&app, &b, &slice))?;
    if let Some(bytes) = cached {
        return Ok(response(bytes));
    }
    #[derive(Serialize)]
    struct TraceReport {
        prompt: usize,
        #[serde(flatten)]
        report: trace::diff::Report,
    }
    let bytes = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let report = trace::diff::compare(left, right).map_err(invalid)?;
        app.diffs.save(
            key,
            Vec::new(),
            &pair.report(TraceReport {
                prompt: slice.prompt,
                report,
            }),
        )
    })
    .await
    .map_err(ApiError::storage)??;
    Ok(response(bytes))
}

pub(crate) async fn trace_read(app: &App, job: &Job, slice: &trace::Slice) -> Result<Value> {
    match app.dispatcher.trace_read(job, slice).await.map_err(|e| {
        eprintln!("Trace diff read failed for {}: {e}", job.id);
        ApiError(
            StatusCode::BAD_GATEWAY,
            "trace worker could not read the result".into(),
        )
    })? {
        trace::ReadReply::Ok(data) => Ok(data),
        trace::ReadReply::Missing => Err(ApiError::not_found("trace slice")),
        trace::ReadReply::TooLarge => Err(too_large()),
    }
}
