//! Append-only hypothesis versions and controller-resolved evidence.
mod replay;

use crate::{
    App,
    domain::{self, State as JobState},
    idempotency::{Replay, Submission},
    scratch::{RevisionState, jobs},
    store::Store,
    web::{ApiError, read, single},
};
use axum::{
    Extension, Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc};

type Result<T> = std::result::Result<T, ApiError>;
type Payload<T> = std::result::Result<Json<T>, JsonRejection>;
type Mutation = Result<(StatusCode, Json<Finding>)>;
const QUALIFICATION: &str = "not release-qualified";
const MAX_EXPERIMENTS: usize = 12;

#[derive(Clone)]
pub(crate) enum Caller {
    ExternalMcp,
    Builtin(String),
}

#[derive(Clone, Serialize, Deserialize)]
struct Agent {
    kind: String,
    id: String,
}

impl Agent {
    fn resolve(caller: Option<Extension<Caller>>, client: Option<&str>) -> Result<Self> {
        if let Some(Extension(Caller::Builtin(id))) = caller {
            return Ok(Self {
                kind: "builtin".into(),
                id,
            });
        }
        let id = client
            .filter(|id| {
                !id.trim().is_empty() && id.len() <= 128 && !id.chars().any(char::is_control)
            })
            .ok_or_else(|| ApiError::bad("supply client_id (1 to 128 bytes)"))?;
        Ok(Self {
            kind: if caller.is_some() {
                "external_mcp"
            } else {
                "external_rest"
            }
            .into(),
            id: id.into(),
        })
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Model {
    id: String,
    sha256: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Ready,
    Running,
    Failed,
}

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Summary {
    pub(super) id: String,
    version: u32,
    hypothesis: String,
    parent_id: Option<String>,
    replay_of: Option<String>,
    created_at_ms: u64,
    state: Status,
    model: Model,
    verdict: Verdict,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Finding {
    #[serde(flatten)]
    pub(super) summary: Summary,
    schema_version: u32,
    qualification: String,
    provenance: Provenance,
    experiments: Vec<Experiment>,
    comparison: Vec<Comparison>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Provenance {
    agent: Agent,
    tool_version: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Verdict {
    status: VerdictStatus,
    reasoning: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum VerdictStatus {
    Supported,
    Refuted,
    Inconclusive,
}

fn inconclusive(reasoning: impl Into<String>) -> Verdict {
    Verdict {
        status: VerdictStatus::Inconclusive,
        reasoning: reasoning.into(),
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Experiment {
    spec: Spec,
    artifact: Artifact,
    metrics: Vec<Metric>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Spec {
    Job {
        request: Value,
    },
    TokenLoss {
        tool: String,
        arguments: Value,
        preparation: TokenPrep,
    },
    Crosscoder {
        tool: String,
        arguments: Value,
    },
}

#[derive(Clone, Serialize, Deserialize)]
struct TokenPrep {
    edit: String,
    arguments: Value,
    assets_id: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct Artifact {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    job_id: Option<String>,
    sha256: String,
    api_url: String,
    view_url: String,
    pins: Value,
}

#[derive(Clone, Serialize, Deserialize)]
struct Metric {
    path: String,
    value: serde_json::Number,
}

#[derive(Clone, Serialize, Deserialize)]
struct Comparison {
    experiment: usize,
    path: String,
    original: serde_json::Number,
    replay: serde_json::Number,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Create {
    hypothesis: String,
    model_id: String,
    model_sha256: String,
    client_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Add {
    evidence: Option<Evidence>,
    verdict: Option<Verdict>,
    client_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Evidence {
    Job {
        job_id: String,
        paths: Vec<String>,
    },
    TokenLoss {
        a: String,
        b: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        sentence: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        start: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        end: Option<usize>,
        paths: Vec<String>,
    },
    Crosscoder {
        artifact_id: String,
        paths: Vec<String>,
    },
}

pub fn router() -> Router<Arc<App>> {
    Router::new()
        .route("/findings", get(list).post(create))
        .route("/findings/{id}", get(detail))
        .route("/findings/{id}/evidence", post(add))
        .route("/findings/{id}/replay", post(replay::start))
}

async fn list(State(app): State<Arc<App>>) -> Result<Json<Vec<Summary>>> {
    Ok(Json(
        app.store
            .lock()
            .map_err(ApiError::storage)?
            .findings()
            .map_err(ApiError::storage)?,
    ))
}

async fn detail(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<Json<Finding>> {
    Ok(Json(lookup(
        &*app.store.lock().map_err(ApiError::storage)?,
        &id,
    )?))
}

fn lookup(store: &Store, id: &str) -> Result<Finding> {
    store
        .finding(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("finding"))
}

fn conflict(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::CONFLICT, msg.into())
}

fn submission<'a>(
    headers: &'a HeaderMap,
    scope: &'static str,
    input: &impl Serialize,
) -> Result<Submission<'a>> {
    let invalid = || ApiError::bad("supply a valid Idempotency-Key");
    let key = single(headers, "idempotency-key")
        .map_err(|_| invalid())?
        .filter(|key| crate::idempotency::valid_key(key))
        .ok_or_else(invalid)?;
    Submission::new(scope, key, input).map_err(ApiError::storage)
}

fn prior(store: &Store, key: &Submission<'_>) -> Result<Option<Finding>> {
    match store.replay(key).map_err(ApiError::storage)? {
        Replay::New => Ok(None),
        Replay::Matched(id) => Ok(Some(lookup(store, &id)?)),
        Replay::Conflict => Err(conflict(
            "Idempotency-Key already used for a different request",
        )),
    }
}

fn provenance(agent: Agent) -> Provenance {
    Provenance {
        agent,
        tool_version: env!("CARGO_PKG_VERSION").into(),
    }
}

fn text(input: &str) -> Result<()> {
    if input.trim().is_empty() || input.len() > 4096 || input.contains('\0') {
        return Err(ApiError::bad("supply non-empty text of at most 4096 bytes"));
    }
    Ok(())
}

fn bind(store: &Store, id: String, sha256: String) -> Result<Model> {
    if !brewery_worker::is_digest(&sha256) {
        return Err(ApiError::bad(
            "supply the model revision SHA-256 in lowercase",
        ));
    }
    if let Some(row) = store.scratch(&id).map_err(ApiError::storage)? {
        if row.state == RevisionState::Discarded || row.sha256 != sha256 {
            return Err(conflict("scratch revision is not available at this hash"));
        }
    } else {
        let library = store
            .model(&id)
            .map_err(ApiError::storage)?
            .ok_or_else(|| ApiError::not_found("model revision"))?;
        if library.sha256.as_ref().is_some_and(|sha| sha != &sha256) {
            return Err(conflict("Library model hash does not match"));
        }
    }
    Ok(Model { id, sha256 })
}

impl Finding {
    fn next(&self, provenance: Provenance) -> Result<Self> {
        if self.summary.state != Status::Ready {
            return Err(conflict("use a completed finding"));
        }
        let mut next = self.clone();
        next.summary.version = self
            .summary
            .version
            .checked_add(1)
            .ok_or_else(|| conflict("version limit"))?;
        next.summary.parent_id = Some(self.summary.id.clone());
        next.summary.id = domain::id();
        next.summary.created_at_ms = domain::now();
        next.provenance = provenance;
        Ok(next)
    }
}

async fn create(
    State(app): State<Arc<App>>,
    caller: Option<Extension<Caller>>,
    headers: HeaderMap,
    payload: Payload<Create>,
) -> Mutation {
    let request = read(payload)?;
    let actor = Agent::resolve(caller, request.client_id.as_deref())?;
    let key = submission(&headers, "findings-create", &(&actor, &request))?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(row) = prior(&store, &key)? {
        return Ok((StatusCode::OK, Json(row)));
    }
    text(&request.hypothesis)?;
    let model = bind(&store, request.model_id, request.model_sha256)?;
    let row = Finding {
        schema_version: 1,
        qualification: QUALIFICATION.into(),
        summary: Summary {
            id: domain::id(),
            version: 1,
            hypothesis: request.hypothesis,
            parent_id: None,
            replay_of: None,
            created_at_ms: domain::now(),
            state: Status::Ready,
            model,
            error: None,
            verdict: inconclusive("No evidence has been recorded."),
        },
        provenance: provenance(actor),
        experiments: Vec::new(),
        comparison: Vec::new(),
    };
    store.add_finding(&row, &key).map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(row)))
}

async fn add(
    State(app): State<Arc<App>>,
    caller: Option<Extension<Caller>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<Add>,
) -> Mutation {
    let request = read(payload)?;
    let actor = Agent::resolve(caller, request.client_id.as_deref())?;
    let key = submission(&headers, "findings-evidence", &(&actor, &id, &request))?;
    let mut row = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        if let Some(row) = prior(&store, &key)? {
            return Ok((StatusCode::OK, Json(row)));
        }
        lookup(&store, &id)?.next(provenance(actor))?
    };
    if request.verdict.is_none() && request.evidence.is_none() {
        return Err(ApiError::bad("supply evidence, a verdict, or both"));
    }
    if let Some(evidence) = request.evidence {
        if row.experiments.len() >= MAX_EXPERIMENTS {
            return Err(ApiError::bad("use at most 12 experiments in a finding"));
        }
        row.experiments
            .push(evidence.collect(&app, &row.summary.model).await?);
        row.summary.verdict = inconclusive("The new evidence needs an agent verdict.");
    }
    if let Some(verdict) = request.verdict {
        text(&verdict.reasoning)?;
        if row.experiments.is_empty() && !matches!(verdict.status, VerdictStatus::Inconclusive) {
            return Err(ApiError::bad(
                "record evidence before giving a decisive verdict",
            ));
        }
        row.summary.verdict = verdict;
    }
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if let Some(row) = prior(&store, &key)? {
        return Ok((StatusCode::OK, Json(row)));
    }
    store.add_finding(&row, &key).map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(row)))
}

impl Evidence {
    async fn collect(self, app: &Arc<App>, model: &Model) -> Result<Experiment> {
        match self {
            Self::Job { job_id, paths } => job_evidence(
                &*app.store.lock().map_err(ApiError::storage)?,
                model,
                &job_id,
                &paths,
            ),
            Self::TokenLoss {
                a,
                b,
                sentence,
                start,
                end,
                paths,
            } => {
                let query = crate::diff::LossSlice {
                    a,
                    b,
                    sentence,
                    start,
                    end,
                };
                token_loss(app, model, query, &paths).await
            }
            Self::Crosscoder { artifact_id, paths } => crosscoder(
                &*app.store.lock().map_err(ApiError::storage)?,
                model,
                &artifact_id,
                &paths,
            ),
        }
    }
}

fn metrics(data: &Value, paths: &[String]) -> Result<Vec<Metric>> {
    let mut unique = BTreeSet::new();
    if paths.is_empty() || paths.len() > 16 {
        return Err(ApiError::bad("select 1 to 16 numeric JSON pointers"));
    }
    paths
        .iter()
        .map(|path| {
            let value = data
                .pointer(path)
                .and_then(Value::as_number)
                .filter(|_| path.starts_with('/') && path.len() <= 256 && unique.insert(path))
                .ok_or_else(|| {
                    ApiError::bad("each metric needs a distinct JSON pointer to a number")
                })?;
            Ok(Metric {
                path: path.clone(),
                value: value.clone(),
            })
        })
        .collect()
}

fn hash(data: &Value) -> Result<String> {
    let mut canonical = data.clone();
    canonical.sort_all_objects();
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&canonical).map_err(ApiError::storage)?)
    ))
}

fn input_hash(store: &Store, model: &Model) -> Result<String> {
    if store
        .scratch(&model.id)
        .map_err(ApiError::storage)?
        .is_some()
    {
        bind(store, model.id.clone(), model.sha256.clone())?;
        Ok(jobs::trace_input(store, &model.id)?.1)
    } else {
        Ok(model.sha256.clone())
    }
}

fn pins(data: &Value, fields: &[&str]) -> Value {
    Value::Object(
        fields
            .iter()
            .filter_map(|&name| Some((name.into(), data.get(name)?.clone())))
            .collect(),
    )
}

fn job_evidence(store: &Store, model: &Model, id: &str, paths: &[String]) -> Result<Experiment> {
    let job = store
        .job(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("job"))?;
    if job.state != JobState::Succeeded || job.input_ids != [model.id.as_str()] {
        return Err(conflict(
            "use a completed single-model job for this finding",
        ));
    }
    let data = store
        .job_result(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("job artifact"))?;
    let sha = input_hash(store, model)?;
    if data["model_sha256"] != sha || data["model_id"] != model.id {
        return Err(conflict("artifact is not bound to the finding model hash"));
    }
    let (route, view_url, pinned) = match job.kind.as_str() {
        "circuit" => (
            "circuit",
            format!("#/jobs/{id}"),
            &["executable_sha256", "manifest_sha256", "backend"][..],
        ),
        "trace" => (
            "trace",
            format!("#/trace?job={id}"),
            &["executable_sha256", "lanternfish_commit"][..],
        ),
        "sensitivity" => (
            "sensitivity",
            format!("#/jobs/{id}"),
            &[
                "executable_sha256",
                "manifest_sha256",
                "backend",
                "corpus_sha256",
            ][..],
        ),
        _ => {
            return Err(ApiError::bad(
                "job evidence must be a circuit, trace or sensitivity job",
            ));
        }
    };
    let envelope = store.envelope(id).map_err(ApiError::storage)?;
    let request = json!({
        "kind": job.kind, "host": job.host, "input_ids": job.input_ids,
        "options": job.options, "prompt": envelope.prompt
    });
    Ok(Experiment {
        spec: Spec::Job { request },
        artifact: Artifact {
            id: id.into(),
            job_id: Some(id.into()),
            sha256: hash(&data)?,
            api_url: if job.kind == "trace" {
                format!("/api/jobs/{id}")
            } else {
                format!("/api/jobs/{id}/{route}")
            },
            view_url,
            pins: pins(&data, pinned),
        },
        metrics: metrics(&data, paths)?,
    })
}

async fn token_loss(
    app: &Arc<App>,
    model: &Model,
    query: crate::diff::LossSlice,
    paths: &[String],
) -> Result<Experiment> {
    if query.a != model.id {
        return Err(conflict("token-loss parent must be the finding model"));
    }
    let (check_job, preparation, pinned) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let row = store
            .scratch(&query.b)
            .map_err(ApiError::storage)?
            .filter(|row| row.state != RevisionState::Discarded)
            .ok_or_else(|| ApiError::not_found("scratch edit"))?;
        if row.provenance.parent_id != model.id || row.provenance.parent_sha256 != model.sha256 {
            return Err(conflict("edit is not bound to the finding model hash"));
        }
        let (edit, arguments) = row.provenance.finding_edit();
        let convert = row
            .gguf
            .as_ref()
            .ok_or_else(|| conflict("convert the scratch edit first"))?;
        let env = store.envelope(&convert.job_id).map_err(ApiError::storage)?;
        let assets = env
            .inputs
            .get(1)
            .ok_or_else(|| ApiError::storage("missing GGUF assets"))?;
        let (check_job, mut report) = row
            .finding_checks()
            .ok_or_else(|| conflict("run measured checks first"))?;
        report["converter_version"] = json!(convert.converter_version);
        (
            check_job.to_owned(),
            TokenPrep {
                edit,
                arguments,
                assets_id: assets.id.clone(),
            },
            report,
        )
    };
    let a = &query.a;
    let b = &query.b;
    let mut api_url = format!("/api/diff/token-loss?a={a}&b={b}");
    for (key, value) in [
        ("sentence", query.sentence),
        ("start", query.start),
        ("end", query.end),
    ] {
        if let Some(value) = value {
            api_url.push_str(&format!("&{key}={value}"));
        }
    }
    let data = crate::diff::finding_loss(app.clone(), query.clone()).await?;
    // Verify the selected check did not change between reading its metadata and scores.
    if data["check_bundle_id"] != pinned["check_bundle_id"]
        || data["artifact_sha256"] != pinned["artifact_sha256"]
    {
        return Err(conflict("token-loss bundle changed while reading evidence"));
    }
    let view_url = format!("#/edits?rev={}", query.b);
    let arguments = serde_json::to_value(query).map_err(ApiError::storage)?;
    Ok(Experiment {
        spec: Spec::TokenLoss {
            tool: "read_token_loss_diff".into(),
            arguments,
            preparation,
        },
        artifact: Artifact {
            id: data["check_bundle_id"].as_str().unwrap_or_default().into(),
            job_id: Some(check_job),
            sha256: data["artifact_sha256"].as_str().unwrap_or_default().into(),
            api_url,
            view_url,
            pins: pinned,
        },
        metrics: metrics(&data, paths)?,
    })
}

fn crosscoder(store: &Store, model: &Model, id: &str, paths: &[String]) -> Result<Experiment> {
    let data = store
        .crosscoder(id)
        .map_err(ApiError::storage)?
        .ok_or_else(|| ApiError::not_found("crosscoder artifact"))?;
    let model_sha = input_hash(store, model)?;
    if !["parent", "child"]
        .into_iter()
        .any(|k| data[k]["id"] == model.id && data[k]["sha256"] == model_sha)
    {
        return Err(conflict(
            "crosscoder is not bound to the finding model hash",
        ));
    }
    Ok(Experiment {
        spec: Spec::Crosscoder {
            tool: "crosscoder_diff".into(),
            arguments: json!({"id":id}),
        },
        artifact: Artifact {
            id: id.into(),
            job_id: None,
            sha256: hash(&data)?,
            api_url: format!("/api/crosscoders/{id}"),
            view_url: format!("#/edits/crosscoder?id={id}"),
            pins: pins(&data, &["training", "weights_sha256"]),
        },
        metrics: metrics(&data, paths)?,
    })
}
