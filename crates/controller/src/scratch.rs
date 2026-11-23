//! Scratch artifacts use the Library store and local CPU editor admission,
//! but do not enter the Library until a separate operator action.
use crate::{
    App,
    domain::{NewModel, id, now},
    idempotency::{self, Replay, Submission},
    local_edit::Lane,
    store::Store,
    web::{ApiError, read, single},
};
use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
use brewery_worker::weight_source::{
    open_regular,
    scratch::{self, Operation, Summary, TensorInfo},
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    os::unix::fs::{DirBuilderExt, MetadataExt},
    path::{Path as FsPath, PathBuf},
    sync::Arc,
};

type Result<T> = std::result::Result<T, ApiError>;
type Payload<T> = std::result::Result<Json<T>, JsonRejection>;
type Mutation = Result<(StatusCode, Json<Revision>)>;
const LIMIT: u32 = 64;
const PROVENANCE: &str = "provenance.json";
#[path = "scratch_checks.rs"]
mod checks;
#[path = "scratch_experts.rs"]
mod experts;
#[path = "scratch_jobs.rs"]
pub(crate) mod jobs;
#[path = "scratch_locate.rs"]
pub(crate) mod locate;
#[path = "scratch_merge.rs"]
mod merge;
#[path = "scratch_program.rs"]
mod program;
#[path = "scratch_prune.rs"]
mod prune;

#[derive(Clone, Copy, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionState {
    Scratch,
    PublishRequested,
    Approved,
    Discarded,
}

impl RevisionState {
    pub fn key(self) -> &'static str {
        match self {
            Self::Scratch => "scratch",
            Self::PublishRequested => "publish_requested",
            Self::Approved => "approved",
            Self::Discarded => "discarded",
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct Revision {
    pub id: String,
    kind: String,
    pub state: RevisionState,
    pub(crate) sha256: String,
    qualification: String,
    pub(crate) library_model_id: Option<String>,
    pub provenance: Provenance,
    pub(crate) delta: Summary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    check_bundle: Option<checks::Bundle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) gguf: Option<jobs::PrivateGguf>,
}

impl Revision {
    pub(crate) fn finding_checks(&self) -> Option<(&str, serde_json::Value)> {
        self.check_bundle.as_ref()?.finding_receipt()
    }

    pub(crate) fn token_loss(&self) -> Option<(&str, &crate::scratch_token_loss::Reference)> {
        self.check_bundle.as_ref()?.token_loss()
    }

    pub(crate) fn is_compiled_program(&self) -> bool {
        self.provenance.compiled_program.is_some()
    }
}

#[derive(Deserialize, Serialize)]
pub struct Provenance {
    schema: String,
    pub parent_id: String,
    pub(crate) parent_sha256: String,
    revision_id: String,
    revision_sha256: String,
    worker: String,
    method: String,
    operations: Vec<Operation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    merge: Option<merge::Spec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pruning: Option<prune::Params>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pruning_trace: Option<prune::Evidence>,
    tensors: Vec<TensorInfo>,
    tool: String,
    tool_version: String,
    runtime_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    locate_edit: Option<serde_json::Value>,
    created_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compiled_program: Option<program::Recipe>,
}

impl Provenance {
    pub(crate) fn finding_edit(&self) -> (String, serde_json::Value) {
        if let Some(pruning) = &self.pruning {
            ("prune".into(), serde_json::json!({"pruning":pruning}))
        } else if !self.operations.is_empty() && self.method == "direct" {
            (
                "edits".into(),
                serde_json::json!({"operations":self.operations}),
            )
        } else {
            (self.method.clone(), serde_json::Value::Null)
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Create {
    parent_id: String,
    parent_sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Edit {
    operations: Vec<Operation>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Review {
    sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PublishRequest {
    sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    acknowledge_bundle: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Discard {}

pub fn router() -> Router<Arc<App>> {
    Router::new()
        .route("/scratch", get(list).post(create))
        .route("/scratch/merge", post(merge::run))
        .route("/scratch/compile_program", post(program::compile))
        .route("/scratch/{id}/run_program", post(program::run))
        .route("/scratch/{id}", get(get_revision))
        .route("/scratch/{id}/edits", post(edit))
        .route("/scratch/{id}/rome", post(locate::rome))
        .route("/scratch/{id}/memit", post(locate::memit))
        .route("/scratch/{id}/alphaedit", post(locate::alphaedit))
        .route("/scratch/{id}/prune", post(prune::model))
        .route("/scratch/experts/{id}/plan", post(experts::plan))
        .route("/scratch/experts/{id}/prune", post(experts::prune))
        .route("/scratch/{id}/delta", get(delta))
        .route("/scratch/{id}/gguf", post(jobs::convert))
        .route("/scratch/{id}/checks", get(checks::get).post(checks::run))
        .route("/scratch/{id}/discard", post(discard))
        .route("/scratch/{id}/publish-request", post(request_publish))
        .route("/admin/scratch/{id}/approve", post(approve))
}

async fn list(State(app): State<Arc<App>>) -> Result<Json<Vec<Revision>>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(store.scratches().map_err(ApiError::storage)?))
}

async fn get_revision(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
) -> Result<Json<Revision>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(visible(&store, &id)?))
}

async fn delta(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<Json<Summary>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(visible(&store, &id)?.delta))
}

async fn create(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    payload: Payload<Create>,
) -> Mutation {
    mutate(
        app,
        headers,
        "scratch-create",
        None,
        read(payload)?,
        |app, _, request, key| {
            build(
                app,
                request.parent_id,
                Some(request.parent_sha256),
                Vec::new(),
                None,
                key,
            )
        },
    )
    .await
}

async fn edit(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<Edit>,
) -> Mutation {
    mutate(
        app,
        headers,
        "scratch-edit",
        Some(id),
        read(payload)?,
        |app, target, request, key| {
            if !(1..=32).contains(&request.operations.len()) {
                return Err(ApiError::bad("expected 1 to 32 edit operations"));
            }
            build(
                app,
                request_id(target)?,
                None,
                request.operations,
                None,
                key,
            )
        },
    )
    .await
}

async fn discard(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<Discard>,
) -> Mutation {
    mutate(
        app,
        headers,
        "scratch-discard",
        Some(id),
        read(payload)?,
        |app, target, _, key| change(app, target, RevisionState::Discarded, None, None, key),
    )
    .await
}

async fn request_publish(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<PublishRequest>,
) -> Mutation {
    mutate(
        app,
        headers,
        "scratch-request",
        Some(id),
        read(payload)?,
        |app, target, request, key| {
            change(
                app,
                target,
                RevisionState::PublishRequested,
                Some(request.sha256),
                request.acknowledge_bundle,
                key,
            )
        },
    )
    .await
}

// This route has the same trust boundary as /api/admin/actions. The agent
// interface must not expose it. It does not grant qualification.
async fn approve(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<Review>,
) -> Mutation {
    mutate(
        app,
        headers,
        "scratch-approve",
        Some(id),
        read(payload)?,
        |app, target, request, key| {
            change(
                app,
                target,
                RevisionState::Approved,
                Some(request.sha256),
                None,
                key,
            )
        },
    )
    .await
}

async fn mutate<T: Serialize + Send + 'static>(
    app: Arc<App>,
    headers: HeaderMap,
    scope: &'static str,
    target: Option<String>,
    request: T,
    action: impl FnOnce(&App, Option<String>, T, Option<&Submission<'_>>) -> Mutation + Send + 'static,
) -> Mutation {
    let invalid = || ApiError::bad("invalid Idempotency-Key");
    let key = single(&headers, "idempotency-key").map_err(|_| invalid())?;
    if key.is_some_and(|key| !idempotency::valid_key(key)) {
        return Err(invalid());
    }
    let slot = app.scratch_slots.clone().try_acquire_owned().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "scratch editor is busy".into(),
        )
    })?;
    let key = key.map(str::to_owned);
    tokio::task::spawn_blocking(move || {
        let _slot = slot;
        // Use typed input and bind path parameters to the key as well.
        let key = key
            .as_deref()
            .map(|key| Submission::new(scope, key, &(&target, &request)))
            .transpose()
            .map_err(ApiError::storage)?;
        if let Some(key) = &key {
            let store = app.store.lock().map_err(ApiError::storage)?;
            match store.replay(key).map_err(ApiError::storage)? {
                Replay::New => {}
                Replay::Conflict => {
                    return Err(conflict(
                        "Idempotency-Key already used for a different request",
                    ));
                }
                Replay::Matched(id) => {
                    let row = store
                        .scratch(&id)
                        .map_err(ApiError::storage)?
                        .ok_or_else(|| {
                            ApiError::storage("idempotent scratch revision is missing")
                        })?;
                    if scope == "scratch-discard" {
                        remove(store.scratch_root(), &id)?;
                    }
                    return Ok((StatusCode::OK, Json(row)));
                }
            }
        }
        action(&app, target, request, key.as_ref())
    })
    .await
    .map_err(ApiError::storage)?
}

fn request_id(target: Option<String>) -> Result<String> {
    target.ok_or_else(|| ApiError::not_found("scratch revision"))
}

fn visible(store: &Store, id: &str) -> Result<Revision> {
    store
        .scratch(id)
        .map_err(ApiError::storage)?
        .filter(|row| row.state != RevisionState::Discarded)
        .ok_or_else(|| ApiError::not_found("scratch revision"))
}

fn conflict(message: &str) -> ApiError {
    ApiError(StatusCode::CONFLICT, message.into())
}

fn admit<'a>(app: &'a App, worker: &str) -> Result<&'a Lane> {
    let enabled = app
        .effective_hosts()
        .map_err(ApiError::storage)?
        .iter()
        .any(|host| {
            host.id == worker
                && host.local_edit
                && !host.scheduling_paused
                && host.capabilities["weight_apply"]
                && host.capabilities["weight_safetensors"]
        });
    app.local_edit
        .get(worker)
        .filter(|_| enabled)
        .ok_or_else(|| conflict("enable the local SafeTensors editor on this worker"))
}

fn require_source(app: &App, worker: &str, path: &FsPath) -> Result<()> {
    if !app
        .scratch_source_allowed(worker, path)
        .map_err(ApiError::storage)?
    {
        return Err(conflict("source is not in an approved model directory"));
    }
    Ok(())
}

fn build(
    app: &App,
    parent_id: String,
    supplied_pin: Option<String>,
    operations: Vec<Operation>,
    merge: Option<merge::Spec>,
    key: Option<&Submission<'_>>,
) -> Mutation {
    build_with_pruning(app, parent_id, supplied_pin, operations, merge, None, key)
}

fn build_with_pruning(
    app: &App,
    parent_id: String,
    supplied_pin: Option<String>,
    operations: Vec<Operation>,
    merge: Option<merge::Spec>,
    pruning: Option<prune::Params>,
    key: Option<&Submission<'_>>,
) -> Mutation {
    let (root, worker, source, expected, library_source, reserve, merge_paths) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        if store.scratch_count().map_err(ApiError::storage)? >= LIMIT {
            return Err(conflict("scratch revision limit reached"));
        }
        let root = store.scratch_root().to_owned();
        let scratch_parent = store.scratch(&parent_id).map_err(ApiError::storage)?;
        let (worker, source, expected, library_source) = if let Some(row) = scratch_parent {
            if row.state == RevisionState::Discarded {
                return Err(ApiError::not_found("scratch revision"));
            }
            if row.is_compiled_program() {
                return Err(conflict(
                    "compiled program revisions do not support scratch edits",
                ));
            }
            if supplied_pin.as_ref().is_some_and(|pin| pin != &row.sha256) {
                return Err(ApiError::bad("parent SHA-256 mismatch"));
            }
            (
                row.provenance.worker,
                root.join(&row.id).join(scratch::MODEL),
                row.sha256,
                false,
            )
        } else {
            let pin = supplied_pin.ok_or_else(|| ApiError::not_found("scratch revision"))?;
            let model = store
                .model(&parent_id)
                .map_err(ApiError::storage)?
                .ok_or_else(|| ApiError::not_found("parent revision"))?;
            let path = PathBuf::from(model.path);
            if model.kind != "safetensors" || store.is_scratch_path(&path) {
                return Err(ApiError::bad(
                    "use a standalone SafeTensors Library source or a scratch ID",
                ));
            }
            (model.host, path, pin, true)
        };
        let lane = admit(app, &worker)?;
        if library_source {
            require_source(app, &worker, &source)?;
        }
        let merge_paths = merge
            .as_ref()
            .map(|spec| spec.paths(app, &store, &root, &worker))
            .transpose()?;
        (
            root,
            worker,
            source,
            expected,
            library_source,
            lane.scratch_disk_reserve(),
            merge_paths,
        )
    };
    if !brewery_worker::is_digest(&expected) {
        return Err(ApiError::bad("expected a lowercase parent SHA-256"));
    }
    private_root(&root)?;
    let bytes = open_regular(&source)
        .map_err(|_| ApiError::bad("cannot open standalone SafeTensors source"))?
        .metadata()
        .map_err(ApiError::storage)?
        .len();
    if !(10..=scratch::MAX_FILE).contains(&bytes) {
        return Err(ApiError::bad("SafeTensors file exceeds scratch limit"));
    }
    let budget = bytes
        .saturating_add(reserve)
        .saturating_add(if merge.is_some() {
            bytes + scratch::MERGE_DELTA_OVERHEAD
        } else {
            scratch::DELTA_OVERHEAD
                + if operations.is_empty() && pruning.is_none() {
                    0
                } else {
                    scratch::MAX_EDIT_BYTES
                }
        });
    if brewery_worker::available(&root).map_err(ApiError::storage)? < budget {
        return Err(conflict(
            "not enough disk space for scratch and the configured reserve",
        ));
    }
    let calibration = match &pruning {
        Some(p) => p.calibration(app, &parent_id)?,
        None => None,
    };
    let id = id();
    let mut output = Uncommitted::new(root.join(&id))?;
    let result = if let Some((spec, paths)) = merge.as_ref().zip(merge_paths.as_ref()) {
        spec.build(&source, &expected, &output.path, paths)?
    } else {
        let output = if let Some(p) = &pruning {
            scratch::build_pruned(
                &source,
                &expected,
                &output.path,
                &p.plan,
                calibration.as_ref().map(|c| c.rows.as_slice()),
            )
        } else {
            scratch::build(&source, &expected, &output.path, &operations)
        };
        output.map_err(|e| {
            if e.downcast_ref::<std::io::Error>().is_some() {
                ApiError::storage(e)
            } else {
                ApiError::bad(e.to_string())
            }
        })?
    };
    let provenance = Provenance {
        schema: "brewery.scratch-provenance.v1".into(),
        parent_id,
        parent_sha256: expected,
        revision_id: id.clone(),
        revision_sha256: result.sha256.clone(),
        worker,
        method: if let Some(spec) = &merge {
            spec.method()
        } else if pruning.is_some() {
            "prune"
        } else if operations.is_empty() {
            "copy"
        } else {
            "direct"
        }
        .into(),
        operations,
        merge,
        pruning,
        pruning_trace: calibration.map(|c| c.evidence),
        tensors: result.tensors,
        tool: "brewery SafeTensors scratch editor".into(),
        tool_version: env!("CARGO_PKG_VERSION").into(),
        runtime_version: format!(
            "native-cpu-v1/{}-{}",
            std::env::consts::ARCH,
            std::env::consts::OS
        ),
        locate_edit: None,
        created_at_ms: now(),
        compiled_program: None,
    };
    brewery_worker::atomic(&output.path.join(PROVENANCE), &provenance)
        .map_err(ApiError::storage)?;
    File::open(&root)
        .and_then(|file| file.sync_all())
        .map_err(ApiError::storage)?;
    let row = Revision {
        id,
        kind: "scratch".into(),
        state: RevisionState::Scratch,
        sha256: result.sha256,
        qualification: "not release-qualified".into(),
        library_model_id: None,
        provenance,
        delta: result.delta,
        check_bundle: None,
        gguf: None,
    };
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    admit(app, &row.provenance.worker)?;
    if library_source {
        require_source(app, &row.provenance.worker, &source)?;
        if store
            .model(&row.provenance.parent_id)
            .map_err(ApiError::storage)?
            .is_none()
        {
            return Err(conflict("parent Library revision was removed"));
        }
    }
    if let Some(spec) = &row.provenance.merge {
        let current = spec.paths(app, &store, &root, &row.provenance.worker)?;
        if merge_paths.as_ref() != Some(&current) {
            return Err(conflict("merge input paths changed"));
        }
    }
    store.insert_scratch(&row, key).map_err(ApiError::storage)?;
    output.keep = true;
    Ok((StatusCode::CREATED, Json(row)))
}

fn change(
    app: &App,
    target: Option<String>,
    new: RevisionState,
    pin: Option<String>,
    acknowledgement: Option<String>,
    key: Option<&Submission<'_>>,
) -> Mutation {
    use RevisionState::*;
    let id = request_id(target)?;
    let (row, root) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let row = visible(&store, &id)?;
        if pin.is_some_and(|pin| pin != row.sha256) {
            return Err(conflict("review hash does not match this scratch revision"));
        }
        (row, store.scratch_root().to_owned())
    };
    let old = row.state;
    match (old, new) {
        (Scratch, PublishRequested)
        | (PublishRequested, Approved)
        | (Scratch | PublishRequested, Discarded) => {}
        _ => return Err(conflict("invalid scratch state transition")),
    }
    let path = root.join(&row.id).join(scratch::MODEL);
    if new != Discarded {
        // Reject incomplete bundles before hashing the full model. Recheck the
        // current record after the hash in case a worker finished in between.
        checks::require_publishable(&row, new, acknowledgement.as_deref())?;
        if row.is_compiled_program() {
            program::integrity(&row, &root.join(&row.id)).map_err(|error| conflict(&error))?;
        }
        let mut source =
            open_regular(&path).map_err(|_| conflict("scratch artifact is not available"))?;
        if scratch::file_sha256(&mut source).map_err(ApiError::storage)? != row.sha256 {
            return Err(conflict(
                "scratch artifact no longer matches the reviewed hash",
            ));
        }
    }
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    let mut row = visible(&store, &id)?;
    if row.state != old {
        return Err(conflict("scratch state changed before commit"));
    }
    if store
        .scratch_has_jobs(&id, None)
        .map_err(ApiError::storage)?
    {
        return Err(conflict("wait for active scratch jobs to finish"));
    }
    if new != Discarded {
        admit(app, &row.provenance.worker)?;
        if checks::require_publishable(&row, new, acknowledgement.as_deref())? {
            checks::acknowledge(&mut row);
        }
    } else if store.has_scratch_children(&id).map_err(ApiError::storage)? {
        return Err(conflict("discard the child scratch revisions first"));
    }
    let model = if new == Approved {
        let model = NewModel {
            host: row.provenance.worker.clone(),
            name: Some(program::library_name(&row)),
            path: path
                .to_str()
                .ok_or_else(|| ApiError::storage("non-UTF-8 scratch path"))?
                .into(),
            kind: "safetensors".into(),
        }
        .validate()
        .map_err(ApiError::bad)?;
        row.library_model_id = Some(model.id.clone());
        Some(model)
    } else {
        None
    };
    row.state = new;
    store
        .update_scratch(&row, old, model.as_ref(), key)
        .map_err(ApiError::storage)?;
    if new == Discarded {
        remove(&root, &id)?;
    }
    Ok((StatusCode::OK, Json(row)))
}

fn private_root(path: &FsPath) -> Result<()> {
    match fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(ApiError::storage(e)),
    }
    let meta = fs::symlink_metadata(path).map_err(ApiError::storage)?;
    if !meta.is_dir() || meta.mode() & 0o7777 != 0o700 || meta.uid() != unsafe { libc::geteuid() } {
        return Err(ApiError::storage(
            "scratch root must be private and controller-owned",
        ));
    }
    Ok(())
}

fn remove(root: &FsPath, id: &str) -> Result<()> {
    match fs::remove_dir_all(root.join(id)) {
        Ok(()) => File::open(root)
            .and_then(|file| file.sync_all())
            .map_err(ApiError::storage),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(ApiError::storage(e)),
    }
}

pub(crate) struct Uncommitted {
    path: PathBuf,
    keep: bool,
}

impl Uncommitted {
    pub(crate) fn keep(mut self) {
        self.keep = true;
    }

    fn new(path: PathBuf) -> Result<Self> {
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .map_err(ApiError::storage)?;
        Ok(Self { path, keep: false })
    }
}

impl Drop for Uncommitted {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
