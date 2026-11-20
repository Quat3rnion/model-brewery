//! Revisioned access to directories on the controller-owned local edit lane.
use super::{ApiError, ApiResult, configured, infra_policy::Policy, record_audit};
use crate::{
    App,
    domain::{Envelope, Job},
    local_edit::{Lane, Lanes},
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::StatusCode,
    routing::post,
};
use brewery_worker::source_grants::{SourceGrant, revalidate, validate};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

const MAX_GRANTS: usize = 32;

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Action {
    Add,
    Replace,
    Remove,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    worker: String,
    revision: u64,
    action: Action,
    path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_path: Option<PathBuf>,
}

#[derive(Deserialize, Serialize)]
struct Plan {
    request: Request,
    confirmation: String,
    grant: Option<SourceGrant>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Confirm {
    worker: String,
    revision: u64,
    confirmation: String,
}

pub(super) fn router() -> Router<Arc<App>> {
    Router::new()
        .route("/admin/model-directories/plan", post(plan))
        .route("/admin/model-directories/confirm", post(confirm))
        .layer(DefaultBodyLimit::max(10 * 1024))
}

pub(super) fn view(lanes: &Lanes, id: &str, policy: &Policy) -> Value {
    let Some(lane) = lanes.get(id) else {
        return json!({"available":false,"reason":lanes.unavailable_reason(id),"configured":[],"granted":[]});
    };
    json!({"available":true,"configured":lane.configured,
        "granted":policy.model_directories.iter().map(|grant| &grant.path).collect::<Vec<_>>()})
}

fn available<'a>(app: &'a App, worker: &str) -> ApiResult<&'a Lane> {
    if !app
        .defined_hosts()
        .map_err(ApiError::storage)?
        .iter()
        .any(|host| host.id == worker)
    {
        return Err(ApiError(StatusCode::NOT_FOUND, "Unknown worker"));
    }
    app.local_edit.get(worker).ok_or(ApiError::conflict(
        "Model directories are supported only on local CPU editor workers",
    ))
}

fn overlap(a: &Path, b: &Path) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

fn proposed_grants(
    policy: &Policy,
    lane: &Lane,
    request: &Request,
    grant: Option<SourceGrant>,
) -> ApiResult<Vec<SourceGrant>> {
    let mut grants = policy.model_directories.clone();
    let invalid = || {
        ApiError(
            StatusCode::BAD_REQUEST,
            "Choose a new directory or an existing grant; configured directories cannot be changed here",
        )
    };
    match (request.action, &request.previous_path) {
        (Action::Add, None) => {}
        (Action::Remove, None) => {
            let index = grants
                .iter()
                .position(|g| g.path.as_os_str() == request.path.as_os_str())
                .ok_or_else(invalid)?;
            grants.remove(index);
        }
        (Action::Replace, Some(previous)) => {
            let index = grants
                .iter()
                .position(|g| g.path.as_os_str() == previous.as_os_str())
                .ok_or_else(invalid)?;
            grants.remove(index);
        }
        _ => return Err(invalid()),
    }
    if let Some(grant) = grant {
        if grants.len() >= MAX_GRANTS
            || grants.iter().any(|g| overlap(&g.path, &grant.path))
            || lane.configured.iter().any(|p| overlap(p, &grant.path))
        {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                "Directory overlaps existing access or the grant limit was reached",
            ));
        }
        grants.push(grant);
    }
    Ok(grants)
}

async fn plan(
    State(app): State<Arc<App>>,
    body: Result<Json<Request>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    let Json(request) =
        body.map_err(|_| ApiError(StatusCode::BAD_REQUEST, "Invalid model directory request"))?;
    let admin = configured(&app)?;
    let lane = available(&app, &request.worker)?;
    let db = admin.db.lock().map_err(ApiError::storage)?;
    let current = Policy::load(&db, &request.worker).map_err(ApiError::storage)?;
    if current.revision != request.revision {
        return Err(ApiError::conflict("Host policy revision is stale"));
    }
    let grant = if request.action == Action::Remove {
        None
    } else {
        Some(validate(&request.worker, &request.path, &app.local_edit.protected).map_err(|_| {
            ApiError(StatusCode::BAD_REQUEST, "Choose a readable absolute model directory on this worker, outside protected locations and without symlinks")
        })?)
    };
    proposed_grants(&current, lane, &request, grant.clone())?;
    let plan = Plan {
        request,
        confirmation: crate::domain::id(),
        grant,
    };
    db.execute(
        "INSERT INTO infra_directory_plan(worker,body) VALUES(?,?)
        ON CONFLICT(worker) DO UPDATE SET body=excluded.body",
        params![
            plan.request.worker,
            serde_json::to_string(&plan).map_err(ApiError::storage)?
        ],
    )
    .map_err(ApiError::storage)?;
    let mut row = serde_json::to_value(&plan.request).map_err(ApiError::storage)?;
    row["confirmation"] = json!(plan.confirmation);
    Ok(Json(row))
}

async fn confirm(
    State(app): State<Arc<App>>,
    body: Result<Json<Confirm>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    let Json(request) = body.map_err(|_| {
        ApiError(
            StatusCode::BAD_REQUEST,
            "Invalid model directory confirmation",
        )
    })?;
    let admin = configured(&app)?;
    {
        let _store = app.store.lock().map_err(ApiError::storage)?;
        let lane = available(&app, &request.worker)?;
        let mut db = admin.db.lock().map_err(ApiError::storage)?;
        let tx = db.transaction().map_err(ApiError::storage)?;
        let mut current = Policy::load(&tx, &request.worker).map_err(ApiError::storage)?;
        let stale = || ApiError::conflict("Directory review is stale; validate and save again");
        if current.revision != request.revision {
            return Err(stale());
        }
        let body = tx
            .query_row(
                "SELECT body FROM infra_directory_plan WHERE worker=?",
                [&request.worker],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(ApiError::storage)?
            .ok_or_else(stale)?;
        let plan: Plan = serde_json::from_str(&body).map_err(ApiError::storage)?;
        if plan.request.worker != request.worker
            || plan.request.revision != request.revision
            || plan.confirmation != request.confirmation
        {
            return Err(stale());
        }
        if let Some(grant) = &plan.grant {
            revalidate(&request.worker, grant, &app.local_edit.protected).map_err(|_| {
                ApiError::conflict("Directory changed after review; validate and save again")
            })?;
        }
        current.model_directories = proposed_grants(&current, lane, &plan.request, plan.grant)?;
        current.revision = current.revision.checked_add(1).ok_or_else(stale)?;
        current
            .save(&tx, &request.worker)
            .map_err(ApiError::storage)?;
        tx.execute(
            "DELETE FROM infra_directory_plan WHERE worker=?",
            [&request.worker],
        )
        .map_err(ApiError::storage)?;
        let change = match plan.request.action {
            Action::Add => "add",
            Action::Replace => "replace",
            Action::Remove => "remove",
        };
        let mut audit = json!({"action":"set_model_directories","worker":request.worker,"revision":current.revision,
            "change":change,"path":plan.request.path});
        if let Some(previous) = plan.request.previous_path {
            audit["previous_path"] = json!(previous);
        }
        record_audit(&tx, &audit, "applied")?;
        tx.commit().map_err(ApiError::storage)?;
    }
    admin.view(&app).await.map(Json).map_err(ApiError::storage)
}

pub(crate) struct Access {
    pub revision: Option<u64>,
    pub grants: Vec<SourceGrant>,
}

pub(super) fn access(
    lanes: &Lanes,
    lane: &Lane,
    job: &Job,
    envelope: &Envelope,
    policy: &Policy,
) -> Option<Access> {
    let mut access = Access {
        revision: None,
        grants: Vec::new(),
    };
    for (i, input) in envelope.inputs.iter().enumerate() {
        let path = Path::new(&input.path);
        if input.host != job.host {
            return None;
        }
        if lane.configured.iter().any(|root| path.starts_with(root)) || path.starts_with(&lane.root)
        {
            continue; // The worker verifies configured roots and private publications independently.
        }
        if i != 0 {
            return None;
        }
        let grant = policy
            .model_directories
            .iter()
            .find(|g| path.starts_with(&g.path))?;
        revalidate(&job.host, grant, &lanes.protected).ok()?;
        access.revision = Some(policy.revision);
        access.grants.push(grant.clone());
    }
    Some(access)
}
