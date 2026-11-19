use super::{
    Provision, Registry, Row, State as RegistryState, Target, id_valid, name_valid, transport,
};
use crate::{App, store::Store, web::ApiError};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State, rejection::JsonRejection},
    http::StatusCode,
    routing::{get, post, put},
};
use serde::Deserialize;
use serde_json::{Value, json, value::RawValue};
use std::sync::{Arc, MutexGuard};

type Result<T> = std::result::Result<T, ApiError>;
type Shared = Arc<App>;

pub fn router() -> Router<Shared> {
    Router::new()
        .route("/admin/ssh-hosts", get(list).post(create))
        .route("/admin/ssh-hosts/{id}", put(update).delete(remove))
        .route("/admin/ssh-hosts/{id}/action", post(action))
        .layer(DefaultBodyLimit::max(8 * 1024))
}

// Flat configuration objects are intentional; deserialize the target
// separately from the profile metadata so unknown fields are always denied.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Definition {
    id: Option<String>,
    revision: Option<u64>,
    name: String,
    host: String,
    user: String,
    port: u16,
    fingerprint: String,
}
impl Definition {
    fn target(&self) -> Target {
        Target {
            host: self.host.clone(),
            user: self.user.clone(),
            port: self.port,
            fingerprint: self.fingerprint.clone(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Revision {
    revision: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    Test,
    Plan,
    Install,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Action {
    action: Operation,
    revision: u64,
    confirmation: Option<String>,
}

type Payload = std::result::Result<Json<Box<RawValue>>, JsonRejection>;

fn body<T: serde::de::DeserializeOwned>(payload: Payload) -> Result<T> {
    let invalid = |code| ApiError(code, "invalid SSH definition request".into());
    let Json(raw) = payload.map_err(|err| invalid(err.status()))?;
    if !raw.get().starts_with('{') {
        return Err(invalid(StatusCode::UNPROCESSABLE_ENTITY));
    }
    serde_json::from_str(raw.get()).map_err(|_| invalid(StatusCode::UNPROCESSABLE_ENTITY))
}

fn conflict(message: &'static str) -> ApiError {
    ApiError(StatusCode::CONFLICT, message.into())
}
fn unavailable() -> ApiError {
    ApiError(
        StatusCode::SERVICE_UNAVAILABLE,
        "SSH provisioning helper unavailable".into(),
    )
}
fn refused() -> ApiError {
    ApiError(
        StatusCode::BAD_GATEWAY,
        "SSH verification or provisioning failed".into(),
    )
}

impl Registry {
    fn mutate(&self) -> Result<tokio::sync::SemaphorePermit<'_>> {
        self.changes
            .try_acquire()
            .map_err(|_| conflict("another SSH definition change is in progress"))
    }

    fn lock(&self) -> Result<MutexGuard<'_, RegistryState>> {
        self.state.lock().map_err(ApiError::storage)
    }

    fn row<'a>(&self, state: &'a RegistryState, id: &str, revision: u64) -> Result<&'a Row> {
        let row = state
            .rows
            .get(id)
            .filter(|_| !self.reserved.contains(&id.to_ascii_lowercase()))
            .ok_or_else(|| ApiError::not_found("SSH definition"))?;
        if row.revision != revision {
            return Err(conflict("stale SSH definition revision"));
        }
        Ok(row)
    }
}

fn busy(store: &Store, id: &str) -> Result<()> {
    if store.ssh_host_in_use(id).map_err(ApiError::storage)? {
        return Err(conflict(
            "worker has queued or running jobs or active conversions",
        ));
    }
    Ok(())
}

fn next_revision(revision: u64) -> Result<u64> {
    revision
        .checked_add(1)
        .ok_or_else(|| conflict("SSH revision limit reached"))
}

async fn list(State(app): State<Shared>) -> Result<Json<Value>> {
    let ssh = &app.ssh_hosts;
    let state = ssh.lock()?;
    Ok(Json(
        json!({"hosts":state.rows.values().map(Row::view).collect::<Vec<_>>(), "available":ssh.helper.is_some()}),
    ))
}

async fn create(State(app): State<Shared>, payload: Payload) -> Result<(StatusCode, Json<Value>)> {
    let definition: Definition = body(payload)?;
    let target = definition.target();
    let id = definition
        .id
        .ok_or_else(|| ApiError::bad("SSH host ID is required"))?;
    if !id_valid(&id)
        || !name_valid(&definition.name)
        || !target.valid()
        || definition.revision.is_some()
    {
        return Err(ApiError::bad("invalid SSH host fields"));
    }
    let ssh = &app.ssh_hosts;
    let _change = ssh.mutate()?;
    let store = app.store.lock().map_err(ApiError::storage)?;
    let mut state = ssh.lock()?;
    let key = id.to_ascii_lowercase();
    if ssh.reserved.contains(&key)
        || state.retired.contains(&key)
        || state.rows.keys().any(|id| id.eq_ignore_ascii_case(&key))
        || store.host_has_history(&id).map_err(ApiError::storage)?
    {
        return Err(conflict("SSH host ID is already in use or retired"));
    }
    if state.rows.len() >= 64 {
        return Err(conflict("SSH definition limit reached"));
    }
    let row = Row {
        id: id.clone(),
        name: definition.name,
        target,
        revision: 1,
        provision: Provision::Defined,
    };
    let mut next = state.clone();
    next.rows.insert(id, row.clone());
    ssh.commit(&mut state, next).map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(row.view())))
}

async fn update(
    State(app): State<Shared>,
    Path(id): Path<String>,
    payload: Payload,
) -> Result<Json<Value>> {
    let definition: Definition = body(payload)?;
    let revision = definition
        .revision
        .ok_or_else(|| ApiError::bad("SSH definition revision is required"))?;
    let target = definition.target();
    if definition.id.is_some() || !name_valid(&definition.name) || !target.valid() {
        return Err(ApiError::bad("invalid SSH host fields"));
    }
    let ssh = &app.ssh_hosts;
    let _change = ssh.mutate()?;
    let store = app.store.lock().map_err(ApiError::storage)?;
    let mut state = ssh.lock()?;
    let mut row = ssh.row(&state, &id, revision)?.clone();
    busy(&store, &id)?;
    if !row.target.matches(&target) {
        if matches!(row.provision, Provision::Installed { .. }) {
            return Err(conflict(
                "installed SSH identity cannot change; use a new host ID",
            ));
        }
        row.target = target;
        row.provision = Provision::Defined;
    }
    row.name = definition.name;
    row.revision = next_revision(revision)?;
    let mut next = state.clone();
    next.rows.insert(id, row.clone());
    ssh.commit(&mut state, next).map_err(ApiError::storage)?;
    Ok(Json(row.view()))
}

async fn remove(
    State(app): State<Shared>,
    Path(id): Path<String>,
    payload: Payload,
) -> Result<StatusCode> {
    let Revision { revision } = body(payload)?;
    let ssh = &app.ssh_hosts;
    let _change = ssh.mutate()?;
    let store = app.store.lock().map_err(ApiError::storage)?;
    let mut state = ssh.lock()?;
    ssh.row(&state, &id, revision)?;
    busy(&store, &id)?;
    let mut next = state.clone();
    next.rows.remove(&id);
    next.retired.insert(id.to_ascii_lowercase());
    ssh.commit(&mut state, next).map_err(ApiError::storage)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn action(
    State(app): State<Shared>,
    Path(id): Path<String>,
    payload: Payload,
) -> Result<Json<Value>> {
    let Action {
        action,
        revision,
        confirmation,
    } = body(payload)?;
    let ssh = &app.ssh_hosts;
    let _change = ssh.mutate()?;
    let mut row = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let state = ssh.lock()?;
        let row = ssh.row(&state, &id, revision)?.clone();
        busy(&store, &id)?;
        row
    };
    let (operation, expected_state, prior_key) =
        match (&action, &row.provision, confirmation.as_deref()) {
            (Operation::Test, Provision::Defined, None) => ("test", "verified", None),
            (Operation::Test, Provision::Verified { key }, None) => ("test", "verified", Some(key)),
            (
                Operation::Plan,
                Provision::Verified { key } | Provision::Planned { key, .. },
                None,
            ) => ("plan", "planned", Some(key)),
            (
                Operation::Install,
                Provision::Planned {
                    key,
                    confirmation: expected,
                    ..
                },
                Some(seen),
            ) if expected == seen => ("install", "installed", Some(key)),
            _ => {
                return Err(conflict(
                    "SSH action requires a matching verified definition and current confirmation",
                ));
            }
        };
    let helper = ssh.helper.as_ref().ok_or_else(unavailable)?;
    let reply = helper
        .call(&row, operation, confirmation.as_deref())
        .await
        .ok_or_else(refused)?;
    let key = reply
        .host_key
        .filter(|key| {
            transport::key_matches(key, &row.target.fingerprint)
                && prior_key.is_none_or(|expected| expected == key)
        })
        .ok_or_else(refused)?;
    if reply.state != expected_state {
        return Err(refused());
    }
    row.provision = match action {
        Operation::Test => Provision::Verified { key },
        Operation::Plan => Provision::Planned {
            key,
            confirmation: reply
                .confirmation
                .filter(|s| brewery_worker::is_digest(s))
                .ok_or_else(refused)?,
            install: reply.install.filter(|i| i.valid(&id)).ok_or_else(refused)?,
        },
        Operation::Install => {
            let Provision::Planned { install, .. } = &row.provision else {
                unreachable!()
            };
            let installed = reply.install.filter(|i| i == install).ok_or_else(refused)?;
            Provision::Installed {
                key,
                install: installed,
            }
        }
    };
    let store = app.store.lock().map_err(ApiError::storage)?;
    let mut state = ssh.lock()?;
    ssh.row(&state, &id, revision)?;
    busy(&store, &id)?;
    row.revision = next_revision(revision)?;
    let mut next = state.clone();
    next.rows.insert(id, row.clone());
    ssh.commit(&mut state, next).map_err(ApiError::storage)?;
    Ok(Json(row.view()))
}
