//! Wrap the existing REAP planner. Managed Library apply is never used here.
use super::*;
use crate::{dispatch::read_limited, domain::Model};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{io, process::Stdio, time::Duration};

const PLANNER: &[u8] = include_bytes!("../../../qualification/expert_plan.py");
const USAGE: &[u8] = include_bytes!("../../../qualification/expert_usage.py");
const MAX_MAP: usize = 128 << 10;
const MAX_SPLIT: usize = 96 << 10;
const MAX_SOURCE: u64 = 32 << 20; // Same isolated GGUF lane as managed preview.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Plan {
    source_sha256: String,
    keep: u16,
    calibration: String,
    held_out: String,
    routing_evidence: Option<crate::routing_atlas::Evidence>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Apply {
    plan: String,
    plan_sha256: String,
    reviewed: bool,
}

pub(super) async fn plan(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    payload: Payload<Plan>,
) -> Result<Json<Value>> {
    let mut body = read(payload)?;
    if !brewery_worker::is_digest(&body.source_sha256)
        || !(1..=1024).contains(&body.keep)
        || [&body.calibration, &body.held_out]
            .iter()
            .any(|s| s.is_empty() || s.len() > MAX_SPLIT)
    {
        return Err(ApiError::bad(
            "invalid source hash, keep count, or evidence size",
        ));
    }
    let model = gguf(&app, &id)?;
    admit(&app, &model.host)?;
    let source = PathBuf::from(&model.path);
    require_source(&app, &model.host, &source)?;
    let routing_evidence = match body.routing_evidence.take() {
        Some(spec) => crate::routing_atlas::evidence(&app, spec, &id, &body.source_sha256).await?,
        None => Value::Null,
    };
    let slot = app.scratch_slots.clone().try_acquire_owned().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "scratch editor is busy".into(),
        )
    })?;
    let root = app
        .store
        .lock()
        .map_err(ApiError::storage)?
        .scratch_root()
        .to_owned();
    let work = tokio::task::spawn_blocking(move || {
        let mut source =
            open_regular(&source).map_err(|_| conflict("cannot open the registered local GGUF"))?;
        let size = source.metadata().map_err(ApiError::storage)?.len();
        if !(1..=MAX_SOURCE).contains(&size) {
            return Err(ApiError::bad(
                "REAP planning requires a source of at most 32 MiB",
            ));
        }
        let mut hash = Sha256::new();
        let n = io::copy(&mut io::Read::take(&mut source, size + 1), &mut hash)
            .map_err(ApiError::storage)?;
        if n != size || format!("{:x}", hash.finalize()) != body.source_sha256 {
            return Err(conflict("registered GGUF SHA-256 mismatch"));
        }
        private_root(&root)?;
        let work = Uncommitted::new(root.join(crate::domain::id()))?;
        for (name, data) in [
            ("expert_plan.py", PLANNER),
            ("expert_usage.py", USAGE),
            ("calibration.json", body.calibration.as_bytes()),
            ("held-out.json", body.held_out.as_bytes()),
        ] {
            fs::write(work.path.join(name), data).map_err(ApiError::storage)?;
        }
        Ok((work, body.source_sha256, body.keep, slot))
    })
    .await
    .map_err(ApiError::storage)??;
    let (work, source_sha256, keep, _slot) = work;
    let mut child = tokio::process::Command::new("/usr/bin/python3")
        .args([
            "-B",
            "-E",
            "-s",
            "expert_plan.py",
            "reap",
            "--source-sha256",
        ])
        .arg(&source_sha256)
        .args([
            "--calibration",
            "calibration.json",
            "--held-out",
            "held-out.json",
            "--keep",
        ])
        .arg(keep.to_string())
        .env_clear()
        .current_dir(&work.path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| {
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "local REAP planner is not available".into(),
            )
        })?;
    let (stdout, stderr) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let (bytes, detail, status) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::try_join!(
            read_limited(stdout, MAX_MAP),
            read_limited(stderr, 4096),
            child.wait()
        )
    })
    .await
    .map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "local REAP planner timed out".into(),
        )
    })?
    .map_err(|_| ApiError::bad("invalid or oversized REAP planner output"))?;
    if !status.success() {
        return Err(ApiError::bad(
            String::from_utf8_lossy(&detail)
                .lines()
                .find(|s| s.starts_with("expert_plan: "))
                .unwrap_or("REAP planner rejected the evidence")
                .to_owned(),
        ));
    }
    let plan_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let exact = String::from_utf8(bytes).map_err(ApiError::storage)?;
    let map: Value = serde_json::from_str(&exact).map_err(ApiError::storage)?;
    if map["schema"] != "brewery.expert-map.v1"
        || map["source_sha256"] != source_sha256
        || map["review"]["method"] != "reap"
        || map["review"]["release_qualified"] != false
    {
        return Err(ApiError::storage("unexpected REAP plan output"));
    }
    Ok(Json(json!({
        "parent_id": id, "source_sha256": source_sha256,
        "plan": exact, "plan_sha256": plan_sha256,
        "planner_sha256": format!("{:x}", Sha256::digest(PLANNER)),
        "expert_usage_sha256": format!("{:x}", Sha256::digest(USAGE)),
        "routing_evidence": routing_evidence,
        "qualification": "not release-qualified"
    })))
}

pub(super) async fn prune(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    payload: Payload<Apply>,
) -> Result<Json<Revision>> {
    let body = read(payload)?;
    if !body.reviewed
        || !(1..=MAX_MAP).contains(&body.plan.len())
        || format!("{:x}", Sha256::digest(body.plan.as_bytes())) != body.plan_sha256
    {
        return Err(ApiError::bad(
            "review and supply the SHA-256 of the exact expert map bytes",
        ));
    }
    let map: Value =
        serde_json::from_str(&body.plan).map_err(|_| ApiError::bad("invalid expert map JSON"))?;
    if map["schema"] != "brewery.expert-map.v1"
        || !map["source_sha256"]
            .as_str()
            .is_some_and(brewery_worker::is_digest)
        || map["review"]["method"] != "reap"
        || map["review"]["release_qualified"] != false
    {
        return Err(ApiError::bad("use a reviewed source-bound REAP map"));
    }
    gguf(&app, &id)?;
    // A future apply must validate the plan again against the admitted source
    // and publish only through the W3 scratch transaction. Do not send this
    // request to weight_expert_apply: it registers a Library revision.
    Err(ApiError(
        StatusCode::SERVICE_UNAVAILABLE,
        "expert pruning is pending a verified writer for private scratch revisions".into(),
    ))
}

fn gguf(app: &App, id: &str) -> Result<Model> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    store
        .model(id)
        .map_err(ApiError::storage)?
        .filter(|row| row.kind == "gguf" && !store.is_scratch_path(FsPath::new(&row.path)))
        .ok_or_else(|| ApiError::not_found("registered GGUF model"))
}
