use crate::{
    App, backend_hf,
    domain::{Model, NewModel},
    web::ApiError,
};
use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use sha2::{Digest as _, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read as _, Write as _},
    os::unix::fs::{MetadataExt as _, OpenOptionsExt as _},
    path::{Path, PathBuf},
    sync::Arc,
};

const MAX_BYTES: usize = 256 << 10;
const MAX_FILES: usize = 128;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Upload {
    name: String,
    activations: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct Batch<'a> {
    activations: &'a [Vec<f64>],
    targets: &'a [Vec<f64>],
}

fn matrix(rows: &[Vec<f64>]) -> bool {
    (1..=1024).contains(&rows.len())
        && (1..=16384).contains(&rows[0].len())
        && rows
            .iter()
            .all(|row| row.len() == rows[0].len() && row.iter().all(|n| n.is_finite()))
}

pub fn is_managed(root: Option<&Path>, model: &Model) -> bool {
    let path = Path::new(&model.path);
    model.host == backend_hf::ID
        && model.kind == "weight-calibration"
        && root.is_some_and(|root| path.parent() == Some(root))
        && path.extension().is_some_and(|ext| ext == "json")
        && path
            .file_stem()
            .and_then(|name| name.to_str())
            .is_some_and(brewery_worker::is_digest)
}

pub async fn register(
    State(app): State<Arc<App>>,
    payload: Result<Json<Box<RawValue>>, JsonRejection>,
) -> Result<(StatusCode, Json<Model>), ApiError> {
    let bad = || ApiError::bad("invalid calibration batch");
    let Json(raw) =
        payload.map_err(|err| ApiError(err.status(), "invalid calibration request".into()))?;
    if !raw.get().starts_with('{') {
        return Err(bad());
    }
    let upload: Upload = serde_json::from_str(raw.get()).map_err(|_| bad())?;
    if !matrix(&upload.activations)
        || !matrix(&upload.targets)
        || upload.activations.len() != upload.targets.len()
    {
        return Err(bad());
    }
    let bytes = serde_json::to_vec(&Batch {
        activations: &upload.activations,
        targets: &upload.targets,
    })
    .map_err(|_| bad())?;
    if bytes.len() > MAX_BYTES {
        return Err(bad());
    }
    let unavailable = || {
        ApiError(
            StatusCode::CONFLICT,
            "backend calibration tuning is unavailable".into(),
        )
    };
    let root = app.backend_calibration.as_deref().ok_or_else(unavailable)?;
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    if !app
        .effective_hosts()
        .map_err(ApiError::storage)?
        .iter()
        .any(|host| {
            host.id == backend_hf::ID && host.capabilities["weight_tune"] && !host.scheduling_paused
        })
    {
        return Err(unavailable());
    }
    let path = root.join(format!("{:x}.json", Sha256::digest(&bytes)));
    let model = NewModel {
        name: Some(upload.name),
        host: backend_hf::ID.into(),
        kind: "weight-calibration".into(),
        path: path.to_str().ok_or_else(bad)?.into(),
    }
    .validate()
    .map_err(ApiError::bad)?;
    if store
        .models()
        .map_err(ApiError::storage)?
        .iter()
        .any(|m| m.host == backend_hf::ID && m.path == model.path)
    {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "this calibration batch is already registered".into(),
        ));
    }
    if !publish(root, &path, &bytes).map_err(ApiError::storage)? {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "backend calibration storage limit reached".into(),
        ));
    }
    store.add_model(&model, None).map_err(ApiError::storage)?;
    Ok((StatusCode::CREATED, Json(model)))
}

fn publish(root: &Path, path: &Path, bytes: &[u8]) -> Result<bool, crate::Error> {
    brewery_worker::operational::private_root(root)?;
    match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
        .open(path)
    {
        Ok(file) => {
            let meta = file.metadata()?;
            let mut saved = Vec::new();
            if !meta.is_file()
                || meta.mode() & 0o7777 != 0o400
                || meta.nlink() != 1
                || meta.uid() != unsafe { libc::geteuid() }
                || file
                    .take((MAX_BYTES + 1) as u64)
                    .read_to_end(&mut saved)
                    .is_err()
                || saved != bytes
            {
                return Err("existing calibration object is not the verified upload".into());
            }
            return Ok(true);
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }
    if fs::read_dir(root)?.take(MAX_FILES).count() >= MAX_FILES {
        return Ok(false);
    }
    let staged: PathBuf = root.join(format!(".{}", uuid::Uuid::new_v4()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o400)
        .open(&staged)?;
    let result = (|| -> io::Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        // Publication must not replace an existing content-addressed object.
        fs::hard_link(&staged, path)?;
        fs::remove_file(&staged)?;
        File::open(root)?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staged);
    }
    result?;
    Ok(true)
}
