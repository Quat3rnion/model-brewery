use super::*;
use crate::{
    domain::Model,
    scratch::{Revision, RevisionState},
    store::Store,
};
use brewery_worker::weight_source::{open_regular, scratch};
use std::path::{Path, PathBuf};

pub(super) fn run(app: &App, q: Pair) -> Result<Bytes> {
    let (a, a_sha, b, b_sha, delta) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let a = scratch(&store, &q.a)?;
        let b = scratch(&store, &q.b)?;
        let library = |id: &str, row: &Option<Revision>| {
            if row.is_some() {
                return Ok(None);
            }
            store
                .model(id)
                .map_err(ApiError::storage)?
                .map(Some)
                .ok_or_else(|| ApiError::not_found("revision"))
        };
        let la = library(&q.a, &a)?;
        let lb = library(&q.b, &b)?;
        let ra = a.as_ref().map(|r| origin(&store, r)).transpose()?;
        let rb = b.as_ref().map(|r| origin(&store, r)).transpose()?;
        let sha = match (&ra, &rb) {
            (Some(a), Some(b)) if a == b => &a.1,
            (None, Some(b)) if q.a == b.0 => &b.1,
            (Some(a), None) if q.b == a.0 => &a.1,
            _ => return Err(invalid("use two revisions of the same scratch source")),
        };
        let edge = a
            .as_ref()
            .filter(|r| r.provenance.parent_id == q.b)
            .or_else(|| b.as_ref().filter(|r| r.provenance.parent_id == q.a));
        let delta = edge.map(|r| (location(&store, &r.id, scratch::DELTA), r.delta.clone()));
        let (ap, ah) = source(app, &store, a.as_ref(), la, sha)?;
        let (bp, bh) = source(app, &store, b.as_ref(), lb, sha)?;
        (ap, ah, bp, bh, delta)
    };
    let (mut a, mut b) = (file(&a)?, file(&b)?);
    let mut delta = delta
        .map(|(path, s)| file(&path).map(|f| (f, s)))
        .transpose()?;
    let inputs = |a: &File, b: &File, delta: Option<&File>| -> Result<Vec<Stamp>> {
        let mut stamps = vec![stamp(a)?, stamp(b)?];
        if let Some(delta) = delta {
            stamps.push(stamp(delta)?);
        }
        Ok(stamps)
    };
    let stamps = inputs(&a, &b, delta.as_ref().map(|(f, _)| f))?;
    let key = Key::Tensors(q.clone());
    if let Some(bytes) = app.diffs.get(&key, &stamps)? {
        return Ok(bytes);
    }
    let tensors = scratch::diff::compare(
        &mut a,
        &a_sha,
        &mut b,
        &b_sha,
        delta
            .as_mut()
            .map(|(file, summary)| scratch::diff::Delta { file, summary }),
    )
    .map_err(invalid)?;
    if inputs(&a, &b, delta.as_ref().map(|(f, _)| f))? != stamps {
        return Err(invalid("tensor inputs changed during the diff"));
    }
    #[derive(Serialize)]
    struct Tensors {
        a_sha256: String,
        b_sha256: String,
        tensors: Vec<scratch::diff::TensorDiff>,
    }
    app.diffs.save(
        key,
        stamps,
        &q.report(Tensors {
            a_sha256: a_sha,
            b_sha256: b_sha,
            tensors,
        }),
    )
}

pub(super) fn scratch(store: &Store, id: &str) -> Result<Option<Revision>> {
    let row = store.scratch(id).map_err(ApiError::storage)?;
    if row
        .as_ref()
        .is_some_and(|r| r.state == RevisionState::Discarded)
    {
        return Err(ApiError::not_found("scratch revision"));
    }
    Ok(row)
}

pub(super) fn location(store: &Store, id: &str, leaf: &str) -> PathBuf {
    store.scratch_root().join(id).join(leaf)
}

fn origin(store: &Store, row: &Revision) -> Result<(String, String)> {
    // The nil parent denotes absence, not a common source for compiled roots.
    if row.is_compiled_program() {
        return Ok((row.id.clone(), row.sha256.clone()));
    }
    let mut id = row.provenance.parent_id.clone();
    let mut sha = row.provenance.parent_sha256.clone();
    for _ in 0..64 {
        let Some(parent) = scratch(store, &id)? else {
            return Ok((id, sha));
        };
        if parent.sha256 != sha {
            return Err(invalid("scratch ancestry does not match its parent hash"));
        }
        id = parent.provenance.parent_id;
        sha = parent.provenance.parent_sha256;
    }
    Err(invalid("scratch ancestry exceeds the diff limit"))
}

fn source(
    app: &App,
    store: &Store,
    row: Option<&Revision>,
    model: Option<Model>,
    pin: &str,
) -> Result<(PathBuf, String)> {
    if let Some(row) = row {
        return Ok((location(store, &row.id, scratch::MODEL), row.sha256.clone()));
    }
    let model = model.ok_or_else(|| ApiError::not_found("Library source"))?;
    let path = PathBuf::from(model.path);
    if model.kind != "safetensors"
        || store.is_scratch_path(&path)
        || !app
            .scratch_source_allowed(&model.host, &path)
            .unwrap_or(false)
    {
        return Err(invalid(
            "Library source is not in an approved SafeTensors directory",
        ));
    }
    Ok((path, pin.to_owned()))
}

fn file(path: &Path) -> Result<File> {
    open_regular(path).map_err(|_| ApiError::not_found("tensor diff artifact"))
}
