use super::*;
use brewery_worker::weight_source::scratch::merge::{self as engine, Method};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    base_id: String,
    base_sha256: String,
    algorithm: Method,
    inputs: Vec<Pin>,
}

#[derive(Deserialize, Serialize)]
pub struct Spec {
    algorithm: Method,
    inputs: Vec<Pin>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    id: String,
    sha256: String,
    weight: f32,
}

pub(super) async fn run(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    payload: Payload<Request>,
) -> Mutation {
    mutate(
        app,
        headers,
        "scratch-merge",
        None,
        read(payload)?,
        |app, _, request, key| {
            if !(1..=engine::MAX_INPUTS).contains(&request.inputs.len()) {
                return Err(ApiError::bad("expected 1 to 16 merge inputs"));
            }
            build(
                app,
                request.base_id,
                Some(request.base_sha256),
                Vec::new(),
                Some(Spec {
                    algorithm: request.algorithm,
                    inputs: request.inputs,
                }),
                key,
            )
        },
    )
    .await
}

impl Spec {
    pub(super) fn method(&self) -> &'static str {
        self.algorithm.name()
    }

    pub(super) fn paths(
        &self,
        app: &App,
        store: &Store,
        root: &FsPath,
        worker: &str,
    ) -> Result<Vec<PathBuf>> {
        self.inputs
            .iter()
            .map(|pin| {
                if !brewery_worker::is_digest(&pin.sha256) {
                    return Err(ApiError::bad("expected a lowercase merge input SHA-256"));
                }
                if let Some(row) = store.scratch(&pin.id).map_err(ApiError::storage)? {
                    if row.state == RevisionState::Discarded {
                        return Err(ApiError::not_found("merge input revision"));
                    }
                    if row.provenance.worker != worker {
                        return Err(conflict("merge inputs must use the base worker"));
                    }
                    if row.sha256 != pin.sha256 {
                        return Err(ApiError::bad("merge input SHA-256 mismatch"));
                    }
                    return Ok(root.join(row.id).join(scratch::MODEL));
                }
                let model = store
                    .model(&pin.id)
                    .map_err(ApiError::storage)?
                    .ok_or_else(|| ApiError::not_found("merge input revision"))?;
                if model.host != worker {
                    return Err(conflict("merge inputs must use the base worker"));
                }
                let path = PathBuf::from(model.path);
                if model.kind != "safetensors" || store.is_scratch_path(&path) {
                    return Err(ApiError::bad(
                        "use a standalone SafeTensors Library input or a scratch ID",
                    ));
                }
                require_source(app, worker, &path)?;
                Ok(path)
            })
            .collect()
    }

    pub(super) fn build(
        &self,
        source: &FsPath,
        expected: &str,
        dir: &FsPath,
        paths: &[PathBuf],
    ) -> Result<scratch::Output> {
        let inputs: Vec<_> = self
            .inputs
            .iter()
            .zip(paths)
            .map(|(pin, path)| engine::Input {
                path,
                sha256: &pin.sha256,
                weight: pin.weight,
            })
            .collect();
        engine::build(source, expected, dir, self.algorithm, &inputs).map_err(|e| {
            if e.downcast_ref::<std::io::Error>().is_some() {
                ApiError::storage(e)
            } else {
                ApiError::bad(format!("{e:#}"))
            }
        })
    }
}
