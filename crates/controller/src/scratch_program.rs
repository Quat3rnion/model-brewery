use super::*;
use axum::response::IntoResponse;
use brewery_worker::compiled_program as runtime;
use serde_json::{Value, json};
use std::{ffi::OsStr, io::Read};

const ROOT_ID: &str = "00000000-0000-0000-0000-000000000000";
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const KIND: &str = "compiled_program";
const SCHEMA: &str = "brewery.compiled-program.v1";
const CONFIG: &str = "config.json";
const GROUND_TRUTH: &str = "ground-truth.v1.json";
const MAX_GROUND_TRUTH: u64 = 64 << 10;
const MAX_MODEL: u64 = 1 << 20;

#[derive(Clone, Copy, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Program {
    Sort,
    Histogram,
    Reverse,
    FracPrevs,
    AltaParityRelative,
    AltaParityAbsolute,
}

impl Program {
    fn vocabulary(self) -> u8 {
        if matches!(self, Self::AltaParityAbsolute | Self::AltaParityRelative) {
            2
        } else {
            4
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Self::Sort => "sort",
            Self::Histogram => "histogram",
            Self::Reverse => "reverse",
            Self::FracPrevs => "frac_prevs",
            Self::AltaParityRelative => "alta_parity_relative",
            Self::AltaParityAbsolute => "alta_parity_absolute",
        }
    }
}

#[derive(Deserialize, Serialize)]
pub(super) struct Recipe {
    program: Program,
    config_sha256: String,
    runner_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ground_truth_sha256: Option<String>,
}

pub(super) fn library_name(row: &Revision) -> String {
    row.provenance.compiled_program.as_ref().map_or_else(
        || format!("Scratch revision ({})", row.id),
        |recipe| format!("Compiled program {} ({})", recipe.program.slug(), row.id),
    )
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Compile {
    worker: String,
    program: Program,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Run {
    sha256: String,
    tokens: Vec<u8>,
}

#[derive(Deserialize)]
struct Receipt {
    sha256: String,
    config_sha256: String,
    ground_truth_sha256: Option<String>,
}

#[derive(Deserialize)]
struct Config {
    schema: String,
    program: Program,
    architecture: String,
    compiler_commit: String,
    runtime: String,
}

struct Runner {
    python: PathBuf,
    script: PathBuf,
    sha256: String,
}

impl Runner {
    fn configured() -> Result<Self> {
        let path = |key| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .filter(|p| p.is_absolute() && p.is_file())
                .ok_or_else(|| conflict("configure the local compiled program runtime"))
        };
        let python = path(runtime::PYTHON)?;
        let script = path(runtime::SCRIPT)?;
        let sha256 = brewery_worker::hash(&script).map_err(ApiError::storage)?;
        Ok(Self {
            python,
            script,
            sha256,
        })
    }

    fn review(recipe: &Recipe) -> Result<Self> {
        let runner = Self::configured()?;
        if recipe.runner_sha256 != runner.sha256 {
            return Err(conflict(
                "compiled program runner hash does not match provenance",
            ));
        }
        Ok(runner)
    }

    fn call<T: serde::de::DeserializeOwned>(&self, args: &[&OsStr]) -> Result<T> {
        runtime::run(&self.python, &self.script, args).map_err(ApiError::storage)
    }
}

pub(super) async fn compile(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    payload: Payload<Compile>,
) -> Mutation {
    mutate(
        app,
        headers,
        "scratch-compile-program",
        None,
        read(payload)?,
        |app, _, request, key| build(app, request, key),
    )
    .await
}

fn build(app: &App, request: Compile, key: Option<&Submission<'_>>) -> Mutation {
    let (root, reserve) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        if store.scratch_count().map_err(ApiError::storage)? >= LIMIT {
            return Err(conflict("scratch revision limit reached"));
        }
        (
            store.scratch_root().to_owned(),
            admit(app, &request.worker)?.scratch_disk_reserve(),
        )
    };
    let runner = Runner::configured()?;
    private_root(&root)?;
    if brewery_worker::available(&root).map_err(ApiError::storage)?
        < reserve.saturating_add(MAX_MODEL)
    {
        return Err(conflict(
            "not enough disk space for scratch and the configured reserve",
        ));
    }
    let id = id();
    let mut output = Uncommitted::new(root.join(&id))?;
    let receipt: Receipt = runner.call(&[
        OsStr::new("compile"),
        OsStr::new(request.program.slug()),
        output.path.as_os_str(),
    ])?;
    let recipe = Recipe {
        program: request.program,
        config_sha256: receipt.config_sha256,
        runner_sha256: runner.sha256,
        ground_truth_sha256: receipt.ground_truth_sha256,
    };
    let (config, tensors) =
        inspect(&output.path, &receipt.sha256, &recipe).map_err(ApiError::storage)?;
    let assets = [
        Some(scratch::MODEL),
        Some(CONFIG),
        recipe.ground_truth_sha256.as_ref().map(|_| GROUND_TRUTH),
    ];
    let provenance = Provenance {
        schema: "brewery.scratch-provenance.v1".into(),
        parent_id: ROOT_ID.into(),
        parent_sha256: EMPTY_SHA256.into(),
        revision_id: id.clone(),
        revision_sha256: receipt.sha256.clone(),
        worker: request.worker,
        method: "compile_program".into(),
        operations: Vec::new(),
        merge: None,
        pruning: None,
        pruning_trace: None,
        tensors,
        tool: config.architecture,
        tool_version: config.compiler_commit,
        runtime_version: format!(
            "numpy-cpu-v1/{}-{} {}",
            std::env::consts::ARCH,
            std::env::consts::OS,
            config.runtime,
        ),
        locate_edit: None,
        created_at_ms: now(),
        compiled_program: Some(recipe),
    };
    for leaf in assets.into_iter().flatten() {
        open_regular(&output.path.join(leaf))
            .map_err(ApiError::storage)?
            .sync_all()
            .map_err(ApiError::storage)?;
    }
    brewery_worker::atomic(&output.path.join(PROVENANCE), &provenance)
        .map_err(ApiError::storage)?;
    File::open(&root)
        .and_then(|f| f.sync_all())
        .map_err(ApiError::storage)?;
    let row = Revision {
        id,
        kind: KIND.into(),
        state: RevisionState::Scratch,
        sha256: receipt.sha256.clone(),
        qualification: "not release-qualified".into(),
        library_model_id: None,
        provenance,
        // Keep the scratch record shape. A compiled root has no reversible delta.
        delta: Summary {
            schema: "brewery.weight-delta.v1".into(),
            parent_sha256: EMPTY_SHA256.into(),
            revision_sha256: receipt.sha256,
            changed_tensors: 0,
            changed_bytes: 0,
            delta_bytes: 0,
            delta_sha256: EMPTY_SHA256.into(),
        },
        check_bundle: None,
        gguf: None,
    };
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    admit(app, &row.provenance.worker)?;
    store.insert_scratch(&row, key).map_err(ApiError::storage)?;
    output.keep = true;
    Ok((StatusCode::CREATED, Json(row)))
}

pub(super) async fn run(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<Run>,
) -> Result<Json<Value>> {
    let (request, _) = crate::web::read_submission(&headers, "scratch-run-program", payload)?;
    let slot = app.scratch_slots.clone().try_acquire_owned().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "scratch editor is busy".into(),
        )
    })?;
    tokio::task::spawn_blocking(move || {
        let _slot = slot;
        let (row, directory) = {
            let store = app.store.lock().map_err(ApiError::storage)?;
            let row = visible(&store, &id)?;
            admit(&app, &row.provenance.worker)?;
            (row, store.scratch_root().join(id))
        };
        let recipe = row
            .provenance
            .compiled_program
            .as_ref()
            .ok_or_else(|| ApiError::bad("use a compiled program revision"))?;
        if row.sha256 != request.sha256 {
            return Err(conflict("run the current hash of the compiled program"));
        }
        if !(1..=4).contains(&request.tokens.len())
            || request
                .tokens
                .iter()
                .any(|&token| token >= recipe.program.vocabulary())
        {
            return Err(ApiError::bad(
                "use 1 to 4 tokens in the compiled vocabulary",
            ));
        }
        integrity(&row, &directory).map_err(|e| conflict(&e))?;
        let runner = Runner::review(recipe)?;
        let tokens = serde_json::to_string(&request.tokens).map_err(ApiError::storage)?;
        let output: Vec<serde_json::Number> = runner.call(&[
            OsStr::new("run"),
            directory.as_os_str(),
            OsStr::new(&tokens),
        ])?;
        if output.len() != request.tokens.len() {
            return Err(ApiError::storage("invalid compiled program response"));
        }
        Ok(Json(json!({
            "id": row.id, "sha256": row.sha256, "config_sha256": recipe.config_sha256,
            "output": output, "qualification": row.qualification,
        })))
    })
    .await
    .map_err(ApiError::storage)?
}

#[derive(Deserialize)]
struct CheckReceipt {
    cases: usize,
    mismatch: Option<Vec<u8>>,
}

pub(super) fn check(
    app: &App,
    mut row: Revision,
    directory: &FsPath,
    key: Option<&Submission<'_>>,
) -> Result<axum::response::Response> {
    let recipe = row
        .provenance
        .compiled_program
        .as_ref()
        .ok_or_else(|| ApiError::bad("use a compiled program revision"))?;
    let failure = integrity(&row, directory).err();
    let (cases, correctness) = if failure.is_some() {
        (
            0,
            checks::Check::NotRun {
                reason: "integrity check did not pass".into(),
            },
        )
    } else {
        let receipt: CheckReceipt =
            Runner::review(recipe)?.call(&[OsStr::new("check"), directory.as_os_str()])?;
        let expected = (1..=4)
            .map(|n| usize::from(recipe.program.vocabulary()).pow(n))
            .sum::<usize>();
        match receipt.mismatch {
            Some(input) => (
                receipt.cases,
                checks::Check::Failed {
                    reason: format!("output mismatch for {input:?}"),
                },
            ),
            None if receipt.cases == expected => (receipt.cases, checks::Check::Passed),
            None => return Err(ApiError::storage("incomplete program correctness receipt")),
        }
    };
    row.check_bundle = Some(checks::program_bundle(&row, failure, correctness, cases));
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    store
        .update_scratch(&row, RevisionState::Scratch, None, key)
        .map_err(ApiError::storage)?;
    Ok((StatusCode::OK, Json(row)).into_response())
}

pub(super) fn integrity(row: &Revision, directory: &FsPath) -> std::result::Result<(), String> {
    let p = &row.provenance;
    let recipe = p
        .compiled_program
        .as_ref()
        .ok_or("no compiler provenance")?;
    if row.kind != KIND
        || p.parent_id != ROOT_ID
        || p.parent_sha256 != EMPTY_SHA256
        || p.revision_id != row.id
        || p.revision_sha256 != row.sha256
    {
        return Err("compiled root does not match provenance".into());
    }
    let (config, tensors) = inspect(directory, &row.sha256, recipe)?;
    if config.architecture != p.tool
        || config.compiler_commit != p.tool_version
        || tensors.len() != p.tensors.len()
        || tensors
            .iter()
            .zip(&p.tensors)
            .any(|(a, b)| a.name != b.name || a.dtype != b.dtype || a.shape != b.shape)
    {
        return Err("compiler config or tensors do not match provenance".into());
    }
    Ok(())
}

fn inspect(
    directory: &FsPath,
    sha: &str,
    recipe: &Recipe,
) -> std::result::Result<(Config, Vec<TensorInfo>), String> {
    let mut file =
        open_regular(&directory.join(CONFIG)).map_err(|_| "cannot open program config")?;
    let mut bytes = Vec::new();
    (&mut file)
        .take(4097)
        .read_to_end(&mut bytes)
        .map_err(|_| "cannot read program config")?;
    use sha2::{Digest, Sha256};
    if bytes.len() > 4096 || format!("{:x}", Sha256::digest(&bytes)) != recipe.config_sha256 {
        return Err("program config hash does not match provenance".into());
    }
    if let Some(digest) = &recipe.ground_truth_sha256 {
        let mut truth = Vec::new();
        open_regular(&directory.join(GROUND_TRUTH))
            .map_err(|_| "cannot open compiler ground truth")?
            .take(MAX_GROUND_TRUTH + 1)
            .read_to_end(&mut truth)
            .map_err(|_| "cannot read compiler ground truth")?;
        if truth.len() as u64 > MAX_GROUND_TRUTH
            || format!("{:x}", Sha256::digest(&truth)) != *digest
        {
            return Err("compiler ground truth hash does not match provenance".into());
        }
    }
    let config: Config = serde_json::from_slice(&bytes).map_err(|_| "invalid program config")?;
    let compiler = match recipe.program.vocabulary() {
        2 => ("alta", "afb0559da1dc08a3819c5ae20e0547686052b75a"),
        _ => ("tracr", "9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd"),
    };
    if config.schema != SCHEMA
        || config.program != recipe.program
        || (
            config.architecture.as_str(),
            config.compiler_commit.as_str(),
        ) != compiler
    {
        return Err("program config does not match the requested compiler and program".into());
    }
    let (mut file, header) = checks::open_pinned(&directory.join(scratch::MODEL), sha, MAX_MODEL)?;
    checks::finite(&mut file, &header)?;
    let tensors = header
        .entries
        .into_iter()
        .map(|(name, t)| TensorInfo {
            name,
            dtype: t.dtype,
            shape: t.shape,
        })
        .collect::<Vec<_>>();
    if tensors.is_empty() {
        return Err("compiled program has no tensors".into());
    }
    Ok((config, tensors))
}
