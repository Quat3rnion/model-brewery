use super::*;
use crate::{
    domain::{Envelope, Job, Model},
    web::{read_submission, replay},
};
use axum::response::{IntoResponse, Response};
use brewery_worker::{
    scratch_eval::{self, BehaviorConfig, BehaviorMetric as Metric, BehaviorReport},
    weight_source::{Header, LIMIT as HEADER_LIMIT},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};

const SCHEMA: &str = "brewery.scratch-checks.v1";
const BEHAVIOR_SCHEMA: &str = "brewery.scratch-behavior.v1";
const FIXTURES: &str = include_str!("../../eval/data/edit-checks/manifest.json");
const NO_GGUF: &str = "convert both scratch revisions to private GGUF";
const METRICS: &[(&str, &str)] = &[
    ("counterfact_efficacy", "higher"),
    ("counterfact_target_probability", "higher"),
    ("counterfact_target_rank", "lower"),
    ("counterfact_paraphrase", "higher"),
    ("counterfact_paraphrase_probability", "higher"),
    ("counterfact_paraphrase_rank", "lower"),
    ("counterfact_locality_probability", "neither"),
    ("counterfact_locality_rank", "neither"),
    ("zsre_efficacy", "higher"),
    ("zsre_target_probability", "higher"),
    ("zsre_target_rank", "lower"),
    ("zsre_paraphrase", "higher"),
    ("zsre_paraphrase_probability", "higher"),
    ("zsre_paraphrase_rank", "lower"),
    ("zsre_locality_probability", "neither"),
    ("zsre_locality_rank", "neither"),
    ("perplexity", "lower"),
];

#[derive(Deserialize, Serialize)]
pub(super) struct Bundle {
    schema: String,
    id: String,
    revision_sha256: String,
    parent_sha256: String,
    created_at_ms: u64,
    qualification: String,
    integrity: Check,
    smoke: Check,
    behavior: Check,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    behavior_direction: Option<BehaviorReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    program_correctness: Option<Check>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    program_cases: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    offline_report: Option<OfflineReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    measured_report: Option<OfflineReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    measured_job_id: Option<String>,
    #[serde(default)]
    job_refs: Vec<JobRef>,
    acknowledged_at_ms: Option<u64>,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(super) enum Check {
    Passed,
    Failed { reason: String },
    NotRun { reason: String },
    Reported { reason: String },
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Run {
    pub(super) sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) behavior_direction: Option<BehaviorConfig>,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    Parent,
    Child,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    job_id: String,
    role: Role,
}

#[derive(Deserialize, Serialize)]
struct JobRef {
    #[serde(flatten)]
    reference: Reference,
    kind: String,
    // Legacy, unverified references are retained for existing bundles only.
    verified_revision: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct OfflineReport {
    schema: String,
    qualification: String,
    parent_sha256: String,
    revision_sha256: String,
    parent_gguf_sha256: String,
    revision_gguf_sha256: String,
    runner_sha256: String,
    fixture_sha256: std::collections::BTreeMap<String, String>,
    metrics: Vec<Metric>,
    parent_generation: String,
    child_generation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    behavior_direction: Option<BehaviorReport>,
}

impl OfflineReport {
    fn validate(&self, row: &Revision) -> Result<()> {
        let fixture_sha256: std::collections::BTreeMap<String, String> =
            serde_json::from_str(FIXTURES).map_err(ApiError::storage)?;
        if self.schema != BEHAVIOR_SCHEMA
            || self.qualification != "not release-qualified"
            || self.parent_sha256 != row.provenance.parent_sha256
            || self.revision_sha256 != row.sha256
            || self.fixture_sha256 != fixture_sha256
            || ![
                &self.parent_gguf_sha256,
                &self.revision_gguf_sha256,
                &self.runner_sha256,
            ]
            .into_iter()
            .all(|sha| brewery_worker::is_digest(sha))
            || self.parent_generation.is_empty()
            || self.child_generation.is_empty()
            || self.parent_generation.len() > 4096
            || self.child_generation.len() > 4096
            || self.metrics.len() != METRICS.len()
        {
            return Err(ApiError::bad(
                "behavior report does not match this revision or protocol",
            ));
        }
        for (name, better) in METRICS {
            let mut found = self.metrics.iter().filter(|m| m.name == *name);
            let Some(m) = found.next() else {
                return Err(ApiError::bad("missing behavior metric"));
            };
            let low = if name.ends_with("_rank") || *name == "perplexity" {
                1.
            } else {
                0.
            };
            let high = if low == 0. { 1. } else { 1e12 };
            if m.better != *better
                || found.next().is_some()
                || ![m.parent, m.child]
                    .into_iter()
                    .all(|v| (low..=high).contains(&v))
            {
                return Err(ApiError::bad("invalid behavior metric"));
            }
        }
        Ok(())
    }
}

pub(super) async fn get(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
) -> Result<Json<Bundle>> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    visible(&store, &id)?
        .check_bundle
        .map(Json)
        .ok_or_else(|| ApiError::not_found("scratch check bundle"))
}

pub(super) async fn run(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    payload: Payload<Run>,
) -> Result<Response> {
    let request = read(payload)?;
    if let Some(config) = &request.behavior_direction {
        config
            .validate()
            .map_err(|e| ApiError::bad(e.to_string()))?;
    }
    // Keep the existing scratch writer serialization for checks and edits.
    let slot = app.scratch_slots.clone().try_acquire_owned().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "scratch editor is busy".into(),
        )
    })?;
    tokio::task::spawn_blocking(move || {
        let _slot = slot;
        let (_, submission) =
            read_submission(&headers, "scratch-checks", Ok(Json((&id, &request))))?;
        check(&app, id, request, submission.as_ref())
    })
    .await
    .map_err(ApiError::storage)?
}

pub(super) fn check(app: &App, id: String, request: Run, key: Option<&Submission<'_>>) -> Result<Response> {
    let (row, directory, parent) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        if let Some(key) = key
            && let Replay::Matched(ref id) = store.replay(key).map_err(ApiError::storage)?
            && let Some(prior) = store.scratch(id).map_err(ApiError::storage)?
        {
            return Ok((StatusCode::OK, Json(prior)).into_response());
        }
        if let Some(response) = replay(&store, key)? {
            return Ok(response);
        }
        let row = visible(&store, &id)?;
        if row.state != RevisionState::Scratch || row.sha256 != request.sha256 {
            return Err(conflict(
                "run checks for the current hash of an unrequested scratch revision",
            ));
        }
        admit(app, &row.provenance.worker)?;
        let root = store.scratch_root();
        if row.is_compiled_program() {
            if request.behavior_direction.is_some() {
                return Err(ApiError::bad("behavior checks require a language model"));
            }
            let directory = root.join(&id);
            drop(store);
            return program::check(app, row, &directory, key);
        }
        let parent_id = &row.provenance.parent_id;
        let parent = if let Some(parent) = store.scratch(parent_id).map_err(ApiError::storage)? {
            (parent.state != RevisionState::Discarded
                && parent.sha256 == row.provenance.parent_sha256)
                .then(|| jobs::input(&store, &parent, scratch::MODEL, "safetensors"))
        } else if let Some(model) = store.model(parent_id).map_err(ApiError::storage)? {
            let path = FsPath::new(&model.path);
            if model.kind != "safetensors" || model.host != row.provenance.worker {
                return Err(conflict("parent is not the reviewed SafeTensors source"));
            }
            require_source(app, &model.host, path)?;
            Some(model)
        } else {
            None
        };
        (row, root.join(&id), parent)
    };
    let integrity = match &parent {
        Some(parent) => integrity(
            &row,
            FsPath::new(&parent.path),
            &directory.join(scratch::MODEL),
        ),
        None => Err("immediate parent is not available".into()),
    };
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    let mut row = visible(&store, &id)?;
    if store
        .scratch_has_jobs(&id, Some(jobs::CHECK))
        .map_err(ApiError::storage)?
    {
        return Err(conflict("checks are already running for this revision"));
    }
    if let (Ok(()), Some(parent)) = (&integrity, parent)
        && let Some((job, envelope)) =
            plan(&store, &row, parent, request.behavior_direction.as_ref())?
    {
        return jobs::enqueue(app, &mut store, job, envelope, key, StatusCode::ACCEPTED);
    }
    if request.behavior_direction.is_some() {
        return Err(conflict(integrity.err().as_deref().unwrap_or(NO_GGUF)));
    }
    let failure = integrity.err();
    row.check_bundle = Some(bundle(&row, failure, None));
    write_behavior(&row, &directory).map_err(ApiError::storage)?;
    store
        .update_scratch(&row, RevisionState::Scratch, None, key)
        .map_err(ApiError::storage)?;
    Ok((StatusCode::OK, Json(row)).into_response())
}

fn plan(
    store: &Store,
    row: &Revision,
    parent: Model,
    behavior: Option<&BehaviorConfig>,
) -> Result<Option<(Job, Envelope)>> {
    if row.gguf.is_none() {
        return Ok(None);
    }
    let (child_gguf, child_pin) = jobs::trace_input(store, &row.id)?;
    let (parent_gguf, parent_pin) = if row.sha256 == row.provenance.parent_sha256 {
        // A byte-identical root copy already has its parent's exact weights.
        (child_gguf.clone(), child_pin.clone())
    } else {
        let Some(parent) = store
            .scratch(&parent.id)
            .map_err(ApiError::storage)?
            .filter(|p| p.gguf.is_some() && p.sha256 == row.provenance.parent_sha256)
        else {
            return Ok(None);
        };
        jobs::trace_input(store, &parent.id)?
    };
    let child = jobs::input(store, row, scratch::MODEL, "safetensors");
    let pins = BTreeMap::from([
        (parent_gguf.path.clone(), parent_pin),
        (child_gguf.path.clone(), child_pin),
        (parent.path.clone(), row.provenance.parent_sha256.clone()),
        (child.path.clone(), row.sha256.clone()),
    ]);
    let (job, mut envelope) = jobs::private_job(
        jobs::CHECK,
        vec![parent_gguf, child_gguf, parent, child],
        behavior.map_or_else(|| json!({}), |config| json!({"behavior_direction":config})),
    );
    envelope.scratch_pins = pins;
    Ok(Some((job, envelope)))
}

pub(super) fn finish(
    mut row: Revision,
    envelope: &Envelope,
    job: &Job,
    result: &Value,
    directory: &FsPath,
) -> std::result::Result<(Revision, Value), crate::Error> {
    let mut measured: OfflineReport = serde_json::from_value(result.clone())?;
    measured
        .validate(&row)
        .map_err(|_| "invalid scratch behavior protocol or measurements")?;
    for (input, sha) in envelope.inputs.iter().zip([
        &measured.parent_gguf_sha256,
        &measured.revision_gguf_sha256,
        &measured.parent_sha256,
        &measured.revision_sha256,
    ]) {
        if envelope.scratch_pins.get(&input.path) != Some(sha) {
            return Err("worker behavior result does not match the pinned inputs".into());
        }
    }
    let config = scratch_eval::check_options(&json!(envelope.options))?;
    let behavior = measured.behavior_direction.take();
    scratch_eval::validate_behavior(behavior.as_ref(), config.as_ref())?;
    let mut completed = bundle(&row, None, Some((job.id.clone(), measured)));
    completed.behavior_direction = behavior;
    row.check_bundle = Some(completed);
    write_behavior(&row, directory)?;
    let safe_result = json!({"scratch_id":row.id, "check_bundle":row.check_bundle});
    Ok((row, safe_result))
}

fn bundle(
    row: &Revision,
    failure: Option<String>,
    measured: Option<(String, OfflineReport)>,
) -> Bundle {
    let (measured_job_id, measured_report) = measured.unzip();
    let not_run = if failure.is_some() {
        "integrity check did not pass"
    } else {
        NO_GGUF
    };
    let measured_status = || {
        if measured_report.is_some() {
            Check::Passed
        } else {
            Check::NotRun {
                reason: not_run.into(),
            }
        }
    };
    Bundle {
        schema: SCHEMA.into(),
        id: crate::domain::id(),
        revision_sha256: row.sha256.clone(),
        parent_sha256: row.provenance.parent_sha256.clone(),
        created_at_ms: now(),
        qualification: "not release-qualified".into(),
        integrity: failure.map_or(Check::Passed, |reason| Check::Failed { reason }),
        smoke: measured_status(),
        behavior: measured_status(),
        behavior_direction: None,
        program_correctness: None,
        program_cases: None,
        offline_report: None,
        measured_report,
        measured_job_id,
        job_refs: Vec::new(),
        acknowledged_at_ms: None,
    }
}

pub(super) fn program_bundle(
    row: &Revision,
    failure: Option<String>,
    correctness: Check,
    cases: usize,
) -> Bundle {
    let mut result = bundle(row, failure, None);
    result.smoke = if cases > 0 || matches!(correctness, Check::Failed { .. }) {
        Check::Passed
    } else {
        Check::NotRun {
            reason: "integrity check did not pass".into(),
        }
    };
    result.behavior = Check::NotRun {
        reason: "language-model behavior checks do not apply to compiled programs".into(),
    };
    result.program_correctness = Some(correctness);
    result.program_cases = Some(cases);
    result
}

fn write_behavior(row: &Revision, directory: &FsPath) -> std::result::Result<(), crate::Error> {
    #[derive(Serialize)]
    struct Behavior<'a> {
        schema_version: u8,
        parent_id: &'a str,
        child_id: &'a str,
        metrics: Vec<&'a Metric>,
    }
    let behavior = Behavior {
        schema_version: 1,
        parent_id: &row.provenance.parent_id,
        child_id: &row.id,
        metrics: row
            .check_bundle
            .iter()
            .flat_map(|bundle| {
                bundle
                    .measured_report
                    .iter()
                    .flat_map(|report| &report.metrics)
                    .chain(
                        bundle
                            .behavior_direction
                            .iter()
                            .flat_map(|report| &report.metrics),
                    )
            })
            .collect(),
    };
    // Metrics have fixed keys and bounded numbers. Keep the projection below the W5 read limit.
    if serde_json::to_vec(&behavior)?.len() >= 256 * 1024 {
        return Err("scratch behavior result exceeds its size limit".into());
    }
    brewery_worker::atomic(&directory.join("behavior.v1.json"), &behavior)?;
    Ok(())
}

pub(super) fn require_publishable(
    row: &Revision,
    new: RevisionState,
    acknowledgement: Option<&str>,
) -> Result<bool> {
    let bundle = row
        .check_bundle
        .as_ref()
        .filter(|b| {
            b.schema == SCHEMA
                && b.revision_sha256 == row.sha256
                && b.parent_sha256 == row.provenance.parent_sha256
                && matches!(b.integrity, Check::Passed)
        })
        .ok_or_else(|| conflict("passing scratch integrity checks are required for this hash"))?;
    if row.is_compiled_program() && !matches!(bundle.program_correctness, Some(Check::Passed)) {
        return Err(conflict(
            "passing program correctness checks are required for this hash",
        ));
    }
    if matches!(
        (&bundle.smoke, &bundle.behavior),
        (Check::Passed, Check::Passed)
    ) {
        return Ok(false);
    }
    if new == RevisionState::PublishRequested && acknowledgement == Some(bundle.id.as_str()) {
        return Ok(true);
    } else if new != RevisionState::Approved || bundle.acknowledged_at_ms.is_none() {
        return Err(conflict(
            "acknowledge the current incomplete scratch check bundle by ID",
        ));
    }
    Ok(false)
}

pub(super) fn acknowledge(row: &mut Revision) {
    if let Some(bundle) = row.check_bundle.as_mut() {
        bundle.acknowledged_at_ms = Some(now());
    }
}

pub(super) fn open_pinned(
    path: &FsPath,
    pin: &str,
    limit: u64,
) -> std::result::Result<(File, Header), String> {
    let mut file = open_regular(path).map_err(|_| "cannot open regular SafeTensors artifact")?;
    let bytes = file
        .metadata()
        .map_err(|_| "cannot read artifact size")?
        .len();
    if bytes > limit {
        return Err("SafeTensors artifact exceeds scratch size limit".into());
    }
    if scratch::file_sha256(&mut file).map_err(|_| "cannot hash artifact")? != pin {
        return Err("SafeTensors hash does not match provenance".into());
    }
    let header = Header::read(&mut file, bytes, HEADER_LIMIT).map_err(|e| e.to_string())?;
    Ok((file, header))
}

fn integrity(row: &Revision, parent: &FsPath, child: &FsPath) -> std::result::Result<(), String> {
    let p = &row.provenance;
    if p.revision_id != row.id
        || p.revision_sha256 != row.sha256
        || row.delta.parent_sha256 != p.parent_sha256
        || row.delta.revision_sha256 != row.sha256
    {
        return Err("revision metadata does not match provenance".into());
    }
    let (mut parent, ph) = open_pinned(parent, &p.parent_sha256, scratch::MAX_FILE)?;
    let (mut child, ch) = open_pinned(child, &row.sha256, scratch::MAX_FILE)?;
    if ph.payload_start != ch.payload_start
        || ph.entries.len() != ch.entries.len()
        || ph.entries.iter().any(|(name, a)| {
            ch.entries.get(name).is_none_or(|b| {
                a.dtype != b.dtype || a.shape != b.shape || a.data_offsets != b.data_offsets
            })
        })
        || p.tensors.iter().any(|t| {
            ch.entries
                .get(&t.name)
                .is_none_or(|actual| actual.dtype != t.dtype || actual.shape != t.shape)
        })
    {
        return Err("SafeTensors headers do not match parent and edit provenance".into());
    }
    finite(&mut parent, &ph)?;
    finite(&mut child, &ch)
}

pub(super) fn finite(file: &mut File, header: &Header) -> std::result::Result<(), String> {
    let mut buf = vec![0; 256 * 1024];
    for (name, t) in &header.entries {
        // C64 contains two F32 components. Inspect exponent bits for each float format.
        let (width, mask) = match t.dtype.as_str() {
            "F8_E4M3" => (1, 0x7f),
            "F8_E5M2" => (1, 0x7c),
            "F8_E8M0" => (1, 0xff),
            "F16" => (2, 0x7c00),
            "BF16" => (2, 0x7f80),
            "F32" | "C64" => (4, 0x7f80_0000),
            "F64" => (8, 0x7ff0_0000_0000_0000),
            _ => continue,
        };
        let [start, end] = t.data_offsets;
        file.seek(SeekFrom::Start(header.payload_start + start))
            .map_err(|_| "cannot seek to SafeTensors payload")?;
        let mut left = end - start;
        while left > 0 {
            let size = left.min(buf.len() as u64) as usize;
            file.read_exact(&mut buf[..size])
                .map_err(|_| "cannot read SafeTensors payload")?;
            if buf[..size].chunks_exact(width).any(|bytes| {
                let mut word = [0; 8];
                word[..width].copy_from_slice(bytes);
                u64::from_le_bytes(word) & mask == mask
            }) {
                return Err(format!("non-finite SafeTensors value in tensor {name}"));
            }
            left -= size as u64;
        }
    }
    Ok(())
}
