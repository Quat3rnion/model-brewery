use super::*;
use crate::domain::State as JobState;
use brewery_worker::weight_source::{Header, LIMIT as HEADER_LIMIT};
use std::io::{Read, Seek, SeekFrom};

const SCHEMA: &str = "brewery.scratch-checks.v1";
const BEHAVIOR_SCHEMA: &str = "brewery.scratch-behavior.v1";
const FIXTURES: &str = include_str!("../../eval/data/edit-checks/manifest.json");
const NO_GGUF: &str = "no pre-approval GGUF path";
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
    offline_report: Option<OfflineReport>,
    job_refs: Vec<JobRef>,
    acknowledged_at_ms: Option<u64>,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Check {
    Passed,
    Failed { reason: String },
    NotRun { reason: String },
    Reported { reason: String },
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Run {
    sha256: String,
    #[serde(default)]
    job_refs: Vec<Reference>,
    offline_report: Option<OfflineReport>,
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
    // Job admission does not pin its input to a scratch revision.
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
    // Leave controller smoke as not_run. These are client-reported samples.
    parent_generation: String,
    child_generation: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Metric {
    name: String,
    parent: f64,
    child: f64,
    better: String,
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
                "offline report does not match this revision or protocol",
            ));
        }
        for (name, better) in METRICS {
            let mut found = self.metrics.iter().filter(|m| m.name == *name);
            let Some(m) = found.next() else {
                return Err(ApiError::bad("missing offline metric"));
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
                return Err(ApiError::bad("invalid offline metric"));
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
) -> Mutation {
    mutate(
        app,
        headers,
        "scratch-checks",
        Some(id),
        read(payload)?,
        check,
    )
    .await
}

fn check(
    app: &App,
    target: Option<String>,
    request: Run,
    key: Option<&Submission<'_>>,
) -> Mutation {
    let id = request_id(target)?;
    let (mut row, directory, parent, jobs) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let row = visible(&store, &id)?;
        if row.state != RevisionState::Scratch || row.sha256 != request.sha256 {
            return Err(conflict(
                "run checks for the current hash of an unrequested scratch revision",
            ));
        }
        admit(app, &row.provenance.worker)?;
        if let Some(report) = &request.offline_report {
            report.validate(&row)?;
        }
        let root = store.scratch_root();
        let parent_id = &row.provenance.parent_id;
        let parent = if let Some(parent) = store.scratch(parent_id).map_err(ApiError::storage)? {
            (parent.state != RevisionState::Discarded
                && parent.sha256 == row.provenance.parent_sha256)
                .then(|| root.join(parent.id).join(scratch::MODEL))
        } else if let Some(model) = store.model(parent_id).map_err(ApiError::storage)? {
            let path = PathBuf::from(model.path);
            if model.kind != "safetensors" || model.host != row.provenance.worker {
                return Err(conflict("parent is not the reviewed SafeTensors source"));
            }
            require_source(app, &model.host, &path)?;
            Some(path)
        } else {
            None
        };
        let jobs = job_refs(&store, request.job_refs)?;
        (row, root.join(id), parent, jobs)
    };
    let integrity = match parent {
        Some(parent) => integrity(&row, &parent, &directory.join(scratch::MODEL)),
        None => Err("immediate parent is not available".into()),
    };
    let (integrity, offline_report) = match integrity {
        Ok(()) => (Check::Passed, request.offline_report),
        Err(reason) => (Check::Failed { reason }, None),
    };
    let behavior = if offline_report.is_some() {
        Check::Reported {
            reason: "client report; GGUF ancestry and measurements are not verified".into(),
        }
    } else {
        Check::NotRun {
            reason: NO_GGUF.into(),
        }
    };
    let bundle = Bundle {
        schema: SCHEMA.into(),
        id: crate::domain::id(),
        revision_sha256: row.sha256.clone(),
        parent_sha256: row.provenance.parent_sha256.clone(),
        created_at_ms: now(),
        qualification: "not release-qualified".into(),
        integrity,
        smoke: Check::NotRun {
            reason: NO_GGUF.into(),
        },
        behavior,
        offline_report,
        job_refs: jobs,
        acknowledged_at_ms: None,
    };
    #[derive(Serialize)]
    struct Behavior<'a> {
        schema_version: u8,
        parent_id: &'a str,
        child_id: &'a str,
        metrics: &'a [Metric],
    }
    let behavior = Behavior {
        schema_version: 1,
        parent_id: &row.provenance.parent_id,
        child_id: &row.id,
        metrics: bundle
            .offline_report
            .as_ref()
            .map_or(&[], |report| &report.metrics),
    };
    // Metrics have fixed keys and bounded numbers. Keep the projection below the W5 read limit.
    if serde_json::to_vec(&behavior)
        .map_err(ApiError::storage)?
        .len()
        >= 256 * 1024
    {
        return Err(ApiError::storage(
            "scratch behavior result exceeds its size limit",
        ));
    }
    brewery_worker::atomic(&directory.join("behavior.v1.json"), &behavior)
        .map_err(ApiError::storage)?;
    row.check_bundle = Some(bundle);
    let mut store = app.store.lock().map_err(ApiError::storage)?;
    store
        .update_scratch(&row, RevisionState::Scratch, None, key)
        .map_err(ApiError::storage)?;
    Ok((StatusCode::OK, Json(row)))
}

fn job_refs(store: &Store, references: Vec<Reference>) -> Result<Vec<JobRef>> {
    let mut seen = std::collections::HashSet::new();
    if references.len() > 8 || references.iter().any(|r| !seen.insert(&r.job_id)) {
        return Err(ApiError::bad("use at most eight distinct check job IDs"));
    }
    references
        .into_iter()
        .map(|reference| {
            let job = store
                .job(&reference.job_id)
                .map_err(ApiError::storage)?
                .filter(|job| {
                    matches!(job.kind.as_str(), "evaluate" | "baseline" | "benchmark")
                        && job.state == JobState::Succeeded
                        && job.has_result
                })
                .ok_or_else(|| ApiError::bad("check job reference needs a completed evaluation or benchmark with a result"))?;
            Ok(JobRef {
                reference,
                kind: job.kind,
                verified_revision: false,
            })
        })
        .collect()
}

pub(super) fn require_publishable(
    row: &mut Revision,
    new: RevisionState,
    acknowledgement: Option<&str>,
) -> Result<()> {
    let bundle = row
        .check_bundle
        .as_mut()
        .filter(|b| {
            b.schema == SCHEMA
                && b.revision_sha256 == row.sha256
                && b.parent_sha256 == row.provenance.parent_sha256
                && matches!(b.integrity, Check::Passed)
        })
        .ok_or_else(|| conflict("passing scratch integrity checks are required for this hash"))?;
    if matches!(
        (&bundle.smoke, &bundle.behavior),
        (Check::Passed, Check::Passed)
    ) {
        return Ok(());
    }
    if new == RevisionState::PublishRequested && acknowledgement == Some(bundle.id.as_str()) {
        bundle.acknowledged_at_ms = Some(now());
    } else if new != RevisionState::Approved || bundle.acknowledged_at_ms.is_none() {
        return Err(conflict(
            "acknowledge the current incomplete scratch check bundle by ID",
        ));
    }
    Ok(())
}

fn integrity(row: &Revision, parent: &FsPath, child: &FsPath) -> std::result::Result<(), String> {
    fn open(path: &FsPath, pin: &str) -> std::result::Result<(File, Header), String> {
        let mut file =
            open_regular(path).map_err(|_| "cannot open regular SafeTensors artifact")?;
        let bytes = file
            .metadata()
            .map_err(|_| "cannot read artifact size")?
            .len();
        if bytes > scratch::MAX_FILE {
            return Err("SafeTensors artifact exceeds scratch size limit".into());
        }
        if scratch::file_sha256(&mut file).map_err(|_| "cannot hash artifact")? != pin {
            return Err("SafeTensors hash does not match provenance".into());
        }
        let header = Header::read(&mut file, bytes, HEADER_LIMIT).map_err(|e| e.to_string())?;
        Ok((file, header))
    }
    let p = &row.provenance;
    if p.revision_id != row.id
        || p.revision_sha256 != row.sha256
        || row.delta.parent_sha256 != p.parent_sha256
        || row.delta.revision_sha256 != row.sha256
    {
        return Err("revision metadata does not match provenance".into());
    }
    let (mut parent, ph) = open(parent, &p.parent_sha256)?;
    let (mut child, ch) = open(child, &row.sha256)?;
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

fn finite(file: &mut File, header: &Header) -> std::result::Result<(), String> {
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
