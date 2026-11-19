use crate::{Job, bounded, child_budget, child_threads, converter, integrity, progress::Progress};
use anyhow::{Context, Result, bail, ensure};
use brewery_worker::weight_source;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    io::{Seek, Write},
    os::unix::{
        fs::{OpenOptionsExt, PermissionsExt},
        process::CommandExt,
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

mod safetensors;

const PROPOSAL: &str = "brewery.weight-edit-proposal.v1";
const RESULT: &str = "brewery.weight-edit-result.v1";
const TUNE: &str = "brewery.weight-tune-report.v1";
const MODEL: u64 = 32 << 20;
const CALIBRATION: u64 = 8 << 20;
const JSON: u64 = 128 << 10;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inspect {
    tensor: String,
    index: Vec<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inventory {
    offset: u64,
    limit: u64,
    source_sha256: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tile {
    tensor: String,
    origin: Vec<u64>,
    shape: Vec<u64>,
    source_sha256: String,
}

#[derive(Deserialize, serde::Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Edit {
    tensor: String,
    index: Vec<u64>,
    op: String,
    value: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Propose {
    edits: Vec<Edit>,
    source_sha256: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tune {
    tensor: String,
    indices: Vec<[u64; 2]>,
    max_update: f64,
    source_sha256: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Apply {
    proposal_sha256: String,
    source_sha256: String,
}

enum Mode {
    Inspect(Inspect),
    Inventory(Inventory),
    Tile(Tile),
    Propose(Propose),
    Tune(Tune),
    Apply(Apply),
}

fn rank_ok(index: &[u64]) -> bool {
    (1..=4).contains(&index.len())
}

fn valid_edit(edit: &Edit, native_safe: bool) -> bool {
    weight_source::tensor_name(&edit.tensor, native_safe)
        && rank_ok(&edit.index)
        && matches!(edit.op.as_str(), "delta" | "set")
        && edit.value.is_finite()
}

impl Mode {
    fn parse(job: &Job, options: Value, native_safe: bool) -> Result<Self> {
        ensure!(job.prompt.is_none(), "weight jobs do not accept prompts");
        Ok(match job.kind.as_str() {
            "weight_inspect" if options.get("mode").is_some() => {
                let mut options = options;
                let mode = options.as_object_mut().unwrap().remove("mode");
                match mode.as_ref().and_then(Value::as_str) {
                    Some("inventory") => {
                        let v: Inventory = serde_json::from_value(options)?;
                        ensure!(
                            v.offset <= 16384 && (1..=128).contains(&v.limit),
                            "invalid inventory page"
                        );
                        Self::Inventory(v)
                    }
                    Some("tile") => {
                        let v: Tile = serde_json::from_value(options)?;
                        let rank = v.shape.len();
                        ensure!(
                            weight_source::tensor_name(&v.tensor, native_safe)
                                && rank_ok(&v.shape)
                                && v.origin.len() == rank
                                && v.origin
                                    .iter()
                                    .zip(&v.shape)
                                    .all(|(a, b)| a.checked_add(*b).is_some())
                                && if rank == 1 {
                                    (1..=1024).contains(&v.shape[0])
                                } else {
                                    v.shape[..rank - 2].iter().all(|n| *n == 1)
                                        && v.shape[rank - 2..].iter().all(|n| (1..=32).contains(n))
                                },
                            "invalid tile dimensions or origin"
                        );
                        Self::Tile(v)
                    }
                    _ => bail!("unsupported weight inspection mode"),
                }
            }
            "weight_inspect" => {
                let v: Inspect = serde_json::from_value(options)?;
                ensure!(
                    weight_source::tensor_name(&v.tensor, native_safe) && rank_ok(&v.index),
                    "invalid scalar coordinate"
                );
                Self::Inspect(v)
            }
            "weight_propose" => {
                let v: Propose = serde_json::from_value(options)?;
                let mut selected = BTreeSet::new();
                ensure!(
                    (1..=256).contains(&v.edits.len())
                        && v.edits.iter().all(|edit| {
                            valid_edit(edit, native_safe)
                                && selected.insert((&edit.tensor, &edit.index))
                        }),
                    "invalid or duplicate scalar edits"
                );
                Self::Propose(v)
            }
            "weight_tune" => {
                let v: Tune = serde_json::from_value(options)?;
                ensure!(
                    weight_source::tensor_name(&v.tensor, native_safe)
                        && (1..=256).contains(&v.indices.len())
                        && v.indices.iter().collect::<BTreeSet<_>>().len() == v.indices.len()
                        && serde_json::to_vec(&v.indices)?.len() <= 8192
                        && v.max_update.is_finite()
                        && v.max_update > 0.0,
                    "invalid tuning selection"
                );
                Self::Tune(v)
            }
            "weight_apply" => Self::Apply(serde_json::from_value(options)?),
            _ => bail!("unsupported weight job"),
        })
    }

    fn auxiliary_inputs(&self) -> &'static [(&'static str, &'static str, u64)] {
        match self {
            Self::Tune(_) => &[
                ("weight-calibration", "calibration.json", CALIBRATION),
                ("weight-calibration", "held-out.json", CALIBRATION),
            ],
            Self::Apply(_) => &[("weight-edit-proposal", "proposal.json", JSON)],
            _ => &[],
        }
    }

    fn visual_sha(&self) -> Option<&str> {
        match self {
            Self::Inventory(v) => v.source_sha256.as_deref(),
            Self::Tile(v) => Some(&v.source_sha256),
            Self::Propose(v) => v.source_sha256.as_deref(),
            Self::Tune(v) => v.source_sha256.as_deref(),
            _ => None,
        }
    }
}

pub(crate) fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    ensure!(
        env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_none(),
        "weight editing is disabled on source-pinned qualification workers"
    );
    let mut options = job.options.clone();
    let packed = match options
        .as_object_mut()
        .context("invalid weight job options")?
        .remove("packed")
    {
        None => false,
        Some(Value::Bool(true)) => true,
        _ => bail!("packed must be true when supplied"),
    };
    ensure!(
        !packed || env::var("BREWERY_WEIGHT_PACKED").as_deref() == Ok("1"),
        "packed weight editing is not enabled on this worker"
    );
    let source_kind = job
        .inputs
        .first()
        .context("missing weight source input")?
        .kind
        .as_str();
    let native_safe = weight_source::supports(source_kind);
    let mode = Mode::parse(job, options, native_safe).context("invalid weight job options")?;
    if matches!(mode, Mode::Inventory(_) | Mode::Tile(_)) || mode.visual_sha().is_some() {
        ensure!(
            env::var("BREWERY_WEIGHT_VISUAL").as_deref() == Ok("1"),
            "visual weight inspection is not enabled on this worker"
        );
    }
    if native_safe {
        ensure!(
            env::var(weight_source::GATE).as_deref() == Ok("1"),
            "SafeTensors weight editing is not enabled on this worker"
        );
        ensure!(
            !packed,
            "packed weight editing is not supported for SafeTensors sources"
        );
    }
    let (source_name, output_name, edited_name, magic) = match source_kind {
        "gguf" => ("source.gguf", "output.gguf", "edited.gguf", Some(b"GGUF")),
        "hgn" => {
            ensure!(
                env::var("BREWERY_WEIGHT_HGN").as_deref() == Ok("1"),
                "HGN weight editing is not enabled on this worker"
            );
            ("source.hgn", "output.hgn", "edited.hgn", Some(b"HGN1"))
        }
        "safetensors" => (
            "source.safetensors",
            "output.safetensors",
            "edited.safetensors",
            None,
        ),
        "hf" => ("source.hf", "output.hf", "edited.hf", None),
        _ => bail!("unsupported registered weight source kind"),
    };
    let policy = CodecPolicy {
        hgn: source_kind == "hgn",
        packed,
        native_safe,
    };
    let required = std::iter::once((source_kind, source_name, MODEL))
        .chain(mode.auxiliary_inputs().iter().copied())
        .collect::<Vec<_>>();
    ensure!(
        job.inputs.len() == required.len(),
        "wrong number of weight job inputs"
    );
    for (index, (input, (kind, _, limit))) in job.inputs.iter().zip(&required).enumerate() {
        ensure!(
            input.kind == *kind && input.integrity_basis.is_none(),
            "invalid weight job input kind"
        );
        if native_safe && index == 0 {
            continue;
        }
        let meta = integrity::canonical(&input.path)?;
        ensure!(
            meta.is_file() && (1..=*limit).contains(&meta.len()),
            "weight job input exceeds the bounded file limit"
        );
        ensure!(
            input.sha256.is_some()
                && input.bundle_sha256.is_some()
                && input.files.as_ref().is_some_and(|files| files.len() == 1),
            "weight jobs require a verified single-file input manifest"
        );
    }
    let var = if matches!(mode, Mode::Tune(_)) {
        "BREWERY_WEIGHT_TUNE"
    } else {
        "BREWERY_WEIGHT_EDIT"
    };
    let executable = env::var_os(var)
        .map(PathBuf::from)
        .with_context(|| format!("weight job requires {var}"))?;
    let meta = integrity::canonical(&executable)?;
    ensure!(
        meta.is_file() && meta.permissions().mode() & 0o111 != 0,
        "{var} must be a canonical executable"
    );
    let (binary, stamp) = integrity::file_digest(&executable, "binary".into())?;
    let ids = if native_safe {
        let source = integrity::Identity::weight_manifest(&job.inputs[0])?;
        ensure!(
            source.bytes <= weight_source::MAX_BYTES,
            "SafeTensors source exceeds model limit"
        );
        let auxiliary = &job.inputs[1..];
        ensure!(
            auxiliary.iter().all(|i| i.host == job.inputs[0].host),
            "all resolved inputs must be on one selected host"
        );
        let mut ids = vec![source];
        if !auxiliary.is_empty() {
            ids.extend(integrity::check_inputs(auxiliary, progress)?);
        }
        ids
    } else {
        integrity::check_inputs(&job.inputs, progress)?
    };
    ensure!(
        ids.iter()
            .skip(usize::from(native_safe))
            .all(|id| id.files.len() == 1),
        "split weight inputs are not supported"
    );
    let source_sha = &ids[0].sha256;
    ensure!(
        mode.visual_sha()
            .is_none_or(|approved| approved == source_sha),
        "visual weight source SHA-256 changed"
    );
    if let Mode::Apply(approved) = &mode {
        ensure!(
            &approved.source_sha256 == source_sha && approved.proposal_sha256 == ids[1].sha256,
            "approved source or proposal SHA-256 mismatch"
        );
    }
    let scratch = if native_safe {
        tempfile::Builder::new()
            .prefix(safetensors::SCRATCH_PREFIX)
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir_in(cwd)?
    } else if source_kind == "gguf" {
        tempfile::Builder::new()
            .prefix(".weight-gguf-")
            .tempdir_in(cwd)?
    } else if source_kind == "hgn" {
        tempfile::Builder::new()
            .prefix(".weight-hgn-")
            .tempdir_in(cwd)?
    } else {
        tempfile::tempdir_in(cwd)?
    };
    let work = scratch.path();
    let safe_source = native_safe
        .then(|| {
            safetensors::Source::prepare(
                &job.inputs[0],
                &ids[0],
                work,
                matches!(mode, Mode::Apply(_)),
            )
        })
        .transpose()?;
    for ((input, id), (_, name, limit)) in job
        .inputs
        .iter()
        .zip(&ids)
        .zip(&required)
        .skip(usize::from(native_safe))
    {
        let source = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&input.path)?;
        let data = bounded(source, *limit)?;
        if *name == source_name {
            ensure!(
                magic.is_some_and(|magic| data.starts_with(magic)),
                "weight source magic does not match registered kind"
            );
        }
        ensure!(
            data.len() as u64 == id.bytes && format!("{:x}", Sha256::digest(&data)) == id.sha256,
            "private staged input does not match verified SHA-256"
        );
        let mut copy = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o400)
            .open(work.join(name))?;
        copy.write_all(&data)?;
        id.check()?;
    }
    let mut cmd = Command::new(&executable);
    cmd.current_dir(work)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", work)
        .env("TMPDIR", work)
        .env("LC_ALL", "C")
        .env("PYTHONNOUSERSITE", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("BREWERY_WEIGHT_PACKED", if packed { "1" } else { "0" })
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("HIP_VISIBLE_DEVICES", "")
        .env("ROCR_VISIBLE_DEVICES", "");
    child_budget(&mut cmd, child_threads(None)?.min(4));
    let source_path = if let Some(source) = &safe_source {
        if source.has_proof() && supports_source_pins(&executable, work, &stamp)? {
            source.pin_for_sandbox(&mut cmd, &ids[0], source_kind == "hf")?
        } else {
            source.path.as_path()
        }
    } else {
        Path::new(source_name)
    };
    let out = work.join(output_name);
    let input_proposal = match &mode {
        Mode::Inspect(v) => {
            cmd.arg("inspect")
                .arg(source_path)
                .args(["--source-sha256", source_sha])
                .arg(format!("--tensor={}", v.tensor))
                .arg(format!("--index={}", serde_json::to_string(&v.index)?));
            None
        }
        Mode::Inventory(v) => {
            cmd.arg("inventory")
                .arg(source_path)
                .args(["--source-sha256", source_sha])
                .arg(format!("--offset={}", v.offset))
                .arg(format!("--limit={}", v.limit));
            None
        }
        Mode::Tile(v) => {
            cmd.arg("tile")
                .arg(source_path)
                .args(["--source-sha256", source_sha])
                .arg(format!("--tensor={}", v.tensor))
                .arg(format!("--origin={}", serde_json::to_string(&v.origin)?))
                .arg(format!("--shape={}", serde_json::to_string(&v.shape)?));
            None
        }
        Mode::Propose(v) => {
            let request = work.join("request.json");
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&request)?;
            serde_json::to_writer(&mut file, &v.edits)?;
            cmd.arg("propose").arg(source_path).args([
                "request.json",
                "--source-sha256",
                source_sha,
            ]);
            None
        }
        Mode::Tune(v) => {
            cmd.arg(source_path)
                .args([
                    "--source-sha256",
                    source_sha,
                    "--calibration",
                    "calibration.json",
                    "--held-out",
                    "held-out.json",
                ])
                .arg(format!("--tensor={}", v.tensor))
                .arg(format!("--indices={}", serde_json::to_string(&v.indices)?))
                .arg(format!("--max-update={}", v.max_update));
            None
        }
        Mode::Apply(_) => {
            let proposal = crate::read_json(&work.join("proposal.json"))?;
            validate_proposal(&proposal, source_sha, policy)?;
            cmd.arg("apply")
                .arg(source_path)
                .args(["proposal.json", output_name]);
            Some(proposal)
        }
    };
    if native_safe {
        cmd.arg("--source-manifest")
            .arg(work.join("source-manifest.json"));
    }
    let phase = match mode {
        Mode::Inspect(_) | Mode::Inventory(_) | Mode::Tile(_) => "inspect",
        Mode::Tune(_) | Mode::Propose(_) => "plan",
        Mode::Apply(_) => "convert",
    };
    let _phase = progress.start(phase, "items", None);
    stamp.check(&executable)?;
    let (stdout, stderr) = capture(
        &mut cmd,
        work,
        native_safe,
        matches!(mode, Mode::Apply(_)),
        ids[0].bytes,
    )?;
    stamp.check(&executable)?;
    ids.iter().try_for_each(integrity::Identity::check)?;
    let mut report: Value =
        serde_json::from_slice(&stdout).context("invalid weight helper result")?;
    ensure!(
        report["source_sha256"] == *source_sha,
        "weight helper source SHA-256 mismatch"
    );
    if !matches!(mode, Mode::Propose(_) | Mode::Tune(_)) {
        ensure!(
            if source_kind != "gguf" {
                report["source_format"] == source_kind
            } else {
                report.get("source_format").is_none_or(|v| v == "gguf")
            },
            "weight helper reported the wrong source format"
        );
    }
    let mut result = json!({
        "kind":job.kind,"host":job.inputs[0].host,"source_sha256":source_sha,
        "inputs":ids.iter().zip(&job.inputs).map(|(id,input)| id.input(input)).collect::<Vec<_>>(),
        "backend_receipt":{"tool":var,"executable_sha256":binary.sha256},
        "release_qualified":false
    });
    if source_kind != "gguf" {
        result["source_format"] = json!(source_kind);
    }
    if packed {
        result["packed"] = json!(true);
    }
    match mode {
        Mode::Inspect(v) => {
            ensure!(
                report["tensor"] == v.tensor
                    && report["index"] == json!(v.index)
                    && policy.editable(&report["dtype"])
                    && report["value"].as_f64().is_some(),
                "weight inspection result differs from request"
            );
        }
        Mode::Inventory(v) => check_inventory(&report, &v, policy)?,
        Mode::Tile(v) => check_tile(&report, &v, policy)?,
        Mode::Propose(_) | Mode::Tune(_) => {
            let edits = validate_proposal(&report, source_sha, policy)?;
            if let Mode::Propose(v) = &mode {
                ensure!(
                    edits == v.edits,
                    "weight proposal differs from requested edits"
                );
            }
            let proposal = report;
            report = json!({"proposal":proposal});
            if let Mode::Tune(v) = mode {
                ensure!(
                    edits.iter().zip(proposal["edits"].as_array().unwrap()).all(
                        |(edit, details)| {
                            edit.tensor == v.tensor
                                && edit.op == "set"
                                && edit
                                    .index
                                    .as_slice()
                                    .try_into()
                                    .is_ok_and(|index: [u64; 2]| v.indices.contains(&index))
                                && (details["after"].as_f64().unwrap()
                                    - details["before"].as_f64().unwrap())
                                .abs()
                                    <= v.max_update
                        }
                    ),
                    "weight tuning proposal did not honor the selection or update cap"
                );
                let tune: Value =
                    serde_json::from_slice(&stderr).context("invalid weight tuning report")?;
                ensure!(
                    tune["schema"] == TUNE
                        && tune["source_sha256"] == *source_sha
                        && tune["calibration_sha256"] == ids[1].sha256
                        && tune["held_out_sha256"] == ids[2].sha256
                        && tune["tensor"] == v.tensor
                        && tune["max_update"].as_f64() == Some(v.max_update)
                        && tune["selected"] == v.indices.len()
                        && tune["proposed"] == proposal["edits"].as_array().unwrap().len(),
                    "weight tuning report does not bind the selected inputs"
                );
                report["tune"] = tune;
            }
            let tmp = work.join("result.json");
            {
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(&tmp)?;
                serde_json::to_writer(&mut file, &proposal)?;
                file.sync_all()?;
            }
            result["artifact"] = publish(
                &tmp,
                &cwd.join("weight-proposal.json"),
                "weight-edit-proposal",
                progress,
                |_| Ok(()),
            )?;
            result["proposal"] = proposal;
        }
        Mode::Apply(_) => {
            let count = input_proposal.as_ref().unwrap()["edits"]
                .as_array()
                .unwrap()
                .len();
            ensure!(
                report["schema"] == RESULT && report["edit_count"] == count,
                "invalid verified edit report"
            );
            if native_safe {
                weight_source::inventory(&out, source_kind)?;
            }
            result["artifact"] =
                publish(&out, &cwd.join(edited_name), source_kind, progress, |id| {
                    ensure!(
                        id.bytes == ids[0].bytes
                            && id.sha256 != *source_sha
                            && report["output_sha256"] == id.sha256,
                        "edited model did not match the verified output receipt"
                    );
                    if let Some(magic) = magic {
                        let mut tag = [0; 4];
                        std::io::Read::read_exact(&mut fs::File::open(&out)?, &mut tag)?;
                        ensure!(
                            &tag == magic,
                            "edited model magic does not match the source format"
                        );
                    } else {
                        safetensors::check_output(&out, source_kind, &ids[0], id)?;
                    }
                    Ok(())
                })?;
            result["verified"] = json!(true);
            if packed || native_safe {
                result["proposal_sha256"] = json!(ids[1].sha256);
            }
            report["verified"] = json!(true);
        }
    }
    result["report"] = report;
    Ok(result)
}

fn dtype_ok(v: &Value) -> bool {
    matches!(v.as_str(), Some("F32" | "F16" | "BF16"))
}

#[derive(Clone, Copy)]
struct CodecPolicy {
    hgn: bool,
    packed: bool,
    native_safe: bool,
}

impl CodecPolicy {
    fn fixed_scale(self, dtype: &Value) -> bool {
        matches!(
            (self.hgn, dtype.as_str()),
            (false, Some("Q4_0" | "Q8_0")) | (true, Some("Q8G64"))
        )
    }

    fn editable(self, dtype: &Value) -> bool {
        dtype_ok(dtype) || self.packed && self.fixed_scale(dtype)
    }
}

fn dimensions(value: &Value) -> Option<Vec<u64>> {
    let values = value.as_array()?;
    if !(1..=4).contains(&values.len()) {
        return None;
    }
    values
        .iter()
        .map(|v| v.as_u64().filter(|n| *n > 0))
        .collect()
}

fn check_inventory(report: &Value, request: &Inventory, policy: CodecPolicy) -> Result<()> {
    let total = report["total"]
        .as_u64()
        .filter(|n| (1..=16384).contains(n))
        .context("invalid weight inventory total")?;
    let tensors = report["tensors"]
        .as_array()
        .context("missing weight inventory entries")?;
    ensure!(
        report["mode"] == "inventory"
            && report["offset"] == request.offset
            && request.offset <= total
            && tensors.len() as u64 == request.limit.min(total - request.offset),
        "weight inventory did not match requested page"
    );
    let end = request.offset + tensors.len() as u64;
    ensure!(
        report.get("next_offset") == Some(&json!((end < total).then_some(end))),
        "invalid next weight inventory offset"
    );
    let mut unique = BTreeSet::new();
    ensure!(
        tensors.iter().all(|tensor| {
            tensor["name"].as_str().is_some_and(|name| {
                !name.is_empty()
                    && name.len() <= 95
                    && name.bytes().all(|b| b.is_ascii_graphic() || b == b' ')
                    && (!policy.native_safe || weight_source::native_tensor_name(name))
                    && unique.insert(name)
            }) && inventory_dtype(tensor, policy)
                && inventory_dimensions(
                    &tensor["shape"],
                    tensor["editable"] == false && policy.native_safe,
                )
                .and_then(|dims| dims.into_iter().try_fold(1u64, u64::checked_mul))
                .is_some_and(|n| tensor["elements"] == n)
        }),
        "invalid tensor metadata in weight inventory"
    );
    Ok(())
}

fn inventory_dimensions(value: &Value, read_only_safetensors: bool) -> Option<Vec<u64>> {
    if !read_only_safetensors {
        return dimensions(value);
    }
    let values = value.as_array()?;
    (values.len() <= 8).then_some(())?;
    values.iter().map(Value::as_u64).collect()
}

fn inventory_dtype(tensor: &Value, policy: CodecPolicy) -> bool {
    match tensor["editable"].as_bool() {
        Some(true) => policy.editable(&tensor["dtype"]),
        Some(false) => {
            tensor["dtype"].as_str().is_some_and(|name| {
                (1..=32).contains(&name.len())
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"_-/".contains(&b))
            }) && tensor["reason"].as_str().is_some_and(|reason| {
                (1..=256).contains(&reason.len()) && !reason.chars().any(char::is_control)
            })
        }
        None => {
            !policy.hgn
                && !policy.native_safe
                && tensor.get("editable").is_none()
                && dtype_ok(&tensor["dtype"])
        }
    }
}

fn check_tile(report: &Value, request: &Tile, policy: CodecPolicy) -> Result<()> {
    let shape = dimensions(&report["shape"]).context("invalid weight tensor dimensions")?;
    let (columns, prefix) = request.shape.split_last().unwrap();
    let rows = prefix.last().copied().unwrap_or(1);
    ensure!(
        report["mode"] == "tile"
            && report["tensor"] == request.tensor
            && report["origin"] == json!(request.origin)
            && report["tile_shape"] == json!(request.shape)
            && policy.editable(&report["dtype"])
            && shape.len() == request.shape.len()
            && request
                .origin
                .iter()
                .zip(&request.shape)
                .zip(shape)
                .all(|((origin, n), dim)| origin.checked_add(*n).is_some_and(|end| end <= dim))
            && report["values"].as_array().is_some_and(|grid| {
                grid.len() as u64 == rows
                    && grid.iter().all(|row| {
                        row.as_array().is_some_and(|values| {
                            values.len() as u64 == *columns
                                && values
                                    .iter()
                                    .all(|value| value.as_f64().is_some_and(f64::is_finite))
                        })
                    })
            }),
        "weight tile did not match requested coordinate or value bounds"
    );
    if policy.fixed_scale(&report["dtype"]) {
        let grid = report["packed_steps"]
            .as_array()
            .context("missing fixed-scale tile steps")?;
        let values = report["values"].as_array().unwrap();
        ensure!(
            grid.len() == values.len()
                && grid.iter().zip(values).all(|(row, values)| {
                    let values = values.as_array().unwrap();
                    row.as_array().is_some_and(|cells| {
                        cells.len() == values.len()
                            && cells.iter().zip(values).all(|(cell, value)| {
                                let value = value.as_f64().unwrap();
                                cell.as_object().is_some_and(|cell| {
                                    cell.len() == 2
                                        && cell.get("lower").is_some_and(|v| {
                                            v.is_null() || v.as_f64().is_some_and(|v| v < value)
                                        })
                                        && cell.get("upper").is_some_and(|v| {
                                            v.is_null() || v.as_f64().is_some_and(|v| v > value)
                                        })
                                })
                            })
                    })
                }),
            "fixed-scale steps do not match weight tile values"
        );
    } else {
        ensure!(
            report.get("packed_steps").is_none(),
            "floating-point tile must not contain packed steps"
        );
    }
    Ok(())
}

fn validate_proposal(proposal: &Value, sha: &str, policy: CodecPolicy) -> Result<Vec<Edit>> {
    ensure!(
        proposal.as_object().is_some_and(|v| v.len() == 3)
            && proposal["schema"] == PROPOSAL
            && proposal["source_sha256"] == sha,
        "proposal is not bound to the verified source"
    );
    let edits = proposal["edits"]
        .as_array()
        .context("missing proposal edits")?;
    ensure!(
        (1..=256).contains(&edits.len()),
        "invalid proposal edit count"
    );
    let mut seen = BTreeSet::new();
    let mut selected = Vec::with_capacity(edits.len());
    for edit in edits {
        let scalar: Edit = serde_json::from_value(json!({
            "tensor":edit["tensor"],"index":edit["index"],"op":edit["op"],"value":edit["value"]
        }))?;
        ensure!(
            edit.as_object().is_some_and(|v| v.len() == 7)
                && valid_edit(&scalar, policy.native_safe)
                && policy.editable(&edit["dtype"])
                && edit["before"].as_f64().is_some()
                && edit["after"].as_f64().is_some()
                && seen.insert((scalar.tensor.clone(), scalar.index.clone())),
            "invalid or duplicate proposal edit"
        );
        selected.push(scalar);
    }
    Ok(selected)
}

fn publish(
    path: &Path,
    target: &Path,
    kind: &str,
    progress: &Progress,
    validate: impl FnOnce(&integrity::Identity) -> Result<()>,
) -> Result<Value> {
    let meta = fs::symlink_metadata(path)?;
    ensure!(
        if kind == "hf" {
            meta.is_dir()
        } else {
            meta.is_file()
        },
        "weight helper artifact has the wrong file kind"
    );
    let id = integrity::identify_output(path, progress)?;
    validate(&id)?;
    id.check()?;
    let mut phase = progress.start("publish_output", "items", Some(1));
    converter::publish_directory(path, target)?;
    phase.advance(1);
    phase.finish();
    Ok(id.artifact(target, kind))
}

fn supports_source_pins(executable: &Path, work: &Path, stamp: &integrity::Stamp) -> Result<bool> {
    let mut probe = Command::new(executable);
    probe.current_dir(work).env_clear().arg("--capabilities");
    stamp.check(executable)?;
    let result = capture(&mut probe, work, true, false, 0);
    stamp.check(executable)?;
    let Ok((stdout, _)) = result else {
        return Ok(false); // Existing operator-provided direct helpers have no capability probe.
    };
    let receipt = serde_json::from_slice::<Value>(&stdout).unwrap_or(Value::Null);
    let packaged = receipt["schema"] == "brewery.weight-runtime.v1";
    ensure!(
        !packaged || receipt["proof_source_fds"] == true,
        "packaged SafeTensors source proofs require a runtime with Bubblewrap --ro-bind-fd"
    );
    Ok(packaged)
}

fn capture(
    cmd: &mut Command,
    work: &Path,
    native_safe: bool,
    apply: bool,
    source_bytes: u64,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut stdout = tempfile::tempfile_in(work)?;
    let mut stderr = tempfile::tempfile_in(work)?;
    // File cap bounds even a burst between monitor polls. It also covers the
    // helper's temporary GGUF output. The supervisor handles job cancellation.
    unsafe {
        cmd.pre_exec(move || {
            let max_file = if native_safe && apply {
                source_bytes.max(JSON)
            } else {
                MODEL
            };
            let limit = libc::rlimit {
                rlim_cur: max_file,
                rlim_max: max_file,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if native_safe {
                let mut memory = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                if libc::getrlimit(libc::RLIMIT_AS, &mut memory) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                memory.rlim_cur = memory.rlim_cur.min(safetensors::MEMORY_BYTES);
                if libc::setrlimit(libc::RLIMIT_AS, &memory) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(stdout.try_clone()?)
        .stderr(stderr.try_clone()?)
        .spawn()?;
    let start = Instant::now();
    let seconds = if native_safe && apply {
        // Stream and reread a complete output, including on slower storage.
        (120 + source_bytes / (4 << 20)).min(24 * 60 * 60)
    } else {
        120
    };
    loop {
        let size = stdout
            .metadata()
            .and_then(|a| stderr.metadata().map(|b| (a.len(), b.len())));
        if !size.is_ok_and(|(a, b)| a <= JSON && b <= JSON)
            || start.elapsed() >= Duration::from_secs(seconds)
        {
            let _ = child.kill();
            let _ = child.wait();
            bail!("weight helper exceeded output or time limit");
        }
        if let Some(status) = child.try_wait()? {
            ensure!(
                status.success(),
                "weight helper rejected the job ({status})"
            );
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    stdout.rewind()?;
    stderr.rewind()?;
    Ok((bounded(stdout, JSON)?, bounded(stderr, JSON)?))
}
