use crate::{
    integrity,
    progress::Progress,
    trace::{hgn, Runtime},
    Job,
};
use anyhow::{ensure, Context, Result};
use brewery_worker::{behavior_direction as direction, trace};
use serde_json::{json, Value};
use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let request = direction::Request::parse(&job.options)?;
    ensure!(
        job.prompt.is_none() && job.inputs.len() == 1,
        "direction needs one Library input and the pinned prompt set"
    );
    let input = &job.inputs[0];
    request.check_input_kind(&input.kind)?;
    let is_hgn = trace::is_hgn(&input.kind);
    let model_id = input.id.as_deref().context("missing Library model ID")?;
    ensure!(
        input.sha256.as_deref() == Some(request.model_sha256.as_str()),
        "direction needs the resolved model revision"
    );
    let cache =
        PathBuf::from(env::var_os(direction::CACHE).context("direction cache is not configured")?);
    ensure!(cache.is_absolute(), "direction cache path must be absolute");
    let pairs = request.load_pairs(&cache)?;
    let splits: Vec<_> = pairs.iter().map(|p| p.split).collect();
    let [model] = integrity::check_inputs(&job.inputs, progress)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("missing verified model"))?;
    ensure!(
        model.files.len() == 1 && input.path.is_file() && model.sha256 == request.model_sha256,
        "direction needs a single-file checkpoint with matching SHA-256"
    );
    let hgn_dir = is_hgn
        .then(|| {
            tempfile::Builder::new()
                .prefix(".direction-hgn-")
                .permissions(fs::Permissions::from_mode(0o700))
                .tempdir_in(cwd)
        })
        .transpose()?;
    let hgn = hgn_dir
        .as_ref()
        .map(|dir| hgn::Runtime::stage(dir.path(), &model.sha256))
        .transpose()?;
    let runtime = (!is_hgn).then(|| Runtime::stage(cwd)).transpose()?;
    let work = if let Some(dir) = &hgn_dir {
        dir.path()
    } else {
        runtime.as_ref().context("missing trace runtime")?.path()
    };
    let out = work.join("direction-trace");
    let mut phase = progress.start("trace_prompts", "items", Some((pairs.len() * 2) as u64));
    let fit = direction::compute(&request.layers, request.role, &splits, |row, positive| {
        let pair = &pairs[row];
        let prompt = if positive {
            &pair.positive
        } else {
            &pair.negative
        };
        let prompt = if is_hgn {
            prompt.hgn_chat_prefix()
        } else {
            prompt.qwen_chat_prefix()
        };
        let capture = |spec: &trace::Capture| {
            model.check()?;
            let verified = if let Some(hgn) = &hgn {
                hgn.capture(work, &input.path, &prompt, spec, &out)?
            } else {
                runtime
                    .as_ref()
                    .context("missing Lanternfish runtime")?
                    .capture(&input.path, &prompt, spec, &out, None)?
            };
            model.check()?;
            let captured = trace::direction_rows(&out, spec, &verified, request.role, is_hgn)?;
            fs::remove_dir_all(&out)?;
            Ok::<_, anyhow::Error>(captured)
        };
        let probe = if request.position.is_none() {
            Some(capture(&trace::direction_probe())?.0)
        } else {
            None
        };
        let positions = if let Some(position) = request.position {
            position..position + 1
        } else {
            let len = probe.as_ref().context("missing direction probe")?.len();
            let start = len
                .checked_sub(request.last_k.unwrap_or(1))
                .context("last_k exceeds the number of prompt tokens")?;
            start..len
        };
        let (tokens, rows) = capture(&trace::direction_capture(
            &request.layers,
            request.role,
            positions,
        ))?;
        ensure!(
            probe.is_none_or(|expected| expected == tokens),
            "direction trace token IDs changed"
        );
        phase.advance(1);
        Ok(rows)
    })?;
    phase.finish();
    let mut result = if let Some(hgn) = &hgn {
        let mut value = hgn.receipt();
        if request.position.is_none() {
            value["last_k"] = json!(request.last_k.unwrap_or(1));
        }
        value
    } else {
        json!({"lanternfish_commit":trace::COMMIT, "executable_sha256":trace::SHA256})
    };
    let fields = json!({
        "kind":direction::JOB, "schema_version":direction::VERSION, "release_qualified":false,
        "model_id":model_id, "model_sha256":model.sha256, "position":request.position,
        "behavior":request.behavior, "contrast_sha256":request.behavior.contrast_sha256(),
        "fit":fit,
    });
    result
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    request.verify_result(&result, model_id, &model.sha256, &input.kind)?;
    Ok(result)
}
