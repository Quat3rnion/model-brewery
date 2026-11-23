use crate::{Job, child_threads, gguf_conversion::Tool, integrity, progress::Progress, write_json};
use anyhow::{Context, Result, ensure};
use brewery_worker::{is_digest, locate_edit, weight_source::open_regular};
use serde_json::{Value, json};
use std::{
    env, fs,
    io::BufReader,
    path::{Path, PathBuf},
};

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let request = locate_edit::Request::parse(&job.options)?;
    ensure!(
        job.prompt.is_none()
            && job
                .inputs
                .iter()
                .map(|i| i.kind.as_str())
                .eq(locate_edit::INPUTS)
            && job
                .inputs
                .iter()
                .all(|i| i.sha256.as_deref().is_some_and(is_digest)),
        "locate-then-edit needs pinned private GGUF and SafeTensors inputs"
    );
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    ensure!(
        ids.iter().all(|id| id.files.len() == 1),
        "expected single-file inputs"
    );
    let model = &job.inputs[0].path;
    let work = tempfile::Builder::new()
        .prefix(".locate-edit-")
        .tempdir_in(cwd)?;
    let tool = Tool::pinned_digest(
        work.path(),
        locate_edit::EXECUTABLE,
        locate_edit::SHA256,
        "lanternfish",
        "Lanternfish",
    )?;
    let configured = PathBuf::from(
        env::var_os(locate_edit::LIB_DIR).context("missing edit runtime library directory")?,
    );
    ensure!(
        integrity::canonical(&configured)?.is_dir(),
        "invalid edit runtime library directory"
    );
    let libs = work.path().join("libs");
    fs::create_dir(&libs)?;
    for (name, sha) in locate_edit::LIBRARIES {
        let library = configured.join(name).canonicalize()?;
        ensure!(
            library.parent() == Some(configured.as_path()),
            "edit runtime library escapes its directory"
        );
        let staged = libs.join(name);
        fs::copy(library, &staged)?;
        ensure!(
            integrity::file_digest(&staged, name.into())?.0.sha256 == sha,
            "edit runtime library {name} SHA-256 mismatch"
        );
    }
    let cache = cwd
        .ancestors()
        .nth(3)
        .context("missing private worker root")?
        .join("edit-moments");
    fs::create_dir_all(&cache)?;
    ensure!(integrity::canonical(&cache)?.is_dir(), "invalid moment cache");
    let request_file = work.path().join("request.json");
    write_json(&request_file, &serde_json::to_value(&request)?)?;
    let out = work.path().join("factors.json");
    let threads = child_threads(None)?.min(16);
    let mut command = tool.command(work.path(), threads);
    command
        .env("LD_LIBRARY_PATH", &libs)
        .arg("locate-edit")
        .arg("--model")
        .arg(model)
        .arg("--request")
        .arg(request_file)
        .arg("--out")
        .arg(&out)
        .arg("--cache")
        .arg(cache)
        .arg("--threads")
        .arg(threads.to_string());
    ids.iter().try_for_each(integrity::Identity::check)?;
    let mut phase = progress.start("locate_edit", "items", Some(1));
    tool.capture(&mut command, work.path())
        .context("Lanternfish locate-then-edit failed")?;
    ids.iter().try_for_each(integrity::Identity::check)?;
    let file = open_regular(&out)?;
    ensure!(
        file.metadata()?.len() <= brewery_worker::LIMIT,
        "edit factor export exceeds byte limit"
    );
    let factors: locate_edit::Factors = serde_json::from_reader(BufReader::new(file))?;
    factors.verify(&request, &ids[0].sha256)?;
    phase.advance(1);
    phase.finish();
    Ok(json!({
        "kind":locate_edit::JOB,"parent_sha256":ids[1].sha256,
        "executable_sha256":locate_edit::SHA256,"lanternfish_commit":locate_edit::COMMIT,
        "factors":factors
    }))
}
