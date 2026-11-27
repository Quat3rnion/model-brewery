use crate::{Job, check_inputs, integrity, progress::Progress};
use anyhow::{Context, Result, ensure};
use brewery_worker::qsa as contract;
use serde_json::{Value, json};
use std::{
    env,
    fs::{self, File},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

fn trainer() -> Result<Command> {
    let path = |key| -> Result<PathBuf> {
        let path = PathBuf::from(env::var_os(key).context("QSA runtime is not configured")?);
        brewery_worker::absolute(&path)?;
        ensure!(fs::metadata(&path)?.is_file(), "invalid QSA runtime file");
        Ok(path)
    };
    // Keep the configured venv invocation path: resolving its Python symlink
    // would bypass the venv and probe the system dependency set instead.
    let mut command = Command::new(path(contract::PYTHON)?);
    command
        .args(["-I", "-B"])
        .arg(path(contract::TRAINER)?)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .stdin(Stdio::null());
    Ok(command)
}

pub fn probe() -> Result<Value> {
    let output = trainer()?.arg("--probe").output()?;
    let result: Value = brewery_worker::decode_limited(output.stdout.as_slice(), 4096)?;
    ensure!(
        output.status.success() && contract::probe_receipt(&result),
        "QSA GPU runtime probe did not pass"
    );
    Ok(result)
}

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let spec = contract::Spec::parse(&job.options)?;
    ensure!(
        job.prompt.is_none()
            && contract::layout(
                job.inputs
                    .iter()
                    .map(|i| { (i.kind.as_str(), i.host.as_str(), i.path.as_path()) })
            ),
        "QSA training requires a direct teacher and two separate corpora"
    );
    let [source, train, eval] = job.inputs.as_slice() else {
        unreachable!()
    };
    ensure!(
        integrity::canonical(&source.path)?.is_dir()
            && job.inputs.iter().all(|input| {
                input.source_host.as_ref() == Some(&input.host)
                    && input.files.is_some()
                    && input.bundle_sha256.is_some()
                    && input.bytes.is_some()
            })
            && [train, eval].iter().all(|input| {
                input.files.as_ref().is_some_and(|files| files.len() == 1)
                    && input.sha256.is_some()
                    && integrity::canonical(&input.path).is_ok_and(|m| {
                        m.is_file() && (1..=contract::CORPUS_LIMIT).contains(&m.len())
                    })
            }),
        "QSA inputs lack resolved direct identities"
    );
    let identities = check_inputs(&job.inputs, progress)?;
    let config = identities[0]
        .files
        .iter()
        .find(|entry| entry.path == "config.json")
        .context("QSA teacher config is missing")?;
    ensure!(
        identities[1].sha256 != identities[2].sha256,
        "training and held-out corpora have identical bytes"
    );
    let out = cwd.join(contract::OUTPUT);
    probe()?;
    let phase = progress.start("qsa_training", "items", None);
    let status = trainer()?
        .args(["--source"])
        .arg(&source.path)
        .args(["--train-data"])
        .arg(&train.path)
        .args(["--eval-data"])
        .arg(&eval.path)
        .args(["--output-dir"])
        .arg(&out)
        .args(["--steps", &spec.steps.to_string()])
        .args(["--seq-len", &spec.seq_len.to_string()])
        .args(["--seed", &spec.seed.to_string()])
        .args(["--lr", &spec.lr.to_string()])
        .stdout(Stdio::null())
        .status()?;
    ensure!(status.success(), "QSA trainer failed: {status}");
    phase.finish();
    for identity in &identities {
        identity.check()?;
    }
    let file = out.join(contract::FILE);
    let metrics_path = out.join(contract::METRICS);
    ensure!(
        fs::read_dir(&out)?.count() == 2
            && [&file, &metrics_path].iter().all(|path| {
                integrity::canonical(path).is_ok_and(|m| m.is_file() && m.nlink() == 1)
            }),
        "QSA trainer did not produce both regular output files"
    );
    let metrics: Value =
        brewery_worker::decode_limited(File::open(&metrics_path)?, contract::METRICS_LIMIT)?;
    contract::check_metrics(
        &spec,
        &metrics,
        &config.sha256,
        &identities[1].sha256,
        &identities[2].sha256,
    )?;
    contract::check_indexer(&mut File::open(&file)?, &config.sha256)?;
    let artifact = integrity::identify_output(&file, progress)?.artifact(&file, contract::INDEXER);
    let metrics_file = integrity::identify(&metrics_path, false)?.artifact(&metrics_path, "json");
    for path in [&file, &metrics_path, &out, cwd] {
        File::open(path)?.sync_all()?;
    }
    let result = json!({
        "kind":contract::JOB, "host":source.host, "verified":true, "release_qualified":false,
        "teacher":{"id":source.id,"sha256":identities[0].sha256},
        "inputs":identities.iter().zip(&job.inputs).map(|(identity, input)| identity.input(input)).collect::<Vec<_>>(),
        "artifact":artifact, "metrics_file":metrics_file, "metrics":metrics,
    });
    contract::check_receipt(&spec, &source.host, &result)?;
    Ok(result)
}
