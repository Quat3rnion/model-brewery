use crate::{check_inputs, integrity, progress::Progress, Job};
use anyhow::{ensure, Context, Result};
use brewery_worker::qsa as contract;
use serde::{de, Deserialize, Deserializer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fmt,
    fs::{self, File},
    io::Read,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const MONO: &str = "model.safetensors";
const INDEX: &str = "model.safetensors.index.json";
const INDEX_LIMIT: u64 = 16 << 20;

#[derive(Deserialize)]
struct Index {
    #[serde(deserialize_with = "unique_shards")]
    weight_map: BTreeSet<String>,
}

fn unique_shards<'de, D: Deserializer<'de>>(deserializer: D) -> Result<BTreeSet<String>, D::Error> {
    struct Shards;
    impl<'de> de::Visitor<'de> for Shards {
        type Value = BTreeSet<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a map from unique weight names to shard filenames")
        }

        fn visit_map<M: de::MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
            let mut names = BTreeSet::new();
            let mut shards = BTreeSet::new();
            while let Some((weight, shard)) = map.next_entry::<String, String>()? {
                if !names.insert(weight) {
                    return Err(de::Error::custom("duplicate weight mapping"));
                }
                shards.insert(shard);
            }
            Ok(shards)
        }
    }
    deserializer.deserialize_map(Shards)
}

fn verify_teacher_weights(path: &Path, members: &[integrity::Member]) -> Result<()> {
    let verified_member = |name: &str| {
        members
            .iter()
            .find(|entry| entry.path == name && entry.bytes > 0)
    };
    // Look at both selector paths, so a directory with a selector name cannot
    // be absent from the file manifest yet influence the loader's choice.
    let exists = |name| match fs::symlink_metadata(path.join(name)) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    };
    match (exists(MONO)?, exists(INDEX)?) {
        (true, false) => {
            ensure!(
                verified_member(MONO).is_some(),
                "QSA teacher file is not a verified bundle member"
            );
        }
        (false, true) => {
            let index = verified_member(INDEX)
                .context("QSA teacher index is not a verified bundle member")?;
            ensure!(
                index.bytes <= INDEX_LIMIT,
                "QSA teacher index exceeds the size limit"
            );
            let file = brewery_worker::weight_source::open_regular(&path.join(INDEX))?;
            let mut bytes = Vec::new();
            file.take(INDEX_LIMIT + 1).read_to_end(&mut bytes)?;
            ensure!(
                bytes.len() as u64 == index.bytes
                    && format!("{:x}", Sha256::digest(&bytes)) == index.sha256,
                "QSA teacher index differs from the verified bundle"
            );
            let index: Index =
                serde_json::from_slice(&bytes).context("Invalid QSA teacher index")?;
            ensure!(
                !index.weight_map.is_empty(),
                "QSA teacher weight map is empty"
            );
            ensure!(
                index
                    .weight_map
                    .iter()
                    .all(|name| brewery_worker::valid_basename(name)
                        && name.ends_with(".safetensors")
                        && verified_member(name).is_some()),
                "QSA teacher shard is not a verified bundle member"
            );
        }
        _ => anyhow::bail!("QSA teacher needs exactly one SafeTensors file or index"),
    }
    Ok(())
}

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
    verify_teacher_weights(&source.path, &identities[0].files)?;
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
    for identity in &identities {
        identity.check()?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teacher_index_requires_each_shard_in_the_verified_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().canonicalize().unwrap();
        let first = "model-00001-of-00002.safetensors";
        let second = "model-00002-of-00002.safetensors";
        fs::write(
            path.join(INDEX),
            json!({"weight_map":{"weight_a":first,"weight_b":second}}).to_string(),
        )
        .unwrap();
        fs::write(path.join(first), b"first fixture shard").unwrap();
        let missing = integrity::identify(&path, false).unwrap();
        assert!(verify_teacher_weights(&path, &missing.files)
            .unwrap_err()
            .to_string()
            .contains("shard"));
        fs::write(path.join(second), b"second fixture shard").unwrap();
        assert!(verify_teacher_weights(&path, &missing.files).is_err());
        let complete = integrity::identify(&path, false).unwrap();
        verify_teacher_weights(&path, &complete.files).unwrap();
        fs::write(
            path.join(INDEX),
            format!(r#"{{"weight_map":{{"weight_a":"{first}","weight_a":"{second}"}}}}"#),
        )
        .unwrap();
        let duplicate = integrity::identify(&path, false).unwrap();
        assert!(verify_teacher_weights(&path, &duplicate.files).is_err());
        assert!(verify_teacher_weights(&path, &complete.files).is_err());
        fs::write(path.join(MONO), b"ambiguous fixture").unwrap();
        let ambiguous = integrity::identify(&path, false).unwrap();
        assert!(verify_teacher_weights(&path, &ambiguous.files).is_err());
        fs::remove_file(path.join(INDEX)).unwrap();
        let monolithic = integrity::identify(&path, false).unwrap();
        verify_teacher_weights(&path, &monolithic.files).unwrap();
    }
}
