use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs::File, io::Read, path::Path};

pub const JOB: &str = "qsa_train";
pub const CORPUS: &str = "qsa-corpus";
pub const INDEXER: &str = "qsa-indexer";
pub const OUTPUT: &str = "qsa";
pub const FILE: &str = "qsa-indexers.safetensors";
pub const METRICS: &str = "metrics.json";
pub const PROBE: &str = "brewery.qsa-probe.v1";
pub const PYTHON: &str = "BREWERY_QSA_PYTHON";
pub const TRAINER: &str = "BREWERY_QSA_TRAINER";
pub const CORPUS_LIMIT: u64 = 64 << 20;
pub const METRICS_LIMIT: u64 = 64 << 10;
pub const INDEXER_LIMIT: u64 = 256 << 20;
const REFERENCE_REVISION: &str = "9eb7192828ebc6234a892d718fc775b3df648596";
const LAYERS: [u64; 16] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63];
const NO_CAPTURE: &str =
    "No held-out query has more fully observed blocks than the capture budget.";

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub steps: u32,
    pub seq_len: u32,
    pub seed: u32,
    pub lr: f64,
}

impl Spec {
    pub fn parse(value: &Value) -> Result<Self> {
        let spec: Self = serde_json::from_value(value.clone()).map_err(|_| {
            anyhow::anyhow!("QSA training requires only steps, seq_len, seed, and lr")
        })?;
        ensure!(
            (1..=10_000).contains(&spec.steps)
                && (8..=16_384).contains(&spec.seq_len)
                && spec.lr > 0.0
                && spec.lr <= 1.0,
            "QSA options require steps 1–10000, seq_len 8–16384, seed 0–4294967295, and lr greater than 0 and at most 1"
        );
        Ok(spec)
    }
}

pub fn layout<'a>(inputs: impl IntoIterator<Item = (&'a str, &'a str, &'a Path)>) -> bool {
    let mut inputs = inputs.into_iter();
    let Some(("hf", host, teacher)) = inputs.next() else {
        return false;
    };
    let Some((CORPUS, train_host, train)) = inputs.next() else {
        return false;
    };
    let Some((CORPUS, eval_host, eval)) = inputs.next() else {
        return false;
    };
    host == train_host
        && host == eval_host
        && host != "auto"
        && train != eval
        && !train.starts_with(teacher)
        && !eval.starts_with(teacher)
        && inputs.next().is_none()
}

pub fn probe_receipt(value: &Value) -> bool {
    *value == json!({"schema":PROBE,"ready":true})
}

fn metric(value: &Value, limit: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|value| value >= 0.0 && value <= limit)
}

pub fn check_metrics(
    spec: &Spec,
    metrics: &Value,
    source: &str,
    train: &str,
    eval: &str,
) -> Result<()> {
    let options = json!({
        "steps": metrics["steps"], "seq_len": metrics["seq_len"],
        "seed": metrics["seed"], "lr": metrics["lr"],
    });
    ensure!(
        metrics["status"] == "passed"
            && Spec::parse(&options).is_ok_and(|options| options == *spec)
            && metrics["source_config_sha256"] == source
            && metrics["train_data_sha256"] == train
            && metrics["eval_data_sha256"] == eval
            && metrics["full_attention_layers"] == json!(LAYERS)
            && metrics["budget"] == 2048
            && metrics["train_loss"].as_f64().is_some_and(f64::is_finite)
            && metrics["heldout_loss"] == metrics["after"]["kl"]
            && metrics["indexer_update_l2"]
                .as_f64()
                .is_some_and(|n| n.is_finite() && n > 0.0)
            && ["before", "after"].iter().all(|key| {
                let row = &metrics[key];
                let capture_queries = row["capture_queries"].as_u64();
                (match capture_queries {
                    Some(0) => {
                        row.get("capture").is_some_and(Value::is_null)
                            && row["capture_unavailable_reason"] == NO_CAPTURE
                    }
                    Some(_) => {
                        metric(&row["capture"], 1.0)
                            && row.get("capture_unavailable_reason").is_none()
                    }
                    None => false,
                }) && row["kl"].as_f64().is_some_and(f64::is_finite)
                    && capture_queries
                        .zip(row["queries"].as_u64())
                        .is_some_and(|(capture, queries)| queries > 0 && capture <= queries)
            })
            && metrics["before"]["queries"] == metrics["after"]["queries"]
            && metrics["before"]["capture_queries"] == metrics["after"]["capture_queries"],
        "QSA trainer metrics do not attest completion on the resolved inputs"
    );
    Ok(())
}

pub fn check_indexer(file: &mut File, source_config_sha256: &str) -> Result<()> {
    let bytes = file.metadata()?.len();
    ensure!(
        (9..=INDEXER_LIMIT).contains(&bytes),
        "invalid QSA indexer size"
    );
    let mut len = [0; 8];
    file.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    ensure!(
        (2..=METRICS_LIMIT.min(bytes - 8)).contains(&len),
        "invalid QSA indexer header size"
    );
    let mut header = vec![0; len as usize];
    file.read_exact(&mut header)?;
    let mut tensors: serde_json::Map<String, Value> = serde_json::from_slice(&header)?;
    let metadata = tensors
        .remove("__metadata__")
        .context("missing QSA indexer metadata")?;
    ensure!(
        metadata["format"] == "pt"
            && metadata["teacher"] == "dense_causal_qwen3_5"
            && metadata["reference_revision"] == REFERENCE_REVISION
            && metadata["source_config_sha256"] == source_config_sha256,
        "QSA indexer metadata does not match the teacher"
    );
    let mut ranges = Vec::new();
    for layer in LAYERS {
        for (key, shape) in [
            ("q_proj.weight", &[512u64, 5120][..]),
            ("k_proj.weight", &[128, 5120][..]),
            ("q_norm", &[128][..]),
            ("k_norm", &[128][..]),
            ("k_block_norm", &[128][..]),
        ] {
            let name = format!("model.language_model.layers.{layer}.self_attn._qsa_indexer.{key}");
            let tensor = tensors
                .remove(&name)
                .context("missing QSA indexer tensor")?;
            let offsets: [u64; 2] = serde_json::from_value(tensor["data_offsets"].clone())?;
            ensure!(
                tensor["dtype"] == "F32"
                    && tensor["shape"] == json!(shape)
                    && offsets[1].checked_sub(offsets[0])
                        == Some(4 * shape.iter().product::<u64>()),
                "invalid QSA indexer tensor"
            );
            ranges.push(offsets);
        }
    }
    ranges.sort_unstable();
    let mut end = 0;
    for [start, next] in ranges {
        ensure!(start == end, "non-contiguous QSA indexer tensor payload");
        end = next;
    }
    ensure!(
        tensors.is_empty() && end == bytes - 8 - len,
        "invalid QSA indexer payload"
    );
    Ok(())
}

pub fn check_receipt(spec: &Spec, host: &str, result: &Value) -> Result<()> {
    let inputs = result["inputs"]
        .as_array()
        .filter(|inputs| inputs.len() == 3)
        .context("QSA receipt is missing input identities")?;
    let ids: Vec<_> = inputs
        .iter()
        .map(|input| {
            let files: Vec<crate::Entry> = serde_json::from_value(input["files"].clone())?;
            let path = input["path"].as_str().context("QSA input path missing")?;
            let digest = input["bundle_sha256"]
                .as_str()
                .context("QSA input digest missing")?;
            let entry = if input["kind"] == "hf" {
                "."
            } else {
                Path::new(path)
                    .file_name()
                    .and_then(|p| p.to_str())
                    .context("QSA corpus leaf missing")?
            };
            let bundle = crate::Bundle {
                entry: entry.into(),
                files,
                digest: digest.into(),
            };
            bundle.validate()?;
            ensure!(
                input["bytes"].as_u64() == Some(bundle.bytes()?)
                    && input["host"] == host
                    && input["source_host"] == host,
                "invalid QSA input identity"
            );
            Ok(bundle)
        })
        .collect::<Result<_>>()?;
    let source = ids[0]
        .files
        .iter()
        .find(|entry| entry.path == "config.json" && entry.bytes > 0)
        .context("QSA teacher config is missing")?;
    let corpus = |index: usize| -> Result<&str> {
        let bundle = &ids[index];
        let [file] = bundle.files.as_slice() else {
            anyhow::bail!("invalid QSA corpus manifest")
        };
        ensure!(
            bundle.entry == file.path
                && (1..=CORPUS_LIMIT).contains(&file.bytes)
                && inputs[index]["sha256"] == file.sha256,
            "invalid QSA corpus receipt"
        );
        Ok(&file.sha256)
    };
    let train = corpus(1)?;
    let eval = corpus(2)?;
    ensure!(
        result["kind"] == JOB
            && result["host"] == host
            && result["verified"] == true
            && result["release_qualified"] == false
            && train != eval
            && inputs[0]["kind"] == "hf"
            && inputs[1]["kind"] == CORPUS
            && inputs[2]["kind"] == CORPUS
            && inputs[0]["sha256"] == ids[0].digest
            && result["teacher"] == json!({"id":inputs[0]["id"], "sha256":ids[0].digest}),
        "QSA result does not bind its verified teacher and corpora"
    );
    check_metrics(spec, &result["metrics"], &source.sha256, train, eval)?;
    for (entry, kind, limit) in [
        (&result["artifact"], INDEXER, INDEXER_LIMIT),
        (&result["metrics_file"], "json", METRICS_LIMIT),
    ] {
        ensure!(
            entry["kind"] == kind
                && entry["sha256"].as_str().is_some_and(crate::is_digest)
                && entry["bytes"]
                    .as_u64()
                    .is_some_and(|n| (1..=limit).contains(&n)),
            "invalid QSA publication receipt"
        );
    }
    Ok(())
}
