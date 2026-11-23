use super::{PROMPTS, Runtime, dot_payload};
use crate::{
    bounded,
    gguf_conversion::inventory::{Model, Tensor},
    sensitivity::matches_device,
    write_json,
};
use anyhow::{Context, Result, ensure};
use brewery_worker::{
    damage::CORPUS_LINES, is_digest, sensitivity::CORPUS_SHA256, weight_source::open_regular,
};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    path::{Component, Path, PathBuf},
};

#[derive(Deserialize)]
struct Placement {
    strict: bool,
    device: String,
}

#[derive(Deserialize)]
struct Entry {
    name: String,
    shape: Vec<u64>,
    file: String,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    objective: String,
    corpus_sha256: String,
    dtype: String,
    tokens: Vec<Vec<i32>>,
    observation_count: usize,
    loss: f64,
    backend: String,
    placement: Placement,
    gradients: Vec<Entry>,
}

pub(super) struct Gradients {
    out: PathBuf,
    pub loss: f64,
    pub observation_count: usize,
    entries: BTreeMap<String, Entry>,
}

pub(super) fn run(
    runtime: &Runtime,
    work: &Path,
    model: &Path,
    names: &[&str],
) -> Result<Gradients> {
    let corpus = work.join("heldout.txt");
    let spec = work.join("corpus-grad.json");
    let out = work.join("corpus-gradients");
    fs::write(&corpus, PROMPTS)?;
    write_json(
        &spec,
        &json!({"corpus_sha256":CORPUS_SHA256, "weights":names}),
    )?;
    runtime.run(
        work,
        "grad-corpus",
        model,
        &[
            "--spec",
            spec.to_str().context("non-UTF8 work path")?,
            "--corpus",
            corpus.to_str().context("non-UTF8 work path")?,
            "--out",
            out.to_str().context("non-UTF8 work path")?,
        ],
    )?;
    let manifest: Manifest = serde_json::from_slice(&bounded(
        open_regular(&out.join("gradient.json"))?,
        256 << 10,
    )?)?;
    let placement = &manifest.placement;
    ensure!(
        manifest.schema_version == 1
            && manifest.objective == "mean_next_token_nll"
            && manifest.corpus_sha256 == CORPUS_SHA256
            && manifest.dtype == "f32_le"
            && manifest.backend == runtime.backend.key()
            && placement.strict
            && matches_device(runtime.backend, &placement.device)
            && manifest.tokens.len() == CORPUS_LINES
            && PROMPTS.lines().count() == CORPUS_LINES
            && manifest
                .tokens
                .iter()
                .all(|row| { (2..=128).contains(&row.len()) && row.iter().all(|&id| id >= 0) })
            && manifest.observation_count
                == manifest.tokens.iter().map(|t| t.len() - 1).sum::<usize>()
            && manifest.loss.is_finite()
            && manifest.loss >= 0.
            && manifest.gradients.len() == names.len(),
        "invalid corpus gradient metadata, placement, or observations"
    );
    let mut seen = BTreeSet::new();
    let mut entries = BTreeMap::new();
    for entry in manifest.gradients {
        let file = Path::new(&entry.file);
        ensure!(
            matches!(file.components().next(), Some(Component::Normal(n)) if n == file.as_os_str())
                && file.extension().is_some_and(|ext| ext == "f32")
                && seen.insert(entry.file.clone())
                && is_digest(&entry.sha256),
            "invalid corpus gradient file or SHA-256"
        );
        ensure!(
            names.binary_search(&entry.name.as_str()).is_ok() && !entries.contains_key(&entry.name),
            "unexpected or duplicate corpus gradient tensor"
        );
        entries.insert(entry.name.clone(), entry);
    }
    Ok(Gradients {
        out,
        entries,
        loss: manifest.loss,
        observation_count: manifest.observation_count,
    })
}

impl Gradients {
    pub(super) fn attribute(
        &self,
        name: &str,
        files: &mut [File; 2],
        reference: &Model,
        quantized: &Model,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f64> {
        let entry = &self.entries[name];
        ensure!(
            entry.shape == a.shape,
            "invalid shape for corpus gradient {name}"
        );
        // The shared dot uses reference - quantized. Report quantized - reference
        // so positive NLL attribution means that the quantized tensor raises loss.
        dot_payload(
            &self.out.join(&entry.file),
            &entry.sha256,
            files,
            reference,
            quantized,
            a,
            b,
        )
        .map(|value| -value)
        .with_context(|| format!("attributing corpus loss for {name}"))
    }
}
