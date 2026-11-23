//! Bounded, read-only spectra for local checkpoint files.
use anyhow::{Context, Result, ensure};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};

mod math;
mod source;
use source::Checkpoint;

pub const JOB: &str = "weight_spectra";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Filter {
    pub layer: Option<u32>,
    pub tensor: Option<String>,
}

pub type Input<'a> = (&'a str, &'a Path, &'a str, Option<&'a str>);

pub fn analyze(
    input: Input<'_>,
    compare: Option<Input<'_>>,
    filter: &Filter,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<Value> {
    let mut a = Checkpoint::open(input, &mut checkpoint)?;
    let mut b = compare
        .map(|spec| Checkpoint::open(spec, &mut checkpoint))
        .transpose()?;
    let keys: BTreeSet<_> = a
        .entries
        .keys()
        .chain(b.iter().flat_map(|side| side.entries.keys()))
        .filter(|key| {
            let name = key.as_str();
            filter
                .layer
                .is_none_or(|layer| Some(layer) == layer_of(name))
                && filter.tensor.as_deref().is_none_or(|selected| {
                    selected == name
                        || a.entries.get(name).is_some_and(|t| selected == t.name)
                        || b.as_ref()
                            .and_then(|b| b.entries.get(name))
                            .is_some_and(|t| selected == t.name)
                })
        })
        .cloned()
        .collect();
    ensure!(!keys.is_empty(), "No tensors match the filter");
    ensure!(
        keys.len() <= 512,
        "Select one layer or tensor (at most 512 tensors)"
    );
    let mut rows = Vec::with_capacity(keys.len());
    for key in keys {
        checkpoint()?;
        let left = a.measure(&key, &mut checkpoint)?;
        let right = b
            .as_mut()
            .map(|b| b.measure(&key, &mut checkpoint))
            .transpose()?
            .flatten();
        let mut difference = Value::Null;
        let status = match (&left, &right) {
            (Some(l), Some(r)) if l.shape != r.shape => "shape_mismatch",
            (Some(l), Some(r)) if l.metrics.is_some() && r.metrics.is_some() => {
                let l = l.metrics.as_ref().unwrap();
                let r = r.metrics.as_ref().unwrap();
                difference = json!({
                    "sigma_max": l.singular_values[0] - r.singular_values[0],
                    "effective_rank": l.effective_rank - r.effective_rank,
                    "stable_rank": l.stable_rank - r.stable_rank,
                    "alpha": l.alpha.zip(r.alpha).map(|(l,r)| l - r)
                });
                "compared"
            }
            (Some(l), None) if b.is_none() && l.metrics.is_some() => "measured",
            (None, _) | (_, None) if b.is_some() => "missing",
            _ => "unsupported",
        };
        rows.push(json!({
            "name":key, "layer":layer_of(&key), "source":left, "reference":right,
            "delta":difference, "status":status
        }));
    }
    checkpoint()?;
    a.verify_unchanged()?;
    if let Some(b) = &b {
        b.verify_unchanged()?;
    }
    Ok(json!({
        "kind":JOB, "schema_version":1, "qualification":"not release-qualified",
        "source":a.receipt, "reference":b.map(|b| b.receipt), "tensors":rows
    }))
}

fn layer_of(name: &str) -> Option<u32> {
    name.strip_prefix("model.layers.")
        .or_else(|| name.strip_prefix("blk."))
        .or_else(|| name.strip_prefix("layers."))?
        .split('.')
        .next()?
        .parse()
        .ok()
}

#[derive(Serialize)]
struct Measurement {
    dtype: String,
    shape: Vec<u64>,
    #[serde(flatten)]
    metrics: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
}

#[derive(Serialize)]
struct Metrics {
    method: &'static str,
    singular_values: Vec<f64>,
    effective_rank: f64,
    stable_rank: f64,
    captured_energy_fraction: Option<f64>,
    alpha: Option<f64>,
    alpha_tail_count: Option<usize>,
    alpha_xmin: Option<f64>,
    alpha_ks: Option<f64>,
}
