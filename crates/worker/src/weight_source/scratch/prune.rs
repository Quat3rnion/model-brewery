//! Shape-preserving, deterministic dense pruning on a pinned scratch copy.
use super::{Header, MAX_EDIT_BYTES, decode, editable_width, native_tensor_name};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

#[derive(Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum Plan {
    Unstructured {
        basis_points: u16,
        tensors: Vec<String>,
    },
    MlpNeurons {
        basis_points: u16,
        layer: usize,
    },
    AttentionHeads {
        basis_points: u16,
        layer: usize,
        heads: usize,
    },
    WandaMlp {
        basis_points: u16,
        layer: usize,
        trace: Trace,
        rms_eps: f64,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub id: String,
    pub prompt: usize,
    pub start: usize,
    pub end: usize,
}

pub(super) enum Mask {
    Elements(Vec<usize>),
    Columns {
        cols: usize,
        group_width: usize,
        groups: Vec<usize>,
    },
}

impl Mask {
    pub(super) fn apply(&self, data: &mut [u8], width: usize) {
        match self {
            Self::Elements(indices) => {
                for i in indices {
                    data[i * width..(i + 1) * width].fill(0);
                }
            }
            Self::Columns {
                cols,
                group_width,
                groups,
            } => {
                for row in data.chunks_exact_mut(cols * width) {
                    for group in groups {
                        let start = group * group_width * width;
                        row[start..start + group_width * width].fill(0);
                    }
                }
            }
        }
    }
}

impl Plan {
    pub(super) fn masks(
        &self,
        file: &mut File,
        header: &Header,
        activations: Option<&[Vec<f32>]>,
    ) -> Result<BTreeMap<String, Mask>> {
        let (targets, points) = match self {
            Self::Unstructured {
                basis_points,
                tensors,
            } => (tensors.clone(), *basis_points),
            Self::MlpNeurons {
                basis_points,
                layer,
            }
            | Self::WandaMlp {
                basis_points,
                layer,
                ..
            } => (
                vec![layer_tensor(*layer, "mlp.down_proj.weight")?],
                *basis_points,
            ),
            Self::AttentionHeads {
                basis_points,
                layer,
                ..
            } => (
                vec![layer_tensor(*layer, "self_attn.o_proj.weight")?],
                *basis_points,
            ),
        };
        ensure!(
            (1..10000).contains(&points),
            "basis_points must be 1 to 9999"
        );
        ensure!(
            (1..=32).contains(&targets.len()),
            "select 1 to 32 pruning tensors"
        );
        let mut budget = MAX_EDIT_BYTES;
        for name in &targets {
            ensure!(native_tensor_name(name), "invalid pruning tensor name");
            let tensor = header.entries.get(name).context("unknown pruning tensor")?;
            ensure!(
                matches!(tensor.shape.as_slice(), [rows, cols] if *rows > 0 && *cols > 0),
                "pruning requires nonempty matrices"
            );
            let [start, end] = tensor.data_offsets;
            budget = budget
                .checked_sub(end - start)
                .context("edited tensors exceed scratch limit")?;
        }
        let mut result = BTreeMap::new();
        for name in targets {
            let matrix = Weights::read(file, header, &name)?;
            let (_, cols) = matrix.dims()?;
            let mask = match self {
                Self::Unstructured { .. } => {
                    // Round only the f32 absolute weight, not the ranking; all
                    // supported stored weights are exactly representable as f32.
                    let scores = matrix.values()?.map(|x| x.abs() as f32).collect::<Vec<_>>();
                    Mask::Elements(lowest(&scores, points, f32::total_cmp)?)
                }
                Self::WandaMlp { layer, rms_eps, .. } => {
                    let (hidden, neurons) = matrix.dims()?;
                    let scores = wanda(
                        file,
                        header,
                        *layer,
                        hidden,
                        neurons,
                        *rms_eps,
                        activations.context("Wanda requires a trace")?,
                    )?;
                    Mask::Columns {
                        cols,
                        group_width: 1,
                        groups: lowest(&scores, points, f64::total_cmp)?,
                    }
                }
                Self::MlpNeurons { .. } | Self::AttentionHeads { .. } => {
                    let groups = if let Self::AttentionHeads { heads, .. } = self {
                        ensure!(
                            *heads > 1 && cols.is_multiple_of(*heads) && matrix.dims()?.0 == cols,
                            "invalid Qwen2 attention projection or query head count"
                        );
                        *heads
                    } else {
                        cols
                    };
                    let group_width = cols / groups;
                    let mut scores = vec![0.; groups];
                    for (index, value) in matrix.values()?.enumerate() {
                        scores[(index % cols) / group_width] += value.abs();
                    }
                    Mask::Columns {
                        cols,
                        group_width,
                        groups: lowest(&scores, points, f64::total_cmp)?,
                    }
                }
            };
            ensure!(
                result.insert(name, mask).is_none(),
                "duplicate pruning tensor"
            );
        }
        Ok(result)
    }
}

fn layer_tensor(layer: usize, suffix: &str) -> Result<String> {
    ensure!(layer < 4096, "invalid Qwen2 layer");
    Ok(format!("model.layers.{layer}.{suffix}"))
}

fn lowest<T>(
    values: &[T],
    points: u16,
    cmp: impl Fn(&T, &T) -> std::cmp::Ordering,
) -> Result<Vec<usize>> {
    let count = values.len() * usize::from(points) / 10000;
    ensure!(count > 0, "pruning fraction selects no elements or groups");
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.select_nth_unstable_by(count, |&a, &b| cmp(&values[a], &values[b]).then(a.cmp(&b)));
    Ok(indices[..count].to_vec())
}

struct Weights<'a> {
    dtype: &'a str,
    shape: &'a [u64],
    width: usize,
    bytes: Vec<u8>,
}

impl<'a> Weights<'a> {
    fn read(file: &mut File, header: &'a Header, name: &str) -> Result<Self> {
        let t = header
            .entries
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let width = editable_width(&t.dtype)?;
        let [start, end] = t.data_offsets;
        ensure!(
            end - start <= MAX_EDIT_BYTES,
            "pruning tensor exceeds scratch limit"
        );
        let mut bytes = vec![0; (end - start) as usize];
        file.seek(SeekFrom::Start(header.payload_start + start))?;
        file.read_exact(&mut bytes)?;
        Ok(Self {
            dtype: &t.dtype,
            shape: &t.shape,
            width,
            bytes,
        })
    }

    fn dims(&self) -> Result<(usize, usize)> {
        match self.shape {
            &[rows, cols] if rows > 0 && cols > 0 => Ok((rows as usize, cols as usize)),
            _ => anyhow::bail!("expected a nonempty pruning matrix"),
        }
    }

    fn values(&self) -> Result<impl Iterator<Item = f64> + '_> {
        let values = self
            .bytes
            .chunks_exact(self.width)
            .map(|bytes| decode(self.dtype, bytes));
        ensure!(
            values.clone().all(f64::is_finite),
            "non-finite pruning weight"
        );
        Ok(values)
    }
}

fn wanda(
    file: &mut File,
    header: &Header,
    layer: usize,
    hidden: usize,
    neurons: usize,
    eps: f64,
    rows: &[Vec<f32>],
) -> Result<Vec<f64>> {
    ensure!(
        (1..=64).contains(&rows.len()) && eps > 0. && eps <= 1.,
        "invalid Wanda trace or RMS epsilon"
    );
    let norm = Weights::read(
        file,
        header,
        &layer_tensor(layer, "post_attention_layernorm.weight")?,
    )?;
    ensure!(
        norm.shape == [hidden as u64],
        "Wanda RMSNorm shape mismatch"
    );
    let gain: Vec<_> = norm.values()?.collect();
    let mut sums = vec![0.; hidden];
    for row in rows {
        ensure!(
            row.len() == hidden && row.iter().all(|x| x.is_finite()),
            "invalid ffn_inp row"
        );
        let rms =
            (row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / hidden as f64 + eps).sqrt();
        for ((sum, &x), &g) in sums.iter_mut().zip(row).zip(&gain) {
            *sum += (f64::from(x) / rms * g).powi(2);
        }
    }
    let norms: Vec<_> = sums.into_iter().map(f64::sqrt).collect();
    let mut scores = vec![0.; neurons];
    for suffix in ["mlp.gate_proj.weight", "mlp.up_proj.weight"] {
        let w = Weights::read(file, header, &layer_tensor(layer, suffix)?)?;
        ensure!(
            w.dims()? == (neurons, hidden),
            "Wanda projection shape mismatch"
        );
        for (i, weight) in w.values()?.enumerate() {
            scores[i / hidden] += weight.abs() * norms[i % hidden];
        }
    }
    ensure!(
        scores.iter().any(|&x| x > 0.) && scores.iter().all(|x| x.is_finite()),
        "invalid or all-zero Wanda saliency"
    );
    Ok(scores)
}
