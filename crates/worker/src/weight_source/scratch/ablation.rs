//! Directional edits act in output space of dense residual-writing matrices.
use super::{Header, MAX_EDIT_BYTES, Tensor, decode, editable_width, encode};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_TOTAL_BYTES: u64 = 256 << 20;
const WRITERS: [&str; 2] = ["self_attn.o_proj", "mlp.down_proj"];

#[derive(Clone, Copy, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Variant {
    Plain,
    Projected,
    NormPreserving,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    Flat,
    Gaussian,
    Triangle,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    pub shape: Shape,
    pub peak: usize,
    pub width: f64,
    pub min_weight: f64,
}

#[derive(Deserialize, Serialize)]
pub struct Settings {
    pub first_layer: usize,
    pub last_layer: usize,
    pub kernel: Kernel,
    pub variant: Variant,
    #[serde(default = "unit_scale", skip_serializing_if = "is_unit_scale")]
    pub alpha: f64,
}

pub fn unit_scale() -> f64 {
    1.0
}
pub fn is_unit_scale(value: &f64) -> bool {
    *value == 1.0
}

#[derive(Serialize)]
pub struct TraceRule {
    pub(crate) layer: usize,
    pub(crate) role: crate::trace::Role,
    pub(crate) positions: BTreeSet<usize>,
    op: &'static str,
    vector: Vec<f32>,
    scale: f64,
    keep_norm: bool,
}

impl TraceRule {
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

pub struct Plan {
    layers: Vec<(usize, Layer)>,
}

pub(super) struct Layer {
    direction: Vec<f64>,
    weight: f64,
    keep_norm: bool,
}

impl Plan {
    pub fn new<'a>(
        settings: &Settings,
        direction: &[f32],
        source_negative_mean: Option<&[f32]>,
        target_negative_mean: impl Fn(usize) -> Option<&'a [f32]>,
    ) -> Result<Self> {
        let Settings {
            first_layer,
            last_layer,
            kernel,
            variant,
            alpha,
        } = *settings;
        ensure!(
            first_layer <= last_layer && last_layer < 4096 && last_layer - first_layer < 64,
            "select an inclusive range of at most 64 layers from 0 to 4095"
        );
        ensure!(
            kernel.peak < 4096
                && kernel.width > 0.0
                && kernel.width <= 4096.0
                && (0.0..=1.0).contains(&kernel.min_weight)
                && alpha > 0.0
                && alpha <= 4.0,
            "invalid ablation kernel peak, width or minimum weight"
        );
        let mut direction = unit(direction)?;
        let source_axis = if variant != Variant::Plain {
            let axis = negative_axis(source_negative_mean, direction.len())?;
            subtract_axis(&mut direction, &axis);
            normalize(&mut direction)?;
            Some(axis)
        } else {
            None
        };
        let keep_norm = variant == Variant::NormPreserving;
        let mut layers = Vec::new();
        for layer in first_layer..=last_layer {
            // Handle the peak explicitly for positive subnormal widths.
            let t = if layer == kernel.peak {
                0.0
            } else {
                layer.abs_diff(kernel.peak) as f64 / kernel.width
            };
            let weight = match kernel.shape {
                Shape::Flat => 1.0,
                Shape::Gaussian => (-0.5 * t * t).exp(),
                Shape::Triangle => (1.0 - t).max(0.0),
            }
            .max(kernel.min_weight);
            if weight == 0.0 {
                continue;
            }
            let mut direction = direction.clone();
            if let Some(source_axis) = source_axis.as_ref().filter(|_| keep_norm) {
                // Gram-Schmidt preserves orthogonality to both mean axes.
                let mut axis = negative_axis(target_negative_mean(layer), direction.len())
                    .with_context(|| format!("target layer {layer}"))?;
                subtract_axis(&mut axis, source_axis);
                subtract_axis(&mut axis, source_axis);
                if normalize(&mut axis).is_ok() {
                    subtract_axis(&mut direction, &axis);
                    normalize(&mut direction).with_context(|| format!("target layer {layer}"))?;
                }
            }
            layers.push((
                layer,
                Layer {
                    direction,
                    weight: weight * alpha,
                    keep_norm,
                },
            ));
        }
        ensure!(!layers.is_empty(), "the ablation kernel selects no layers");
        Ok(Self { layers })
    }

    pub fn trace_rules(&self, positions: BTreeSet<usize>) -> Result<Vec<TraceRule>> {
        ensure!(
            !positions.is_empty() && positions.iter().all(|p| *p < 4096),
            "invalid screen positions"
        );
        Ok(self
            .layers
            .iter()
            .map(|(layer, plan)| TraceRule {
                layer: *layer,
                role: crate::trace::Role::LOut,
                positions: positions.clone(),
                op: "project_out",
                vector: plan.direction.iter().map(|&v| v as f32).collect(),
                scale: plan.weight,
                keep_norm: plan.keep_norm,
            })
            .collect())
    }

    pub(super) fn writers<'a>(
        &'a self,
        header: &'a Header,
    ) -> Result<BTreeMap<&'a str, &'a Layer>> {
        let mut writers = BTreeMap::new();
        let mut total = 0;
        for (number, layer) in &self.layers {
            for stem in WRITERS {
                let stem = format!("model.layers.{number}.{stem}");
                ensure!(
                    !header.entries.contains_key(&format!("{stem}.bias")),
                    "ablation does not support biased residual writer {stem}"
                );
                let name = format!("{stem}.weight");
                let (name, tensor) = header
                    .entries
                    .get_key_value(&name)
                    .with_context(|| format!("missing residual writer {name}"))?;
                let [rows, cols] = tensor.shape.as_slice() else {
                    anyhow::bail!("residual writer {name} must be a matrix")
                };
                ensure!(
                    *rows == layer.direction.len() as u64 && *cols > 0,
                    "residual writer {name} does not match direction width"
                );
                editable_width(&tensor.dtype)?;
                let bytes = tensor.data_offsets[1] - tensor.data_offsets[0];
                ensure!(
                    bytes <= MAX_EDIT_BYTES,
                    "residual writer {name} exceeds 32 MiB"
                );
                total += bytes;
                ensure!(
                    total <= MAX_TOTAL_BYTES,
                    "ablation exceeds 256 MiB of tensors"
                );
                writers.insert(name.as_str(), layer);
            }
        }
        Ok(writers)
    }
}

impl Layer {
    pub(super) fn apply(&self, tensor: &Tensor, bytes: &mut [u8]) -> Result<()> {
        let dtype = tensor.dtype.as_str();
        let width = editable_width(dtype)?;
        let cols = tensor.shape[1] as usize;
        let stride = width * cols;
        let mut projection = vec![0.0; cols];
        for (&d, row) in self.direction.iter().zip(bytes.chunks_exact(stride)) {
            for (p, value) in projection.iter_mut().zip(row.chunks_exact(width)) {
                let value = decode(dtype, value);
                ensure!(value.is_finite(), "non-finite input weight");
                *p += d * value;
            }
        }
        for p in &mut projection {
            *p *= self.weight;
        }
        let mut scales = Vec::new();
        if self.keep_norm {
            let mut norms = vec![(0.0, 0.0); cols];
            for (&d, row) in self.direction.iter().zip(bytes.chunks_exact(stride)) {
                for (((old_sq, new_sq), p), value) in norms
                    .iter_mut()
                    .zip(&projection)
                    .zip(row.chunks_exact(width))
                {
                    let old = decode(dtype, value);
                    *old_sq += old.powi(2);
                    *new_sq += (old - d * p).powi(2);
                }
            }
            for (old_sq, new_sq) in norms {
                ensure!(
                    old_sq == 0.0 || new_sq > old_sq * 1e-24,
                    "cannot preserve norm of a writing vector parallel to the ablation direction"
                );
                scales.push(if old_sq == 0.0 {
                    1.0
                } else {
                    (old_sq / new_sq).sqrt()
                });
            }
        }
        for (&d, row) in self.direction.iter().zip(bytes.chunks_exact_mut(stride)) {
            for (col, (p, value)) in projection
                .iter()
                .zip(row.chunks_exact_mut(width))
                .enumerate()
            {
                let edited = decode(dtype, value) - d * p;
                encode(
                    dtype,
                    edited * scales.get(col).copied().unwrap_or(1.0),
                    value,
                )?;
            }
        }
        Ok(())
    }
}

fn negative_axis(mean: Option<&[f32]>, width: usize) -> Result<Vec<f64>> {
    let mean = mean
        .context("fit the direction again with training negative means at the required layers")?;
    ensure!(
        mean.len() == width,
        "negative mean does not match direction width"
    );
    unit(mean).context("invalid training negative mean")
}

fn unit(values: &[f32]) -> Result<Vec<f64>> {
    ensure!(
        (1..=16384).contains(&values.len()),
        "invalid direction width"
    );
    let mut values = values.iter().copied().map(f64::from).collect::<Vec<_>>();
    // Directions and means can have different scales. Set a relative baseline.
    let max = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    ensure!(
        max.is_finite() && max > 0.0,
        "direction must be finite and nonzero"
    );
    for v in &mut values {
        *v /= max;
    }
    normalize(&mut values)?;
    Ok(values)
}

fn normalize(values: &mut [f64]) -> Result<()> {
    let norm = values.iter().map(|x| x * x).sum::<f64>().sqrt();
    ensure!(
        norm.is_finite() && norm > 1e-12,
        "ablation direction is degenerate"
    );
    for v in values {
        *v /= norm;
    }
    Ok(())
}

fn subtract_axis(values: &mut [f64], unit: &[f64]) {
    let dot: f64 = values.iter().zip(unit).map(|(v, u)| v * u).sum();
    for (v, u) in values.iter_mut().zip(unit) {
        *v -= dot * u;
    }
}
