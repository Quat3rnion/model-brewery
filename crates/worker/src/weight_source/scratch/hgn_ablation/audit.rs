//! Reopen both checkpoints. Reuse the native decoder, not the editor's metrics.
use super::*;
use crate::hgn_checks::{AuditStatus, CONTROLS, SEED, TensorAudit, agrees};

fn control_axes(width: usize) -> [Vec<f64>; CONTROLS] {
    let mut seed = SEED;
    std::array::from_fn(|_| {
        let mut axis: Vec<_> = (0..width)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                f64::from(seed >> 8) / f64::from(1 << 23) - 1.
            })
            .collect();
        let norm = axis.iter().map(|v| v * v).sum::<f64>().sqrt();
        axis.iter_mut().for_each(|v| *v /= norm);
        axis
    })
}

fn component_sum(values: &[f32], axis: &[f64], cols: usize) -> f64 {
    let mut projected = vec![0.; cols];
    for (row, &r) in values.chunks_exact(cols).zip(axis) {
        for (acc, &v) in projected.iter_mut().zip(row) {
            *acc += f64::from(v) * r;
        }
    }
    projected.into_iter().map(f64::abs).sum()
}

struct Stored {
    blocks: Blocks,
    codebook: [f32; 16],
    su: Vec<f32>,
    sv: Vec<f32>,
}

impl Stored {
    fn read(reader: &mut Reader, w: &Writer) -> Result<Self> {
        let enc = w.entry.encoding;
        let mut codebook = [0.; 16];
        if enc == Encoding::Q4Cp {
            let raw = reader.read_at(w.index, 0, 64)?;
            for (v, b) in codebook.iter_mut().zip(raw.chunks_exact(4)) {
                *v = f32::from_le_bytes(b.try_into()?);
            }
        }
        let root = w.entry.name.strip_suffix(".weight").unwrap();
        let groups = if enc.ht_bits().is_some() {
            w.experts
        } else {
            1
        };
        let (su, sv) = if enc.ht_bits().is_some() || enc == Encoding::I4r {
            (
                side_plane(reader, &format!("{root}.suh"), groups * w.cols)?,
                side_plane(reader, &format!("{root}.svh"), groups * w.rows)?,
            )
        } else {
            (vec![], vec![])
        };
        Ok(Self {
            blocks: Blocks::new(w)?,
            codebook,
            su,
            sv,
        })
    }

    fn decode(&self, reader: &mut Reader, w: &Writer, expert: usize) -> Result<Vec<f32>> {
        let group = if w.entry.encoding.ht_bits().is_some() {
            expert
        } else {
            0
        };
        let meta = QuantMeta {
            su: self
                .su
                .get(group * w.cols..(group + 1) * w.cols)
                .unwrap_or_default(),
            sv: self
                .sv
                .get(group * w.rows..(group + 1) * w.rows)
                .unwrap_or_default(),
            codebook: &self.codebook,
            ..QuantMeta::default()
        };
        let values = decode(
            w.entry.encoding,
            &self.blocks.read(reader, w.index, expert)?,
            w.rows,
            w.cols,
            &meta,
        )?;
        ensure!(
            values.len() == w.rows * w.cols && values.iter().all(|v| v.is_finite()),
            "invalid decoded HGN tensor"
        );
        Ok(values)
    }
}

pub fn audit(
    parent: &Path,
    child: &Path,
    parent_sha: &str,
    child_sha: &str,
    layers: RangeInclusive<usize>,
    feedback: &Report,
) -> Result<BTreeMap<String, TensorAudit>> {
    let r = &feedback.direction;
    ensure!(
        feedback.schema == SCHEMA
            && feedback.encoder_sha256 == ENCODER_SHA256
            && feedback.parent_sha256 == parent_sha
            && feedback.revision_sha256 == child_sha
            && !r.is_empty()
            && agrees(r.iter().map(|v| v * v).sum::<f64>(), 1.),
        "invalid HGN feedback binding"
    );
    for (path, hash) in [(parent, parent_sha), (child, child_sha)] {
        ensure!(
            file_sha256(&mut open_regular(path)?)? == hash,
            "stored HGN SHA-256 mismatch"
        );
    }
    let (mut parent, mut child) = (Reader::open(parent)?, Reader::open(child)?);
    ensure!(
        parent.version == 2
            && child.version == 2
            && parent.layout.entries.len() == child.layout.entries.len()
            && parent
                .layout
                .entries
                .iter()
                .zip(&child.layout.entries)
                .all(|(a, b)| {
                    a.name == b.name
                        && a.shape == b.shape
                        && a.encoding == b.encoding
                        && a.size == b.size
                }),
        "stored HGN layout differs from its parent"
    );
    let selected = writers(&parent, layers, r.len())?;
    ensure!(
        !selected.is_empty() && selected.len() == feedback.tensors.len(),
        "HGN feedback does not cover the selected writers"
    );
    let axes = control_axes(r.len());
    let mut result = BTreeMap::new();
    for w in selected {
        let name = &w.entry.name;
        let expected = feedback
            .tensors
            .get(name)
            .context("missing HGN feedback tensor")?;
        let child_fingerprint = child.fingerprint(w.index)?;
        ensure!(
            expected.bytes == w.entry.size
                && expected.shape == w.entry.shape
                && expected.store == w.entry.encoding.code()
                && expected.variant == w.entry.encoding.variant()
                && child_fingerprint.sha256 == expected.payload_sha256
                && child_fingerprint.xor32 == child.layout.entries[w.index].checksum
                && parent.fingerprint(w.index)?.xor32 == w.entry.checksum,
            "stored HGN tensor digest or layout mismatch: {name}"
        );
        let (a, b) = (
            Stored::read(&mut parent, &w)?,
            Stored::read(&mut child, &w)?,
        );
        let (mut before, mut after, mut diff_sq, mut norm_sq) = (0., 0., 0., 0.);
        let mut worst = 0.0_f64;
        let mut controls_a = [0.; CONTROLS];
        let mut controls_b = [0.; CONTROLS];
        for e in 0..w.experts {
            let a = a.decode(&mut parent, &w, e)?;
            let b = b.decode(&mut child, &w, e)?;
            let a_component = component_sum(&a, r, w.cols);
            let b_component = component_sum(&b, r, w.cols);
            before += a_component;
            after += b_component;
            worst = worst.max(ratio(b_component, a_component));
            for (i, axis) in axes.iter().enumerate() {
                controls_a[i] += component_sum(&a, axis, w.cols);
                controls_b[i] += component_sum(&b, axis, w.cols);
            }
            for (&x, &y) in a.iter().zip(&b) {
                let x = f64::from(x);
                norm_sq += x * x;
                diff_sq += (x - f64::from(y)).powi(2);
            }
        }
        let leftover = ratio(after, before);
        let rel_error = ratio(diff_sq.sqrt(), norm_sq.sqrt());
        let max_expert_leftover = (w.part == Part::Experts).then_some(worst);
        let parent_mean_abs_component = before / (w.experts * w.cols) as f64;
        let control_shifts =
            std::array::from_fn(|i| ratio((controls_b[i] - controls_a[i]).abs(), controls_a[i]));
        ensure!(
            agrees(leftover, expected.leftover_after)
                && agrees(rel_error, expected.rel_error)
                && agrees(
                    parent_mean_abs_component,
                    expected.parent_mean_abs_component
                )
                && control_shifts.iter().all(|v| v.is_finite())
                && worst.is_finite(),
            "decoded HGN audit disagrees with feedback: {name}"
        );
        let mut row = TensorAudit {
            leftover,
            max_expert_leftover,
            parent_mean_abs_component,
            rel_error,
            control_shifts,
            status: AuditStatus::NoComponent,
        };
        row.status = row.measured_status();
        result.insert(name.clone(), row);
    }
    Ok(result)
}
