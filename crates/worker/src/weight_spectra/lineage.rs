//! Fingerprint differences, not provenance records.
use super::{
    Input, math,
    source::{Checkpoint, float_width},
};
use crate::weight_source::scratch::decode;
use anyhow::{Result, ensure};
use nalgebra::DMatrix;
use serde_json::{Value, json};

pub const MAX_CANDIDATES: usize = 3;
const MAX_BYTES: u64 = 8 << 30;
const TILE: usize = 1 << 16;
const MAX_TENSORS_SHOWN: usize = 8;
const MAX_SPECTRA: usize = 3;
const RELATIVE_DELTA_LIMIT: f64 = 0.20;

pub fn lineage(
    target: Input<'_>,
    candidates: &[Input<'_>],
    mut check: impl FnMut() -> Result<()>,
) -> Result<Value> {
    ensure!(
        candidates.len() <= MAX_CANDIDATES,
        "Select at most three candidates"
    );
    let mut target = Checkpoint::open_bounded(target, MAX_BYTES, &mut check)?;
    let names: Vec<_> = target.entries.keys().cloned().collect();
    ensure!(!names.is_empty(), "Checkpoint has no tensors");
    ensure!(
        target
            .entries
            .values()
            .all(|t| float_width(&t.dtype).is_some()
                && (1..=4).contains(&t.shape.len())
                && !t.shape.contains(&0)),
        "Lineage needs floating-point tensors with one to four axes"
    );

    let mut sources = Vec::new();
    let mut rejected = Vec::new();
    for &spec in candidates {
        check()?;
        match Checkpoint::open_bounded(spec, MAX_BYTES, &mut check) {
            Ok(base)
                if base.entries.len() == names.len()
                    && target.entries.iter().all(|(name, t)| {
                        base.entries
                            .get(name)
                            .is_some_and(|b| b.shape == t.shape && b.dtype == t.dtype)
                    })
                    && base.receipt.format == target.receipt.format =>
            {
                sources.push((spec.0, base, Pair::default()));
            }
            Ok(_) => rejected.push(json!({
                "id":spec.0, "status":"incompatible_inventory"
            })),
            Err(_) => {
                check()?;
                rejected.push(json!({"id":spec.0, "status":"unreadable_checkpoint"}));
            }
        }
    }
    let mut target_raw = Vec::new();
    let mut target_values = Vec::new();
    let mut base_raw = Vec::new();
    let mut base_values = Vec::new();
    let mut tile_deltas = vec![Vec::new(); sources.len()];
    let mut gram = [[0.; MAX_CANDIDATES]; MAX_CANDIDATES];
    let mut target_energy = 0.;
    let mut total_count = 0;
    if !sources.is_empty() {
        for key in names {
            let t = &target.entries[&key];
            let shape: Vec<usize> = t.shape.iter().map(|&n| n as usize).collect();
            let dtype = t.dtype.clone();
            let count = shape.iter().product();
            total_count += count;
            let mut rows: Vec<_> = sources.iter().map(|_| Tensor::new(&key, &shape)).collect();
            for offset in (0..count).step_by(TILE) {
                check()?;
                let len = TILE.min(count - offset);
                read(
                    &mut target,
                    &key,
                    &dtype,
                    offset,
                    len,
                    &mut target_raw,
                    &mut target_values,
                )?;
                target_energy += target_values.iter().map(|v| v * v).sum::<f64>();
                for (((_, base, _), row), delta) in
                    sources.iter_mut().zip(&mut rows).zip(&mut tile_deltas)
                {
                    delta.clear();
                    base.read_float_tile(&key, offset, len, &mut base_raw)?;
                    if target_raw == base_raw {
                        continue;
                    }
                    decode_values(&dtype, &base_raw, &mut base_values)?;
                    for (i, (&a, &b)) in target_values.iter().zip(&base_values).enumerate() {
                        if a != b {
                            row.record(offset + i, a, b);
                        }
                        delta.push(a - b);
                    }
                }
                for i in 0..tile_deltas.len() {
                    for j in (i + 1)..tile_deltas.len() {
                        gram[i][j] += tile_deltas[i]
                            .iter()
                            .zip(&tile_deltas[j])
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                    }
                }
            }
            for ((_, _, pair), row) in sources.iter_mut().zip(rows) {
                pair.add(row);
            }
        }
    }
    for (i, (_, _, pair)) in sources.iter().enumerate() {
        gram[i][i] = pair.energy;
        for j in 0..i {
            gram[i][j] = gram[j][i];
        }
    }
    let ids: Vec<_> = sources.iter().map(|(id, _, _)| *id).collect();
    let fits: Vec<_> = (0..ids.len())
        .filter_map(|i| {
            (ids.len() == 3 && relative_delta(gram[i][i], target_energy) < RELATIVE_DELTA_LIMIT)
                .then(|| fit_merge(&gram, &ids, i))
                .flatten()
                .map(|fit| (i, fit))
        })
        .collect();
    // Affine decompositions can fit from multiple anchors. Without a unique
    // anchor, association is observable but the original base is not.
    let unique_merge = fits.len() == 1;
    let ambiguous_merge = fits.len() > 1;
    let identical = sources.iter().any(|(_, _, pair)| pair.changed == 0);
    let mut results = Vec::new();
    for (i, (id, mut base, pair)) in sources.into_iter().enumerate() {
        let relative = relative_delta(pair.energy, target_energy);
        let close = relative < RELATIVE_DELTA_LIMIT && pair.energy > 0.;
        let (mut evidence, rank_one_energy, low_rank_energy) = pair.evidence(
            &mut target,
            &mut base,
            total_count,
            relative,
            close,
            &mut check,
        )?;
        let merge = fits.iter().find(|(j, _)| i == *j).map(|(_, fit)| fit);
        evidence["merge"] = json!(merge);
        let kind = if merge.is_some() {
            Some("merge")
        } else if !close {
            None
        } else if pair.zero_energy >= 0.98 * pair.energy
            && pair.zero_count as f64 >= 0.98 * pair.changed as f64
        {
            Some("pruning")
        } else if rank_one_energy >= 0.98 {
            Some("rank_one_edit")
        } else if low_rank_energy >= 0.95 {
            Some("low_rank_edit")
        } else if pair.changed as f64 / total_count as f64 <= 0.001 {
            Some("sparse_edit")
        } else {
            Some("fine_tune")
        };
        let status = match kind {
            Some(_) => "related",
            None if relative == 0. => "identical_weights",
            None => "no_match",
        };
        let confidence = if let Some(fit) = merge {
            0.99 - 2. * fit["relative_residual"].as_f64().unwrap()
        } else if kind.is_some() {
            (1. - relative / RELATIVE_DELTA_LIMIT).clamp(0., 0.95)
        } else {
            0.
        };
        base.verify_unchanged()?;
        results.push((
            merge.is_some(),
            relative,
            confidence,
            json!({
                "id":id, "checkpoint":base.receipt, "status":status, "kind":kind,
                "confidence":confidence, "evidence":evidence
            }),
        ));
    }
    check()?;
    target.verify_unchanged()?;
    results.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.2.total_cmp(&a.2).then(a.1.total_cmp(&b.1)))
    });
    let best = results.first().filter(|(_, distance, score, _)| {
        !ambiguous_merge
            && !identical
            && *score > 0.
            && (unique_merge
                || results
                    .get(1)
                    .is_none_or(|second| second.2 == 0. || second.1 > distance * 1.05))
    });
    Ok(json!({
        "schema_version":1, "qualification":"not release-qualified", "target":target.receipt,
        "most_likely":best.map(|(_, _, _, b)| json!({
            "base_id":b["id"], "kind":b["kind"], "confidence":b["confidence"],
        })),
        "candidates":results.into_iter().map(|(_, _, _, r)| r).chain(rejected).collect::<Vec<_>>(),
    }))
}

fn fit_merge(gram: &[[f64; 3]; 3], ids: &[&str], base: usize) -> Option<Value> {
    let [a, b]: [usize; 2] = (0..3)
        .filter(|i| *i != base)
        .collect::<Vec<_>>()
        .try_into()
        .ok()?;
    // Let d_i = target - i. Each parent direction relative to base is d_base - d_i.
    let energy = gram[base][base];
    let rhs = |i: usize| energy - gram[base][i];
    let dot = |i: usize, j: usize| rhs(i) + rhs(j) - energy + gram[i][j];
    let (aa, ab, bb) = (dot(a, a), dot(a, b), dot(b, b));
    let det = aa * bb - ab * ab;
    if energy <= 0. || aa <= 0. || bb <= 0. || det <= 0.01 * aa * bb {
        return None;
    }
    let (p, q) = (rhs(a), rhs(b));
    let (x, y) = ((bb * p - ab * q) / det, (aa * q - ab * p) / det);
    let residual_sq = (energy - x * p - y * q).max(0.);
    let residual = (residual_sq / energy).sqrt();
    if residual > 0.05
        || !(0.02..=4.).contains(&x)
        || !(0.02..=4.).contains(&y)
        || x * x * aa < 0.01 * energy
        || y * y * bb < 0.01 * energy
    {
        return None;
    }
    Some(json!({
        "parent_ids":[ids[a],ids[b]], "weights":[x,y],
        "parent_delta_cosine":ab / (aa * bb).sqrt(), "relative_residual":residual,
    }))
}

fn relative_delta(delta_energy: f64, target_energy: f64) -> f64 {
    // Bound the ratio even for an all-zero target.
    (delta_energy / target_energy.max(f64::MIN_POSITIVE))
        .sqrt()
        .min(f64::MAX)
}

fn decode_values(dtype: &str, raw: &[u8], values: &mut Vec<f64>) -> Result<()> {
    values.clear();
    values.extend(
        raw.chunks_exact(float_width(dtype).unwrap())
            .map(|v| decode(dtype, v)),
    );
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "Lineage requires finite weights"
    );
    Ok(())
}

fn read(
    checkpoint: &mut Checkpoint,
    key: &str,
    dtype: &str,
    offset: usize,
    len: usize,
    raw: &mut Vec<u8>,
    values: &mut Vec<f64>,
) -> Result<()> {
    checkpoint.read_float_tile(key, offset, len, raw)?;
    decode_values(dtype, raw, values)
}

struct Tensor {
    name: String,
    shape: Vec<usize>,
    changed: usize,
    energy: f64,
    zero_count: usize,
    zero_energy: f64,
    start: Vec<usize>,
    end: Vec<usize>,
}

impl Tensor {
    fn new(name: &str, shape: &[usize]) -> Self {
        Self {
            name: name.into(),
            shape: shape.to_owned(),
            changed: 0,
            energy: 0.,
            zero_count: 0,
            zero_energy: 0.,
            start: shape.to_owned(),
            end: vec![0; shape.len()],
        }
    }

    fn record(&mut self, mut index: usize, target: f64, base: f64) {
        self.changed += 1;
        let energy = (target - base).powi(2);
        self.energy += energy;
        if target == 0. {
            self.zero_count += 1;
            self.zero_energy += energy;
        }
        for i in (0..self.shape.len()).rev() {
            let width = self.shape[i];
            let quotient = index / width;
            if self.start[i] != 0 || self.end[i] != width {
                let coordinate = index - quotient * width;
                self.start[i] = self.start[i].min(coordinate);
                self.end[i] = self.end[i].max(coordinate + 1);
            }
            index = quotient;
        }
    }

    fn spectrum(
        &self,
        target: &mut Checkpoint,
        base: &mut Checkpoint,
        check: &mut impl FnMut() -> Result<()>,
    ) -> Result<Option<super::Metrics>> {
        let (rows, cols) = match self.shape.as_slice() {
            [cols] => (1, *cols),
            [rows, cols] => (*rows, *cols),
            _ => return Ok(None),
        };
        if rows * cols > 16 << 20 || rows.max(cols) > math::MAX_AXIS {
            return Ok(None);
        }
        let dtype = target.entries[&self.name].dtype.clone();
        let (mut a, mut b, mut av, mut bv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        math::measure(rows, cols, |row, count| {
            check()?;
            let len = count * cols;
            read(target, &self.name, &dtype, row * cols, len, &mut a, &mut av)?;
            read(base, &self.name, &dtype, row * cols, len, &mut b, &mut bv)?;
            let deltas: Vec<_> = av.iter().zip(&bv).map(|(a, b)| a - b).collect();
            Ok(DMatrix::from_row_slice(count, cols, &deltas))
        })
        .map(Some)
    }
}

#[derive(Default)]
struct Pair {
    changed: usize,
    changed_tensors: usize,
    energy: f64,
    zero_count: usize,
    zero_energy: f64,
    top: Vec<Tensor>,
}

impl Pair {
    fn add(&mut self, row: Tensor) {
        if row.changed == 0 {
            return;
        }
        self.changed += row.changed;
        self.changed_tensors += 1;
        self.energy += row.energy;
        self.zero_count += row.zero_count;
        self.zero_energy += row.zero_energy;
        self.top.push(row);
        self.top.sort_by(|a, b| b.energy.total_cmp(&a.energy));
        self.top.truncate(MAX_TENSORS_SHOWN);
    }

    fn evidence(
        &self,
        target: &mut Checkpoint,
        base: &mut Checkpoint,
        total_count: usize,
        relative: f64,
        close: bool,
        check: &mut impl FnMut() -> Result<()>,
    ) -> Result<(Value, f64, f64)> {
        let mut rank_one = 0.;
        let mut low_rank = 0.;
        let mut rows = Vec::new();
        for (i, row) in self.top.iter().enumerate() {
            let spectrum = if close && i < MAX_SPECTRA {
                row.spectrum(target, base, check)?
            } else {
                None
            };
            if row.shape.len() == 2
                && row.shape.iter().all(|&n| n >= 2)
                && let Some(spectrum) = &spectrum
                && spectrum.stable_rank > 0.
            {
                rank_one += row.energy / spectrum.stable_rank;
                // Use fewer than half the possible singular modes; small or
                // narrow matrices are not automatically "low-rank".
                let k = 8.min(row.shape[0].min(row.shape[1]) / 2);
                low_rank += spectrum
                    .singular_values
                    .iter()
                    .take(k)
                    .map(|s| s * s)
                    .sum::<f64>();
            }
            rows.push(json!({
                "name":row.name, "shape":row.shape, "changed_count":row.changed,
                "changed_fraction":row.changed as f64 / row.shape.iter().product::<usize>() as f64,
                "delta_energy":row.energy, "bounds":{"start":row.start,"end_exclusive":row.end},
                "delta_spectrum":spectrum,
            }));
        }
        let rank_one = if self.energy > 0. {
            (rank_one / self.energy).min(1.)
        } else {
            0.
        };
        let low_rank = if self.energy > 0. {
            (low_rank / self.energy).min(1.)
        } else {
            0.
        };
        Ok((
            json!({
                "checked_count":total_count, "changed_count":self.changed,
                "changed_fraction":self.changed as f64 / total_count as f64,
                "changed_tensors":self.changed_tensors, "relative_delta":relative,
                "changed_to_zero_fraction":if self.changed > 0 {self.zero_count as f64 / self.changed as f64} else {0.},
                "rank_one_energy_fraction":rank_one, "low_rank_energy_fraction":low_rank, "tensors":rows,
            }),
            rank_one,
            low_rank,
        ))
    }
}
