use super::*;

// Peer dense/shared and expert-down writers have 640 input columns and 2560
// output rows. Use that shape for row codecs and one native 128-row HT band
// for the expensive full-gain sweep. All data is synthetic.
const COLS: usize = 640;
const MAX_PASSES: usize = 32;
// Balanced coefficients cancel the common rows. The first update to the
// single discrepant row is 4/10 of one code step, below the midpoint.
const DEAD_BAND_AXIS: [f64; 7] = [2., 1., 1., -1., -1., -1., -1.];

fn sign(i: usize, j: usize) -> f64 {
    if (i & j).count_ones() % 2 == 0 {
        1.
    } else {
        -1.
    }
}

fn hadamard(values: &mut [f64]) {
    for stride in [1, 2, 4, 8, 16, 32, 64] {
        for chunk in values.chunks_exact_mut(2 * stride) {
            let (a, b) = chunk.split_at_mut(stride);
            for (a, b) in a.iter_mut().zip(b) {
                (*a, *b) = (*a + *b, *a - *b);
            }
        }
    }
}

fn undo_i4r_rotation(values: &[f32], su: &[f32], sv: &[f32]) -> Vec<f32> {
    let mut unrotated = vec![0.; values.len()];
    for rb in (0..sv.len()).step_by(128) {
        for cb in (0..COLS).step_by(128) {
            let mut block = [[0.; 128]; 128];
            for (i, row) in block.iter_mut().enumerate() {
                for (j, w) in row.iter_mut().enumerate() {
                    *w = f64::from(values[(rb + i) * COLS + cb + j]);
                }
                hadamard(row);
            }
            for j in 0..128 {
                let mut col = std::array::from_fn::<_, 128, _>(|i| block[i][j]);
                hadamard(&mut col);
                for (i, x) in col.into_iter().enumerate() {
                    unrotated[(rb + i) * COLS + cb + j] =
                        (x / 128.) as f32 * sv[rb + i] * su[cb + j];
                }
            }
        }
    }
    unrotated
}

struct Random(u32);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
}

fn row_store(enc: Encoding, rows: usize, meta: &QuantMeta<'_>) -> EncodedBlock {
    let mut rng = Random(238);
    let mut values = vec![0.; rows * COLS];
    let (group, scale) = match enc {
        Encoding::I4r => (128, 1. / 256.),
        Encoding::Q4Cp => (32, 1. / 8.),
        Encoding::Q8G64 => (64, 1. / 4096.),
        _ => unreachable!(),
    };
    let cb = meta.codebook;
    let level = |n: i32| match enc {
        Encoding::Q4Cp => cb[(n + 8) as usize],
        _ => n as f32,
    };
    for group_values in values.chunks_exact_mut(group) {
        for v in group_values.iter_mut() {
            *v = level(if enc == Encoding::Q8G64 {
                (rng.next() % 191) as i32 + 32
            } else {
                (rng.next() % 13) as i32 - 6
            }) * scale;
        }
        // Keep group scale (and Q8 bias) fixed under the small projection.
        group_values[0] = if enc == Encoding::Q8G64 {
            0.
        } else {
            level(-8) * scale
        };
        if enc == Encoding::Q8G64 {
            group_values[group - 1] = 255. * scale;
        }
    }
    let template = values[..COLS].to_vec();
    for (i, row) in values.chunks_exact_mut(COLS).take(7).enumerate() {
        row.copy_from_slice(&template);
        for j in (1..COLS).step_by(if enc == Encoding::Q4Cp { 32 } else { 16 }) {
            row[j] = match enc {
                Encoding::Q4Cp => level(if i == 0 { 0 } else { -1 }),
                Encoding::Q8G64 => {
                    if i == 0 {
                        129.
                    } else {
                        128.
                    }
                }
                _ => {
                    if i == 0 {
                        1.
                    } else {
                        0.
                    }
                }
            } * scale;
        }
    }
    if enc == Encoding::I4r {
        values = undo_i4r_rotation(&values, meta.su, meta.sv);
    }
    encode_rows(enc, &values, rows, COLS, meta).unwrap()
}

fn ht_stock() -> EncodedBlock {
    // Fixed cyclic HT4 tile, selected once on the pinned encoder. A one-symbol
    // difference in its paired tile supplies a small nonzero component.
    const TILE: &[u8; 256] = b"d12a402c560b077b1f7c2b5dcbafcd1e2b323f0d919d33ca2860ca68a369b159a9d2e8ed22022ad19067da39b81a6f11b392b4418d4eca909f459ec7f634ec53d6eb940aa5d73334aac59dbe2815520f05a6e6cc665040cf6a181d630258e5ade236b9c31dd32c8ea5149c3d994ae2f386d0674a87b115be215ff76d9bc64bfb";
    let nibble = |b: u8| (b as char).to_digit(16).unwrap();
    let mut rng = Random(775);
    let mut codes = Vec::new();
    for col_tile in 0..COLS / 16 {
        for row_tile in 0..8 {
            for i in (0..256).step_by(8) {
                let word = (i..i + 8).fold(0u32, |acc, j| {
                    let n = match (col_tile, row_tile, j) {
                        (0, 1, 190) => 8,
                        (_, 0 | 1, _) => nibble(TILE[j]),
                        _ => rng.next() % 16,
                    };
                    (acc << 4) | n
                });
                codes.extend(word.to_le_bytes());
            }
        }
    }
    EncodedBlock {
        codes,
        scales: vec![],
    }
}

fn unit(axis: &[f32]) -> Vec<f64> {
    let norm = axis
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    axis.iter().map(|&x| f64::from(x) / norm).collect()
}

fn component(weights: &[f32], axis: &[f64]) -> Vec<f64> {
    (0..COLS)
        .map(|j| {
            axis.iter()
                .enumerate()
                .map(|(i, r)| r * f64::from(weights[i * COLS + j]))
                .sum()
        })
        .collect()
}

fn abs_sum(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

fn independently_encode(
    original: &[f32],
    enc: Encoding,
    meta: &QuantMeta<'_>,
    axis: &[f64],
    gain: f64,
    passes: usize,
) -> Vec<f64> {
    let c0 = component(original, axis);
    let mut target = original.to_vec();
    let mut subtract = |c: &[f64], strength: f64| {
        for (row, r) in target.chunks_exact_mut(COLS).zip(axis) {
            for (v, p) in row.iter_mut().zip(c) {
                *v = (f64::from(*v) - strength * r * p) as f32;
            }
        }
        let bytes = encode_rows(enc, &target, axis.len(), COLS, meta).unwrap();
        component(
            &decode_rows(enc, &bytes, axis.len(), COLS, meta).unwrap(),
            axis,
        )
    };
    let parent = abs_sum(&c0);
    assert!(
        parent > 0.,
        "{enc:?}: the fixture must have a target component"
    );
    let mut c = c0;
    (0..passes)
        .map(|i| {
            c = subtract(&c, if i == 0 { 1. } else { gain });
            abs_sum(&c) / parent
        })
        .collect()
}

fn case(enc: Encoding) {
    let rows = if enc == Encoding::Ht { 128 } else { 2560 };
    let mut su: Vec<_> = (0..COLS)
        .map(|i| if i % 3 == 0 { -1. } else { 1. })
        .collect();
    let mut sv: Vec<_> = (0..rows)
        .map(|i| if i % 5 == 0 { -1. } else { 1. })
        .collect();
    if enc == Encoding::Ht {
        // Power-of-two scales keep the fixture in the native half plane.
        su.fill(1.);
        sv.fill(1. / 32.);
    }
    let meta = QuantMeta {
        su: &su,
        sv: &sv,
        threads: 4,
        ..QuantMeta::default()
    };
    let axis_raw: Vec<_> = (0..rows)
        .map(|i| match enc {
            Encoding::Ht => (sign(i, 7) - sign(i, 23)) as f32,
            Encoding::I4r if i < 128 => {
                (DEAD_BAND_AXIS
                    .iter()
                    .enumerate()
                    .map(|(j, x)| x * sign(i, j))
                    .sum::<f64>() as f32)
                    * sv[i]
            }
            Encoding::I4r => 0.,
            _ => DEAD_BAND_AXIS.get(i).copied().unwrap_or(0.) as f32,
        })
        .collect();
    let stock = if enc == Encoding::Ht {
        ht_stock()
    } else {
        row_store(enc, rows, &meta)
    };
    let before = decode_rows(enc, &stock, rows, COLS, &meta).unwrap();
    let axis = unit(&axis_raw);
    let tol = if enc == Encoding::Ht { 0.08 } else { 0.05 };
    let baseline = independently_encode(&before, enc, &meta, &axis, 1., 1)[0];
    assert!(baseline >= 0.8 && baseline > tol, "{enc:?} RTN: {baseline}");
    if enc == Encoding::Ht {
        let full = independently_encode(&before, enc, &meta, &axis, 1., MAX_PASSES);
        let best = full.into_iter().fold(f64::INFINITY, f64::min);
        println!("HT4 gain 1.0: best={best:.6} after {MAX_PASSES} passes");
        assert!(best > tol, "HT gain 1.0 unexpectedly converged: {best}");
    }

    let fixture = tempfile::tempdir().unwrap();
    let parent = fixture.path().join("parent.hgn");
    let child = fixture.path().join("child.hgn");
    let (name, other) = if enc == Encoding::I4r {
        (EXPERTS, ATTENTION)
    } else {
        (ATTENTION, EXPERTS)
    };
    let plan = |name: &str, shape: Vec<u64>, encoding| TensorPlan {
        name: name.into(),
        shape,
        encoding,
        origin: Origin::Tensor { name: name.into() },
        rotation: None,
    };
    let matrix = |name| {
        if name == EXPERTS {
            vec![1, rows as u64, COLS as u64]
        } else {
            vec![rows as u64, COLS as u64]
        }
    };
    let mut tensors = vec![
        plan(name, matrix(name), enc),
        plan(other, matrix(other), Encoding::Bf16),
    ];
    let root = name.strip_suffix(".weight").unwrap();
    let rotated = matches!(enc, Encoding::I4r | Encoding::Ht);
    if rotated {
        tensors.extend([
            plan(&format!("{root}.suh"), vec![COLS as u64], Encoding::F16),
            plan(&format!("{root}.svh"), vec![rows as u64], Encoding::F16),
        ]);
    }
    let mut writer = Writer::create(
        &parent,
        &ContainerPlan {
            filename: "model.hgn".into(),
            identity: "h4-deadband-fixture".into(),
            tensors,
        },
    )
    .unwrap();
    let layout = payload_layout(enc, rows, COLS).unwrap();
    if enc == Encoding::Q4Cp {
        writer
            .write_at(
                0,
                0,
                &meta
                    .codebook
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    writer
        .write_at(0, layout.codes_offset, &stock.codes)
        .unwrap();
    if !stock.scales.is_empty() {
        writer
            .write_at(0, layout.scales_offset, &stock.scales)
            .unwrap();
    }
    if rotated {
        for (i, plane) in [(2, &su), (3, &sv)] {
            let raw = encode_rows(Encoding::F16, plane, 1, plane.len(), &meta).unwrap();
            writer.write_at(i, 0, &raw.codes).unwrap();
        }
    }
    writer.finish().unwrap();
    let pin = file_sha256(&mut File::open(&parent).unwrap()).unwrap();
    let report = hgn_ablation::build(
        &parent,
        &pin,
        &child,
        &fixture.path().join("cache"),
        0..=0,
        &axis_raw,
        0,
    )
    .unwrap_or_else(|e| panic!("{enc:?}: {e:#}"));
    let metric = &report.tensors[name];
    let mut stored = Reader::open(&child).unwrap();
    let after = decode_rows(
        enc,
        &EncodedBlock {
            codes: stored
                .read_at(0, layout.codes_offset, stock.codes.len())
                .unwrap(),
            scales: stored
                .read_at(0, layout.scales_offset, stock.scales.len())
                .unwrap(),
        },
        rows,
        COLS,
        &meta,
    )
    .unwrap();
    let retained = abs_sum(&component(&after, &axis)) / abs_sum(&component(&before, &axis));
    let err = (after
        .iter()
        .zip(&before)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / before.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>())
    .sqrt();
    println!(
        "{enc:?} [{rows},{COLS}]: RTN {baseline:.8} -> feedback {retained:.8} in {} passes, rel_error={err:.8}",
        metric.iterations
    );
    assert!((baseline - metric.leftover_before).abs() < 1e-6);
    assert!(
        metric.iterations > 1 && retained <= tol && (retained - metric.leftover_after).abs() < 1e-6
    );
    assert!(
        err <= 0.04 && (err - metric.rel_error).abs() < 1e-8,
        "{enc:?} rel_error {err}"
    );
    let mut rng = Random(26842);
    for _ in 0..5 {
        let control: Vec<_> = (0..rows)
            .map(|_| f64::from(rng.next()) / f64::from(1 << 23) - 1.)
            .collect();
        // The norm cancels in the ratio. Do not remove overlap with the edit
        // direction; the peer's independent random controls do not either.
        let ratio = abs_sum(&component(&after, &control)) / abs_sum(&component(&before, &control));
        println!("{enc:?}: random control={ratio:.6}");
        assert!(
            (0.95..=1.05).contains(&ratio),
            "{enc:?} random control {ratio}"
        );
    }
}

#[test]
fn q4cp_retains_direction_in_rtn_but_not_feedback() {
    case(Encoding::Q4Cp);
}
#[test]
fn i4r_retains_direction_in_rtn_but_not_feedback() {
    case(Encoding::I4r);
}
#[test]
fn q8g64_retains_direction_in_rtn_but_not_feedback() {
    case(Encoding::Q8G64);
}
#[test]
fn ht_requires_damped_feedback_to_pass() {
    case(Encoding::Ht);
}
