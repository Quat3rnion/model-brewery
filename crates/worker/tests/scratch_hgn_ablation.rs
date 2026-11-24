use bf16_to_hgn::{
    ContainerPlan, Origin, TensorPlan,
    codec::{EncodedBlock, Encoding, QuantMeta, decode_rows, encode_rows, payload_layout},
    container::{Reader, Writer},
};
use brewery_worker::weight_source::scratch::{file_sha256, hgn_ablation};
use std::{fs::File, path::Path};

const ATTENTION: &str = "layers.0.linear_attn.out_proj.weight";
const EXPERTS: &str = "layers.0.mlp.experts.down_proj.weight";

fn write_bf16_model(path: &Path) {
    let tensors = [ATTENTION, EXPERTS, "layers.0.mlp.gate.weight"]
        .into_iter()
        .map(|name| TensorPlan {
            name: name.into(),
            shape: if name == EXPERTS {
                vec![1, 2, 2]
            } else {
                vec![2, 2]
            },
            encoding: Encoding::Bf16,
            origin: Origin::Tensor { name: name.into() },
            rotation: None,
        })
        .collect();
    let plan = ContainerPlan {
        filename: "model.hgn".into(),
        identity: "h4-fixture".into(),
        tensors,
    };
    let mut writer = Writer::create(path, &plan).unwrap();
    let stock = encode_rows(
        Encoding::Bf16,
        &[4., 2., 1., -3.],
        2,
        2,
        &QuantMeta::default(),
    )
    .unwrap();
    for i in 0..3 {
        writer.write_at(i, 0, &stock.codes).unwrap();
    }
    writer.finish().unwrap();
}

#[test]
fn projects_dense_and_routed_writers_exactly_in_bf16() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("stock.hgn");
    let child = dir.path().join("child.hgn");
    write_bf16_model(&source);
    let sha = file_sha256(&mut File::open(&source).unwrap()).unwrap();
    let report = hgn_ablation::build(
        &source,
        &sha,
        &child,
        &dir.path().join("cache"),
        0..=0,
        &[1., 0.],
        0,
    )
    .unwrap();
    assert_eq!(report.tensors.len(), 2);

    let mut original = Reader::open(&source).unwrap();
    let mut stored = Reader::open(&child).unwrap();
    assert_eq!(original.layout.file_size, stored.layout.file_size);
    for i in 0..3 {
        let entry = stored.layout.entries[i].clone();
        let parent = &original.layout.entries[i];
        assert_eq!(
            (
                &entry.name,
                entry.encoding,
                &entry.shape,
                entry.size,
                entry.offset
            ),
            (
                &parent.name,
                parent.encoding,
                &parent.shape,
                parent.size,
                parent.offset
            ),
        );
        assert_eq!(entry.checksum, stored.fingerprint(i).unwrap().xor32);
        let block = EncodedBlock {
            codes: stored.read_at(i, 0, entry.size as usize).unwrap(),
            scales: vec![],
        };
        let decoded = decode_rows(entry.encoding, &block, 2, 2, &QuantMeta::default()).unwrap();
        if i < 2 {
            assert_eq!(decoded, [0., 0., 1., -3.]); // Orthogonal control is unchanged.
            assert_eq!(report.tensors[&entry.name].leftover_after, 0.);
        } else {
            assert_eq!(
                block.codes,
                original.read_at(i, 0, entry.size as usize).unwrap()
            );
        }
    }
}

#[test]
fn stored_quantized_writers_pass_feedback_and_preserve_random_control() {
    const N: usize = 128;
    let dir = tempfile::tempdir().unwrap();
    let mut state = 42u32;
    let mut random = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 8) as f32 / ((1 << 23) as f32) - 1.
    };
    let r: Vec<_> = (0..N).map(|_| random()).collect();
    let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
    let r: Vec<_> = r.into_iter().map(|x| x / norm).collect();
    let mut control: Vec<_> = (0..N).map(|_| random()).collect();
    let dot = r.iter().zip(&control).map(|(a, b)| a * b).sum::<f32>();
    for (v, d) in control.iter_mut().zip(&r) {
        *v -= d * dot;
    }
    let component = |weights: &[f32], direction: &[f32]| -> Vec<f64> {
        (0..N)
            .map(|j| {
                (0..N)
                    .map(|i| f64::from(weights[i * N + j]) * f64::from(direction[i]))
                    .sum()
            })
            .collect()
    };
    let l1 = |v: &[f64]| v.iter().map(|x| x.abs()).sum::<f64>();
    let to_plan = |name: &str, shape, encoding| TensorPlan {
        name: name.into(),
        shape,
        encoding,
        origin: Origin::Tensor { name: name.into() },
        rotation: None,
    };
    let su: Vec<_> = (0..N).map(|i| if i % 3 == 0 { -1. } else { 1. }).collect();
    let sv: Vec<_> = (0..N).map(|i| if i % 5 == 0 { -1. } else { 1. }).collect();
    let meta = QuantMeta {
        su: &su,
        sv: &sv,
        threads: 2,
        ..QuantMeta::default()
    };
    let shape = vec![N as u64, N as u64];
    let base: Vec<_> = (0..N * N).map(|_| random()).collect();
    let original = component(&base, &r);
    let biases: Vec<_> = (0..N).map(|_| random()).collect();
    for enc in [
        Encoding::Q4Cp,
        Encoding::Q8G64,
        Encoding::I4r,
        Encoding::Ht,
        Encoding::Ht3,
    ] {
        let strength = match enc {
            Encoding::Q4Cp => 2.1,
            Encoding::I4r => 2.1,
            Encoding::Ht => 1.1,
            Encoding::Ht3 => 2.2,
            Encoding::Q8G64 => 0.05,
            _ => unreachable!(),
        };
        let mut base = base.clone();
        for i in 0..N {
            for j in 0..N {
                base[i * N + j] += r[i] * (strength * biases[j] - original[j] as f32);
            }
        }
        let source = dir.path().join(format!("{enc:?}.hgn"));
        let child = dir.path().join(format!("{enc:?}-child.hgn"));
        let (name, other, dims) = if enc == Encoding::I4r {
            (EXPERTS, ATTENTION, vec![2, N as u64, N as u64])
        } else {
            (ATTENTION, EXPERTS, shape.clone())
        };
        let root = name.strip_suffix(".weight").unwrap();
        let mut tensors = vec![
            to_plan(name, dims, enc),
            to_plan(
                other,
                if other == EXPERTS {
                    vec![1, N as u64, N as u64]
                } else {
                    shape.clone()
                },
                Encoding::Bf16,
            ),
        ];
        let rotated = enc == Encoding::I4r || enc.ht_bits().is_some();
        if rotated {
            tensors.push(to_plan(
                &format!("{root}.suh"),
                vec![N as u64],
                Encoding::F16,
            ));
            tensors.push(to_plan(
                &format!("{root}.svh"),
                vec![N as u64],
                Encoding::F16,
            ));
        }
        let mut writer = Writer::create(
            &source,
            &ContainerPlan {
                filename: "model.hgn".into(),
                identity: "h4-feedback-fixture".into(),
                tensors,
            },
        )
        .unwrap();
        let stock = encode_rows(enc, &base, N, N, &meta).unwrap();
        let count = if enc == Encoding::I4r { 2 } else { 1 };
        let full = payload_layout(enc, count * N, N).unwrap();
        if enc == Encoding::Q4Cp {
            let raw: Vec<_> = meta.codebook.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.write_at(0, 0, &raw).unwrap();
        }
        for e in 0..count {
            writer
                .write_at(
                    0,
                    full.codes_offset + (e * stock.codes.len()) as u64,
                    &stock.codes,
                )
                .unwrap();
            if !stock.scales.is_empty() {
                writer
                    .write_at(
                        0,
                        full.scales_offset + (e * stock.scales.len()) as u64,
                        &stock.scales,
                    )
                    .unwrap();
            }
        }
        writer.write_at(1, 0, &vec![0; 2 * N * N]).unwrap();
        if rotated {
            for (i, values) in [(2, &su), (3, &sv)] {
                let block = encode_rows(Encoding::F16, values, 1, N, &meta).unwrap();
                writer.write_at(i, 0, &block.codes).unwrap();
            }
        }
        writer.finish().unwrap();
        let sha = file_sha256(&mut File::open(&source).unwrap()).unwrap();
        let result = hgn_ablation::build(
            &source,
            &sha,
            &child,
            &dir.path().join("cache"),
            0..=0,
            &r,
            0,
        )
        .unwrap_or_else(|err| panic!("{enc:?}: {err:#}"));
        let mut reader = Reader::open(&child).unwrap();
        let row = &result.tensors[name];
        println!(
            "{enc:?}: {:.6} -> {:.6} in {} passes (rel_error {:.6})",
            row.leftover_before, row.leftover_after, row.iterations, row.rel_error
        );
        assert!(
            row.iterations > 1 && row.leftover_before > row.tolerance,
            "{enc:?}"
        );
        if enc == Encoding::Q4Cp {
            let mut parent = Reader::open(&source).unwrap();
            assert_eq!(
                parent.read_at(0, 0, 64).unwrap(),
                reader.read_at(0, 0, 64).unwrap()
            );
        }
        assert_eq!(
            (row.store, row.variant, row.bytes),
            (enc.code(), enc.variant(), full.len)
        );
        let actual = reader.fingerprint(0).unwrap().xor32;
        assert_eq!(reader.layout.entries[0].checksum, actual);
        for e in 0..count {
            let block = EncodedBlock {
                codes: reader
                    .read_at(
                        0,
                        full.codes_offset + (e * stock.codes.len()) as u64,
                        stock.codes.len(),
                    )
                    .unwrap(),
                scales: reader
                    .read_at(
                        0,
                        full.scales_offset + (e * stock.scales.len()) as u64,
                        stock.scales.len(),
                    )
                    .unwrap(),
            };
            let before = decode_rows(enc, &stock, N, N, &meta).unwrap();
            let after = decode_rows(enc, &block, N, N, &meta).unwrap();
            let leftover = l1(&component(&after, &r)) / l1(&component(&before, &r));
            assert!(leftover <= row.tolerance, "{enc:?} expert {e}: {leftover}");
            assert!((leftover - row.leftover_after).abs() < 1e-6, "{enc:?}");
            let ratio = l1(&component(&after, &control)) / l1(&component(&before, &control));
            assert!(
                (0.95..=1.05).contains(&ratio),
                "{enc:?}: random control changed: {ratio}"
            );
        }
    }
}

#[test]
fn wrong_parent_pin_leaves_no_child_or_cache() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("stock.hgn");
    let output = dir.path().join("child.hgn");
    let cache = dir.path().join("cache");
    write_bf16_model(&source);
    let error = hgn_ablation::build(
        &source,
        &"a".repeat(64),
        &output,
        &cache,
        0..=0,
        &[1., 0.],
        0,
    )
    .unwrap_err();
    assert!(error.to_string().contains("SHA-256 mismatch"));
    assert!(!output.exists());
    assert!(!cache.exists());
}

#[test]
fn feedback_crosses_a_rounding_deadband() {
    let dir = tempfile::tempdir().unwrap();
    let stock = dir.path().join("deadband.hgn");
    let child = dir.path().join("child.hgn");
    let (rows, cols) = (7, 64);
    let mut values = vec![0.; rows * cols];
    for (i, row) in values.chunks_exact_mut(cols).enumerate() {
        for (j, item) in row.iter_mut().enumerate() {
            let code = match j {
                0 => 0,
                63 => 255,
                _ => 128 + (i == 0) as u8,
            };
            *item = f32::from(code) / 256.;
        }
    }
    let plan = ContainerPlan {
        filename: "model.hgn".into(),
        identity: "h4-deadband".into(),
        tensors: [ATTENTION, EXPERTS]
            .into_iter()
            .map(|name| TensorPlan {
                name: name.into(),
                shape: if name == EXPERTS {
                    vec![1, rows as u64, cols as u64]
                } else {
                    vec![rows as u64, cols as u64]
                },
                encoding: Encoding::Q8G64,
                origin: Origin::Tensor { name: name.into() },
                rotation: None,
            })
            .collect(),
    };
    let mut writer = Writer::create(&stock, &plan).unwrap();
    let encoded = encode_rows(Encoding::Q8G64, &values, rows, cols, &QuantMeta::default()).unwrap();
    writer.write_at(0, 0, &encoded.codes).unwrap();
    writer.write_at(1, 0, &encoded.codes).unwrap();
    writer.finish().unwrap();
    let sha = file_sha256(&mut File::open(&stock).unwrap()).unwrap();
    let report = hgn_ablation::build(
        &stock,
        &sha,
        &child,
        &dir.path().join("cache"),
        0..=0,
        &[2., 1., 1., -1., -1., -1., -1.],
        0,
    )
    .unwrap();
    for tensor in report.tensors.values() {
        // Every value in row 0 is at most 0.4 of a quantization step from
        // its original code after one encode. Feedback crosses the midpoint.
        assert!((tensor.leftover_before - 1.).abs() < 1e-12);
        assert!(tensor.leftover_after < 1e-12);
        assert_eq!(tensor.iterations, 2);
    }
}

#[test]
fn resumes_verified_payloads_and_replaces_a_corrupt_cache_entry() {
    use std::fs::{self, OpenOptions};
    use std::io::{Seek, SeekFrom, Write};
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("stock.hgn");
    let cache = dir.path().join("cache");
    write_bf16_model(&source);
    let sha = file_sha256(&mut File::open(&source).unwrap()).unwrap();
    let build = |name: &str| {
        hgn_ablation::build(
            &source,
            &sha,
            &dir.path().join(name),
            &cache,
            0..=0,
            &[1., 0.],
            0,
        )
        .unwrap()
    };
    let first = build("first.hgn");
    let key = fs::read_dir(&cache)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let files: Vec<_> = fs::read_dir(key)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "payload"))
        .collect();
    let [a, b] = &files[..] else {
        panic!("expected two cached writers")
    };
    let good = fs::read(a).unwrap();
    let atime = fs::metadata(b).unwrap().modified().unwrap();
    let mut corrupt = OpenOptions::new().write(true).open(a).unwrap();
    corrupt.seek(SeekFrom::End(-1)).unwrap();
    corrupt.write_all(&[0xff]).unwrap();
    let resumed = build("resumed.hgn");
    assert_eq!(first.revision_sha256, resumed.revision_sha256);
    assert_eq!(fs::read(a).unwrap(), good);
    assert_eq!(fs::metadata(b).unwrap().modified().unwrap(), atime);
}
