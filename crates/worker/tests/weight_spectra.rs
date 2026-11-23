use brewery_worker::weight_spectra::{Filter, analyze};
use serde_json::{Value, json};
use std::{fs::File, io::Write, path::Path};

const NAME: &str = "model.layers.3.self_attn.q_proj.weight";
const FILTER: Filter = Filter {
    layer: Some(3),
    tensor: None,
};

fn safe(path: &Path, rows: usize, cols: usize, values: impl IntoIterator<Item = f32>) {
    let header = json!({(NAME): {
        "dtype":"F32", "shape":[rows,cols], "data_offsets":[0,rows * cols * 4]
    }})
    .to_string();
    let mut out = File::create(path).unwrap();
    out.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    out.write_all(header.as_bytes()).unwrap();
    for value in values {
        out.write_all(&value.to_le_bytes()).unwrap();
    }
}

fn close(actual: &Value, expected: f64) {
    assert!(
        (actual.as_f64().unwrap() - expected).abs() <= 1e-10 * expected.abs().max(1.),
        "{actual} != {expected}"
    );
}

#[test]
fn exact_spectra_match_numpy_and_compare_only_equal_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("weights.safetensors");
    safe(&path, 2, 3, [3., 1., -2., 2., 1., 4.]);
    let result = analyze(
        ("a", &path, "safetensors", None),
        Some(("b", &path, "safetensors", None)),
        &FILTER,
        || Ok(()),
    )
    .unwrap();
    let row = &result["tensors"][0];
    // np.linalg.svd([[3., 1., -2.], [2., 1., 4.]], compute_uv=False).
    close(&row["source"]["singular_values"][0], 4.597_831_548_093_106);
    close(
        &row["source"]["singular_values"][1],
        3.722_894_714_514_464_6,
    );
    close(&row["source"]["effective_rank"], 1.988_953_316_246_932_2);
    close(&row["source"]["stable_rank"], 1.655_624_835_964_474);
    assert_eq!(result["qualification"], "not release-qualified");
    assert_eq!(row["source"]["method"], "exact");
    assert_eq!(row["source"]["alpha"], Value::Null);
    assert_eq!(row["delta"]["effective_rank"], 0.);
    assert_eq!(row["delta"]["sigma_max"], 0.);
    let other = tmp.path().join("other.safetensors");
    safe(&other, 3, 2, [3., 1., -2., 2., 1., 4.]);
    let result = analyze(
        ("a", &path, "safetensors", None),
        Some(("b", &other, "safetensors", None)),
        &FILTER,
        || Ok(()),
    )
    .unwrap();
    assert_eq!(result["tensors"][0]["status"], "shape_mismatch");
    assert_eq!(result["tensors"][0]["delta"], Value::Null);
    assert!(
        analyze(
            ("a", &path, "safetensors", Some(&"0".repeat(64))),
            None,
            &FILTER,
            || Ok(())
        )
        .is_err()
    );
}

#[test]
fn alpha_minimum_ks_and_seeded_rank_one_spectrum_match_numpy() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("weights.safetensors");
    let diag = [
        15., 12., 9., 7., 5., 4., 3.5, 3.1, 2.9, 2.8, 2.7, 2.65, 2.61, 2.6, 2.59, 2.57,
    ];
    safe(
        &path,
        16,
        16,
        (0..16).flat_map(|r| (0..16).map(move |c| if r == c { diag[r] } else { 0. })),
    );
    let result = analyze(("a", &path, "safetensors", None), None, &FILTER, || Ok(())).unwrap();
    let row = &result["tensors"][0]["source"];
    close(&row["alpha"], 1.767_562_788_151_047);
    close(&row["alpha_ks"], 0.155_346_014_853_912_92);
    close(&row["alpha_xmin"], 7.839_999_732_971_194);
    assert_eq!(row["alpha_tail_count"], 10);
    safe(
        &path,
        192,
        160,
        (1..=192).flat_map(|r| (1..=160).map(move |c| ((r % 7) * (c % 5)) as f32)),
    );
    let result = analyze(("a", &path, "safetensors", None), None, &FILTER, || Ok(())).unwrap();
    let row = &result["tensors"][0]["source"];
    assert_eq!(row["method"], "randomized_top_32");
    assert_eq!(row["singular_values"].as_array().unwrap().len(), 32);
    close(&row["singular_values"][0], 1540.181_807_450_016_5);
    close(&row["singular_values"][1], 0.);
    close(&row["effective_rank"], 1.);
    close(&row["stable_rank"], 1.);
    close(&row["captured_energy_fraction"], 1.);
    assert_eq!(row["alpha"], Value::Null);
}

#[test]
fn gguf_q8_0_dequantizes_and_matches_safetensors_by_layer() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("weights.gguf");
    let reference = tmp.path().join("reference.safetensors");
    let name = b"blk.3.attn_q.weight";
    let q: Vec<_> = (-16i8..16).chain([-1i8, 2].repeat(16)).collect();
    let mut data = b"GGUF".to_vec();
    data.extend(3u32.to_le_bytes());
    for n in [1, 0, name.len() as u64] {
        data.extend(n.to_le_bytes());
    }
    data.extend(name);
    data.extend(2u32.to_le_bytes());
    data.extend(32u64.to_le_bytes());
    data.extend(2u64.to_le_bytes());
    data.extend(8u32.to_le_bytes());
    data.extend(0u64.to_le_bytes());
    data.resize(data.len().next_multiple_of(32), 0);
    for row in q.chunks_exact(32) {
        data.extend(0x3800u16.to_le_bytes());
        data.extend(row.iter().map(|&n| n as u8));
    }
    std::fs::write(&path, &data).unwrap();
    safe(&reference, 2, 32, q.iter().map(|&n| f32::from(n) * 0.5));
    let result = analyze(
        ("a", &path, "gguf", None),
        Some(("b", &reference, "safetensors", None)),
        &FILTER,
        || Ok(()),
    )
    .unwrap();
    let row = &result["tensors"][0];
    assert_eq!(row["name"], NAME);
    assert_eq!(row["status"], "compared");
    assert_eq!(row["source"]["dtype"], "Q8_0");
    close(&row["source"]["singular_values"][0], 26.153_854_314_635_044);
    close(&row["source"]["singular_values"][1], 4.469_441_183_061_465);
    close(&row["delta"]["sigma_max"], 0.);
    data.pop();
    std::fs::write(&path, data).unwrap();
    assert!(analyze(("a", &path, "gguf", None), None, &FILTER, || Ok(())).is_err());
}
