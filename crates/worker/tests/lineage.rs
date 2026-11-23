use brewery_worker::weight_spectra::lineage;
use serde_json::{Value, json};
use std::{fs::File, io::Write, path::Path};

const N: usize = 16;

fn write(path: &Path, weights: &[f32], extra: f32) {
    let bytes = N * N * 4;
    let header = json!({
        "projection": {"dtype":"F32","shape":[N,N],"data_offsets":[0,bytes]},
        "other": {"dtype":"F32","shape":[N,N],"data_offsets":[bytes,2*bytes]},
    })
    .to_string();
    let mut f = File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    for value in weights
        .iter()
        .copied()
        .chain(std::iter::repeat_n(extra, N * N))
    {
        f.write_all(&value.to_le_bytes()).unwrap();
    }
}

#[test]
fn infers_rank_one_edits_and_rejects_unrelated_architecture_twins() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.safetensors");
    let child = dir.path().join("child.safetensors");
    let stranger = dir.path().join("stranger.safetensors");
    let original: Vec<_> = (0..N * N)
        .map(|i| ((i * 17 % 53) as f32 - 27.) / 16.)
        .collect();
    let mut edited = original.clone();
    for r in 1..4 {
        for c in 2..6 {
            edited[r * N + c] += (r * c) as f32 / 32.;
        }
    }
    write(&base, &original, 0.25);
    write(&child, &edited, 0.25);
    write(
        &stranger,
        &original.iter().map(|x| -x).collect::<Vec<_>>(),
        -0.25,
    );
    let a = ("base", base.as_path(), "safetensors", None);
    let b = ("child", child.as_path(), "safetensors", None);
    let c = ("unrelated", stranger.as_path(), "safetensors", None);
    let answer = lineage(b, &[c, a], || Ok(())).unwrap();
    let best = &answer["most_likely"];
    assert_eq!(best["base_id"], "base", "{answer}");
    assert_eq!(best["kind"], "rank_one_edit", "{answer}");
    let evidence = &answer["candidates"][0]["evidence"];
    assert_eq!(evidence["changed_count"], 12);
    let row = &evidence["tensors"][0];
    assert_eq!(row["bounds"], json!({"start":[1,2],"end_exclusive":[4,6]}));
    assert_eq!(row["delta_spectrum"]["method"], "exact");
    assert!((row["delta_spectrum"]["stable_rank"].as_f64().unwrap() - 1.).abs() < 1e-10);
    let unrelated = lineage(c, &[a, b], || Ok(())).unwrap();
    assert_eq!(unrelated["most_likely"], Value::Null, "{unrelated}");
}

#[test]
fn detects_pruning_direction_and_merge_from_independent_parent_deltas() {
    let dir = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (0..5)
        .map(|i| dir.path().join(format!("{i}.safetensors")))
        .collect();
    let original: Vec<_> = (0..N * N)
        .map(|i| (16 + (i * 17 % 53)) as f32 / 16.)
        .collect();
    let mut a = original.clone();
    let mut b = original.clone();
    let mut merged = original.clone();
    let mut pruned = original.clone();
    for i in 0..8 {
        a[i] += 0.5;
        merged[i] += 0.25;
        b[i + 5] -= 0.75;
        merged[i + 5] -= 0.5625;
        pruned[32 + i] = 0.;
    }
    let ids = ["base", "parent-a", "parent-b", "merged", "pruned"];
    let input = |i: usize| (ids[i], paths[i].as_path(), "safetensors", None);
    for (path, weights) in paths.iter().zip([original, a, b, merged, pruned]) {
        write(path, &weights, 0.5);
    }
    let result = lineage(input(3), &[input(1), input(2), input(0)], || Ok(())).unwrap();
    assert_eq!(result["most_likely"]["kind"], "merge", "{result}");
    assert_eq!(result["most_likely"]["base_id"], "base", "{result}");
    let evidence = &result["candidates"][0]["evidence"]["merge"];
    assert_eq!(evidence["parent_ids"], json!(["parent-a", "parent-b"]));
    assert!((evidence["weights"][0].as_f64().unwrap() - 0.5).abs() < 1e-10);
    assert!((evidence["weights"][1].as_f64().unwrap() - 0.75).abs() < 1e-10);
    assert!(evidence["relative_residual"].as_f64().unwrap() < 1e-10);

    let result = lineage(input(4), &[input(0)], || Ok(())).unwrap();
    assert_eq!(result["most_likely"]["kind"], "pruning", "{result}");
    assert_eq!(
        result["candidates"][0]["evidence"]["changed_to_zero_fraction"],
        1.
    );
    let inverse = lineage(input(0), &[input(4)], || Ok(())).unwrap();
    assert_ne!(inverse["most_likely"]["kind"], "pruning", "{inverse}");
}

#[test]
fn distinguishes_low_rank_from_dense_changes_and_does_not_force_a_base() {
    let dir = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (0..4)
        .map(|i| dir.path().join(format!("{i}.safetensors")))
        .collect();
    let input = |i: usize| {
        (
            ["base", "low", "dense", "duplicate"][i],
            paths[i].as_path(),
            "safetensors",
            None,
        )
    };
    let original: Vec<_> = (0..N * N)
        .map(|i| (16 + (i * 17 % 53)) as f32 / 16.)
        .collect();
    let mut low = original.clone();
    let mut dense = original.clone();
    for i in 0..N {
        low[i] += 0.25;
        low[N + i] += if i % 2 == 0 { 0.25 } else { -0.25 };
        dense[i * N + i] += 0.25;
    }
    for (path, weights) in paths.iter().zip([original, low.clone(), dense, low]) {
        write(path, &weights, 0.5);
    }
    for (i, kind) in [(1, "low_rank_edit"), (2, "fine_tune")] {
        let answer = lineage(input(i), &[input(0)], || Ok(())).unwrap();
        assert_eq!(answer["most_likely"]["kind"], kind, "{answer}");
    }
    let answer = lineage(input(1), &[input(0), input(3)], || Ok(())).unwrap();
    assert_eq!(answer["most_likely"], Value::Null, "{answer}");
    assert_eq!(answer["candidates"][1]["status"], "identical_weights");
}
