use brewery_worker::{
    behavior_direction::{self as direction, Observation, Request, Split},
    trace::{self, Role},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;

#[test]
fn holds_out_pairs_when_fitting_and_selects_by_signed_projection() {
    let pairs = [Split::Train, Split::HeldOut, Split::HeldOut];
    let artifact = direction::compute(&[0, 1], Role::LOut, &pairs, |row, positive| {
        let sign = if positive { 1.0 } else { 0.0 };
        let rows = match row {
            0 => vec![vec![1.0 + 3.0 * sign, 2.0 + 4.0 * sign], vec![sign, 0.0]],
            1 => vec![vec![5.0 * sign, -5.0 * sign], vec![2.0 * sign, 10.0]],
            _ => vec![vec![0.0, 0.0], vec![4.0 * sign, -11.0]],
        };
        Ok(rows.into_iter().map(Observation::residual).collect())
    })
    .unwrap();

    assert_eq!(artifact.layers[0].unit_vector, Some(vec![0.6, 0.8]));
    assert_eq!(artifact.layers[0].norm, 5.0);
    assert_eq!(artifact.layers[0].held_out_margin, Some(-0.5));
    assert_eq!(artifact.layers[0].negative_mean, Some(vec![1., 2.]));
    assert_eq!(artifact.layers[1].held_out_margin, Some(3.0));
    assert_eq!(artifact.selected_layer, Some(1));
}

#[test]
fn prefers_held_out_separation_to_large_deep_layer_margin() {
    let pairs = [Split::Train, Split::HeldOut, Split::HeldOut];
    let fit = direction::compute(&[8, 23], Role::LOut, &pairs, |row, positive| {
        let (mid, deep) = match (row, positive) {
            (0, true) => (2.0, 100.0),
            (0, false) => (0.0, 0.0),
            (1, true) => (3.0, 100.0),
            (1, false) => (0.0, 0.0),
            (2, true) => (4.0, 200.0),
            (2, false) => (1.0, 200.0),
            _ => unreachable!(),
        };
        Ok(vec![
            Observation::residual(vec![mid]),
            Observation::residual(vec![deep]),
        ])
    })
    .unwrap();
    assert_eq!(fit.layers[1].held_out_margin, Some(50.0));
    assert_eq!(fit.layers[1].norm, 100.0);
    assert_eq!(fit.layers[1].mean_residual_norm, Some(125.0));
    assert_eq!(fit.layers[0].mean_residual_norm, Some(2.0));
    assert!((fit.layers[0].held_out_cohens_d.unwrap() - 4.242640687).abs() < 1e-9);
    assert!((fit.layers[1].held_out_cohens_d.unwrap() - 0.447213595).abs() < 1e-9);
    assert_eq!(fit.selected_layer, Some(8));
}

#[test]
fn does_not_select_an_undefined_standardized_score() {
    let fit = direction::compute(
        &[8],
        Role::LOut,
        &[Split::Train, Split::HeldOut, Split::HeldOut],
        |_, positive| {
            Ok(vec![Observation::residual(vec![if positive {
                2.0
            } else {
                0.0
            }])])
        },
    )
    .unwrap();
    assert_eq!(fit.layers[0].held_out_margin, Some(2.0));
    assert_eq!(fit.layers[0].held_out_cohens_d, None);
    assert_eq!(fit.selected_layer, None);
}

#[test]
fn cache_checks_payload_digest_and_job_rejects_unapproved_behaviors() {
    let mut options = json!({
        "behavior": "slop",
        "model_sha256": "a".repeat(64),
        "layers": [0, 2],
    });
    let request = Request::parse(&options).unwrap();
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path()
            .join(format!("{}.json", request.behavior.contrast_sha256())),
        b"not JSON",
    )
    .unwrap();
    assert!(
        request
            .load_pairs(dir.path())
            .err()
            .unwrap()
            .to_string()
            .contains("SHA-256")
    );
    options["behavior"] = json!("unreviewed");
    assert!(Request::parse(&options).is_err());
}

#[test]
fn hgn_direction_sums_streams_then_normalizes_each_captured_position() {
    let dir = tempfile::tempdir().unwrap();
    let spec = trace::direction_capture(&[3], Role::LOut, [1, 2]);
    let rows = [
        // The first coordinate cancels. The stream sum is [0, 2, 0, ...].
        (1, [4., 1., -4., 1., 0., 0., 0., 0.]),
        // The stream sum is [3, 4, 0, ...].
        (2, [1., 1., 2., 2., 0., 1., 0., 0.]),
    ];
    let tensors: Vec<_> = rows
        .into_iter()
        .map(|(position, row)| {
            let mut full = vec![0.0_f32; 4 * 2560];
            for (stream, pair) in row.chunks_exact(2).enumerate() {
                full[stream * 2560..][..2].copy_from_slice(pair);
            }
            let bytes: Vec<_> = full.into_iter().flat_map(f32::to_le_bytes).collect();
            let name = format!("tensor-00000{position}.f32");
            fs::write(dir.path().join(&name), &bytes).unwrap();
            json!({
                "layer": 3, "role": "l_out", "position": position, "node": "l_out-3",
                "file": name, "shape": [4, 2560], "dtype": "f32_le",
                "bytes": bytes.len(), "sha256": format!("{:x}", Sha256::digest(&bytes)),
            })
        })
        .collect();
    fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1, "kind": "trace", "spec": spec,
            "model": {"arch": "qwen4_exp_text", "hidden": 2560, "l_out_axes": ["stream","hidden"]},
            "tokens": [{"position": 0, "token_id": 9}, {"position": 1, "token_id": 7},
                       {"position": 2, "token_id": 3}],
            "tensors": tensors, "missing": [], "interventions": [], "model_logits": [], "logit_lens": [],
        }))
        .unwrap(),
    )
    .unwrap();
    let receipt = trace::verify(dir.path(), &spec).unwrap();
    let (tokens, obs) =
        trace::direction_rows(dir.path(), &spec, &receipt, Role::LOut, true).unwrap();
    assert_eq!(tokens, [9, 7, 3]);
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0].residual_norm, 3.5);
    assert_eq!(obs[0].activation.len(), 2560);
    assert_eq!(obs[0].activation[..2], [0.3, 0.9]);
    assert!(obs[0].activation[2..].iter().all(|&x| x == 0.));
    assert!(trace::direction_rows(dir.path(), &spec, &receipt, Role::LOut, false).is_err());
}

#[test]
fn hgn_direction_accepts_published_pins_only_for_saved_reads() {
    let digest = "a".repeat(64);
    let request = Request::parse(&json!({
        "behavior": "slop", "model_sha256": digest, "layers": [1], "last_k": 3,
    }))
    .unwrap();
    let mut receipt = json!({
        "kind":"behavior_direction", "schema_version":2, "release_qualified":false,
        "model_id":"model", "model_sha256":digest, "behavior":"slop",
        "position":null, "last_k":3, "contrast_sha256":request.behavior.contrast_sha256(),
        "runtime":trace::HGN_RUNTIME, "runtime_sha256":trace::HGN_RUNTIME_SHA256.trim_end(),
        "executable_sha256":digest,
        "assets_sha256":{"sidecar":digest,"config":digest,"tokenizer":digest},
        "fit":{"role":"l_out","train_pairs":12,"held_out_pairs":8,"selected_layer":1,
            "layers":[{"layer":1,"norm":2.0,"unit_vector":[0.0,1.0],
                "held_out_margin":0.25,"held_out_cohens_d":0.5,"mean_residual_norm":1.0}]}
    });
    request
        .verify_result(&receipt, "model", &digest, "hgn-v2")
        .unwrap();
    for pin in [
        "29c4699cfcf3170fdf5daeced397fc168f7488244ee8f92d24dc39613693b767", // H2
        "e09a61b5ef47004dcb6d95b5de24e6666d46d7896f5cf1a32d777755a6e40b00", // H3
        "8d4b6d5a685130b4603bbad5825f853c5d42b0275442541f4ea39fa327d8c0b3", // H5
    ] {
        receipt["runtime_sha256"] = json!(pin);
        request
            .verify_version(&receipt, "model", &digest, "hgn-v2", 2)
            .unwrap_or_else(|err| panic!("cannot read saved pin {pin}: {err}"));
        assert!(
            request
                .verify_result(&receipt, "model", &digest, "hgn-v2")
                .is_err()
        );
    }
    receipt["runtime_sha256"] = json!(digest);
    assert!(
        request
            .verify_version(&receipt, "model", &digest, "hgn-v2", 2)
            .is_err()
    );
}
