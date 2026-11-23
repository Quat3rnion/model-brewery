use brewery_worker::{
    behavior_direction::{self as direction, Observation, Request, Split},
    trace::Role,
};
use serde_json::json;
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
