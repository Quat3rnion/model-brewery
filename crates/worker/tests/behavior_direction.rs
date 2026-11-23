use brewery_worker::{
    behavior_direction::{self as direction, Request, Split},
    trace::Role,
};
use serde_json::json;
use std::fs;

#[test]
fn holds_out_pairs_when_fitting_and_selects_by_signed_projection() {
    let pairs = [Split::Train, Split::HeldOut, Split::HeldOut];
    let artifact = direction::compute(&[0, 1], Role::LOut, &pairs, |row, positive| {
        let sign = if positive { 1.0 } else { 0.0 };
        Ok(match row {
            0 => vec![vec![3.0 * sign, 4.0 * sign], vec![sign, 0.0]],
            1 => vec![vec![5.0 * sign, -5.0 * sign], vec![2.0 * sign, 10.0]],
            _ => vec![vec![0.0, 0.0], vec![4.0 * sign, -11.0]],
        })
    })
    .unwrap();

    assert_eq!(artifact.layers[0].unit_vector, Some(vec![0.6, 0.8]));
    assert_eq!(artifact.layers[0].norm, 5.0);
    assert_eq!(artifact.layers[0].held_out_margin, Some(-0.5));
    assert_eq!(artifact.layers[1].held_out_margin, Some(3.0));
    assert_eq!(artifact.selected_layer, Some(1));
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
