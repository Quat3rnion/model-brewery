use brewery_worker::heretic::{Options, REPRODUCE, RUN};
use serde_json::json;

#[test]
fn reproduction_requires_the_attested_commit_and_base_digest() {
    let mut options = json!({
        "kind": REPRODUCE,
        "base_sha256": "a".repeat(64),
        "base_hf_commit": "b".repeat(40),
        "invocation": null,
        "manifest": {"version":"4","settings":{"model_commit":"b".repeat(40)}},
    });
    assert!(Options::parse(REPRODUCE, &options).is_ok());
    options["manifest"]["settings"]["model_commit"] = json!("c".repeat(40));
    assert!(Options::parse(REPRODUCE, &options).is_err());
    options["manifest"]["settings"]["model_commit"] = json!("b".repeat(40));
    options["base_sha256"] = json!("d".repeat(40));
    assert!(Options::parse(REPRODUCE, &options).is_err());
}

#[test]
fn trial_summary_is_bounded_and_contains_the_selected_trial() {
    let invocation = json!({"profile":"noslop-dolly","trials":2,"trial_index":0,"seed":83});
    let pin = "a".repeat(64);
    let options = Options::parse(
        RUN,
        &json!({
            "kind": RUN, "base_sha256": pin, "invocation": invocation,
        }),
    )
    .unwrap();
    let mut receipt = json!({
        "kind": RUN, "run_directory":"/isolated/runs/brewery-Abc123", "export_sha256":pin,
        "journal": {
            "kind": RUN, "schema":"brewery.heretic-journal.v1", "base_sha256":pin, "export_sha256":pin,
            "invocation":invocation, "effective_config":"", "source_config":"", "trial_index":0, "seed":83,
            "selected_study_trial":2, "runtime":{
                "heretic":"2.1.0","commit":brewery_worker::heretic::COMMIT,"licence":"AGPL-3.0-or-later",
                "torch":"2.9.1","transformers":"5.24.0","lock_sha256":pin,"environment_sha256":pin,"launcher_sha256":pin
            }
        }
    });
    assert!(
        options.verify(&receipt, None).is_ok(),
        "A6 journal still reads"
    );
    receipt["journal"]["trials"] =
        json!([{"index":2,"scores":[{"name":"KL","value":0.1,"baseline":0.0}]}]);
    assert!(options.verify(&receipt, None).is_ok());
    receipt["journal"]["trials"][0]["index"] = json!(1);
    assert!(
        options.verify(&receipt, None).is_err(),
        "selected trial must be present"
    );
}
