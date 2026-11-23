use brewery_worker::heretic::{Options, REPRODUCE};
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
