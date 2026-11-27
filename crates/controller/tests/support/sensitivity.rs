use super::*;

#[test]
fn sensitivity_binds_artifacts_to_the_job_and_exposes_the_mcp_submission() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"sensitivity":true}},"beta":{"local":true}}}"#,
    )
    .unwrap();
    app.launch();
    let model = app.register("alpha", "gguf", "/models/model.gguf");
    let options = json!({
        "model_sha256":"a".repeat(64),
        "corpus_sha256":"9678f9afdc31b2a9be7fec2045997e2360693ee72b893287c0d539983bff4923",
        "tensors":["blk.0.attn_q.weight"], "observations":2
    });
    let body = json!({"kind":"sensitivity","input_ids":[model],"options":options});
    js_invalid(&app, &body, "corpus_sha256", json!("b".repeat(64)));
    js_invalid(&app, &body, "revision_sha256", json!("b".repeat(64)));
    js_invalid(&app, &body, "layers", json!([0]));
    let mut different_host = body.clone();
    different_host["host"] = json!("beta");
    app.json("POST", "/api/jobs", Some(different_host), 409);
    assert!(app.launches().is_empty());

    let reply = super::mcp::call(
        &app,
        "tensor_sensitivity",
        json!({"idempotency_key":"sensitivity", "input_ids":[model], "options":options}),
    );
    assert_eq!(reply["result"]["isError"], false, "{reply}");
    let id = reply["result"]["structuredContent"]["id"].as_str().unwrap();
    let job = app.await_job(id, |job| job["state"] == "succeeded");
    let path = format!("/api/jobs/{id}/sensitivity");
    let artifact = app.json("GET", &path, None, 200);
    assert_eq!(artifact["revision_sha256"], options["model_sha256"]);
    assert_eq!(artifact["tensors"][0]["fisher"], 0.25);
    assert_eq!(artifact["qualification"], "not release-qualified");
    assert_eq!(job["result"], artifact);

    app.touch("sensitivity-wrong-revision");
    let mut bad = body;
    bad["options"]["observations"] = json!(1);
    let id = app.enqueue(bad);
    let job = app.await_job(&id, |job| job["state"] == "failed");
    assert!(job["result"].is_null());
    app.json("GET", &format!("/api/jobs/{id}/sensitivity"), None, 404);
}

fn js_invalid(app: &Fixture, body: &Value, key: &str, invalid: Value) {
    let mut bad = body.clone();
    bad["options"][key] = invalid;
    app.json("POST", "/api/jobs", Some(bad), 400);
}
