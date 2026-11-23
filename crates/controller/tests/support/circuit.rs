use super::*;

#[test]
fn circuit_mcp_submission_binds_the_saved_rest_artifact() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"circuit":true}},"beta":{"local":true}}}"#,
    )
    .unwrap();
    app.launch();
    let model = app.register("alpha", "gguf", "/models/model.gguf");
    let options = json!({
        "model_sha256":"a".repeat(64), "clean_prompt":"clean", "corrupt_prompt":"corrupt",
        "target_id":4, "contrast_id":5, "layers":[0], "positions":[0],
        "roles":["l_out","attn_out"], "top_k":2
    });
    let body = json!({"kind":"circuit","input_ids":[model],"options":options});
    for (name, value) in [
        ("positions", json!([0, 0])),
        ("roles", json!(["ffn_inp"])),
        ("contrast_id", json!(4)),
        ("top_k", json!(3)),
        ("sample_lower", json!(1)),
    ] {
        let mut bad = body.clone();
        bad["options"][name] = value;
        app.json("POST", "/api/jobs", Some(bad), 400);
    }
    let mut wrong_host = body.clone();
    wrong_host["host"] = json!("beta");
    app.json("POST", "/api/jobs", Some(wrong_host), 409);
    assert!(app.launches().is_empty());

    let reply = super::mcp::call(
        &app,
        "find_circuit",
        json!({"idempotency_key":"circuit", "input_ids":[model], "options":options}),
    );
    assert_eq!(reply["result"]["isError"], false, "{reply}");
    let id = reply["result"]["structuredContent"]["id"].as_str().unwrap();
    let job = app.await_job(id, |job| job["state"] == "succeeded");
    let artifact = app.json("GET", &format!("/api/jobs/{id}/circuit"), None, 200);
    assert_eq!(artifact["sites"][0]["patched_delta"], -5.0);
    assert_eq!(artifact["pearson_r"], 1.0);
    assert!((artifact["spearman_rho"].as_f64().unwrap() - 1.).abs() < 1e-12);
    assert_eq!(artifact, job["result"]);

    app.touch("circuit-wrong-correlation");
    let mut bad = body;
    bad["options"]["target_id"] = json!(6);
    let id = app.enqueue(bad);
    let job = app.await_job(&id, |job| job["state"] == "failed");
    assert!(job["result"].is_null());
    app.json("GET", &format!("/api/jobs/{id}/circuit"), None, 404);
}
