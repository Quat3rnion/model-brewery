use super::*;

#[test]
fn hgn_scratch_output_cannot_enter_library() {
    let mut fixture = Fixture::new();
    fixture.launch();
    for path in [
        "/srv/brewery/worker/jobs/a/work/brewery-private-hgn-scratch/model.hgn",
        "/srv/brewery/worker/brewery-private-hgn-cache/abc/0.payload",
    ] {
        let blocked = fixture.json(
            "POST",
            "/api/models",
            Some(json!({"host":"beta","path":path,"kind":"hgn-v2"})),
            400,
        );
        assert!(
            blocked["error"]
                .as_str()
                .unwrap()
                .contains("private scratch")
        );
    }
}

#[test]
fn hgn_ablation_queues_private_child_and_pinned_descendant_through_mcp() {
    use brewery_worker::{behavior_direction::Behavior, trace};

    let mut app = Fixture::new();
    fs::write(app.path("dispatch.json"), json!({"hosts":{
        "beta":{"ssh":"beta","capabilities":{"behavior_direction":true,"scratch_hgn_ablation":true,"trace":true}},
        "gamma":{"ssh":"gamma"},
    }}).to_string()).unwrap();
    let key = app.path("hgn-key");
    fs::write(&key, [42; 32]).unwrap();
    fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(app.path("hgn-runtime-sha256"), trace::HGN_RUNTIME_SHA256).unwrap();
    app.launch_with(&[(brewery_worker::hgn_edit::KEY_FILE, key)]);
    let hgn = app.register("beta", "hgn-v2", "/approved/stock.hgn");
    let hy4 = app.register("gamma", "hgn-v2", "/approved/stock.hgn");
    let digest = "a".repeat(64);
    fs::write(app.path("direction-result.json"), json!({
        "kind":"behavior_direction","schema_version":2,"release_qualified":false,
        "model_id":hgn,"model_sha256":digest,"behavior":"slop","position":null,"last_k":1,
        "contrast_sha256":Behavior::Slop.contrast_sha256(),
        "runtime":trace::HGN_RUNTIME,"runtime_sha256":trace::HGN_RUNTIME_SHA256.trim_end(),
        "executable_sha256":digest,"assets_sha256":{"sidecar":digest,"config":digest,"tokenizer":digest},
        "fit":{"role":"l_out","train_pairs":12,"held_out_pairs":8,"selected_layer":0,
                "layers":[{"layer":0,"norm":2.0,"unit_vector":[1.,0.],
                    "held_out_margin":0.25,"held_out_cohens_d":0.5,"mean_residual_norm":1.0}]}
    }).to_string()).unwrap();
    let direction = app.enqueue(json!({"kind":"behavior_direction","input_ids":[hgn],
        "options":{"layers":[0],"behavior":"slop","model_sha256":digest}}));
    app.await_job(&direction, |job| job["state"] == "succeeded");
    let request = |id: &str, sha: &str, key: &str| {
        json!({
            "id":id,"sha256":sha,"idempotency_key":key,
            "direction":{"id":direction,"version":2,"layer":0},
            "first_layer":0,"last_layer":0,"variant":"plain",
            "kernel":{"shape":"flat","width":1,"peak":0,"min_weight":0}
        })
    };
    let call = |name: &str, body| {
        let (_, raw) = app.request(
            "POST",
            "/mcp",
            Some(
                json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
                    "name":name,"arguments":body
                }}),
            ),
            "Accept: application/json, text/event-stream\r\n",
        );
        let raw: Value = serde_json::from_str(&raw).unwrap();
        raw["result"]["structuredContent"].clone()
    };
    let mcp = |body| call("ablate_direction", body);
    assert_eq!(
        mcp(request(&hgn, &"0".repeat(64), "wrong-pin"))["status"],
        400
    );
    assert_eq!(mcp(request(&hy4, &digest, "other-worker"))["status"], 400);
    let first = request(&hgn, &digest, "first");
    let queued = mcp(first.clone());
    assert_eq!(queued["kind"], "scratch_hgn_ablation", "{queued}");
    assert_eq!(queued["host"], "beta");
    let id = id(&queued);
    assert_eq!(mcp(first)["id"], id);
    let done = app.await_job(id, |j| j["state"] == "succeeded" || j["state"] == "failed");
    assert_eq!(done["state"], "succeeded", "{done}");
    assert_eq!(done["result"]["scratch_id"], id);
    assert!(done["result"].get("artifact").is_none());
    assert!(!done.to_string().contains("hgn_edit_grant"));
    let route = format!("/api/scratch/{id}");
    let row = app.json("GET", &route, None, 200);
    assert_eq!(row["kind"], "scratch_hgn");
    assert!(row["library_model_id"].is_null());
    assert_eq!(
        row["provenance"]["ablation"]["hgn"]["report"]["parent_sha256"],
        digest
    );
    let child_sha = row["sha256"].as_str().unwrap();
    let check = app.json(
        "POST",
        &format!("{route}/checks"),
        Some(json!({"sha256":child_sha})),
        202,
    );
    assert_eq!(check["kind"], "scratch_hgn_checks");
    assert_eq!(check["host"], "beta");
    assert!(!check.to_string().contains("hgn_check_grant"));
    let done = app.await_job(check["id"].as_str().unwrap(), |j| {
        j["state"] == "succeeded" || j["state"] == "failed"
    });
    assert_eq!(done["state"], "succeeded", "{done}");
    let bundle = app.json("GET", &format!("{route}/checks"), None, 200);
    assert_eq!(bundle["integrity"]["status"], "passed");
    assert_eq!(bundle["smoke"]["status"], "passed");
    assert_eq!(bundle["behavior"]["status"], "not_run");
    assert_eq!(bundle["qualification"], "not release-qualified");
    assert_eq!(
        bundle["hgn_stored"]["tensors"].as_object().unwrap().len(),
        2
    );
    assert_eq!(bundle["hgn_stored"]["prompts"][0]["kl_parent_child"], 0.015);
    assert!(!bundle.to_string().contains("/approved/"));
    assert_eq!(call("get_checks", json!({"id":id})), bundle);
    app.json(
        "POST",
        &format!("{route}/publish-request"),
        Some(json!({"sha256":child_sha})),
        409,
    );
    app.json(
        "POST",
        &format!("{route}/edits"),
        Some(json!({"operations":[
            {"op":"set","tensor":"w","origin":[0],"shape":[1],"values":[0]}
        ]})),
        409,
    );
    let child = mcp(request(id, child_sha, "second"));
    let descendant = child["id"].as_str().unwrap();
    let done = app.await_job(descendant, |j| {
        j["state"] == "succeeded" || j["state"] == "failed"
    });
    assert_eq!(done["state"], "succeeded", "{done}");
    let row = app.json("GET", &format!("/api/scratch/{descendant}"), None, 200);
    assert_eq!(row["provenance"]["parent_id"], id);
    assert_eq!(
        row["provenance"]["ablation"]["hgn"]["report"]["parent_sha256"],
        child_sha
    );
    app.touch("hgn-audit-failed-control");
    let check = call(
        "run_checks",
        json!({"id":descendant,"sha256":row["sha256"],"idempotency_key":"check-second"}),
    );
    assert_eq!(check["kind"], "scratch_hgn_checks", "{check}");
    let done = app.await_job(check["id"].as_str().unwrap(), |j| {
        j["state"] == "succeeded" || j["state"] == "failed"
    });
    assert_eq!(done["state"], "succeeded", "{done}");
    let bundle = call("get_checks", json!({"id":descendant}));
    assert_eq!(bundle["integrity"]["status"], "failed");
    assert_eq!(bundle["smoke"]["status"], "passed");
    assert_eq!(bundle["hgn_stored"]["parent_sha256"], child_sha);
    app.touch("bad-hgn-check");
    let check = call(
        "run_checks",
        json!({"id":descendant,"sha256":row["sha256"],"idempotency_key":"bad-check"}),
    );
    let done = app.await_job(check["id"].as_str().unwrap(), |j| {
        j["state"] == "succeeded" || j["state"] == "failed"
    });
    assert_eq!(done["state"], "failed", "{done}");
    let failed = call("get_checks", json!({"id":descendant}));
    assert_eq!(failed["measured_job_id"], check["id"]);
    assert_eq!(failed["integrity"]["status"], "failed");
    assert!(failed.get("hgn_stored").is_none());
    app.touch("bad-hgn-receipt");
    let bad = mcp(request(&hgn, &digest, "bad-receipt"));
    let bad_id = bad["id"].as_str().unwrap();
    let rejected = app.await_job(bad_id, |j| {
        j["state"] == "failed" || j["state"] == "succeeded"
    });
    assert_eq!(rejected["state"], "failed", "{rejected}");
    app.json("GET", &format!("/api/scratch/{bad_id}"), None, 404);
    assert_eq!(app.model_records().as_array().unwrap().len(), 2);
}
