use super::*;

#[test]
fn hgn_scratch_output_cannot_enter_library() {
    let mut fixture = Fixture::new();
    fixture.launch();
    let path = "/srv/brewery/worker/jobs/a/work/brewery-private-hgn-scratch/model.hgn";
    let blocked = fixture.json(
        "POST",
        "/api/models",
        Some(json!({"host":"beta","path":path,"kind":"hgn-v2"})),
        400,
    );
    assert!(blocked["error"].as_str().unwrap().contains("private scratch"));
}

#[test]
fn hgn_ablation_queues_private_child_and_pinned_descendant_through_mcp() {
    let mut app = Fixture::new();
    fs::write(app.path("dispatch.json"), json!({"hosts":{
        "alpha":{"local":true,"capabilities":{"behavior_direction":true}},
        "beta":{"ssh":"beta","capabilities":{"scratch_hgn_ablation":true}},
        "gamma":{"ssh":"gamma"},
    }}).to_string()).unwrap();
    let key = app.path("hgn-key");
    fs::write(&key, [42;32]).unwrap();
    fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();
    app.launch_with(&[(brewery_worker::hgn_edit::KEY_FILE, key.clone())]);
    let hgn = app.register("beta", "hgn-v2", "/approved/stock.hgn");
    let hy4 = app.register("gamma", "hgn-v2", "/approved/stock.hgn");
    let source = app.register("alpha", "gguf", "/approved/direction.gguf");
    let digest = "a".repeat(64);
    fs::write(app.path("direction-result.json"), json!({
        "kind":"behavior_direction","schema_version":2,"release_qualified":false,
        "model_id":source,"model_sha256":digest,"behavior":"slop","position":null,
        "contrast_sha256":brewery_worker::behavior_direction::Behavior::Slop.contrast_sha256(),
        "lanternfish_commit":brewery_worker::trace::COMMIT,
        "executable_sha256":brewery_worker::trace::SHA256,
        "fit":{"role":"l_out","train_pairs":12,"held_out_pairs":8,"selected_layer":0,
                "layers":[{"layer":0,"norm":2.0,"unit_vector":[1.,0.],
                    "held_out_margin":0.25,"held_out_cohens_d":0.5,"mean_residual_norm":1.0}]}
    }).to_string()).unwrap();
    let direction = app.enqueue(json!({"kind":"behavior_direction","input_ids":[source],
        "options":{"layers":[0],"behavior":"slop","model_sha256":digest}}));
    app.await_job(&direction, |job| job["state"] == "succeeded");
    app.stop();
    {
        // H2 owns HGN direction capture. Use the verified v2 direction seam
        // with a synthetic vector bound to the mock HGN Library record.
        let db = rusqlite::Connection::open(app.path("state/controller.db")).unwrap();
        db.execute("UPDATE jobs SET envelope=json_set(envelope,'$.inputs[0].id',?1),
            result=json_set(result,'$.model_id',?1) WHERE id=?2", [&hgn, &direction]).unwrap();
    }
    app.launch_with(&[(brewery_worker::hgn_edit::KEY_FILE, key)]);
    let request = |id: &str, sha: &str, key: &str| json!({
        "id":id,"sha256":sha,"idempotency_key":key,
        "direction":{"id":direction,"version":2,"layer":0},
        "first_layer":0,"last_layer":0,"variant":"plain",
        "kernel":{"shape":"flat","width":1,"peak":0,"min_weight":0}
    });
    let mcp = |body| {
        let (_, raw) = app.request("POST", "/mcp",
            Some(json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
                "name":"ablate_direction","arguments":body
            }})), "Accept: application/json, text/event-stream\r\n");
        let raw: Value = serde_json::from_str(&raw).unwrap();
        raw["result"]["structuredContent"].clone()
    };
    assert_eq!(mcp(request(&hgn, &"0".repeat(64), "wrong-pin"))["status"], 400);
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
    assert_eq!(row["provenance"]["ablation"]["hgn"]["report"]["parent_sha256"], digest);
    let child_sha = row["sha256"].as_str().unwrap();
    app.json("POST", &format!("{route}/publish-request"), Some(json!({"sha256":child_sha})), 409);
    app.json("POST", &format!("{route}/edits"), Some(json!({"operations":[
        {"op":"set","tensor":"w","origin":[0],"shape":[1],"values":[0]}
    ]})), 409);
    let child = mcp(request(id, child_sha, "second"));
    let descendant = child["id"].as_str().unwrap();
    let done = app.await_job(descendant, |j| j["state"] == "succeeded" || j["state"] == "failed");
    assert_eq!(done["state"], "succeeded", "{done}");
    let row = app.json("GET", &format!("/api/scratch/{descendant}"), None, 200);
    assert_eq!(row["provenance"]["parent_id"], id);
    assert_eq!(row["provenance"]["ablation"]["hgn"]["report"]["parent_sha256"], child_sha);
    app.touch("bad-hgn-receipt");
    let bad = mcp(request(&hgn, &digest, "bad-receipt"));
    let bad_id = bad["id"].as_str().unwrap();
    let rejected = app.await_job(bad_id, |j| j["state"] == "failed" || j["state"] == "succeeded");
    assert_eq!(rejected["state"], "failed", "{rejected}");
    app.json("GET", &format!("/api/scratch/{bad_id}"), None, 404);
    assert_eq!(app.models().as_array().unwrap().len(), 3);
}
