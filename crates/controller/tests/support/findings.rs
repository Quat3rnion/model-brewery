use super::*;

fn tool(t: &Fixture, name: &str, arguments: Value) -> Value {
    let reply = mcp::call(t, name, arguments);
    assert_eq!(reply["result"]["isError"], false, "{name}: {reply}");
    reply["result"]["structuredContent"].clone()
}

#[test]
fn findings_record_and_replay_mock_circuit_without_rewriting_history() {
    let mut t = Fixture::new();
    fs::write(
        t.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"circuit":true}}}}"#,
    )
    .unwrap();
    t.launch();
    let model = t.register("alpha", "gguf", "/models/original.gguf");
    let target = t.register("alpha", "gguf", "/models/target.gguf");
    let original_sha: String = "a".repeat(64);
    let target_sha: String = "b".repeat(64);
    let root = tool(
        &t,
        "create_finding",
        json!({
            "idempotency_key":"finding-root","hypothesis":"Top sites explain the contrast.",
            "model_id":model,"model_sha256":original_sha,"client_id":"finding-test"
        }),
    );
    let root_id = root["id"].as_str().unwrap();
    assert_eq!(root["version"], 1);
    assert_eq!(
        root["provenance"]["agent"],
        json!({"kind":"external_mcp","id":"finding-test"})
    );
    assert_eq!(root["experiments"], json!([]));

    let options = json!({
        "model_sha256":original_sha,"clean_prompt":"clean","corrupt_prompt":"corrupt",
        "target_id":4,"contrast_id":5,"layers":[0],"positions":[0],
        "roles":["l_out","attn_out"],"top_k":2
    });
    let job = tool(
        &t,
        "find_circuit",
        json!({"idempotency_key":"finding-source-job","input_ids":[model],"options":options}),
    );
    let job_id = job["id"].as_str().unwrap();
    t.await_job(job_id, |j| j["state"] == "succeeded");
    for (key, path) in [
        ("bad-pointer", "/sites/0/role"),
        ("missing-pointer", "/sites/0/missing"),
    ] {
        let reply = mcp::call(
            &t,
            "add_evidence",
            json!({
                "idempotency_key":key,"id":root_id,"client_id":"finding-test",
                "evidence":{"kind":"job","job_id":job_id,"paths":[path]}
            }),
        );
        assert_eq!(
            reply["result"]["structuredContent"]["status"], 400,
            "{reply}"
        );
    }
    let add = json!({
        "idempotency_key":"finding-evidence","id":root_id,"client_id":"finding-test",
        "evidence":{"kind":"job","job_id":job_id,"paths":["/sites/0/patched_delta","/pearson_r"]},
        "verdict":{"status":"supported","reasoning":"The strongest patch changed the margin."}
    });
    let original = tool(&t, "add_evidence", add.clone());
    let original_id = original["id"].as_str().unwrap();
    assert_eq!(tool(&t, "add_evidence", add), original);
    assert_eq!(original["version"], 2);
    assert_eq!(
        original["experiments"][0]["spec"]["request"]["host"],
        "alpha"
    );
    assert_eq!(original["parent_id"], root["id"]);
    assert_eq!(original["experiments"][0]["artifact"]["job_id"], job_id);
    assert_eq!(
        original["experiments"][0]["artifact"]["sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(
        original["experiments"][0]["spec"]["request"]["options"],
        options
    );
    assert_eq!(
        original["experiments"][0]["metrics"][0],
        json!({"path":"/sites/0/patched_delta","value":-5.0})
    );
    assert_eq!(original["verdict"]["status"], "supported");

    let replay = tool(
        &t,
        "replay_finding",
        json!({
            "idempotency_key":"finding-replay","id":original_id,"client_id":"finding-test",
            "model_id":target,"model_sha256":target_sha,
        }),
    );
    let replay_id = replay["id"].as_str().unwrap();
    assert_eq!(replay["parent_id"], original["id"]);
    assert_eq!(replay["version"], 3);
    let replay = wait_for(|| {
        let value = t.json("GET", &format!("/api/findings/{replay_id}"), None, 200);
        assert_ne!(value["state"], "failed", "{value}");
        (value["state"] == "ready").then_some(value)
    });
    assert_ne!(replay["experiments"][0]["artifact"]["job_id"], job_id);
    assert_ne!(
        replay["experiments"][0]["artifact"]["sha256"],
        original["experiments"][0]["artifact"]["sha256"]
    );
    assert_eq!(
        replay["experiments"][0]["spec"]["request"]["input_ids"],
        json!([target])
    );
    let mut rebound = options;
    rebound["model_sha256"] = json!(target_sha);
    assert_eq!(
        replay["experiments"][0]["spec"]["request"]["options"],
        rebound
    );
    assert_eq!(
        replay["comparison"][0],
        json!({"experiment":1,"path":"/sites/0/patched_delta","original":-5.0,"replay":-5.0})
    );
    assert_eq!(replay["verdict"]["status"], "inconclusive");
    assert_eq!(tool(&t, "get_finding", json!({"id":root_id})), root);
    assert_eq!(tool(&t, "get_finding", json!({"id":original_id})), original);
    let list = tool(&t, "list_findings", json!({}));
    assert_eq!(list["findings"].as_array().unwrap().len(), 3);
}
