use super::*;

/// Replay the pinned engine's receipt through the controller's mock dispatcher.
/// Run with BREWERY_HGN_DIRECTION_RECEIPT set to the output of hgn_direction_job.py.
#[test]
#[ignore = "requires the pinned HGN engine acceptance receipt"]
fn hgn_direction_mcp_compares_real_engine_receipt() {
    let path = std::env::var_os("BREWERY_HGN_DIRECTION_RECEIPT").expect("real HGN engine receipt");
    let mut receipt: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let layer = receipt["fit"]["selected_layer"].as_u64().unwrap();
    assert!(
        receipt["fit"]["layers"][0]["held_out_cohens_d"]
            .as_f64()
            .unwrap()
            > 1.0
    );

    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"behavior_direction":true}}}}"#,
    )
    .unwrap();
    app.launch();
    let model = app.register("alpha", "hgn-v2", "/models/real-receipt-fixture.hgn");
    // Remap only the Library ID, which this isolated controller owns.
    receipt["model_id"] = json!(model);
    fs::write(app.path("direction-result.json"), receipt.to_string()).unwrap();
    let input = json!({"kind":"behavior_direction","input_ids":[model],"options":{
        "behavior": receipt["behavior"], "model_sha256": receipt["model_sha256"],
        "last_k": 3, "layers": [0, 1],
    }});
    let reference = app.enqueue(input.clone());
    app.await_job(&reference, |job| job["state"] == "succeeded");
    let id = app.enqueue(input);
    app.await_job(&id, |job| job["state"] == "succeeded");
    let compared = super::mcp::call(
        &app,
        "get_direction",
        json!({"id":id,"version":2,"reference_id":reference,"reference_layer":layer}),
    );
    assert_eq!(compared["result"]["isError"], false, "{compared}");
    let comparison = &compared["result"]["structuredContent"];
    assert_eq!(comparison["direction"], receipt);
    assert_eq!(comparison["reference"]["id"], reference);
    assert_eq!(comparison["reference"]["layer"], layer);
    for score in comparison["reference"]["cosine_by_layer"]
        .as_array()
        .unwrap()
    {
        assert!((score["cosine"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    }
    println!("Real receipt SHA-256: {}", receipt["model_sha256"]);
    println!("MCP comparison: {}", comparison["reference"]);
}

#[test]
fn hgn_direction_mcp_queues_last_k_and_reads_version_two() {
    use brewery_worker::{behavior_direction::Behavior, trace};

    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"behavior_direction":true}}}}"#,
    )
    .unwrap();
    app.launch();
    let model = app.register("alpha", "hgn-v2", "/models/tiny.hgn");
    let digest = "a".repeat(64);
    let options = json!({"layers":[1],"behavior":"slop","model_sha256":digest,"last_k":3});
    let mut invalid = json!({"kind":"behavior_direction","input_ids":[model],"options":options});
    invalid["options"]["position"] = json!(0);
    app.json("POST", "/api/jobs", Some(invalid.clone()), 400);
    invalid["options"]["position"] = Value::Null;
    invalid["options"]["role"] = json!("ffn_out");
    app.json("POST", "/api/jobs", Some(invalid), 400);
    assert!(app.launches().is_empty());

    let mut receipt = json!({
            "kind":"behavior_direction","schema_version":2,"release_qualified":false,
            "model_id":model,"model_sha256":digest,"behavior":"slop","position":null,"last_k":3,
            "contrast_sha256":Behavior::Slop.contrast_sha256(),
            "runtime":trace::HGN_RUNTIME, "runtime_sha256":trace::HGN_RUNTIME_SHA256.trim_end(),
            "executable_sha256":digest,"assets_sha256":{"sidecar":digest,"config":digest,"tokenizer":digest},
            "fit":{"role":"l_out","train_pairs":12,"held_out_pairs":8,"selected_layer":1,
                 "layers":[{"layer":1,"norm":2.0,"unit_vector":[0.0,1.0],
                            "held_out_margin":0.25,"held_out_cohens_d":0.5,"mean_residual_norm":1.0}]}
    });
    fs::write(app.path("direction-result.json"), receipt.to_string()).unwrap();
    let reply = super::mcp::call(
        &app,
        "compute_direction",
        json!({"idempotency_key":"hgn-fit","input_ids":[model],"options":options}),
    );
    assert_eq!(reply["result"]["isError"], false, "{reply}");
    let id = reply["result"]["structuredContent"]["id"].as_str().unwrap();
    let job = app.await_job(id, |job| job["state"] == "succeeded");
    let saved = app.json(
        "GET",
        &format!("/api/directions/{id}/versions/2"),
        None,
        200,
    );
    assert_eq!(saved["direction"], job["result"]);
    app.json(
        "GET",
        &format!("/api/directions/{id}/versions/1"),
        None,
        404,
    );

    receipt["last_k"] = json!(2);
    receipt["fit"]["layers"][0]["unit_vector"] = json!([-0.6, 0.8]);
    fs::write(app.path("direction-result.json"), receipt.to_string()).unwrap();
    let prior = app.enqueue(json!({"kind":"behavior_direction","input_ids":[model],
        "options":{"layers":[1],"behavior":"slop","model_sha256":digest,"last_k":2}}));
    app.await_job(&prior, |job| job["state"] == "succeeded");
    let compared = super::mcp::call(
        &app,
        "get_direction",
        json!({"id":id,"version":2,"reference_id":prior,"reference_layer":1}),
    );
    assert_eq!(compared["result"]["isError"], false, "{compared}");
    let comparison = &compared["result"]["structuredContent"];
    assert_eq!(comparison["direction"], saved["direction"]);
    assert_eq!(comparison["reference"]["id"], prior);
    assert_eq!(comparison["reference"]["layer"], 1);
    assert!(
        (comparison["reference"]["cosine_by_layer"][0]["cosine"]
            .as_f64()
            .unwrap()
            - 0.8)
            .abs()
            < 1e-7
    );
    app.json(
        "GET",
        &format!("/api/directions/{id}/versions/2?reference_id={prior}"),
        None,
        400,
    );
    app.json(
        "GET",
        &format!("/api/directions/{id}/versions/2?reference_id={prior}&reference_layer=3"),
        None,
        404,
    );
    assert_eq!(
        app.json(
            "GET",
            &format!("/api/directions/{id}/versions/2"),
            None,
            200
        ),
        saved
    );
}

#[test]
fn intervention_uses_a_verified_direction_and_exports_only_its_subtraction() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{"alpha":{"local":true,"capabilities":{
            "trace":true,"behavior_direction":true
        }}}})
        .to_string(),
    )
    .unwrap();
    fs::write(app.path("trace-interventions"), "").unwrap();
    app.launch();
    let model = app.register("alpha", "gguf", "/models/steer.gguf");
    let digest = "a".repeat(64);
    fs::write(
        app.path("direction-result.json"),
        json!({
            "kind":"behavior_direction","schema_version":2,"release_qualified":false,
            "model_id":model,"model_sha256":digest,"behavior":"slop","position":null,
            "contrast_sha256":brewery_worker::behavior_direction::Behavior::Slop.contrast_sha256(),
            "lanternfish_commit":"0ea30b37729a136a76b8412e7ad1ec91bd7de9b5",
            "executable_sha256":"0e4ca38c55a9afa0436eeadd06e89bf295eb201b986dd2f40305f95d3a61e46f",
            "fit":{"role":"l_out","train_pairs":12,"held_out_pairs":8,"selected_layer":1,
                "layers":[{"layer":1,"norm":2.0,"unit_vector":[0.0,1.0],
                    "held_out_margin":0.25,"held_out_cohens_d":0.5,"mean_residual_norm":1.0}]}
        })
        .to_string(),
    )
    .unwrap();
    let direction = app.enqueue(json!({"kind":"behavior_direction","input_ids":[model],
        "options":{"layers":[1],"behavior":"slop","model_sha256":digest}}));
    app.await_job(&direction, |job| job["state"] == "succeeded");
    let reference = json!({"direction_id":direction,"direction_version":2,"source_layer":1,
        "layer":2,"role":"l_out","positions":[1],"op":"project_out"});
    let mut body = json!({"name":"Remove slop","model_id":model,"model_sha256":digest,
        "direction_rules":[reference]});
    let saved = app.json("POST", "/api/interventions", Some(body.clone()), 201);
    assert_eq!(saved["direction_rules"], body["direction_rules"]);
    assert_eq!(
        saved["interventions"],
        json!([
            {"layer":2,"role":"l_out","positions":[1],"op":"project_out","vector":[0.0,1.0]}
        ])
    );
    let versions = format!("/api/interventions/{}/versions", id(&saved));
    assert_eq!(app.json("GET", &format!("{versions}/1"), None, 200), saved);
    body["idempotency_key"] = json!("mcp-direction");
    let (_, mcp) = app.request(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc":"2.0","id":1,"method":"tools/call","params":{
                "name":"create_direction_intervention","arguments":body
            }
        })),
        "Accept: application/json, text/event-stream\r\n",
    );
    let mcp: Value = serde_json::from_str(&mcp).unwrap();
    assert_eq!(mcp["result"]["isError"], false, "{mcp}");
    assert_eq!(
        mcp["result"]["structuredContent"]["direction_rules"],
        body["direction_rules"]
    );
    body.as_object_mut().unwrap().remove("idempotency_key");
    let projection = app.json("GET", &format!("{versions}/1/gguf"), None, 400);
    assert!(projection["error"].as_str().unwrap().contains("projection"));
    let trace = app.enqueue(
        json!({"kind":"trace","input_ids":[model],"prompt":"one two",
        "options":{"layers":[2],"roles":["l_out"],"positions":[1],"top_k":2,
            "intervention_artifact":{"id":saved["id"],"version":1}}}),
    );
    let done = app.await_job(&trace, |job| job["state"] == "succeeded");
    assert_eq!(
        done["result"]["envelope"]["options"]["interventions"],
        saved["interventions"]
    );
    assert_eq!(
        done["result"]["envelope"]["options"]["expected_model_sha256"],
        digest
    );

    let mut subtraction = reference;
    subtraction["op"] = json!("subtract");
    subtraction["scale"] = json!(0.25);
    app.json(
        "POST",
        &versions,
        Some(json!({"direction_rules":[subtraction]})),
        201,
    );
    let (status, bytes) = app.bytes("GET", &format!("{versions}/2/gguf"));
    assert_eq!(status, 200);
    let data = &bytes[bytes.len() - 32..];
    assert_eq!(f32::from_le_bytes(data[..4].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_le_bytes(data[4..8].try_into().unwrap()), -0.25);

    body["model_sha256"] = json!("b".repeat(64));
    app.json("POST", "/api/interventions", Some(body.clone()), 400);
    body["model_sha256"] = json!(digest);
    body["direction_rules"][0]["source_layer"] = json!(7);
    app.json("POST", "/api/interventions", Some(body), 404);
}

#[test]
fn intervention_versions_are_immutable_and_bound_to_a_library_model() {
    let mut app = Fixture::new();
    app.launch();
    let model = app.register("alpha", "gguf", "/models/steer.gguf");
    let digest = "a".repeat(64);
    let rules =
        json!([{"op":"add","layer":1,"role":"l_out","positions":[0],"vector":[1.0,-2.0,3.0]}]);
    let mut create = json!({"name":"Direction","model_id":model,"model_sha256":digest,
        "interventions":rules});
    let collection = "/api/interventions";
    create["model_sha256"] = json!("wrong");
    app.json("POST", collection, Some(create.clone()), 400);
    create["model_sha256"] = json!(digest);
    let saved = app.json("POST", collection, Some(create), 201);
    let versions = format!("{collection}/{}/versions", id(&saved));
    assert_eq!(saved["version"], 1);
    assert_eq!(saved["model_id"], model);
    assert_eq!(saved["model_sha256"], digest);
    assert_eq!(saved["interventions"], rules);
    let ablation = json!([{"op":"zero","layer":1,"role":"ffn_out","positions":[0]}]);
    let body = json!({"interventions":ablation});
    let key = "Idempotency-Key: direction-v2\r\n";
    let (status, second) = app.request("POST", &versions, Some(body.clone()), key);
    assert_eq!(status, 201, "{second}");
    let second: Value = serde_json::from_str(&second).unwrap();
    assert_eq!(second["version"], 2);
    assert_eq!(second["model_sha256"], digest);
    let (status, replay) = app.request("POST", &versions, Some(body), key);
    assert_eq!(status, 200);
    assert_eq!(serde_json::from_str::<Value>(&replay).unwrap(), second);
    app.json(
        "POST",
        &versions,
        Some(
            json!({"interventions":[{"op":"zero","layer":1,"role":"kq_soft_max","positions":[0]}]}),
        ),
        400,
    );
    app.stop();
    app.launch();
    assert_eq!(
        app.json("GET", collection, None, 200),
        json!([{"id":saved["id"],"name":"Direction","model_id":model,
            "model_sha256":digest,"latest_version":2}])
    );
    let first = format!("{versions}/1");
    assert_eq!(app.json("GET", &first, None, 200), saved);
    assert_eq!(app.json("GET", &format!("{versions}/2"), None, 200), second);
    app.json("GET", &format!("{versions}/3"), None, 404);
}

#[test]
fn trace_admission_resolves_saved_directions_and_verified_patch_rows() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({
            "hosts":{"alpha":{"local":true,"capabilities":{"trace":true}}}
        })
        .to_string(),
    )
    .unwrap();
    fs::write(app.path("trace-interventions"), "").unwrap();
    app.launch();
    let model = app.register("alpha", "gguf", "/models/steer.gguf");
    let other = app.register("alpha", "gguf", "/models/other.gguf");
    let mut request = json!({"kind":"trace","input_ids":[model],"prompt":"one two",
        "options":{"layers":[0,1,2],"roles":["l_out"],"positions":[0,1],"top_k":2}});
    let source = app.enqueue(request.clone());
    app.await_job(&source, |job| job["state"] == "succeeded");
    let add = json!({"op":"add","layer":1,"role":"l_out","positions":[0],"vector":[4.0]});
    let artifact = app.json(
        "POST",
        "/api/interventions",
        Some(json!({
            "name":"Steer","model_id":model,"model_sha256":"a".repeat(64),"interventions":[add],
        })),
        201,
    );
    request["options"]["intervention_artifact"] = json!({"id":artifact["id"],"version":1});
    request["options"]["activation_patches"] = json!([{
        "source_job_id":source,"layer":0,"role":"l_out","positions":[0,1],
    }]);
    request["options"]["interventions"] = json!([{
        "op":"zero","layer":2,"role":"l_out","positions":[0],
    }]);
    let trace = app.enqueue(request.clone());
    let done = app.await_job(&trace, |job| job["state"] == "succeeded");
    assert_eq!(
        done["options"]["activation_patches"],
        request["options"]["activation_patches"]
    );
    let sent = &done["result"]["envelope"]["options"];
    assert!(sent.get("activation_patches").is_none());
    assert!(sent.get("intervention_artifact").is_none());
    assert_eq!(sent["expected_model_sha256"], "a".repeat(64));
    assert_eq!(
        sent["interventions"],
        json!([
            {"op":"zero","layer":2,"role":"l_out","positions":[0]},
            add,
            {"op":"replace","layer":0,"role":"l_out","positions":[0],"vector":[1.0]},
            {"op":"replace","layer":0,"role":"l_out","positions":[1],"vector":[-2.0]},
        ])
    );

    let mut wrong_model = request.clone();
    wrong_model["input_ids"] = json!([other]);
    app.json("POST", "/api/jobs", Some(wrong_model), 400);

    let mut forged = request.clone();
    forged["options"]["expected_model_sha256"] = json!("b".repeat(64));
    app.json("POST", "/api/jobs", Some(forged), 400);

    let mut overlap = request.clone();
    overlap["options"]["activation_patches"][0]["layer"] = json!(1);
    app.json("POST", "/api/jobs", Some(overlap), 400);

    let mut missing = request.clone();
    missing["options"]["activation_patches"][0]["positions"] = json!([0, 2]);
    app.json("POST", "/api/jobs", Some(missing), 404);

    let mut wrong_revision = request.clone();
    let artifact = app.json("POST", "/api/interventions", Some(json!({
        "name":"Other digest","model_id":model,"model_sha256":"b".repeat(64),"interventions":[add],
    })), 201);
    wrong_revision["options"]["intervention_artifact"]["id"] = artifact["id"].clone();
    app.json("POST", "/api/jobs", Some(wrong_revision), 400);

    fs::write(app.path("trace-corrupt"), "").unwrap();
    let repeat = |body| {
        app.request(
            "POST",
            "/api/jobs",
            Some(body),
            "Brewery-Allow-Duplicate: 1\r\n",
        )
    };
    let (status, reply) = repeat(request.clone());
    assert_eq!(status, 502, "{reply}");
    fs::write(app.path("trace-pruned"), "").unwrap();
    let (status, reply) = repeat(request);
    assert_eq!(status, 404, "{reply}");
    assert_eq!(app.launches().len(), 2);
}

#[test]
fn intervention_exports_only_representable_control_vectors() {
    let mut app = Fixture::new();
    app.launch();
    let model = app.register("alpha", "gguf", "/models/steer.gguf");
    let add = json!({"op":"add","layer":2,"role":"l_out","positions":[0],"vector":[1.0,-2.0,3.0]});
    let earlier =
        json!({"op":"add","layer":1,"role":"l_out","positions":[0],"vector":[0.0,0.5,-0.5]});
    let artifact = app.json(
        "POST",
        "/api/interventions",
        Some(json!({
            "name":"Steer","model_id":model,"model_sha256":"a".repeat(64),"interventions":[add,earlier],
        })),
        201,
    );
    let versions = format!("/api/interventions/{}/versions", id(&artifact));
    let (status, bytes) = app.bytes("GET", &format!("{versions}/1/gguf"));
    assert_eq!(status, 200);
    assert_eq!(&bytes[..8], b"GGUF\x03\x00\x00\x00");
    assert_eq!(
        &bytes[8..24],
        &[2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]
    );
    for field in [
        "general.architecture",
        "controlvector",
        "controlvector.model_hint",
        "controlvector.layer_count",
        "direction.1",
        "direction.2",
    ] {
        assert!(
            bytes.windows(field.len()).any(|w| w == field.as_bytes()),
            "{field}"
        );
    }
    assert_eq!(
        &bytes[bytes.len() - 64..][..12],
        &[0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 191]
    );
    assert_eq!(
        &bytes[bytes.len() - 32..][..12],
        &[0, 0, 128, 63, 0, 0, 0, 192, 0, 0, 64, 64]
    );
    for (rule, reason) in [
        (
            json!({"layer":0,"op":"add","role":"l_out","positions":[0],"vector":[1]}),
            "layer 0",
        ),
        (
            json!({"layer":1,"op":"add","role":"attn_out","positions":[0],"vector":[1]}),
            "l_out",
        ),
        (
            json!({"layer":1,"op":"zero","role":"l_out","positions":[0]}),
            "add",
        ),
    ] {
        let saved = app.json(
            "POST",
            &versions,
            Some(json!({"interventions":[rule]})),
            201,
        );
        let bad = app.json(
            "GET",
            &format!("{versions}/{}/gguf", saved["version"]),
            None,
            400,
        );
        assert!(bad["error"].as_str().unwrap().contains(reason), "{bad}");
    }
}
