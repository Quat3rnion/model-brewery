use super::*;

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
