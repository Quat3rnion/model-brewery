use super::*;

fn weights() -> Vec<u8> {
    let header = br#"{"W_enc":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},"W_dec":{"dtype":"F32","shape":[2,1],"data_offsets":[8,16]},"b_enc":{"dtype":"F32","shape":[2],"data_offsets":[16,24]},"b_dec":{"dtype":"F32","shape":[1],"data_offsets":[24,28]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(
        [2_f32, -3., 5., -1., 0., 0., 0.]
            .into_iter()
            .flat_map(f32::to_le_bytes),
    );
    bytes
}

fn jump_weights() -> Vec<u8> {
    let original = weights();
    let end = 8 + u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let mut header: Value = serde_json::from_slice(&original[8..end]).unwrap();
    header["threshold"] = json!({"dtype":"F32","shape":[2],"data_offsets":[28,36]});
    let header = serde_json::to_vec(&header).unwrap();
    let mut output = (header.len() as u64).to_le_bytes().to_vec();
    output.extend(header);
    output.extend(&original[end..]);
    output.extend(2_f32.to_le_bytes());
    output.extend(5_f32.to_le_bytes());
    output
}

fn upload(app: &Fixture, route: &str, metadata: &Value, weights: &[u8], status: u16) -> Value {
    let mut body = format!("--brewery\r\nContent-Disposition: form-data; name=\"metadata\"\r\n\r\n{metadata}\r\n--brewery\r\nContent-Disposition: form-data; name=\"weights\"; filename=\"sae.safetensors\"\r\n\r\n").into_bytes();
    body.extend(weights);
    body.extend(b"\r\n--brewery--\r\n");
    let headers = format!(
        "Content-Type: multipart/form-data; boundary=brewery\r\nContent-Length: {}\r\nIdempotency-Key: sae-fixture\r\nOrigin: http://{}\r\n",
        body.len(),
        app.addr.unwrap()
    );
    let mut conn = app.open_request("POST", route, None, &headers);
    conn.write_all(&body).unwrap();
    let mut data = String::new();
    conn.read_to_string(&mut data).unwrap();
    let (head, body) = data.split_once("\r\n\r\n").unwrap();
    assert_eq!(
        head.split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<u16>()
            .unwrap(),
        status,
        "{body}"
    );
    serde_json::from_str(body).unwrap()
}

#[test]
fn sae_version_pins_feature_reads_and_decoder_steering() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"trace":true}}}}"#,
    )
    .unwrap();
    fs::write(app.path("trace-interventions"), "").unwrap();
    app.launch();
    let model = app.register("alpha", "gguf", "/models/sae.gguf");
    let mut meta = json!({
        "schema_version":1,"name":"Example","model_id":model,"model_sha256":"a".repeat(64),
        "layer":0,"role":"l_out","d_in":1,"d_sae":2,"activation":{"kind":"top_k","k":1},
        "normalization":{"scale":1.0,"subtract_decoder_bias":false},"provenance":{"method":"fixture"}
    });
    let blob = weights();
    upload(&app, "/api/saes", &meta, &blob[..blob.len() - 1], 400);
    let artifact = upload(&app, "/api/saes", &meta, &blob, 201);
    assert_eq!(upload(&app, "/api/saes", &meta, &blob, 200), artifact);
    let versions = format!("/api/saes/{}/versions", id(&artifact));
    meta["activation"] = json!({"kind":"relu"});
    let v2 = upload(&app, &versions, &meta, &blob, 201);
    assert_eq!(v2["version"], 2);
    assert_eq!(
        app.json("GET", &format!("{versions}/1"), None, 200),
        artifact
    );
    let (status, exported) = app.bytes("GET", &format!("{versions}/1/safetensors"));
    assert_eq!(status, 200);
    assert_eq!(exported, blob);
    assert_eq!(app.json("GET", "/api/saes", None, 200), json!([v2]));

    let mut request = json!({"kind":"trace","input_ids":[model],"prompt":"one two",
        "options":{"layers":[0],"roles":["l_out"],"positions":[0,1],"top_k":2,
            "sae":{"id":artifact["id"],"version":1,"top_k":2},
            "feature_steering":{"id":artifact["id"],"version":1,"feature":0,"alpha":2.0,"positions":[0]}}});
    let mut args = request.clone();
    args.as_object_mut().unwrap().remove("kind");
    args["idempotency_key"] = json!("feature-steering");
    let mcp = super::mcp::call(&app, "steer_feature", args);
    assert_eq!(mcp["result"]["isError"], false, "{mcp}");
    let trace = id(&mcp["result"]["structuredContent"]).to_string();
    let done = app.await_job(&trace, |job| job["state"] == "succeeded");
    let sent = &done["result"]["envelope"]["options"];
    assert!(sent.get("sae").is_none() && sent.get("feature_steering").is_none());
    assert_eq!(sent["expected_model_sha256"], "a".repeat(64));
    assert_eq!(
        sent["interventions"],
        json!([
            {"layer":0,"role":"l_out","positions":[0],"op":"add","vector":[10.0]}
        ])
    );
    let route = format!("/api/jobs/{trace}/trace/features?start=0&end=2");
    let features = app.json("GET", &route, None, 200);
    let mcp = super::mcp::call(
        &app,
        "read_feature_trace",
        json!({"id":trace,"start":0,"end":2}),
    );
    assert_eq!(mcp["result"]["structuredContent"], features);
    assert_eq!(
        features["rows"],
        json!([
            {"position":0,"top":[{"feature":0,"activation":2.0}]},
            {"position":1,"top":[{"feature":1,"activation":6.0}]}
        ])
    );
    app.json(
        "GET",
        &format!("/api/jobs/{trace}/trace/features?start=0&end=17"),
        None,
        400,
    );
    let mut wrong_layer = request.clone();
    wrong_layer["options"]["layers"] = json!([1]);
    app.json("POST", "/api/jobs", Some(wrong_layer), 400);
    let mut overlap = request.clone();
    overlap["options"]["interventions"] =
        json!([{"layer":0,"role":"l_out","positions":[0],"op":"zero"}]);
    app.json("POST", "/api/jobs", Some(overlap), 400);
    fs::write(app.path("trace-corrupt"), "").unwrap();
    app.json("GET", &route, None, 502);
    request["options"]["feature_steering"]["feature"] = json!(2);
    app.json("POST", "/api/jobs", Some(request), 400);
}

#[test]
fn sae_jump_relu_uses_strict_thresholds_and_immutable_bindings() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"trace":true}}}}"#,
    )
    .unwrap();
    fs::write(app.path("trace-interventions"), "").unwrap();
    app.launch();
    let model = app.register("alpha", "gguf", "/models/sae.gguf");
    let mut meta = json!({
        "schema_version":1,"name":"Thresholds","model_id":model,"model_sha256":"a".repeat(64),
        "layer":0,"role":"l_out","d_in":1,"d_sae":2,"activation":{"kind":"jump_relu"},
        "normalization":{"scale":1.0,"subtract_decoder_bias":true},"provenance":{"method":"fixture"}
    });
    upload(&app, "/api/saes", &meta, &weights(), 400);
    let blob = jump_weights();
    let artifact = upload(&app, "/api/saes", &meta, &blob, 201);
    let versions = format!("/api/saes/{}/versions", id(&artifact));
    meta["model_sha256"] = json!("b".repeat(64));
    upload(&app, &versions, &meta, &blob, 400);
    meta["layer"] = json!(1);
    upload(&app, "/api/saes", &meta, &blob, 409);

    let trace = app.enqueue(
        json!({"kind":"trace","input_ids":[model],"prompt":"one two",
        "options":{"layers":[0],"roles":["l_out"],"positions":[0,1],"top_k":2,
            "sae":{"id":artifact["id"],"version":1,"top_k":2}}}),
    );
    app.await_job(&trace, |job| job["state"] == "succeeded");
    let route = format!("/api/jobs/{trace}/trace/features?start=0&end=2");
    assert_eq!(
        app.json("GET", &route, None, 200)["rows"],
        json!([
            {"position":0,"top":[]}, {"position":1,"top":[{"feature":1,"activation":6.0}]}
        ])
    );
}
