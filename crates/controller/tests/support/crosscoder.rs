use super::*;

fn weights() -> Vec<u8> {
    let header = br#"{"W_enc":{"dtype":"F32","shape":[4,4],"data_offsets":[0,64]},"W_dec":{"dtype":"F32","shape":[2,4,2],"data_offsets":[64,128]},"b_enc":{"dtype":"F32","shape":[4],"data_offsets":[128,144]},"b_dec":{"dtype":"F32","shape":[2,2],"data_offsets":[144,160]}}"#;
    let mut blob = (header.len() as u64).to_le_bytes().to_vec();
    blob.extend(header);
    blob.extend([0_u8; 64]);
    blob.extend(
        [
            1_f32, 0., 0., 0., 0.707, 0., 0.707, 0., 0., 0., 1., 0., 0., 0.707, 0.707, 0.,
        ]
        .into_iter()
        .flat_map(f32::to_le_bytes),
    );
    blob.extend([0_u8; 32]);
    blob
}

#[test]
fn crosscoder_report_is_bound_to_both_trace_revisions_and_readable_with_mcp() {
    let mut t = Fixture::new();
    fs::write(
        t.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"trace":true}}}}"#,
    )
    .unwrap();
    t.touch("crosscoder-trace");
    t.launch();
    let make_trace = |model: &str, prompt: &str| {
        let trace = t.json(
            "POST",
            "/api/jobs",
            Some(json!({"kind":"trace","input_ids":[model],"prompt":prompt,
                "options":{"layers":[8],"roles":["l_out"],"positions":[0,1],"top_k":1}})),
            201,
        );
        let result = t.await_job(id(&trace), |j| j["state"] == "succeeded");
        result["id"].clone()
    };
    let parent = t.register("alpha", "gguf", "/models/parent.gguf");
    let child = t.register("alpha", "gguf", "/models/child.gguf");
    let pairs = [("train", "training text"), ("heldout", "test text")].map(|(split, prompt)| {
        json!({"split":split,"parent_job_id":make_trace(&parent,prompt),
            "child_job_id":make_trace(&child,prompt)})
    });
    let metrics = json!({"mse":0.1,"explained_variance":0.8,"l0":1.5,"dead_features":1});
    let mut meta = json!({
        "schema_version":1,"parent":{"id":parent,"sha256":"a".repeat(64)},
        "child":{"id":child,"sha256":"b".repeat(64)},
        "layer":8,"role":"l_out","pairs":pairs,"width":2,"k":2,"scale":1.0,
        "seen":[true,true,true,true],"metrics":{"parent":metrics,"child":metrics},
        "training":{"torch":"2.7.1+cpu","steps":100,"difference_weight":1.0,
            "train_rows":2,"heldout_rows":2,"dataset_sha256":"a".repeat(64)}
    });
    let blob = weights();
    let import = |meta: &Value, blob: &[u8], status| {
        super::sae::upload(&t, "/api/crosscoders", meta, blob, status)
    };
    import(&meta, &blob[..blob.len() - 1], 400);
    meta["child"]["sha256"] = json!("c".repeat(64));
    import(&meta, &blob, 400);
    meta["child"]["sha256"] = json!("b".repeat(64));
    let child_trace = meta["pairs"][1]["child_job_id"].clone();
    meta["pairs"][1]["child_job_id"] = make_trace(&child, "different text");
    import(&meta, &blob, 400);
    meta["pairs"][1]["child_job_id"] = child_trace;
    let artifact = import(&meta, &blob, 201);
    assert_eq!(import(&meta, &blob, 200), artifact);
    assert_eq!(artifact["version"], 1);
    assert_eq!(artifact["qualification"], "not release-qualified");
    assert_eq!(
        artifact["scratch_revision_sha256"],
        json!({"parent":null,"child":null})
    );
    let classes: Vec<_> = artifact["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["classification"].as_str().unwrap())
        .collect();
    assert_eq!(
        classes,
        ["parent_unique", "child_unique", "changed", "shared"]
    );
    assert_eq!(artifact["features"][0]["child_norm_ratio"], 0.);
    assert_eq!(artifact["features"][1]["child_norm_ratio"], 1.);
    assert_eq!(artifact["features"][2]["decoder_cosine"], 0.);
    let path = format!("/api/crosscoders/{}", id(&artifact));
    assert_eq!(t.json("GET", &path, None, 200), artifact);
    assert_eq!(t.bytes("GET", &format!("{path}/safetensors")), (200, blob));
    let mcp = super::mcp::call(&t, "crosscoder_diff", json!({"id":artifact["id"]}));
    assert_eq!(mcp["result"]["isError"], false, "{mcp}");
    assert_eq!(mcp["result"]["structuredContent"], artifact);
    let bad_id = super::mcp::call(&t, "crosscoder_diff", json!({"id":"../state"}));
    assert_eq!(bad_id["error"]["code"], -32602);
}
