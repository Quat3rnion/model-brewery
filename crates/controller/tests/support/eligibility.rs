use super::*;

fn matches_admission(app: &Fixture, row: &Value, path: &str, body: Value, expected: bool) {
    let (status, response) = app.request("POST", path, Some(body), "");
    assert_eq!(row["eligible"], expected, "{path}: {row}, {response}");
    assert_eq!(status == 201, expected, "{path}: {status} {response}");
    if expected {
        assert_eq!(row["reasons"], json!([]));
    } else {
        assert!(matches!(status, 400 | 409));
        let refusal: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(refusal["error"], row["reasons"][0]["text"], "{path}");
        assert!(
            row["reasons"][0]["fix_anchor"]
                .as_str()
                .unwrap()
                .starts_with("#/")
        );
    }
}

#[test]
fn workflow_eligibility_matches_admission() {
    let mut app = Fixture::new();
    let companion = app.path("dispatch");
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true, "quantization_pool":true,"capabilities":{
                "generate":true,"evaluate":true,"baseline":true,"benchmark":true,
                "gguf_output":true,"gguf_hf_output":true,"custom_profiles":true
            },"benchmark_targets":[{"id":"target"}],"benchmarks":[{"id":"suite"}]},
            "beta":{"local":true,"capabilities":{
                "compare":false,"transfer":false,"transfer_preview":false,"inspect":false,"plan":false,"verify":false,"gguf_output":true
            }},
            "cpu":{"local":true,"component_cpu_only":true,"capabilities":{
                "convert_components":true,"gguf_output":true,"gguf_safetensors_output":true,"gguf_rocmfpx_output":true
            }},
            "legacy":{"ssh":"legacy","capabilities":{
                "generate":true,"evaluate":true,"baseline":true,"benchmark":true,
                "gguf_output":true,"gguf_hf_output":true
            },"benchmark_targets":[{"id":"target"}],"benchmarks":[{"id":"suite"}]},
            "hy4":{"local":true,"capabilities":{"hy4_encode":true}}
        }}).to_string(),
    )
    .unwrap();
    app.launch_with(&[
        ("BREWERY_HF_DISPATCH", companion.clone()),
        ("BREWERY_HF_WORKER", companion.clone()),
        ("BREWERY_HF_ENGINE", companion.clone()),
        ("BREWERY_HF_HFR", companion),
    ]);

    let snapshot = app.json("GET", "/api/state", None, 200);
    let eligibility = &snapshot["eligibility"];
    let a = &eligibility["hosts"]["alpha"];
    let b = &eligibility["hosts"]["beta"];
    assert_eq!(eligibility["global"]["compare"]["eligible"], true);
    assert_eq!(a["pool_conversion"]["eligible"], true);
    assert_eq!(b["pool_conversion"]["eligible"], false);
    assert_eq!(
        b["managed_conversion"]["reasons"].as_array().unwrap().len(),
        3
    );
    assert_eq!(a["gguf_output_by_kind"]["safetensors"]["eligible"], false);
    let hy4 = app.register("hy4", "hf", "/fixture/hy4");
    matches_admission(
        &app,
        &eligibility["hosts"]["hy4"]["managed_conversion"],
        "/api/conversions",
        json!({
            "input_ids":[hy4]
        }),
        false,
    );
    let cpu = app.register("cpu", "hf", "/fixture/checkpoint");
    let assets = app.register("cpu", "assets", "/fixture/assets");
    let weights = app.register("cpu", "safetensors", "/fixture/weights.safetensors");
    matches_admission(
        &app,
        &eligibility["hosts"]["cpu"]["component_conversion"],
        "/api/conversions",
        json!({
            "input_ids":[cpu,assets],"options":{"assets":1}
        }),
        true,
    );
    for (entry, options) in [
        ("gguf_output_by_kind", json!({})),
        ("gguf_rocmfpx_by_kind", json!({"preset":"Q8_0_ROCMFPX"})),
    ] {
        matches_admission(
            &app,
            &eligibility["hosts"]["cpu"][entry]["safetensors"],
            "/api/conversions",
            json!({"format":"gguf","input_ids":[weights,assets],"options":options}),
            true,
        );
    }

    let settings = app.json("GET", "/api/settings", None, 200);
    let legacy = settings["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|host| host["id"] == "legacy")
        .unwrap();
    assert_eq!(legacy["evaluation"]["status"], "profile_only");
    for (host, row, enabled) in [
        ("alpha", a, true),
        ("beta", b, false),
        ("legacy", &eligibility["hosts"]["legacy"], true),
    ] {
        let hf = app.register(host, "hf", "/fixture/checkpoint");
        let assets = app.register(host, "assets", "/fixture/assets");
        let gguf_a = app.register(host, "gguf", "/fixture/a.gguf");
        let gguf_b = app.register(host, "gguf", "/fixture/b.gguf");
        let hgn = app.register(host, "hgn", "/fixture/bundle");
        let baseline = app.register(host, "gsm8k-predictions", "/fixture/predictions");
        for (kind, inputs, options, prompt) in [
            (
                "compare",
                vec![&gguf_a, &gguf_b],
                json!({}),
                Some("Compare these models"),
            ),
            ("generate", vec![&hgn], json!({}), Some("Generate text")),
            ("baseline", vec![&baseline], json!({}), None),
            ("evaluate", vec![&hgn], json!({}), None),
            (
                "benchmark",
                vec![],
                json!({"benchmark_target":"target","benchmarks":["suite"]}),
                None,
            ),
            ("transfer", vec![&gguf_a], json!({}), None),
            ("transfer_preview", vec![&gguf_a], json!({}), None),
        ] {
            matches_admission(
                &app,
                &row[kind],
                "/api/jobs",
                json!({
                    "kind":kind,"host":host,"input_ids":inputs,"options":options,"prompt":prompt
                }),
                enabled,
            );
            if host == "legacy" && ["generate", "evaluate", "baseline", "benchmark"].contains(&kind)
            {
                assert_eq!(
                    legacy["readiness"][kind]["ready"], row[kind]["eligible"],
                    "{legacy}"
                );
            }
        }
        matches_admission(
            &app,
            &row["managed_conversion"],
            "/api/conversions",
            json!({
                "host":host,"input_ids":[hf]
            }),
            enabled,
        );
        matches_admission(
            &app,
            &row["component_conversion"],
            "/api/conversions",
            json!({
                "host":host,"input_ids":[hf,assets],"options":{"assets":1}
            }),
            false,
        );
        matches_admission(
            &app,
            &row["gguf_output_by_kind"]["gguf"],
            "/api/conversions",
            json!({
                "format":"gguf","input_ids":[gguf_a]
            }),
            enabled,
        );
        matches_admission(
            &app,
            &row["gguf_output_by_kind"]["hf"],
            "/api/conversions",
            json!({
                "format":"gguf","input_ids":[hf]
            }),
            enabled,
        );
        let weights = app.register(host, "safetensors", "/fixture/weights.safetensors");
        matches_admission(
            &app,
            &row["gguf_output_by_kind"]["safetensors"],
            "/api/conversions",
            json!({"format":"gguf","input_ids":[weights,assets]}),
            false,
        );
        matches_admission(
            &app,
            &row["register_manual"],
            "/api/models",
            json!({
                "host":host,"kind":"gguf","path":"/fixture/manual.gguf"
            }),
            true,
        );
    }
    assert_eq!(legacy["ready"], true);
    matches_admission(
        &app,
        &eligibility["hosts"]["backend-hf"]["register_manual"],
        "/api/models",
        json!({
            "host":"backend-hf","kind":"gguf","path":"/fixture/manual.gguf"
        }),
        false,
    );
}

#[test]
fn conversion_eligibility_matches_selected_profile_inputs_and_pool() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"component_cpu_only":true,"quantization_pool":true,
                     "capabilities":{"custom_profiles":true,"convert_components":true}},
            "beta":{"local":true,"capabilities":{"custom_profiles":false}}
        }})
        .to_string(),
    )
    .unwrap();
    app.launch();
    let hf = app.register("beta", "hf", "/fixture/checkpoint");
    let assets = app.register("beta", "assets", "/fixture/assets");
    let gguf = app.register("beta", "gguf", "/fixture/source.gguf");
    let catalog = app.json("GET", "/api/profiles/catalog", None, 200);
    let tensor = catalog["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["editable"] == true)
        .unwrap()["name"]
        .as_str()
        .unwrap();
    let profile = app.json(
        "POST",
        "/api/profiles",
        Some(json!({
            "name":"Test profile","base":"v2","catalog_sha256":catalog["catalog_sha256"],
            "overrides":{(tensor):"ht43"}
        })),
        201,
    );
    let revision = &profile["revision"];
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(
        state["eligibility"]["hosts"]["alpha"]["pool_custom_profile_conversion"]["eligible"],
        true
    );
    assert_eq!(
        state["eligibility"]["hosts"]["beta"]["custom_profile_conversion"]["eligible"],
        false
    );
    for (body, expected) in [
        (
            json!({"host":"auto","input_ids":[hf,assets],"options":{"assets":1},"profile_revision":revision}),
            true,
        ),
        (
            json!({"host":"beta","input_ids":[hf],"profile_revision":revision}),
            false,
        ),
        (
            json!({"host":"auto","input_ids":[gguf],"profile_revision":revision}),
            false,
        ),
        (
            json!({"host":"auto","input_ids":[hf],"profile_revision":revision,"options":{"ht_backend":"hip"}}),
            false,
        ),
        (
            json!({"host":"auto","input_ids":[hf],"decode_backend":"cuda"}),
            false,
        ),
    ] {
        let before = app.json("GET", "/api/conversions", None, 200);
        let preview = app.json(
            "POST",
            "/api/conversions/eligibility",
            Some(body.clone()),
            200,
        );
        assert_eq!(app.json("GET", "/api/conversions", None, 200), before);
        matches_admission(&app, &preview, "/api/conversions", body, expected);
    }
}

#[test]
fn paused_hosts_block_job_eligibility_but_not_manual_registration() {
    let mut app = Fixture::new();
    app.launch();
    let model = app.register("alpha", "gguf", "/fixture/source.gguf");
    app.json(
        "POST",
        "/api/admin/actions",
        Some(
            json!({"action":"set_host_policy","worker":"alpha","revision":0,
            "paused":true,"disabled_jobs":[]}),
        ),
        200,
    );
    let state = app.json("GET", "/api/state", None, 200);
    let host = &state["eligibility"]["hosts"]["alpha"];
    matches_admission(
        &app,
        &host["transfer"],
        "/api/jobs",
        json!({"host":"alpha","kind":"transfer","input_ids":[model]}),
        false,
    );
    matches_admission(
        &app,
        &host["register_manual"],
        "/api/models",
        json!({"host":"alpha","kind":"gguf","path":"/fixture/manual.gguf"}),
        true,
    );
}
