use super::*;

#[test]
fn layer_recipe_requires_an_explicitly_capable_source_worker() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"capabilities":{"gguf_output":true,"gguf_hf_output":true}}
        }})
        .to_string(),
    )
    .unwrap();
    app.launch();
    let source = app.register("alpha", "hf", "/source/base");
    let policy = json!({"schema":"brewery.layer-quant.v1","source_sha256":"1".repeat(64),
                        "default_type":"BF16","layers":{"0":"F16"}});
    let request = json!({"format":"gguf","input_ids":[source],"options":{"layer_recipe":policy}});
    for field in ["preset", "imatrix", "threads"] {
        let mut wrong = request.clone();
        wrong["options"][field] = json!(1);
        app.json("POST", "/api/conversions", Some(wrong), 400);
    }
    let mut wrong = request.clone();
    wrong["format"] = json!("hgn");
    app.json("POST", "/api/conversions", Some(wrong), 400);
    app.json("POST", "/api/conversions", Some(request), 409);
    assert!(app.launches().is_empty());
}

#[test]
fn profile_export_options_are_preview_only_for_both_encoders() {
    let mut app = Fixture::new();
    app.launch();
    let catalog = app.json("GET", "/api/profiles/catalog", None, 200);
    let tensor = catalog["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["editable"] == true)
        .unwrap()["name"]
        .as_str()
        .unwrap();
    let hgn = app.json(
        "POST",
        "/api/profiles",
        Some(json!({
            "name":"HGN policy","base":"v2","catalog_sha256":catalog["catalog_sha256"],
            "overrides":{(tensor):"ht43"}
        })),
        201,
    );
    let gguf = app.json(
        "POST",
        "/api/profiles",
        Some(json!({
            "format":"gguf","name":"GGUF policy","preset":"Q4_K_M","overrides":{}
        })),
        201,
    );
    let before = app.json("GET", "/api/state", None, 200);
    for (encoder, profile) in [("hgn", &hgn), ("gguf", &gguf)] {
        let revision = profile["revision"].as_str().unwrap();
        let path = format!("/api/profiles/{revision}/export-options");
        let option = |format| {
            if format == encoder {
                json!({"format":format,"admission":"standard_workflow","stock_preservation":"by_design",
                "extension_requirement":"not_required","runtime_proof":"not_assessed"})
            } else {
                json!({"format":format,"admission":"unavailable","stock_preservation":"not_established",
                "extension_requirement":"not_established","experimental_extension":"brewery.cross-container.v1",
                "runtime_proof":"not_assessed"})
            }
        };
        assert_eq!(
            app.json("GET", &path, None, 200),
            json!({
                "revision":revision,"encoder":encoder,"options":[option("hgn"),option("gguf")]
            })
        );
        assert_eq!(
            app.request("GET", &path, None, "Origin: http://outside.test\r\n")
                .0,
            403
        );
    }
    app.json("GET", "/api/profiles/not-a-sha/export-options", None, 400);
    app.json(
        "GET",
        &format!("/api/profiles/{}/export-options", "0".repeat(64)),
        None,
        404,
    );
    assert_eq!(
        app.json("GET", "/api/profiles", None, 200),
        json!([hgn, gguf])
    );
    assert_eq!(app.json("GET", "/api/state", None, 200), before);
    assert_eq!(app.json("GET", "/api/conversions", None, 200), json!([]));
    assert!(app.launches().is_empty());
}

#[test]
fn gguf_workflow_accepts_direct_default_and_rejects_private_options() {
    let mut app = Fixture::new();
    app.launch();
    let source = app.register("alpha", "gguf", "/source/base.gguf");
    let request = json!({"format":"gguf","input_ids":[source]});
    app.json("POST", "/api/conversions", Some(request.clone()), 409);
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"capabilities":{"gguf_output":true}}
        }})
        .to_string(),
    )
    .unwrap();
    app.stop();
    app.launch();
    app.touch("conversion-fixture");
    let output = app.json("POST", "/api/conversions", Some(request), 201);
    assert_eq!(output["format"], "gguf");
    let done = await_conversion(&app, &output);
    assert_eq!(done["state"], "succeeded", "{done}");
    let verify = app.detail(done["job_ids"]["verify"].as_str().unwrap());
    assert_eq!(verify["result"]["gguf_type"], "Q4_K_M");
    let state = app.json("GET", "/api/state", None, 200);
    assert!(
        state["models"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == done["artifact_model_id"]
                && m["kind"] == "gguf"
                && m["path"].as_str().unwrap().ends_with("/converted.gguf"))
    );
    app.json(
        "POST",
        "/api/jobs",
        Some(json!({"kind":"convert","input_ids":[source],"options":{
            "output_format":"gguf","gguf_type":"Q4_K_M"
        }})),
        400,
    );
}

fn await_conversion(app: &Fixture, created: &Value) -> Value {
    let path = format!("/api/conversions/{}", created["id"].as_str().unwrap());
    wait_for(|| {
        let updated = app.json("GET", &path, None, 200);
        (updated["state"] != "running").then_some(updated)
    })
}

#[test]
fn gguf_rocmfpx_selection_requires_a_fork_worker_and_pins_mixed_profiles() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"capabilities":{"gguf_output":true}},
            "beta":{"local":true,"capabilities":{"gguf_output":true,"gguf_rocmfpx_output":true}}
        }})
        .to_string(),
    )
    .unwrap();
    app.launch();
    app.touch("conversion-fixture");
    let alpha = app.register("alpha", "gguf", "/source/base.gguf");
    let beta = app.register("beta", "gguf", "/source/base.gguf");
    let catalog = app.json("GET", "/api/profiles/gguf-catalog", None, 200);
    let fork_revision = "fb08d7cdb67082ddda15a25e64dc7ec413afc41c";
    assert_eq!(catalog["rocmfpx_encoder_revision"], fork_revision);
    for (name, id) in [
        ("Q4_0_ROCMFP4", 100),
        ("Q4_0_ROCMFP4_FAST", 101),
        ("Q6_0_ROCMFPX", 102),
        ("Q8_0_ROCMFPX", 103),
        ("Q3_0_ROCMFPX", 104),
    ] {
        assert!(
            catalog["presets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == name && row["encoder_family"] == "rocmfpx")
        );
        assert!(
            catalog["tensor_types"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == name
                    && row["type_id"] == id
                    && row["encoder_family"] == "rocmfpx")
        );
    }
    let profile = app.json(
        "POST",
        "/api/profiles",
        Some(json!({
            "format":"gguf","name":"Mixed fork recipe","preset":"Q8_0",
            "overrides":{"blk.0.attn_q.weight":"Q3_0_ROCMFPX","blk.0.attn_k.weight":"Q8_0"}
        })),
        201,
    );
    assert_eq!(profile["encoder_family"], "rocmfpx");
    assert_eq!(profile["encoder_revision"], fork_revision);
    for (preset, override_type) in [("Q2_0", "Q3_0_ROCMFPX"), ("Q4_0_ROCMFP4", "Q2_0")] {
        app.json(
            "POST",
            "/api/profiles",
            Some(json!({
                "format":"gguf","name":"Unsupported mix","preset":preset,
                "overrides":{"blk.0.attn_q.weight":override_type}
            })),
            400,
        );
    }
    let stock = app.json(
        "POST",
        "/api/profiles",
        Some(json!({
            "format":"gguf","name":"Stock recipe","preset":"Q2_0"
        })),
        201,
    );
    assert_eq!(
        stock["revision"],
        "d0401d7780b8d80f42355009651e0acfa8f3d3e09eed37d1d0c15348e056cfef"
    );
    assert!(stock.get("encoder_family").is_none());
    app.json("POST", "/api/conversions", Some(json!({
        "format":"gguf","input_ids":[beta],"options":{"preset":"Q8_0","encoder_family":"rocmfpx"}
    })), 400);
    app.json(
        "POST",
        "/api/conversions",
        Some(json!({
            "format":"gguf","input_ids":[alpha],"options":{"preset":"Q4_0_ROCMFP4_FAST"}
        })),
        409,
    );
    app.json(
        "POST",
        "/api/conversions",
        Some(json!({
            "format":"gguf","input_ids":[alpha],"profile_revision":profile["revision"]
        })),
        409,
    );
    let done = await_conversion(
        &app,
        &app.json(
            "POST",
            "/api/conversions",
            Some(json!({
                "format":"gguf","input_ids":[beta],"profile_revision":profile["revision"]
            })),
            201,
        ),
    );
    assert_eq!(done["state"], "succeeded", "{done}");
    for stage in ["inspect", "plan", "convert", "verify"] {
        let detail = app.detail(done["job_ids"][stage].as_str().unwrap());
        let options = &detail["result"]["envelope"]["options"];
        assert_eq!(options["encoder_family"], "rocmfpx");
        assert_eq!(options["encoder_revision"], fork_revision);
        assert_eq!(options["tensor_types"], profile["overrides"]);
    }
    app.touch("changed-rocmfpx-tool");
    let stale = await_conversion(
        &app,
        &app.json(
            "POST",
            "/api/conversions",
            Some(json!({
                "format":"gguf","input_ids":[beta],"options":{"preset":"Q4_0_ROCMFP4_FAST"}
            })),
            201,
        ),
    );
    assert_eq!(stale["state"], "failed", "{stale}");
    assert_eq!(stale["stage"], "plan", "{stale}");
    assert!(stale.get("artifact_model_id").is_none());
}

#[test]
fn standalone_gguf_workflow_uses_registered_assets_and_imatrix() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"capabilities":{
                "gguf_output":true,"gguf_hf_output":true,"gguf_safetensors_output":true
            }},
            "beta":{"local":true,"capabilities":{"gguf_output":true,"gguf_hf_output":true}}
        }})
        .to_string(),
    )
    .unwrap();
    app.launch();
    app.touch("conversion-fixture");
    let weights = app.register("alpha", "safetensors", "/weights/standalone.safetensors");
    let assets = app.register("alpha", "assets", "/metadata/assets");
    let matrix = app.register("alpha", "imatrix", "/calibration/matrix.dat");
    let remote_assets = app.register("beta", "assets", "/metadata/assets");
    let old_weights = app.register("beta", "safetensors", "/weights/standalone.safetensors");
    for (input_ids, options, status) in [
        (json!([weights]), json!({}), 400),
        (json!([weights, matrix]), json!({}), 400),
        (json!([weights, assets, assets]), json!({}), 400),
        (json!([weights, remote_assets]), json!({}), 400),
        (json!([weights, assets]), json!({"preset":"IQ2_XXS"}), 400),
        (
            json!([weights, assets]),
            json!({"assets":"/metadata/assets"}),
            400,
        ),
        (json!([old_weights, remote_assets]), json!({}), 409),
    ] {
        app.json(
            "POST",
            "/api/conversions",
            Some(json!({"format":"gguf","input_ids":input_ids,"options":options})),
            status,
        );
    }
    let created = app.json(
        "POST",
        "/api/conversions",
        Some(json!({"format":"gguf","input_ids":[weights,assets,matrix],"options":{"preset":"IQ2_XXS"}})),
        201,
    );
    let done = await_conversion(&app, &created);
    assert_eq!(done["state"], "succeeded", "{done}");
    for (stage, expected) in [
        ("inspect", json!(["safetensors", "assets", "imatrix"])),
        ("plan", json!(["safetensors", "assets", "imatrix"])),
        ("convert", json!(["safetensors", "assets", "imatrix"])),
        (
            "verify",
            json!(["safetensors", "gguf", "assets", "imatrix"]),
        ),
    ] {
        let detail = app.detail(done["job_ids"][stage].as_str().unwrap());
        let envelope = &detail["result"]["envelope"];
        let inputs = envelope["inputs"].as_array().unwrap();
        assert_eq!(
            json!(
                inputs
                    .iter()
                    .map(|input| &input["kind"])
                    .collect::<Vec<_>>()
            ),
            expected
        );
        let offset = usize::from(stage == "verify");
        assert_eq!(inputs[0]["id"], weights);
        assert_eq!(inputs[1 + offset]["id"], assets);
        assert_eq!(inputs[2 + offset]["id"], matrix);
        assert_eq!(envelope["options"]["imatrix"], 2 + offset);
    }
}

#[test]
fn gguf_saved_profiles_require_imatrix_and_do_not_register_failed_verifications() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"capabilities":{"gguf_output":true}}
        }})
        .to_string(),
    )
    .unwrap();
    app.launch();
    app.touch("conversion-fixture");
    app.touch("bad-verify");
    let source = app.register("alpha", "gguf", "/source/base.gguf");
    let matrix = app.register("alpha", "imatrix", "/source/matrix.dat");
    let hf = app.register("alpha", "hf", "/source/hf");
    app.json(
        "POST",
        "/api/conversions",
        Some(json!({"format":"gguf","input_ids":[hf]})),
        409,
    );
    let catalog = app.json("GET", "/api/profiles/gguf-catalog", None, 200);
    assert_eq!(catalog["format"], "gguf");
    let body = json!({"format":"gguf","name":"Matrix recipe","preset":"IQ2_XXS","overrides":{"blk.0.attn_q.weight":"Q8_0"}});
    let saved = app.json("POST", "/api/profiles", Some(body.clone()), 201);
    assert_eq!(saved["requires_imatrix"], true);
    assert_eq!(app.json("POST", "/api/profiles", Some(body), 200), saved);
    app.stop();
    app.launch();
    assert_eq!(app.json("GET", "/api/profiles", None, 200), json!([saved]));
    app.json(
        "POST",
        "/api/conversions",
        Some(json!({
            "format":"gguf","input_ids":[source],"profile_revision":saved["revision"]
        })),
        400,
    );
    let created = app.json(
        "POST",
        "/api/conversions",
        Some(json!({
            "format":"gguf","input_ids":[source,matrix],"profile_revision":saved["revision"]
        })),
        201,
    );
    let done = await_conversion(&app, &created);
    assert_eq!(done["state"], "failed");
    assert!(done.get("artifact_model_id").is_none());
    let verify = app.detail(done["job_ids"]["verify"].as_str().unwrap());
    assert_eq!(verify["result"]["envelope"]["options"]["imatrix"], 2);
    assert_eq!(
        verify["result"]["envelope"]["options"]["tensor_types"],
        saved["overrides"]
    );
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["models"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn saved_profiles_are_immutable_and_only_workflows_can_reference_them() {
    let mut app = Fixture::new();
    app.launch();
    let catalog = app.json("GET", "/api/profiles/catalog", None, 200);
    let tensor = catalog["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["editable"] == true)
        .unwrap();
    let name = tensor["name"].as_str().unwrap();
    let mut request = json!({"name":"Test profile","base":"v2","catalog_sha256":catalog["catalog_sha256"],
        "overrides":{(name): "ht43"}});
    let saved = app.json("POST", "/api/profiles", Some(request.clone()), 201);
    assert_eq!(saved["size"]["delta_bytes"], tensor["delta_bytes"]);
    let source = app.register("alpha", "hf", "/source/base");
    app.json(
        "POST",
        "/api/conversions",
        Some(json!({"input_ids":[source],"profile_revision":saved["revision"]})),
        409,
    );
    app.json(
        "POST",
        "/api/jobs",
        Some(json!({"kind":"convert","input_ids":[source],"options":{"profile_recipe":saved}})),
        400,
    );
    request["name"] = json!("Another profile");
    let second = app.json("POST", "/api/profiles", Some(request.clone()), 201);
    assert_ne!(second["revision"], saved["revision"]);
    app.stop();
    app.launch();
    assert_eq!(
        app.json("POST", "/api/profiles", Some(request.clone()), 200),
        second
    );
    assert_eq!(
        app.json("GET", "/api/profiles", None, 200),
        json!([saved, second])
    );
    request["overrides"] = json!({"unknown.tensor":"ht43"});
    app.json("POST", "/api/profiles", Some(request), 400);
}

#[test]
fn profile_workflow_snapshots_recipe_for_all_four_stages() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"capabilities":{"custom_profiles":true}}
        }})
        .to_string(),
    )
    .unwrap();
    app.launch();
    app.touch("conversion-fixture");
    app.touch("hold");
    let source = app.register("alpha", "hf", "/source/base");
    let catalog = app.json("GET", "/api/profiles/catalog", None, 200);
    let tensor = catalog["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["editable"] == true)
        .unwrap()["name"]
        .as_str()
        .unwrap();
    let request = json!({"name":"My policy","base":"v2","catalog_sha256":catalog["catalog_sha256"],
        "overrides":{(tensor):"ht43"}});
    let saved = app.json("POST", "/api/profiles", Some(request.clone()), 201);
    let workflow = app.json(
        "POST",
        "/api/conversions",
        Some(json!({
            "host":"alpha","input_ids":[source],"profile_revision":saved["revision"]
        })),
        201,
    );
    let inspect = workflow["job_ids"]["inspect"].as_str().unwrap();
    app.await_job(inspect, |job| job["progress"] == "worker progress");
    let mut edited = request;
    edited["base"] = json!("ht43");
    edited["overrides"][tensor] = json!("v2");
    let latest = app.json("POST", "/api/profiles", Some(edited), 201);
    assert_ne!(saved["revision"], latest["revision"]);
    app.stop();
    app.launch();
    app.remove("hold");
    let route = format!("/api/conversions/{}", workflow["id"].as_str().unwrap());
    let complete = wait_for(|| {
        let value = app.json("GET", &route, None, 200);
        (value["state"] != "running").then_some(value)
    });
    assert_eq!(complete["state"], "succeeded", "{complete}");
    assert_eq!(complete["profile_name"], "My policy");
    assert_eq!(complete["profile_revision"], saved["revision"]);
    for stage in ["inspect", "plan", "convert", "verify"] {
        let job = app.detail(complete["job_ids"][stage].as_str().unwrap());
        assert_eq!(job["profile_revision"], saved["revision"]);
        assert!(job["options"].get("profile_recipe").is_none());
        let options = &job["result"]["envelope"]["options"];
        assert_eq!(
            options["profile_recipe"],
            json!({
                "schema":1,"revision":saved["revision"],"name":saved["name"],
                "catalog_sha256":saved["catalog_sha256"],"architecture":catalog["architecture"],
                "base":"v2","overrides":saved["overrides"]
            })
        );
        if stage == "verify" {
            assert!(options.get("profile").is_none());
        } else {
            assert_eq!(options["profile"], "v2");
        }
    }
}

#[test]
fn hf_components_require_host_opt_in_for_manual_and_durable_jobs() {
    let mut app = Fixture::new();
    fs::write(app.path("dispatch.json"), json!({"hosts":{
        "alpha":{"local":true},
        "beta":{"local":true,"quantization_pool":true,"component_cpu_only":true,"capabilities":{"convert_components":true}}
    }}).to_string()).unwrap();
    app.launch();
    app.touch("conversion-fixture");
    let base = app.register("alpha", "hf", "/source/base");
    let vision = app.register("beta", "vision", "/side/vision.safetensors");
    let options = json!({"vision":1});
    app.json(
        "POST",
        "/api/conversions",
        Some(json!({
            "host":"beta","input_ids":[base,vision],"options":options,"decode_backend":"hip"
        })),
        409,
    );
    app.json(
        "POST",
        "/api/jobs",
        Some(json!({
            "kind":"inspect","host":"beta","input_ids":[base],"options":{"ht_backend":"cuda"}
        })),
        409,
    );
    for route in ["/api/jobs", "/api/conversions"] {
        let mut body = json!({"host":"alpha","input_ids":[base,vision],"options":options});
        if route == "/api/jobs" {
            body["kind"] = json!("inspect");
        }
        app.json("POST", route, Some(body), 409);
    }
    let job = app.enqueue(
        json!({"kind":"inspect","host":"beta","input_ids":[base,vision],"options":options}),
    );
    app.await_job(&job, |job| job["state"] == "succeeded");
    let row = app.json(
        "POST",
        "/api/conversions",
        Some(json!({
            "host":"auto","input_ids":[base,vision],"options":options,"decode_backend":"cpu"
        })),
        201,
    );
    assert_eq!(row["host"], "beta");
    let path = format!("/api/conversions/{}", row["id"].as_str().unwrap());
    let completed = wait_for(|| {
        let row = app.json("GET", &path, None, 200);
        (row["state"] != "running").then_some(row)
    });
    assert_eq!(completed["state"], "succeeded", "{completed}");
    let verify = app.detail(completed["job_ids"]["verify"].as_str().unwrap());
    assert_eq!(verify["host"], "beta");
    assert_eq!(
        verify["options"],
        json!({"vision":2,"decode_backend":"cpu"})
    );
}

#[test]
fn conversion_submission_is_durable_across_restart() {
    let mut app = Fixture::new();
    app.launch();
    let source = app.register("alpha", "gguf", "/source/weights.gguf");
    let assets = app.register("alpha", "assets", "/source/assets");
    app.touch("conversion-fixture");
    app.touch("hold");
    let body = json!({"host":"auto","input_ids":[source,assets],"options":{"profile":"ht43","assets":1,"seed":3},"decode_backend":"cpu"});
    let send = |app: &Fixture, body: &Value, expected| {
        let (status, result) = app.request(
            "POST",
            "/api/conversions",
            Some(body.clone()),
            "Idempotency-Key: workflow-1\r\n",
        );
        assert_eq!(status, expected, "{result}");
        serde_json::from_str::<Value>(&result).unwrap()
    };
    let first = send(&app, &body, 201);
    let route = format!("/api/conversions/{}", first["id"].as_str().unwrap());
    let inspect = first["job_ids"]["inspect"].as_str().unwrap();
    app.await_job(inspect, |job| job["progress"] == "worker progress");
    app.stop();
    app.launch();
    let replay = send(&app, &body, 200);
    assert_eq!(replay["job_ids"]["inspect"], inspect);
    assert_eq!(replay["host"], "alpha");
    let mut changed = body.clone();
    changed["decode_backend"] = json!("cuda");
    send(&app, &changed, 409);
    app.remove("hold");
    let complete = wait_for(|| {
        let row = app.json("GET", &route, None, 200);
        match row["state"].as_str().unwrap() {
            "succeeded" => Some(row),
            "running" => None,
            _ => panic!("{row}"),
        }
    });
    assert_eq!(complete["stage"], "verify");
    assert_eq!(app.launches().len(), 4);
    assert_eq!(
        app.json("GET", "/api/conversions", None, 200),
        json!([complete])
    );
    assert!(complete["artifact_model_id"].is_string());
    let verify = app.detail(complete["job_ids"]["verify"].as_str().unwrap());
    assert_eq!(
        verify["input_ids"],
        json!([source, complete["artifact_model_id"], assets])
    );
    assert_eq!(
        verify["options"],
        json!({"assets":2,"decode_backend":"cpu"})
    );
}

#[test]
fn conversion_stops_on_cancel_or_bad_receipt_without_extra_jobs() {
    for (failure, expected, child_count) in [
        ("hold", "cancelled", 1),
        ("no-fit-inspect", "failed", 1),
        ("no-fit-plan", "failed", 2),
        ("bad-path", "failed", 3),
        ("bad-verify", "failed", 4),
    ] {
        let mut app = Fixture::new();
        app.launch();
        let source = app.register("alpha", "gguf", "/source/weights.gguf");
        app.touch("conversion-fixture");
        app.touch(failure);
        let result = app.json(
            "POST",
            "/api/conversions",
            Some(json!({"input_ids":[source]})),
            201,
        );
        let route = format!("/api/conversions/{}", result["id"].as_str().unwrap());
        if failure == "hold" {
            app.await_job(result["job_ids"]["inspect"].as_str().unwrap(), |job| {
                job["progress"] == "worker progress"
            });
            let cancel = app.json("POST", &format!("{route}/cancel"), Some(json!({})), 200);
            assert_eq!(cancel["cancel_requested"], true);
        }
        let terminal = wait_for(|| {
            let row = app.json("GET", &route, None, 200);
            (row["state"] != "running").then_some(row)
        });
        assert_eq!(terminal["state"], expected, "{terminal}");
        assert_eq!(terminal["job_ids"].as_object().unwrap().len(), child_count);
        assert_eq!(
            app.json("GET", "/api/state", None, 200)["jobs"]
                .as_array()
                .unwrap()
                .len(),
            child_count
        );
        if failure != "bad-verify" {
            assert!(terminal.get("artifact_model_id").is_none());
        }
    }
}
