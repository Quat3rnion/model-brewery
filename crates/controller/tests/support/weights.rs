use super::*;

fn configure_safetensors(app: &Fixture, enabled: bool) {
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"capabilities":{
                "weight_inspect":true,"weight_propose":true,"weight_tune":true,
                "weight_apply":true,"weight_visual":true,"weight_packed":true,
                "weight_safetensors":enabled
            }},
            "beta":{"local":true,"capabilities":{"weight_inspect":true,"weight_safetensors":true}}
        }})
        .to_string(),
    )
    .unwrap();
}

#[test]
fn safetensors_sources_require_their_own_worker_opt_in() {
    let mut app = Fixture::new();
    configure_safetensors(&app, false);
    app.touch("weight-fixture");
    app.launch();
    let files = [
        app.register("alpha", "safetensors", "/models/model.safetensors"),
        app.register("alpha", "hf", "/models/checkpoint"),
    ];
    let inspect = |source: &str| {
        json!({"kind":"weight_inspect","input_ids":[source],
        "options":{"mode":"inventory","offset":0,"limit":32}})
    };
    for source in &files {
        app.json("POST", "/api/jobs", Some(inspect(source)), 409);
    }
    app.stop();
    configure_safetensors(&app, true);
    app.launch();
    for source in &files {
        let mut invalid = inspect(source);
        invalid["options"]["packed"] = json!(true);
        app.json("POST", "/api/jobs", Some(invalid), 400);
        let mut other_worker = inspect(source);
        other_worker["host"] = json!("beta");
        app.json("POST", "/api/jobs", Some(other_worker), 400);
        let job = app.enqueue(inspect(source));
        app.await_job(&job, |row| row["state"] == "succeeded");
    }
}

#[test]
fn safetensors_apply_registers_same_kind_only_after_exact_proposal_and_manifest_checks() {
    let mut app = Fixture::new();
    configure_safetensors(&app, true);
    app.touch("weight-fixture");
    app.launch();
    let models = || app.json("GET", "/api/state", None, 200)["models"].clone();
    for (kind, path) in [
        ("safetensors", "/models/model.safetensors"),
        ("hf", "/models/hf"),
    ] {
        let source = app.register("alpha", kind, path);
        let snapshot = models();
        let proposal = app.enqueue(json!({"kind":"weight_propose","input_ids":[source],
            "options":{"edits":[{"tensor":"blk.0.weight","index":[0,0],"op":"set","value":1.5}]}}));
        app.await_job(&proposal, |row| row["state"] == "succeeded");
        let mut apply = json!({"kind":"weight_apply","input_ids":[source],
            "options":{"proposal_job_id":proposal,"reviewed":false}});
        app.json("POST", "/api/jobs", Some(apply.clone()), 400);
        apply["options"]["reviewed"] = json!(true);
        let mut faults = vec!["wrong-weight-output-kind", "wrong-approved-weight-digest"];
        if kind == "hf" {
            faults.extend(["wrong-hf-manifest-hash", "missing-hf-member"]);
        }
        for fault in faults {
            app.touch(fault);
            let job = app.enqueue(apply.clone());
            let terminal = app.await_job(&job, |row| {
                matches!(row["state"].as_str(), Some("failed" | "succeeded"))
            });
            assert_eq!(terminal["state"], "failed", "{kind}: {fault}");
            assert!(terminal.get("result").is_none());
            assert_eq!(models(), snapshot);
            app.remove(fault);
        }
        let job = app.enqueue(apply);
        let terminal = app.await_job(&job, |row| {
            matches!(row["state"].as_str(), Some("failed" | "succeeded"))
        });
        assert_eq!(terminal["state"], "succeeded", "{kind}: {terminal}");
        let library = models();
        let models = library.as_array().unwrap();
        assert_eq!(models.len(), snapshot.as_array().unwrap().len() + 1);
        assert!(models.iter().any(|m| {
            m["id"] == terminal["result"]["library_model_id"]
                && m["kind"] == kind
                && m["path"] == format!("/worker/jobs/{job}/work/edited.{kind}")
        }));
    }
}

#[test]
fn packed_weights_need_capability_and_a_worker_receipt() {
    let mut app = Fixture::new();
    configure_hgn_weights(&app, true, true);
    app.touch("weight-fixture");
    app.launch();
    let source = app.register("alpha", "gguf", "/models/packed.gguf");
    let request = json!({"kind":"weight_inspect","input_ids":[source],
        "options":{"tensor":"blk.0.weight","index":[0,0],"packed":true}});
    app.json("POST", "/api/jobs", Some(request.clone()), 409);
    app.stop();
    enable_packed_weights(&app);
    app.launch();
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["hosts"][0]["capabilities"]["weight_packed"],
        true
    );
    let mut invalid = request.clone();
    invalid["options"]["packed"] = json!(false);
    app.json("POST", "/api/jobs", Some(invalid), 400);

    let legacy = app.enqueue(request);
    let terminal = app.await_job(&legacy, |job| {
        matches!(
            job["state"].as_str(),
            Some("failed" | "succeeded" | "cancelled")
        )
    });
    assert_eq!(terminal["state"], "failed");
    assert!(terminal.get("result").is_none());
}

fn enable_packed_weights(app: &Fixture) {
    let path = app.path("dispatch.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    config["hosts"]["alpha"]["capabilities"]["weight_packed"] = json!(true);
    fs::write(path, config.to_string()).unwrap();
}

#[test]
fn packed_apply_requires_the_exact_approved_receipt_and_supported_source_codec() {
    let mut app = Fixture::new();
    configure_hgn_weights(&app, true, true);
    enable_packed_weights(&app);
    app.touch("weight-fixture");
    app.touch("packed-weight-receipts");
    app.launch();
    let models = || app.json("GET", "/api/state", None, 200)["models"].clone();
    let mut applies = Vec::new();
    for (kind, dtype) in [("gguf", "Q4_0"), ("gguf", "Q8_0"), ("hgn", "Q8G64")] {
        if dtype == "Q8_0" {
            app.touch("weight-q8");
        }
        let source = app.register("alpha", kind, &format!("/models/{dtype}.{kind}"));
        let options = json!({"packed":true,"source_sha256":"a".repeat(64),"edits":[
            {"tensor":"blk.0.weight","index":[0,0],"op":"set","value":1.5}
        ]});
        let propose = json!({"kind":"weight_propose","input_ids":[source],"options":options});
        let original = models();
        app.touch("wrong-weight-codec");
        let id = app.enqueue(propose.clone());
        let failed = app.await_job(&id, |job| {
            matches!(job["state"].as_str(), Some("failed" | "succeeded"))
        });
        assert_eq!(failed["state"], "failed");
        assert!(failed.get("result").is_none());
        app.remove("wrong-weight-codec");

        let proposal = app.enqueue(propose);
        let done = app.await_job(&proposal, |job| job["state"] == "succeeded");
        assert_eq!(
            done["result"]["report"]["proposal"]["edits"][0]["dtype"],
            dtype
        );
        let mut apply = json!({"kind":"weight_apply","input_ids":[source],"options":{
            "proposal_job_id":proposal,"reviewed":true,"packed":true
        }});
        app.json("POST", "/api/jobs", Some(apply.clone()), 400);
        apply["options"].as_object_mut().unwrap().remove("packed");
        app.touch("wrong-approved-weight-digest");
        let id = app.enqueue(apply.clone());
        let failed = app.await_job(&id, |job| {
            matches!(job["state"].as_str(), Some("failed" | "succeeded"))
        });
        assert_eq!(failed["state"], "failed");
        assert!(failed.get("result").is_none());
        assert_eq!(models(), original);
        app.remove("wrong-approved-weight-digest");

        let id = app.enqueue(apply.clone());
        applies.push(apply);
        let done = app.await_job(&id, |job| job["state"] == "succeeded");
        assert_eq!(done["result"]["packed"], true);
        assert_eq!(done["result"]["proposal_sha256"], "b".repeat(64));
        assert!(
            models()
                .as_array()
                .unwrap()
                .iter()
                .any(|m| { m["id"] == done["result"]["library_model_id"] && m["kind"] == kind })
        );
    }
    app.stop();
    configure_hgn_weights(&app, true, true);
    app.launch();
    for apply in applies {
        app.json("POST", "/api/jobs", Some(apply), 409);
    }
}

fn configure_hgn_weights(app: &Fixture, hgn: bool, visual: bool) {
    let mut capabilities = json!({
        "weight_inspect":true,"weight_propose":true,"weight_tune":true,"weight_apply":true,
        "weight_visual":visual
    });
    if hgn {
        capabilities["weight_hgn"] = json!(true);
    }
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{"alpha":{"local":true,"capabilities":capabilities}}}).to_string(),
    )
    .unwrap();
}

#[test]
fn hgn_weight_jobs_require_separate_host_opt_in() {
    let mut app = Fixture::new();
    configure_hgn_weights(&app, false, true);
    app.touch("weight-fixture");
    app.launch();
    let source = app.register("alpha", "hgn", "/models/source.hgn");
    let calibration = app.register("alpha", "weight-calibration", "/models/calibration.json");
    let held_out = app.register("alpha", "weight-calibration", "/models/held-out.json");
    let scalar = json!({"kind":"weight_inspect","input_ids":[source],
        "options":{"tensor":"blk.0.weight","index":[0,0]}});
    let inventory = json!({"kind":"weight_inspect","input_ids":[source],
        "options":{"mode":"inventory","offset":0,"limit":64}});
    let propose = json!({"kind":"weight_propose","input_ids":[source],"options":{"edits":[
        {"tensor":"blk.0.weight","index":[0,0],"op":"delta","value":0.5}
    ]}});
    let tune = json!({"kind":"weight_tune","input_ids":[source,calibration,held_out],
        "options":{"tensor":"blk.0.weight","indices":[[0,0]],"max_update":0.5}});
    let state = || app.json("GET", "/api/state", None, 200);
    assert_eq!(state()["hosts"][0]["capabilities"]["weight_hgn"], false);
    for request in [&scalar, &inventory, &propose, &tune] {
        app.json("POST", "/api/jobs", Some(request.clone()), 409);
    }

    app.stop();
    configure_hgn_weights(&app, true, false);
    app.launch();
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["hosts"][0]["capabilities"]["weight_hgn"],
        true
    );
    app.json("POST", "/api/jobs", Some(inventory.clone()), 409);
    for alias in ["brewery-bundle", "hgn-v2", "ht43"] {
        let mut request = scalar.clone();
        request["input_ids"] = json!([app.register("alpha", alias, "/models/unsupported.hgn")]);
        app.json("POST", "/api/jobs", Some(request), 400);
    }

    app.stop();
    configure_hgn_weights(&app, true, true);
    app.launch();
    for request in [scalar, inventory, tune] {
        let id = app.enqueue(request);
        app.await_job(&id, |job| job["state"] == "succeeded");
    }
    let proposal = app.enqueue(propose);
    app.await_job(&proposal, |job| job["state"] == "succeeded");

    app.stop();
    configure_hgn_weights(&app, false, true);
    app.launch();
    app.json(
        "POST",
        "/api/jobs",
        Some(json!({"kind":"weight_apply","input_ids":[source],
            "options":{"proposal_job_id":proposal,"reviewed":true}})),
        409,
    );
}

#[test]
fn hgn_apply_registers_only_a_verified_reviewed_native_revision() {
    let mut app = Fixture::new();
    configure_hgn_weights(&app, true, false);
    app.touch("weight-fixture");
    app.launch();
    let source = app.register("alpha", "hgn", "/models/source.hgn");
    let proposed = app.enqueue(
        json!({"kind":"weight_propose","input_ids":[source],"options":{
            "edits":[{"tensor":"blk.0.weight","index":[0,0],"op":"delta","value":0.5}]
        }}),
    );
    app.await_job(&proposed, |job| job["state"] == "succeeded");
    let apply = |reviewed| {
        json!({"kind":"weight_apply","input_ids":[source],
        "options":{"proposal_job_id":proposed,"reviewed":reviewed}})
    };
    let models = || app.json("GET", "/api/state", None, 200)["models"].clone();
    let original = models();
    app.json("POST", "/api/jobs", Some(apply(false)), 400);

    for fault in [
        "wrong-weight-output-kind",
        "wrong-weight-output-leaf",
        "wrong-weight-source-format",
    ] {
        app.touch(fault);
        let id = app.enqueue(apply(true));
        let failed = app.await_job(&id, |job| {
            matches!(
                job["state"].as_str(),
                Some("failed" | "succeeded" | "cancelled")
            )
        });
        assert_eq!(failed["state"], "failed", "{fault}: {failed}");
        assert!(failed.get("result").is_none(), "{fault}");
        assert_eq!(models(), original, "{fault}");
        app.remove(fault);
    }

    app.touch("hold");
    app.touch("weight-apply-races-cancel");
    let id = app.enqueue(apply(true));
    app.await_job(&id, |job| job["progress"] == "worker progress");
    assert_eq!(app.cancel(&id)["cancel_requested"], true);
    let cancelled = app.await_job(&id, |job| job["state"] == "cancelled");
    assert!(cancelled.get("result").is_none());
    assert_eq!(models(), original);
    app.remove("hold");

    let id = app.enqueue(apply(true));
    let done = app.await_job(&id, |job| job["state"] == "succeeded");
    assert_eq!(done["result"]["source_format"], "hgn");
    assert_eq!(done["result"]["artifact"]["kind"], "hgn");
    let library = models();
    let library = library.as_array().unwrap();
    assert_eq!(library.len(), 2);
    assert!(library.contains(&original[0]));
    let revision = library
        .iter()
        .find(|model| model["id"] == done["result"]["library_model_id"])
        .unwrap();
    assert_eq!(revision["host"], "alpha");
    assert_eq!(revision["kind"], "hgn");
    assert_eq!(
        revision["path"],
        format!("/worker/jobs/{id}/work/edited.hgn")
    );
}

#[test]
fn visual_reads_require_capability_bounds_and_a_current_source_fingerprint() {
    let mut app = Fixture::new();
    let configure = |app: &Fixture, visual| {
        fs::write(
            app.path("dispatch.json"),
            json!({"hosts":{"alpha":{"local":true,"capabilities":{
                "weight_inspect":true,"weight_propose":true,"weight_visual":visual
            }}}})
            .to_string(),
        )
        .unwrap();
    };
    configure(&app, false);
    app.touch("weight-fixture");
    app.launch();
    let source = app.register("alpha", "gguf", "/models/source.gguf");
    let inspect = |options| json!({"kind":"weight_inspect","input_ids":[source],"options":options});
    let inventory = inspect(json!({"mode":"inventory","offset":0,"limit":64}));
    app.json("POST", "/api/jobs", Some(inventory.clone()), 409);
    let legacy = app.enqueue(inspect(json!({"tensor":"blk.0.weight","index":[0,0]})));
    app.await_job(&legacy, |job| job["state"] == "succeeded");
    app.stop();
    configure(&app, true);
    app.launch();
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["hosts"][0]["capabilities"]["weight_visual"], true);
    let catalog = app.enqueue(inventory);
    let catalog = app.await_job(&catalog, |job| job["state"] == "succeeded");
    let digest = &catalog["result"]["report"]["source_sha256"];
    assert_eq!(digest, &json!("a".repeat(64)));
    for options in [
        json!({"mode":"inventory","offset":0,"limit":129}),
        json!({"mode":"inventory","offset":0,"limit":16,"path":"/bad"}),
        json!({"mode":"tile","tensor":"blk.0.weight","origin":[0,0],"shape":[1,1]}),
        json!({"mode":"tile","tensor":"blk.0.weight","origin":[0,0],"shape":[33,1],"source_sha256":digest}),
    ] {
        app.json("POST", "/api/jobs", Some(inspect(options)), 400);
    }
    let tile = app.enqueue(inspect(json!({
        "mode":"tile","tensor":"blk.0.weight","origin":[0,0],"shape":[1,1],"source_sha256":digest
    })));
    let tile = app.await_job(&tile, |job| job["state"] == "succeeded");
    assert_eq!(tile["result"]["report"]["values"], json!([[1.0]]));

    app.touch("changed-weight-source");
    let stale = app.enqueue(json!({"kind":"weight_propose","input_ids":[source],"options":{
        "source_sha256":digest,"edits":[{"tensor":"blk.0.weight","index":[0,0],"op":"set","value":2.0}]
    }}));
    let failed = app.await_job(&stale, |job| job["state"] == "failed");
    assert!(failed.get("result").is_none());
    assert!(failed["error"].as_str().unwrap().contains("source"));
}

#[test]
fn weight_jobs_require_opt_in_and_reviewed_same_source() {
    let mut app = Fixture::new();
    app.launch();
    let source = app.register("alpha", "gguf", "/models/source.gguf");
    let inspect = json!({"kind":"weight_inspect","input_ids":[source],"options":{"tensor":"blk.0.weight","index":[0,0]}});
    app.json("POST", "/api/jobs", Some(inspect.clone()), 409);
    fs::write(app.path("dispatch.json"), json!({"hosts":{
        "alpha":{"local":true,"capabilities":{"weight_inspect":true,"weight_propose":true,"weight_tune":true,"weight_apply":true}},
        "beta":{"local":true}
    }}).to_string()).unwrap();
    app.stop();
    app.touch("weight-fixture");
    app.launch();
    let inspected = app.enqueue(inspect);
    app.await_job(&inspected, |job| job["state"] == "succeeded");
    let proposed = app.enqueue(
        json!({"kind":"weight_propose","input_ids":[source],"options":{"edits":[
            {"tensor":"blk.0.weight","index":[0,0],"op":"delta","value":0.5}
        ]}}),
    );
    app.await_job(&proposed, |job| job["state"] == "succeeded");
    app.stop();
    app.launch();
    let other = app.register("alpha", "gguf", "/models/other.gguf");
    for (id, proposal, reviewed) in [
        (&source, &proposed, false),
        (&other, &proposed, true),
        (&source, &inspected, true),
    ] {
        app.json(
            "POST",
            "/api/jobs",
            Some(json!({"kind":"weight_apply","input_ids":[id],
            "options":{"proposal_job_id":proposal,"reviewed":reviewed}})),
            400,
        );
    }
    let apply = json!({"kind":"weight_apply","input_ids":[source],
        "options":{"proposal_job_id":proposed,"reviewed":true}});
    app.touch("bad-weight-verify");
    let invalid = app.enqueue(apply.clone());
    app.await_job(&invalid, |job| job["state"] == "failed");
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["models"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    app.remove("bad-weight-verify");
    let applied = app.enqueue(apply);
    let done = app.await_job(&applied, |job| job["state"] == "succeeded");
    let state = app.json("GET", "/api/state", None, 200);
    assert!(state["models"].as_array().unwrap().iter().any(|model| {
        model["kind"] == "gguf"
            && model["id"] == done["result"]["library_model_id"]
            && model["path"]
                .as_str()
                .unwrap()
                .ends_with(&format!("/jobs/{applied}/work/edited.gguf"))
    }));
}
