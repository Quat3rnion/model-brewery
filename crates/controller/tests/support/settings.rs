use super::*;

pub(super) fn target_profile(row: &Value, port: u16) -> Value {
    json!({"revision":row["revision"],"profile_binding":row["profile_binding"],
        "benchmark_targets":[{"id":"desk","endpoint":format!("http://[::1]:{port}/v1"),
        "model":"demo"}],"enabled_suites":[],"baseline_endpoints":[],
        "evaluation_enabled":false})
}

#[test]
fn profile_only_benchmark_target_survives_restart_without_job_admission() {
    let mut app = Fixture::new();
    app.launch();
    let get = app.json("GET", "/api/settings", None, 200);
    let alpha = &get["hosts"][0];
    assert_eq!(alpha["evaluation"]["status"], "profile_only");
    let payload = target_profile(alpha, 9280);
    let mut invalid = payload.clone();
    invalid["benchmark_targets"][0]["endpoint"] = json!("http://192.0.2.1:9280/v1");
    app.json("PUT", "/api/settings/alpha", Some(invalid), 400);
    let mut invalid = payload.clone();
    invalid["enabled_suites"] = json!(["suite"]);
    app.json("PUT", "/api/settings/alpha", Some(invalid), 400);
    let mut invalid = payload.clone();
    invalid["evaluation_enabled"] = json!(true);
    app.json("PUT", "/api/settings/alpha", Some(invalid), 400);
    let saved = app.json("PUT", "/api/settings/alpha", Some(payload.clone()), 200);
    assert_eq!(saved["revision"], 1);
    assert_eq!(saved["benchmark_targets"], payload["benchmark_targets"]);
    assert_eq!(saved["approved_ports"], json!([]));
    app.json("PUT", "/api/settings/alpha", Some(payload), 409);
    app.stop();
    app.launch();
    assert_eq!(
        app.json("GET", "/api/settings", None, 200)["hosts"][0],
        saved
    );
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["hosts"][0]["capabilities"]["benchmark"], false);
    assert_eq!(state["hosts"][0]["benchmark_targets"], json!([]));
}

#[test]
fn backend_and_legacy_hosts_have_management_without_evaluation_enrollment() {
    let mut app = Fixture::new();
    let companion = app.path("dispatch");
    let env = [
        ("BREWERY_HF_WORKER", companion.clone()),
        ("BREWERY_HF_ENGINE", companion.clone()),
        ("BREWERY_HF_HFR", companion.clone()),
        ("BREWERY_HF_DISPATCH", companion),
    ];
    app.launch_with(&env);
    let settings = app.json("GET", "/api/settings", None, 200);
    let hosts = settings["hosts"].as_array().unwrap();
    assert_eq!(hosts.len(), 3);
    for (index, (id, reason, job)) in [
        ("alpha", "managed evaluation settings", "inspect"),
        ("beta", "managed evaluation settings", "evaluate"),
        ("backend-hf", "does not run evaluation", "hf_download"),
    ]
    .into_iter()
    .enumerate()
    {
        let row = hosts.iter().find(|row| row["id"] == id).unwrap();
        assert_eq!(row["evaluation"]["status"], "profile_only");
        assert!(
            row["evaluation"]["reason"]
                .as_str()
                .unwrap()
                .contains(reason)
        );
        assert_eq!(row["evaluation_allowed"], false);
        assert_eq!(row["evaluation_enabled"], false);
        assert_eq!(row["management"]["available"], true);
        assert_eq!(
            row["management"]["href"],
            format!("#/settings/infrastructure?host={id}")
        );
        assert_eq!(
            row["management"]["actions"],
            json!(["set_host_policy", "set_host_resources"])
        );
        let state_url = row["management"]["state_url"].as_str().unwrap();
        let action_url = row["management"]["action_url"].as_str().unwrap();
        let state = app.json("GET", state_url, None, 200);
        let policy = state["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == id)
            .unwrap();
        let change = json!({"action":"set_host_policy","worker":id,
            "revision":policy["revision"],"paused":false,"disabled_jobs":[job]});
        app.json("POST", action_url, Some(change.clone()), 200);
        app.json("POST", action_url, Some(change), 409);
        let payload = target_profile(row, 9000 + index as u16);
        let saved = app.json(
            "PUT",
            &format!("/api/settings/{id}"),
            Some(payload.clone()),
            200,
        );
        assert_eq!(saved["benchmark_targets"], payload["benchmark_targets"]);
    }
    let state = app.json("GET", "/api/state", None, 200);
    let legacy = state["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == "beta")
        .unwrap();
    assert_eq!(legacy["capabilities"]["evaluate"], false);
    assert_eq!(legacy["capabilities"]["generate"], true);
    app.stop();
    app.launch_with(&env);
    let settings = app.json("GET", "/api/settings", None, 200);
    for (index, id) in ["alpha", "beta", "backend-hf"].into_iter().enumerate() {
        let row = settings["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == id)
            .unwrap();
        assert_eq!(row["revision"], 1);
        assert_eq!(
            row["benchmark_targets"],
            target_profile(row, 9000 + index as u16)["benchmark_targets"]
        );
    }
}

#[test]
fn managed_settings_persist_and_gate_job_admission() {
    let mut app = Fixture::new();
    let worker_root = app.path("worker");
    fs::create_dir(&worker_root).unwrap();
    fs::set_permissions(&worker_root, fs::Permissions::from_mode(0o700)).unwrap();
    let catalog = app.path("catalog.json");
    fs::write(
        &catalog,
        json!({"adapters":{"suite":{"executable":"/bin/true","version":"1","protocol":"test"}}})
            .to_string(),
    )
    .unwrap();
    let worker_config = app.path("worker.json");
    fs::write(&worker_config, json!({"root":worker_root,"managed_settings":{"catalog":catalog,"allowed_ports":[9101],"evaluation_allowed":false}}).to_string()).unwrap();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"config":worker_config,"managed_settings":true},
            "beta":{"local":true}
        }})
        .to_string(),
    )
    .unwrap();
    app.launch();
    let initial = app.json("GET", "/api/settings", None, 200);
    let alpha = &initial["hosts"][0];
    assert_eq!(alpha["ready"], false);
    assert_eq!(alpha["readiness"]["benchmark"]["ready"], false);
    assert_eq!(
        alpha["readiness"]["benchmark"]["reason"],
        json!({
            "text":"Add a benchmark target in Settings > Hosts > alpha > Evaluation.",
            "fix_anchor":"#/settings/evaluation?host=alpha"
        })
    );
    assert_eq!(
        alpha["evaluation"]["reason_detail"],
        alpha["readiness"]["evaluate"]["reason"]
    );
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["hosts"][0]["capability_reasons"]["benchmark"],
        alpha["readiness"]["benchmark"]["reason"]
    );
    let new = json!({"revision":0,"benchmark_targets":[{"id":"saved","endpoint":"http://127.0.0.1:9101/v1","model":"fixture"}],
        "enabled_suites":["suite"],"baseline_endpoints":[],"evaluation_enabled":false});
    let saved = app.json("PUT", "/api/settings/alpha", Some(new.clone()), 200);
    assert_eq!(saved["revision"], 1);
    assert_eq!(saved["evaluation"]["status"], "managed");
    assert_eq!(saved["ready"], true);
    assert_eq!(saved["readiness"]["benchmark"]["ready"], true);
    assert_eq!(saved["evaluation_allowed"], false);
    assert!(
        saved["evaluation"]["reason"]
            .as_str()
            .unwrap()
            .contains("worker policy")
    );
    assert_eq!(saved["management"]["available"], true);
    assert_eq!(saved["approved_ports"], json!([9101]));
    let management = app.json(
        "POST",
        saved["management"]["action_url"].as_str().unwrap(),
        Some(
            json!({"action":"set_host_policy","worker":"alpha","revision":0,
            "paused":false,"disabled_jobs":["inspect"]}),
        ),
        200,
    );
    assert_eq!(management["hosts"][0]["disabled_jobs"], json!(["inspect"]));
    app.json("PUT", "/api/settings/alpha", Some(new.clone()), 409);
    app.json("PUT", "/api/settings/beta", Some(new), 409);
    app.stop();
    app.launch();
    assert_eq!(
        app.json("GET", "/api/settings", None, 200)["hosts"][0],
        saved
    );
    app.touch("hold");
    let job = app.enqueue(json!({"kind":"benchmark","host":"alpha","options":{"benchmark_target":"saved","benchmarks":["suite"]}}));
    app.await_job(&job, |j| j["progress"] == "worker progress");
    let mut edited = saved;
    edited.as_object_mut().unwrap().retain(|k, _| {
        [
            "revision",
            "benchmark_targets",
            "enabled_suites",
            "baseline_endpoints",
            "evaluation_enabled",
        ]
        .contains(&k.as_str())
    });
    app.json("PUT", "/api/settings/alpha", Some(edited.clone()), 409);
    app.cancel(&job);
    app.await_job(&job, |j| j["state"] == "cancelled");
    edited["evaluation_enabled"] = json!(true);
    app.json("PUT", "/api/settings/alpha", Some(edited.clone()), 400);
    edited["evaluation_enabled"] = json!(false);
    edited["benchmark_targets"][0]["endpoint"] = json!("http://127.0.0.1:9102/v1");
    app.json("PUT", "/api/settings/alpha", Some(edited.clone()), 400);
    edited["benchmark_targets"][0]["endpoint"] = json!("http://localhost:9101/v1");
    app.json("PUT", "/api/settings/alpha", Some(edited), 400);
    assert_eq!(
        app.json("GET", "/api/settings", None, 200)["hosts"][0]["revision"],
        1
    );
    fs::remove_file(catalog).unwrap();
    let settings = app.json("GET", "/api/settings", None, 200);
    let alpha = &settings["hosts"][0];
    assert_eq!(alpha["evaluation"]["status"], "profile_only");
    assert!(
        alpha["evaluation"]["reason"]
            .as_str()
            .unwrap()
            .contains("suite catalog")
    );
    assert_eq!(alpha["management"]["available"], true);
    let draft = target_profile(alpha, 9290);
    let saved_draft = app.json("PUT", "/api/settings/alpha", Some(draft.clone()), 200);
    assert_eq!(saved_draft["benchmark_targets"], draft["benchmark_targets"]);
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["hosts"][0]["capabilities"]["benchmark"], false);
}

#[test]
fn source_pinned_worker_saves_a_separate_profile_without_mutating_worker_files() {
    let mut app = Fixture::new();
    let root = app.path("worker");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let catalog = app.path("catalog.json");
    fs::write(&catalog, r#"{"adapters":{}}"#).unwrap();
    let config = app.path("worker.json");
    fs::write(
        &config,
        json!({"root":root,"managed_settings":{"catalog":catalog,
            "allowed_ports":[9101],"evaluation_allowed":false},
            "engine_env":{"BREWERY_EXPECTED_SOURCE_SHA256":"a".repeat(64)}})
        .to_string(),
    )
    .unwrap();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{"alpha":{"local":true,"config":config,"managed_settings":true}}})
            .to_string(),
    )
    .unwrap();
    app.launch();
    let get = app.json("GET", "/api/settings", None, 200);
    let row = &get["hosts"][0];
    assert_eq!(row["management"]["available"], true);
    assert_eq!(row["evaluation"]["status"], "profile_only");
    assert!(
        row["evaluation"]["reason"]
            .as_str()
            .unwrap()
            .contains("source-pinned")
    );
    let draft = target_profile(row, 9281);
    let before = fs::read(&config).unwrap();
    let saved = app.json("PUT", "/api/settings/alpha", Some(draft.clone()), 200);
    assert_eq!(saved["benchmark_targets"], draft["benchmark_targets"]);
    assert_eq!(fs::read(&config).unwrap(), before);
    assert!(!root.join("operational-settings.json").exists());
    let mut changed: Value = serde_json::from_slice(&before).unwrap();
    changed["engine_env"]["BREWERY_EXPECTED_SOURCE_SHA256"] = json!("b".repeat(64));
    fs::write(&config, changed.to_string()).unwrap();
    let replacement = app.json("GET", "/api/settings", None, 200);
    assert_ne!(
        row["profile_binding"],
        replacement["hosts"][0]["profile_binding"]
    );
    assert_eq!(replacement["hosts"][0]["benchmark_targets"], json!([]));
    app.json("PUT", "/api/settings/alpha", Some(draft), 409);
}

#[test]
fn worker_policy_without_dispatch_opt_in_keeps_saved_profile_for_manual_adoption() {
    let mut app = Fixture::new();
    let root = app.path("worker");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let config = app.path("worker.json");
    fs::write(&config, json!({"root":root}).to_string()).unwrap();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{"alpha":{"local":true,"config":config,"capabilities":{"generate":false}}}})
            .to_string(),
    )
    .unwrap();
    app.launch();
    let settings = app.json("GET", "/api/settings", None, 200);
    let draft = target_profile(&settings["hosts"][0], 9283);
    let saved_draft = app.json("PUT", "/api/settings/alpha", Some(draft.clone()), 200);
    app.stop();
    let catalog = app.path("catalog.json");
    fs::write(&catalog, r#"{"adapters":{}}"#).unwrap();
    fs::write(
        &config,
        json!({"root":root,
            "managed_settings":{"catalog":catalog,"allowed_ports":[9283],"evaluation_allowed":true},
            "engine_env":{"BREWERY_ALLOW_GPU_LOAD":"1",
                "BREWERY_HALOGEN_IMAGE_ID":format!("sha256:{}", "a".repeat(64))}})
        .to_string(),
    )
    .unwrap();
    app.launch();
    let settings = app.json("GET", "/api/settings", None, 200);
    let row = &settings["hosts"][0];
    assert_eq!(row["evaluation"]["status"], "managed");
    assert_eq!(row["revision"], 0);
    assert_eq!(row["benchmark_targets"], json!([]));
    assert_eq!(row["saved_profile"]["revision"], 1);
    assert_eq!(
        row["saved_profile"]["benchmark_targets"],
        saved_draft["benchmark_targets"]
    );
    app.json("PUT", "/api/settings/alpha", Some(draft), 409);
    let mut native = row["saved_profile"].clone();
    native["revision"] = row["revision"].clone();
    native["evaluation_enabled"] = json!(true);
    native["benchmark_targets"][0]["endpoint"] = json!("http://[::1]:9284/v1");
    app.json("PUT", "/api/settings/alpha", Some(native.clone()), 400);
    native["benchmark_targets"] = saved_draft["benchmark_targets"].clone();
    let confirmed = app.json("PUT", "/api/settings/alpha", Some(native), 200);
    assert_eq!(
        confirmed["benchmark_targets"],
        saved_draft["benchmark_targets"]
    );
    assert_eq!(confirmed["saved_profile"], row["saved_profile"]);
    let worker_snapshot: Value =
        serde_json::from_slice(&fs::read(root.join("operational-settings.json")).unwrap()).unwrap();
    assert_eq!(
        worker_snapshot["benchmark_targets"],
        confirmed["benchmark_targets"]
    );
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["hosts"][0]["capabilities"]["evaluate"], true);
    assert_eq!(state["hosts"][0]["capabilities"]["generate"], false);
}

#[test]
fn managed_reference_precision_does_not_grant_attestation_or_unapproved_ports() {
    let mut app = Fixture::new();
    let root = app.path("worker");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let catalog = app.path("catalog.json");
    fs::write(&catalog, r#"{"adapters":{}}"#).unwrap();
    let config = app.path("worker.json");
    fs::write(
        &config,
        json!({"root":root,"managed_settings":{"catalog":catalog,"allowed_ports":[9101],
            "evaluation_allowed":false,"baseline_allowed":true}})
        .to_string(),
    )
    .unwrap();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{"alpha":{"local":true,"config":config}}}).to_string(),
    )
    .unwrap();
    app.launch();
    let request = json!({"revision":0,"benchmark_targets":[],"enabled_suites":[],
        "evaluation_enabled":false,"baseline_endpoints":[{"id":"reference","model":"fixture",
            "endpoint":"http://127.0.0.1:9101/v1","source_sha256":"b".repeat(64),
            "runtime_id":"fixture","precision":"int8"}]});
    let mut invalid = request.clone();
    invalid["baseline_endpoints"][0]["endpoint"] = json!("http://127.0.0.1:9102/v1");
    app.json("PUT", "/api/settings/alpha", Some(invalid), 400);
    let mut invalid = request.clone();
    invalid["baseline_endpoints"][0]["precision"] = json!("not a token");
    app.json("PUT", "/api/settings/alpha", Some(invalid), 400);
    let mut invalid = request.clone();
    invalid["baseline_endpoints"][0]["bf16_attested"] = json!(true);
    app.json("PUT", "/api/settings/alpha", Some(invalid), 422);
    let saved = app.json("PUT", "/api/settings/alpha", Some(request.clone()), 200);
    assert_eq!(saved["revision"], 1);
    assert_eq!(saved["baseline_endpoints"], request["baseline_endpoints"]);
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["hosts"][0]["capabilities"]["baseline"], true);
    assert_eq!(state["hosts"][0]["capabilities"]["evaluate"], false);
}
