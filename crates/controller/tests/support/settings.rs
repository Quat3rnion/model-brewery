use super::*;

#[test]
fn backend_and_legacy_hosts_have_management_without_evaluation_enrollment() {
    let mut app = Fixture::new();
    let companion = app.path("dispatch");
    app.launch_with(&[
        ("BREWERY_HF_WORKER", companion.clone()),
        ("BREWERY_HF_ENGINE", companion.clone()),
        ("BREWERY_HF_HFR", companion.clone()),
        ("BREWERY_HF_DISPATCH", companion),
    ]);
    let settings = app.json("GET", "/api/settings", None, 200);
    let hosts = settings["hosts"].as_array().unwrap();
    assert_eq!(hosts.len(), 3);
    for (id, reason, job) in [
        ("alpha", "not enrolled", "inspect"),
        ("beta", "not enrolled", "evaluate"),
        ("backend-hf", "does not run evaluation", "hf_download"),
    ] {
        let row = hosts.iter().find(|row| row["id"] == id).unwrap();
        assert_eq!(row["evaluation"]["status"], "not_enrolled");
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
    let new = json!({"revision":0,"benchmark_targets":[{"id":"saved","endpoint":"http://127.0.0.1:9101/v1","model":"fixture"}],
        "enabled_suites":["suite"],"baseline_endpoints":[],"evaluation_enabled":false});
    let saved = app.json("PUT", "/api/settings/alpha", Some(new.clone()), 200);
    assert_eq!(saved["revision"], 1);
    assert_eq!(saved["evaluation"]["status"], "managed");
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
    assert_eq!(alpha["evaluation"]["status"], "unavailable");
    assert!(
        alpha["evaluation"]["reason"]
            .as_str()
            .unwrap()
            .contains("suite catalog")
    );
    assert_eq!(alpha["management"]["available"], true);
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["hosts"][0]["capabilities"]["benchmark"], false);
}

#[test]
fn source_pinned_worker_does_not_become_editable_when_enrolled() {
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
    assert_eq!(row["evaluation"]["status"], "unavailable");
    let rejected = app.json(
        "PUT",
        "/api/settings/alpha",
        Some(
            json!({"revision":0,"evaluation_enabled":false,"benchmark_targets":[],
            "enabled_suites":[],"baseline_endpoints":[]}),
        ),
        503,
    );
    assert_eq!(rejected["error"], row["evaluation"]["reason"]);
    assert!(!root.join("operational-settings.json").exists());
}
