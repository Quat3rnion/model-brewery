use super::*;

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
    edited["benchmark_targets"][0]["endpoint"] = json!("http://localhost:9101/v1");
    app.json("PUT", "/api/settings/alpha", Some(edited), 400);
    fs::remove_file(catalog).unwrap();
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["hosts"][0]["capabilities"]["benchmark"], false);
}
