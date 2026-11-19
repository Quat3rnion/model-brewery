use super::*;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

#[test]
fn backend_downloads_without_a_configured_worker() {
    let mut app = Fixture::new();
    fs::write(app.path("dispatch.json"), r#"{"hosts":{}}"#).unwrap();
    let companion = app.path("dispatch");
    app.launch_with(&[
        ("BREWERY_HF_WORKER", companion.clone()),
        ("BREWERY_HF_ENGINE", companion.clone()),
        ("BREWERY_HF_HFR", companion.clone()),
        ("BREWERY_HF_WEIGHT_EDIT", companion.clone()),
        ("BREWERY_HF_DISPATCH", companion),
    ]);
    let job = app.enqueue(json!({"kind":"hf_download","options":{"repo_id":"example/model"}}));
    let job = terminal(&app, &job);
    assert_eq!(job["host"], "backend-hf");
    assert_eq!(job["state"], "succeeded");
    assert_eq!(library(&app)[0]["id"], job["result"]["library_model_id"]);
    assert_eq!(
        app.json("GET", "/api/settings/hf-tokens", None, 200),
        json!({"hosts":[{"id":"backend-hf","configured":false}]})
    );
    assert_eq!(
        fs::read_to_string(app.path("dispatch.json")).unwrap(),
        r#"{"hosts":{}}"#
    );
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["hosts"][0]["capabilities"]["weight_inspect"], true);
    assert_eq!(state["hosts"][0]["capabilities"]["weight_tune"], false);
    let model = &job["result"]["library_model_id"];
    app.json(
        "POST",
        "/api/jobs",
        Some(json!({
            "kind":"weight_inspect","input_ids":[model],
            "options":{"mode":"inventory","offset":0,"limit":16}
        })),
        201,
    );
    app.json(
        "POST",
        "/api/models",
        Some(json!({
            "host":"backend-hf","kind":"safetensors","path":"/private/not-a-publication.safetensors"
        })),
        400,
    );
}

fn fixture() -> Fixture {
    let app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true},
            "beta":{"ssh":"beta","capabilities":{"hf_download":true}}
        }})
        .to_string(),
    )
    .unwrap();
    app
}

fn request(options: Value) -> Value {
    json!({"kind":"hf_download","host":"beta","input_ids":[],"options":options})
}

#[test]
fn hf_download_requires_an_explicit_opted_in_host_and_closed_options() {
    let mut app = fixture();
    app.launch();
    let state = app.json("GET", "/api/state", None, 200);
    let hosts = &state["hosts"];
    assert_eq!(hosts[0]["capabilities"]["hf_download"], false);
    assert_eq!(hosts[1]["capabilities"]["hf_download"], true);
    let mut body = request(json!({"repo_id":"org/model"}));
    body["host"] = json!("alpha");
    app.json("POST", "/api/jobs", Some(body.clone()), 409);
    body["host"] = Value::Null;
    app.json("POST", "/api/jobs", Some(body.clone()), 409);
    body["host"] = json!("auto");
    app.json("POST", "/api/jobs", Some(body), 400);
    for options in [
        json!({}),
        json!({"repo_id":"https://huggingface.co/org/model"}),
        json!({"repo_id":"org/model","revision":"../main"}),
        json!({"repo_id":"org/model","revision":null}),
        json!({"repo_id":"org/model","filename":"*.gguf"}),
        json!({"repo_id":"org/model","filename":"../model.gguf"}),
        json!({"repo_id":"org/model","filename":"/private/model.gguf"}),
        json!({"repo_id":"org/model","filename":"script.py"}),
        json!({"repo_id":"org/model","filename":null}),
        json!({"repo_id":"org/model","token":"secret"}),
        json!({"repo_id":"org/model","endpoint":"http://other"}),
        json!({"repo_id":"org/model","path":"/private"}),
        json!({"repo_id":"org/model","flags":["--other"]}),
    ] {
        app.json("POST", "/api/jobs", Some(request(options)), 400);
    }
    let mut body = request(json!({"repo_id":"org/model"}));
    body["prompt"] = json!("unexpected");
    app.json("POST", "/api/jobs", Some(body.clone()), 400);
    body["prompt"] = Value::Null;
    body["input_ids"] = json!([app.register("beta", "gguf", "/models/source.gguf")]);
    app.json("POST", "/api/jobs", Some(body), 400);
    assert_eq!(app.json("GET", "/api/state", None, 200)["jobs"], json!([]));
}

fn library(app: &Fixture) -> Value {
    app.json("GET", "/api/state", None, 200)["models"].clone()
}

fn terminal(app: &Fixture, job: &str) -> Value {
    app.await_job(job, |row| {
        matches!(
            row["state"].as_str(),
            Some("succeeded" | "failed" | "cancelled")
        )
    })
}

fn launches(app: &Fixture) -> usize {
    fs::read_to_string(app.path("launches"))
        .unwrap()
        .lines()
        .count()
}

#[test]
fn hf_download_registers_only_verified_artifacts_and_recovers_without_relaunch() {
    let mut app = fixture();
    fs::write(app.path("hold"), "").unwrap();
    app.launch();
    let source = request(json!({"repo_id":"org/model"}));
    let (status, body) = app.request(
        "POST",
        "/api/jobs",
        Some(source.clone()),
        "Idempotency-Key: hf-whole\r\n",
    );
    assert_eq!(status, 201);
    let whole = id(&serde_json::from_str(&body).unwrap()).to_owned();
    app.await_job(&whole, |job| job["progress"] == "worker progress");
    assert_eq!(library(&app), json!([]));
    app.stop();
    fs::remove_file(app.path("hold")).unwrap();
    app.launch();

    let detail = terminal(&app, &whole);
    assert_eq!(detail["state"], "succeeded");
    let result = &detail["result"];
    assert_eq!(result["revision"], "main");
    assert_eq!(result["commit"], COMMIT);
    assert_eq!(result["repo_id"], "org/model");
    assert_eq!(result["verified"], true);
    let saved = library(&app);
    assert_eq!(saved.as_array().unwrap().len(), 1);
    assert_eq!(saved[0]["id"], result["library_model_id"]);
    assert_eq!(saved[0]["kind"], "hf");
    assert_eq!(saved[0]["host"], "beta");
    assert_eq!(
        saved[0]["path"],
        format!("/worker/jobs/{whole}/work/download")
    );
    assert_eq!(launches(&app), 1);

    app.stop();
    app.launch();
    let (status, body) = app.request(
        "POST",
        "/api/jobs",
        Some(source),
        "Idempotency-Key: hf-whole\r\n",
    );
    assert_eq!(status, 200);
    assert_eq!(id(&serde_json::from_str(&body).unwrap()), whole);
    assert_eq!(library(&app), saved);
    for (filename, kind) in [
        ("quant/model.gguf", "gguf"),
        ("model.safetensors", "safetensors"),
    ] {
        let job = app.enqueue(request(json!({
            "repo_id":"org/model","revision":COMMIT,"filename":filename
        })));
        let detail = terminal(&app, &job);
        assert_eq!(detail["state"], "succeeded");
        assert_eq!(detail["result"]["filename"], filename);
        let id = &detail["result"]["library_model_id"];
        let models = library(&app);
        let registered = models
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == *id)
            .unwrap();
        assert_eq!(registered["kind"], kind);
        assert_eq!(
            registered["path"],
            format!("/worker/jobs/{job}/work/download/{filename}")
        );
    }
    assert_eq!(library(&app).as_array().unwrap().len(), 3);
}

#[test]
fn hf_download_never_registers_a_bad_receipt_or_a_success_racing_cancellation() {
    let mut app = fixture();
    app.launch();
    let body = request(json!({
        "repo_id":"org/model","revision":COMMIT,"filename":"model.gguf"
    }));
    for bad in [
        json!({"verified":false}),
        json!({"repo_id":"org/other"}),
        json!({"revision":"main"}),
        json!({"commit":"main"}),
        json!({"commit":"1111111111111111111111111111111111111111"}),
        json!({"filename":"other.gguf"}),
        json!({"artifact":{"kind":"hf"}}),
        json!({"artifact":{"path":"/worker/jobs/other/work/download/model.gguf"}}),
        json!({"artifact":{"sha256":"missing"}}),
        json!({"artifact":{"bytes":0}}),
        json!({"library_model_id":"worker-cannot-pick-id"}),
    ] {
        fs::write(app.path("hf-receipt-override.json"), bad.to_string()).unwrap();
        let job = app.enqueue(body.clone());
        let detail = terminal(&app, &job);
        assert_eq!(detail["state"], "failed", "{bad}: {detail}");
        assert_eq!(detail["has_result"], false);
        assert!(detail.get("result").is_none());
        assert_eq!(library(&app), json!([]));
    }
    fs::remove_file(app.path("hf-receipt-override.json")).unwrap();
    fs::write(app.path("hold"), "").unwrap();
    fs::write(app.path("hf-races-cancel"), "").unwrap();
    let job = app.enqueue(body);
    app.await_job(&job, |job| job["progress"] == "worker progress");
    app.cancel(&job);
    let detail = terminal(&app, &job);
    assert_eq!(detail["state"], "cancelled");
    assert_eq!(detail["cancel_requested"], true);
    assert!(detail.get("result").is_none());
    assert_eq!(library(&app), json!([]));
}

#[test]
fn hf_download_recovered_missing_job_is_not_relaunched_or_registered() {
    let mut app = fixture();
    fs::write(app.path("hold"), "").unwrap();
    fs::write(app.path("hf-races-cancel"), "").unwrap();
    app.launch();
    let job = app.enqueue(request(json!({"repo_id":"org/model"})));
    app.await_job(&job, |job| job["progress"] == "worker progress");
    app.stop();
    fs::write(app.path(&format!("{job}.forget")), "").unwrap();
    app.launch();
    let detail = terminal(&app, &job);
    assert_eq!(detail["state"], "failed");
    assert!(detail.get("result").is_none());
    assert_eq!(library(&app), json!([]));
    assert_eq!(launches(&app), 1);
}
