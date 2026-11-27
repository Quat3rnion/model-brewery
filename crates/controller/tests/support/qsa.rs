use super::*;

#[test]
fn qsa_training_requires_separate_inputs_and_verified_publication() {
    let mut fx = Fixture::new();
    let mut config: Value =
        serde_json::from_str(&fs::read_to_string(fx.path("dispatch.json")).unwrap()).unwrap();
    config["hosts"]["alpha"]["capabilities"]["qsa_train"] = json!(true);
    config["hosts"]["beta"]["capabilities"]["qsa_train"] = json!(true);
    fs::write(fx.path("dispatch.json"), config.to_string()).unwrap();
    fx.launch();
    let model = |host: &str, path: &str, kind: &str| {
        fx.json(
            "POST",
            "/api/models",
            Some(json!({"host":host,"path":path,"kind":kind})),
            201,
        )["id"]
            .clone()
    };
    let teacher = model("alpha", "/models/qwen", "hf");
    let train = model("alpha", "/models/train.jsonl", "qsa-corpus");
    let eval = model("alpha", "/models/eval.jsonl", "qsa-corpus");
    let foreign = model("beta", "/models/eval.jsonl", "qsa-corpus");
    fx.json(
        "POST",
        "/api/models",
        Some(json!({"host":"alpha","path":"/models/fake.safetensors","kind":"qsa-indexer"})),
        400,
    );
    let job = |heldout: &Value| {
        json!({"kind":"qsa_train","input_ids":[teacher, train, heldout],
            "options":{"steps":2,"seq_len":32,"seed":42,"lr":0.001}})
    };
    fx.json("POST", "/api/jobs", Some(job(&eval)), 409);
    fx.stop();
    fs::write(fx.path("qsa-runtime-ready"), "").unwrap();
    fx.launch();
    fx.json("POST", "/api/jobs", Some(job(&train)), 400);
    fx.json("POST", "/api/jobs", Some(job(&foreign)), 400);
    let mut bad_options = job(&eval);
    bad_options["options"]["output_format"] = json!("gguf");
    fx.json("POST", "/api/jobs", Some(bad_options), 400);
    let submit = || {
        fx.await_job(&fx.enqueue(job(&eval)), |detail| {
            matches!(detail["state"].as_str(), Some("failed" | "succeeded"))
        })
    };
    fs::write(fx.path("qsa-wrong-teacher"), "").unwrap();
    assert_eq!(submit()["state"], "failed");
    fs::remove_file(fx.path("qsa-wrong-teacher")).unwrap();
    fs::write(fx.path("qsa-fake-capture"), "").unwrap();
    assert_eq!(submit()["state"], "failed");
    fs::remove_file(fx.path("qsa-fake-capture")).unwrap();
    fs::write(fx.path("hold"), "").unwrap();
    fs::write(fx.path("qsa-races-cancel"), "").unwrap();
    let id = fx.enqueue(job(&eval));
    fx.await_job(&id, |job| job["progress"] == "worker progress");
    fx.cancel(&id);
    let cancelled = fx.await_job(&id, |job| job["state"] == "cancelled");
    assert!(cancelled.get("result").is_none());
    fs::remove_file(fx.path("hold")).unwrap();
    let completed = submit();
    assert_eq!(completed["state"], "succeeded", "{completed}");
    let library_id = &completed["result"]["library_model_id"];
    assert!(library_id.is_string());
    let state = fx.json("GET", "/api/state", None, 200);
    let artifacts: Vec<_> = state["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["kind"] == "qsa-indexer")
        .collect();
    assert_eq!(artifacts.len(), 1);
    assert_eq!(artifacts[0]["id"], *library_id);
    assert_eq!(completed["result"]["teacher"]["id"], teacher);
    assert_eq!(artifacts[0]["qsa_provenance"]["teacher_model_id"], teacher);
    assert_eq!(
        artifacts[0]["qsa_provenance"]["teacher_sha256"],
        completed["result"]["teacher"]["sha256"]
    );
    fx.json(
        "POST",
        "/api/jobs",
        Some(json!({
            "kind":"convert","input_ids":[teacher, library_id]
        })),
        400,
    );
}
