use super::{Fixture, fs, id, json, wait_for};
use serde_json::Value;
use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::Barrier,
    thread,
    time::Duration,
};

const KEY: &str = "Idempotency-Key: submit-1\r\n";

fn keyed(app: &Fixture, path: &str, body: Value, status: u16) -> Value {
    let (actual, response) = app.request("POST", path, Some(body), KEY);
    assert_eq!(actual, status, "{path}: {response}");
    serde_json::from_str(&response).unwrap()
}

#[test]
fn lost_job_response_replays_concurrently_after_restart_and_input_removal() {
    let mut app = Fixture::new();
    app.launch();
    let input = app.register("beta", "hgn", "/models/generate.hgn");
    let body = json!({
        "kind": "generate", "input_ids": [input], "options": {"max_tokens": 8},
        "prompt": "one test prompt"
    });
    // Send the first request but never read its response.
    let unread = app.open_request("POST", "/api/jobs", Some(body.clone()), KEY);
    let job = wait_for(|| {
        let snapshot = app.json("GET", "/api/state", None, 200);
        snapshot["jobs"].as_array().unwrap().first().cloned()
    });
    drop(unread);
    let job_id = id(&job);
    let completed = app.await_job(job_id, |job| job["state"] == "succeeded");
    app.json("DELETE", &format!("/api/models/{input}"), None, 204);
    app.stop();
    fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true}}}"#,
    )
    .unwrap();
    app.launch();

    let barrier = Barrier::new(4);
    let app = &app;
    let body = &body;
    thread::scope(|tasks| {
        for _ in 0..4 {
            let barrier = &barrier;
            tasks.spawn(move || {
                barrier.wait();
                assert_eq!(
                    keyed(app, "/api/jobs", body.clone(), 200),
                    json!({"id": job_id})
                );
            });
        }
    });
    for (field, changed) in [
        ("input_ids", json!(["other"])),
        ("options", json!({"max_tokens": 9})),
        ("prompt", json!("a different prompt")),
        ("prompt_id", json!("a saved prompt")),
    ] {
        let mut mismatch = body.clone();
        mismatch[field] = changed;
        keyed(app, "/api/jobs", mismatch, 409);
    }
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["jobs"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(app.detail(job_id)["events"], completed["events"]);
    assert_eq!(app.launches().len(), 1);
}

#[test]
fn key_is_route_scoped_and_deleted_models_are_not_recreated() {
    let mut app = Fixture::new();
    app.launch();
    let model_body = json!({"host": "alpha", "path": "/model.gguf", "kind": "gguf"});
    let model = keyed(&app, "/api/models", model_body.clone(), 201);
    let prompt_body = json!({"name": "prompt", "text": "some prompt text"});
    let prompt = keyed(&app, "/api/prompts", prompt_body.clone(), 201);
    app.json("DELETE", &format!("/api/models/{}", id(&model)), None, 204);
    app.stop();
    fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"beta":{"local":true}}}"#,
    )
    .unwrap();
    app.launch();
    let expected = json!({"id": id(&model)});
    // The original omitted `name`; explicit null is the same typed request.
    // This wire form also changes the member order and whitespace.
    let raw = r#"{ "kind": "gguf", "path": "/model.gguf", "name": null, "host": "alpha" }"#;
    let addr = app.addr.unwrap();
    let mut conn = TcpStream::connect(addr).unwrap();
    conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    write!(
        conn,
        "POST /api/models HTTP/1.0\r\nHost: {addr}\r\n{KEY}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{raw}",
        raw.len()
    )
    .unwrap();
    let mut response = String::new();
    conn.read_to_string(&mut response).unwrap();
    let (head, body) = response.split_once("\r\n\r\n").unwrap();
    assert_eq!(head.split_whitespace().nth(1), Some("200"));
    assert_eq!(serde_json::from_str::<Value>(body).unwrap(), expected);

    let mut different = model_body;
    different["path"] = json!("/other.gguf");
    keyed(&app, "/api/models", different, 409);
    assert_eq!(
        keyed(&app, "/api/prompts", prompt_body, 200),
        json!({"id": id(&prompt)})
    );
    keyed(
        &app,
        "/api/prompts",
        json!({"name": "prompt", "text": "different text"}),
        409,
    );
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["models"], json!([]));
    assert_eq!(state["prompts"], json!([prompt]));
}

#[test]
fn malformed_keys_and_invalid_creates_do_not_bind_a_key() {
    let mut app = Fixture::new();
    app.launch();
    let path = "/api/prompts";
    let body = json!({"name": "prompt", "text": "some prompt text"});
    for headers in [
        "Idempotency-Key:\r\n".into(),
        "Idempotency-Key: two values\r\n".into(),
        format!("{KEY}Idempotency-Key: submit-1\r\n"),
        format!("Idempotency-Key: {}\r\n", "x".repeat(129)),
    ] {
        let (status, response) = app.request("POST", path, Some(body.clone()), &headers);
        assert_eq!(status, 400);
        assert_eq!(
            serde_json::from_str::<Value>(&response).unwrap(),
            json!({"error": "invalid Idempotency-Key"})
        );
    }
    keyed(&app, path, json!({"name": "", "text": "invalid"}), 400);
    let created = keyed(&app, path, body.clone(), 201);
    let (status, response) = app.request(
        "POST",
        path,
        Some(body.clone()),
        &format!("Idempotency-Key: {}\r\n", "x".repeat(128)),
    );
    assert_eq!(status, 201);
    let other: Value = serde_json::from_str(&response).unwrap();
    assert_ne!(id(&other), id(&created));
    let unkeyed = app.json("POST", path, Some(body), 201);
    assert_ne!(id(&unkeyed), id(&created));
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["prompts"],
        json!([created, other, unkeyed])
    );
}
