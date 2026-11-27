use super::{Fixture, id, json};
use serde_json::Value;
use std::{io::Read, sync::Barrier, thread, time::Duration};

const JOBS: &str = "/api/jobs";
const ALLOW: &str = "Brewery-Allow-Duplicate: 1\r\n";

fn submit(app: &Fixture, route: &str, body: &Value, key: &str, extra: &str, status: u16) -> Value {
    let (actual, reply) = app.request(
        "POST",
        route,
        Some(body.clone()),
        &format!("Idempotency-Key: {key}\r\n{extra}"),
    );
    assert_eq!(actual, status, "{route}: {reply}");
    serde_json::from_str(&reply).unwrap()
}

fn assert_duplicate(app: &Fixture, route: &str, body: &Value, key: &str, original: &str) {
    let mut conn = app.open_request(
        "POST",
        route,
        Some(body.clone()),
        &format!("Idempotency-Key: {key}\r\n"),
    );
    let mut response = String::new();
    conn.read_to_string(&mut response).unwrap();
    let (head, payload) = response.split_once("\r\n\r\n").unwrap();
    assert_eq!(head.split_whitespace().nth(1), Some("200"), "{response}");
    assert!(
        head.lines().any(|line| {
            line.split_once(':').is_some_and(|(name, value)| {
                name.eq_ignore_ascii_case("Brewery-Duplicate") && value.trim() == original
            })
        }),
        "{head}"
    );
    let payload: Value = serde_json::from_str(payload).unwrap();
    assert_eq!(payload["id"], original);
    assert_eq!(payload["duplicate_of"], true);
}

#[test]
fn fresh_key_matches_recent_job_but_explicit_opt_out_enqueues() {
    let mut app = Fixture::new();
    app.launch();
    let source = app.register("beta", "hgn", "/models/dedupe.hgn");
    let body =
        json!({"kind":"generate","input_ids":[source],"prompt":"test","options":{"max_tokens":8}});
    let created = submit(&app, JOBS, &body, "first", "", 201);
    let original = id(&created);
    app.await_job(original, |job| job["state"] == "succeeded");
    app.stop();
    app.launch();

    assert_duplicate(&app, JOBS, &body, "second", original);
    assert_eq!(
        app.json("POST", JOBS, Some(body.clone()), 200),
        json!({"id": original, "duplicate_of": true})
    );
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["jobs"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    for key in ["first", "second"] {
        assert_eq!(
            submit(&app, JOBS, &body, key, ALLOW, 200),
            json!({"id": original})
        );
    }
    let another = submit(&app, JOBS, &body, "another", ALLOW, 201);
    assert_ne!(id(&another), original);
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["jobs"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn concurrent_identical_jobs_enqueue_once_and_active_match_outlives_window() {
    let mut app = Fixture::new();
    app.touch("hold");
    app.launch();
    let source = app.register("beta", "hgn", "/models/active.hgn");
    let body =
        json!({"kind":"generate","input_ids":[source],"prompt":"test","options":{"max_tokens":8}});
    let barrier = Barrier::new(2);
    let replies: Vec<_> = thread::scope(|scope| {
        ["one", "two"]
            .map(|key| {
                let (app, body, barrier) = (&app, &body, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    app.request(
                        "POST",
                        JOBS,
                        Some(body.clone()),
                        &format!("Idempotency-Key: {key}\r\n"),
                    )
                })
            })
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    let mut statuses: Vec<_> = replies.iter().map(|reply| reply.0).collect();
    statuses.sort_unstable();
    assert_eq!(statuses, [200, 201]);
    let created: Value =
        serde_json::from_str(&replies.iter().find(|reply| reply.0 == 201).unwrap().1).unwrap();
    let original = id(&created);
    app.await_job(original, |job| job["state"] == "running");
    thread::sleep(Duration::from_secs(11));
    assert_duplicate(&app, JOBS, &body, "after-window", original);
    app.remove("hold");
    app.await_job(original, |job| job["state"] == "succeeded");
    assert_eq!(
        submit(&app, JOBS, &body, "after-window", "", 200),
        json!({"id": original})
    );
    assert_ne!(id(&submit(&app, JOBS, &body, "expired", "", 201)), original);
}

#[test]
fn conversion_dedupes_without_enqueuing_another_stage() {
    let mut app = Fixture::new();
    app.touch("hold");
    super::fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"gguf_output":true}}}}"#,
    )
    .unwrap();
    app.launch();
    let source = app.register("alpha", "gguf", "/models/source.gguf");
    let body = json!({"host":"alpha","input_ids":[source],"format":"gguf"});
    let route = "/api/conversions";
    let first = submit(&app, route, &body, "conversion", "", 201);
    assert_duplicate(&app, route, &body, "repeat", id(&first));
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["jobs"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        id(&submit(&app, route, &body, "conversion", ALLOW, 200)),
        id(&first)
    );
    assert_ne!(
        id(&submit(&app, route, &body, "another", ALLOW, 201)),
        id(&first)
    );
}
