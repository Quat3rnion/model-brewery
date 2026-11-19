use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    net::{SocketAddr, TcpStream},
    os::unix::fs::PermissionsExt,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

struct Rig {
    root: tempfile::TempDir,
    controller: Option<Child>,
    addr: SocketAddr,
}

impl Rig {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("config.json"),
            r#"{"hosts":{"local":{"local":true}}}"#,
        )
        .unwrap();
        let dispatcher = root.path().join("dispatcher");
        fs::write(&dispatcher, include_str!("support/dispatch.py")).unwrap();
        fs::set_permissions(&dispatcher, fs::Permissions::from_mode(0o700)).unwrap();
        let mut rig = Self {
            root,
            controller: None,
            addr: "127.0.0.1:0".parse().unwrap(),
        };
        rig.start();
        rig
    }

    fn start(&mut self) {
        self.stop();
        let log = self.root.path().join("start.log");
        let child = Command::new(env!("CARGO_BIN_EXE_brewery"))
            .env("BREWERY_LISTEN", "127.0.0.1:0")
            .env("BREWERY_DATA", self.root.path().join("state"))
            .env(
                "BREWERY_DISPATCH_CONFIG",
                self.root.path().join("config.json"),
            )
            .env("BREWERY_DISPATCH", self.root.path().join("dispatcher"))
            .env("BREWERY_POLL_MS", "25")
            .stdout(Stdio::from(fs::File::create(&log).unwrap()))
            .spawn()
            .unwrap();
        self.controller = Some(child);
        let deadline = Instant::now() + Duration::from_secs(5);
        self.addr = loop {
            assert!(
                self.controller
                    .as_mut()
                    .unwrap()
                    .try_wait()
                    .unwrap()
                    .is_none()
            );
            if let Some(addr) = fs::read_to_string(&log).unwrap().lines().find_map(|line| {
                line.strip_prefix("brewery listening on http://")?
                    .parse()
                    .ok()
            }) {
                break addr;
            }
            assert!(Instant::now() < deadline, "no controller listener");
            thread::sleep(Duration::from_millis(10));
        };
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.controller.take() {
            let _ = child.kill();
            child.wait().unwrap();
        }
    }

    fn request(&self, verb: &str, path: &str, headers: &str, payload: Option<Value>) -> Http {
        let mut conn = TcpStream::connect(self.addr).unwrap();
        conn.set_read_timeout(Some(Duration::from_secs(18)))
            .unwrap();
        // HTTP/1.0 uses EOF framing rather than chunked encoding for SSE.
        write!(
            conn,
            "{verb} {path} HTTP/1.0\r\nHost: {}\r\n{headers}",
            self.addr
        )
        .unwrap();
        if let Some(body) = payload {
            let data = body.to_string();
            write!(
                conn,
                "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{data}",
                data.len()
            )
            .unwrap();
        } else {
            write!(conn, "\r\n").unwrap();
        }
        Http::read(conn)
    }

    fn json(&self, verb: &str, path: &str, payload: Option<Value>, status: u16) -> Value {
        let mut response = self.request(verb, path, "", payload);
        assert_eq!(response.status, status);
        if status == 204 {
            Value::Null
        } else {
            serde_json::from_reader(&mut response.body).unwrap()
        }
    }

    fn prompt(&self) -> Value {
        self.json(
            "POST",
            "/api/prompts",
            Some(json!({"name":"hidden name","text":"hidden text"})),
            201,
        )
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Http {
    status: u16,
    headers: String,
    body: BufReader<TcpStream>,
}

impl Http {
    fn read(conn: TcpStream) -> Self {
        let mut body = BufReader::new(conn);
        let mut status = String::new();
        body.read_line(&mut status).unwrap();
        let status = status.split_whitespace().nth(1).unwrap().parse().unwrap();
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            assert_ne!(body.read_line(&mut line).unwrap(), 0);
            if line == "\r\n" {
                break;
            }
            headers.push_str(&line);
        }
        Self {
            status,
            headers,
            body,
        }
    }

    fn frame(&mut self) -> Option<Frame> {
        let mut frame = Frame::default();
        loop {
            let mut line = String::new();
            if self.body.read_line(&mut line).unwrap() == 0 {
                return None;
            }
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                return Some(frame);
            }
            if let Some(name) = line.strip_prefix("event:") {
                frame.event = Some(name.trim().into());
            } else if let Some(id) = line.strip_prefix("id:") {
                frame.id = Some(id.trim().parse().unwrap());
            } else if let Some(data) = line.strip_prefix("data:") {
                frame.data = serde_json::from_str(data).unwrap();
            } else if line.starts_with(':') {
                frame.comment = true;
            }
        }
    }

    fn event(&mut self, name: &str) -> Frame {
        assert_eq!(self.status, 200);
        assert!(self.headers.contains("content-type: text/event-stream"));
        while let Some(frame) = self.frame() {
            if frame.comment {
                continue;
            }
            assert_eq!(frame.event.as_deref(), Some(name));
            return frame;
        }
        panic!("stream ended before {name}")
    }
}

#[derive(Default)]
struct Frame {
    id: Option<i64>,
    event: Option<String>,
    data: Value,
    comment: bool,
}

#[test]
fn feed_replays_exact_snapshot_boundary_across_restart() {
    let mut app = Rig::new();
    let first = app.prompt();
    let snapshot = app.json("GET", "/api/state", None, 200);
    let cursor = snapshot["event_cursor"].as_i64().unwrap();
    assert!(cursor > 0);
    assert_eq!(snapshot["prompts"], json!([first]));
    let model = app.json(
        "POST",
        "/api/models",
        Some(json!({
            "kind":"gguf","host":"local","path":"/private/model.gguf"
        })),
        201,
    );
    let model_id = model["id"].as_str().unwrap();
    app.json("DELETE", &format!("/api/models/{model_id}"), None, 204);
    app.start();
    let mut replay = app.request(
        "GET",
        "/api/events?after=0",
        &format!("Last-Event-ID: {cursor}\r\n"),
        None,
    );
    for (index, kind) in ["model.created", "model.deleted"].into_iter().enumerate() {
        let frame = replay.event("change");
        let id = cursor + index as i64 + 1;
        assert_eq!(frame.id, Some(id));
        assert_eq!(
            frame.data,
            json!({"id":id,"kind":kind,"entity_id":model_id})
        );
    }
    let prompt = app.prompt();
    let frame = replay.event("change");
    assert_eq!(frame.id, Some(cursor + 3));
    assert_eq!(
        frame.data,
        json!({"id":cursor+3,"kind":"prompt.created","entity_id":prompt["id"]})
    );
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["event_cursor"],
        cursor + 3
    );
}

#[test]
fn replay_is_bounded_and_out_of_range_cursors_explicitly_resync() {
    let app = Rig::new();
    for _ in 0..1029 {
        app.prompt();
    }
    let tip = app.json("GET", "/api/state", None, 200)["event_cursor"]
        .as_i64()
        .unwrap();
    assert_eq!(tip, 1029);
    for (after, reason) in [(4, "stale"), (tip + 1, "ahead")] {
        let mut stream = app.request("GET", &format!("/api/events?after={after}"), "", None);
        let resync = stream.event("resync");
        assert!(resync.id.is_none());
        assert_eq!(resync.data, json!({"reason":reason,"event_cursor":tip}));
        assert!(stream.frame().is_none());
    }
    let mut stream = app.request("GET", "/api/events?after=5", "", None);
    for id in 6..=tip {
        let frame = stream.event("change");
        assert_eq!(frame.id, Some(id));
        assert_eq!(frame.data["id"], id);
        assert_eq!(frame.data["kind"], "prompt.created");
    }
}

#[test]
fn job_progress_invalidates_snapshot_without_leaking_payloads() {
    let app = Rig::new();
    fs::write(app.root.path().join("sample_launch"), "").unwrap();
    let source = app.json(
        "POST",
        "/api/models",
        Some(json!({"kind":"gguf","host":"local","path":"/private/input.gguf"})),
        201,
    );
    let mut stream = app.request("GET", "/api/events?after=1", "", None);
    let job = app.json(
        "POST",
        "/api/jobs",
        Some(json!({"kind":"transfer","host":"local","input_ids":[source["id"]]})),
        201,
    );
    let job_id = job["id"].as_str().unwrap();
    for (id, kind, state) in [
        (2, "job.created", "queued"),
        (3, "job.updated", "running"),
        (4, "job.updated", "running"),
    ] {
        let event = stream.event("change");
        assert_eq!(event.id, Some(id));
        assert_eq!(
            event.data,
            json!({"id":id,"kind":kind,"entity_id":job_id,"state":state})
        );
    }
    let snapshot = app.json("GET", "/api/state", None, 200);
    assert_eq!(snapshot["event_cursor"], 4);
    assert_eq!(snapshot["jobs"][0]["progress"], "Copying: 120 / 360 GB");
    let detail = &snapshot["jobs"][0]["progress_detail"];
    assert_eq!(detail["phase"], "copy_input");
    assert_eq!(detail["completed"], 120000000000_u64);
    assert_eq!(detail["current_file"], "model-001.safetensors");
    assert_eq!(
        app.json("GET", &format!("/api/jobs/{job_id}"), None, 200)["progress_detail"],
        *detail
    );
    let cancelling = app.json(
        "POST",
        &format!("/api/jobs/{job_id}/cancel"),
        Some(json!({})),
        200,
    );
    assert!(cancelling.get("progress_detail").is_none());
    let requested = stream.event("change");
    assert_eq!(
        requested.data,
        json!({"id":5,"kind":"job.updated","entity_id":job_id,"state":"running"})
    );
    let cancelled = stream.event("change");
    assert_eq!(
        cancelled.data,
        json!({"id":6,"kind":"job.updated","entity_id":job_id,"state":"cancelled"})
    );
}

#[test]
fn running_progress_accepts_legacy_text_and_drops_invalid_details_without_failing() {
    let app = Rig::new();
    fs::write(app.root.path().join("hold"), "").unwrap();
    let model = app.json(
        "POST",
        "/api/models",
        Some(json!({
            "kind":"gguf", "host":"local", "path":"/private/input.gguf"
        })),
        201,
    );
    let job = app.json(
        "POST",
        "/api/jobs",
        Some(json!({
            "kind":"inspect", "host":"local", "input_ids":[model["id"]]
        })),
        201,
    );
    let path = format!("/api/jobs/{}", job["id"].as_str().unwrap());
    let wait = |text: &str| {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let current = app.json("GET", &path, None, 200);
            if current["progress"] == text {
                break current;
            }
            assert!(
                Instant::now() < deadline,
                "expected progress {text}: {current}"
            );
            thread::sleep(Duration::from_millis(20));
        }
    };
    let reply_path = app.root.path().join("progress-reply.json");
    let send = |raw: String| {
        let temp = reply_path.with_extension("tmp");
        fs::write(&temp, raw).unwrap();
        fs::rename(temp, &reply_path).unwrap();
    };
    assert!(wait("worker progress").get("progress_detail").is_none());
    let valid = json!({
        "phase":"inspect_input","unit":"files","completed":2,"total":5,"updated_at_ms":1760000000000_u64,
        "threads":[{"slot":1,"task":"a.gguf"},{"slot":3,"task":"c.gguf"}]
    });
    send(json!({"state":"running","progress":"working","progress_detail":valid}).to_string());
    let working = wait("working");
    assert_eq!(working["progress_detail"], valid);
    let cursor = app.json("GET", "/api/state", None, 200)["event_cursor"].clone();
    send(json!({"state":"running","progress":"working","progress_detail":{
        "phase":"inspect_input","unit":"files","completed":2,"total":5,"updated_at_ms":1760000001000_u64,
        "threads":[{"slot":1,"task":"a.gguf"},{"slot":3,"task":"c.gguf"}]
    }}).to_string());
    thread::sleep(Duration::from_millis(400));
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["event_cursor"],
        cursor,
        "timestamp-only samples must not flood SSE"
    );

    for (index, invalid) in [
        r#"{"phase":"inspect_input","unit":"files","completed":-1,"updated_at_ms":1}"#,
        r#"{"phase":"inspect_input","unit":"files","completed":1,"updated_at_ms":1,"current_file":"/private/secret"}"#,
        r#"{"phase":"inspect_input","unit":"files","completed":1,"updated_at_ms":1,"threads":[{"slot":1,"task":"a"},{"slot":1,"task":"b"}]}"#,
        r#"{"phase":"inspect_input","unit":"files","completed":1,"updated_at_ms":1,"threads":[{"slot":1,"task":"../a"}]}"#,
        r#"{"phase":"inspect_input","unit":"files","completed":1,"updated_at_ms":1,"input_index":0}"#,
        r#"{"phase":"inspect_input","unit":"files","completed":1,"updated_at_ms":1,"log":"private log"}"#,
        r#"{"phase":"inspect_input","unit":"files","completed":1e500,"updated_at_ms":1}"#,
        r#""not an object""#,
    ].into_iter().enumerate() {
        let text = format!("still working {index}");
        send(format!(r#"{{"state":"running","progress":"{text}","progress_detail":{invalid}}}"#));
        let observed = wait(&text);
        assert_eq!(observed["state"], "running");
        assert!(observed.get("error").is_none());
        assert!(observed.get("progress_detail").is_none());
        assert_eq!(observed["events"], working["events"]);
    }
    send(json!({"state":"running","progress":"nearly done","progress_detail":valid}).to_string());
    assert_eq!(wait("nearly done")["progress_detail"], valid);
    fs::remove_file(&reply_path).unwrap();
    fs::remove_file(app.root.path().join("hold")).unwrap();
    let finished = wait("published");
    assert_eq!(finished["state"], "succeeded");
    assert!(finished.get("progress_detail").is_none());
}

#[test]
fn idle_subscription_emits_a_comment_heartbeat() {
    let app = Rig::new();
    let mut stream = app.request("GET", "/api/events", "", None);
    let heartbeat = stream.frame().unwrap();
    assert!(heartbeat.comment);
    assert!(heartbeat.id.is_none());
    assert!(heartbeat.event.is_none());
}

#[test]
fn subscriptions_are_bounded_and_idle_clients_do_not_block_writes() {
    let app = Rig::new();
    app.prompt(); // Omitted cursor starts live and must not replay this prompt.
    for query in ["-1", "", "1.5", "9223372036854775808", "1&after=2"] {
        app.json("GET", &format!("/api/events?after={query}"), None, 400);
    }
    assert_eq!(
        app.request("GET", "/api/events", "Origin: http://other.test\r\n", None)
            .status,
        403
    );
    assert_eq!(
        app.request(
            "GET",
            "/api/events",
            "Last-Event-ID: 0\r\nLast-Event-ID: 1\r\n",
            None
        )
        .status,
        400
    );
    let mut subscribers: Vec<_> = (0..16)
        .map(|_| app.request("GET", "/api/events", "", None))
        .collect();
    app.json("GET", "/api/events", None, 503);
    let prompt = app.prompt();
    for stream in &mut subscribers {
        let frame = stream.event("change");
        assert_eq!(frame.id, Some(2));
        assert_eq!(frame.data["entity_id"], prompt["id"]);
    }
    drop(subscribers.pop());
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if app.request("GET", "/api/events", "", None).status == 200 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "disconnected subscriber leaked its slot"
        );
        thread::sleep(Duration::from_millis(20));
    }
}
