use super::*;
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader},
    net::TcpListener,
    sync::mpsc,
    thread,
};

const SETTINGS: &str = "/api/settings/agent-provider";
const KEY: &str = "/api/settings/agent-provider/key";
const SESSIONS: &str = "/api/agent/sessions";
const TOKEN: &str = "test-credential-do-not-copy";

fn put(t: &Test, path: &str, body: Value) -> (u16, Value) {
    t.request_method("PUT", path, "", "application/json", Some(body))
}

fn provider(kind: &str, endpoint: &str) -> Value {
    json!({
        "kind":kind,"base_url":endpoint,"model":"fixture-tools","max_tokens":1024,
        "max_steps":12,"max_session_tokens":20000,"timeout_seconds":10,"request_timeout_seconds":4
    })
}

struct Request {
    path: String,
    headers: BTreeMap<String, String>,
    body: Value,
    connection: TcpStream,
}

impl Request {
    fn reply(mut self, status: &str, headers: &str, value: Value) {
        let body = value.to_string();
        write!(self.connection, "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{headers}\r\n{body}", body.len()).unwrap();
    }
}

fn mock() -> (String, mpsc::Receiver<Request>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        listener.set_nonblocking(true).unwrap();
        let until = Instant::now() + Duration::from_secs(20);
        while Instant::now() < until {
            let (connection, _) = match listener.accept() {
                Ok(value) => value,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    sleep(Duration::from_millis(5));
                    continue;
                }
                Err(e) => panic!("{e}"),
            };
            connection
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut reader = BufReader::new(connection);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let path = line.split_whitespace().nth(1).unwrap().to_owned();
            let mut headers = BTreeMap::new();
            loop {
                line.clear();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
                let (key, value) = line.split_once(':').unwrap();
                headers.insert(key.to_ascii_lowercase(), value.trim().to_owned());
            }
            let mut body = vec![0; headers["content-length"].parse().unwrap()];
            reader.read_exact(&mut body).unwrap();
            if tx
                .send(Request {
                    path,
                    headers,
                    body: serde_json::from_slice(&body).unwrap(),
                    connection: reader.into_inner(),
                })
                .is_err()
            {
                return;
            }
        }
    });
    (base, rx)
}

fn next(rx: &mpsc::Receiver<Request>) -> Request {
    rx.recv_timeout(Duration::from_secs(4))
        .expect("hosted request")
}

fn terminal(t: &Test, path: &str) -> Value {
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        let (status, row) = t.request(path, "", None);
        assert_eq!(status, 200, "{row}");
        if row["status"] != "running" {
            return row;
        }
        assert!(Instant::now() < until, "{row}");
        sleep(Duration::from_millis(20));
    }
}

fn start(t: &Test, model: &Value, key: &str, goal: &str) -> (u16, Value) {
    t.request(
        SESSIONS,
        &format!("Idempotency-Key: {key}\r\n"),
        Some(json!({"goal":goal,"model_id":model["id"]})),
    )
}

fn model(t: &Test, source: &std::path::Path) -> Value {
    let (status, parent) = t.request(
        "/api/models",
        "",
        Some(json!({"host":"existing","kind":"safetensors","path":source})),
    );
    assert_eq!(status, 201, "{parent}");
    parent
}

fn use_tool(kind: &str, turn: usize, name: &str, args: Value) -> Value {
    if kind == "anthropic" {
        json!({"stop_reason":"tool_use","usage":{"input_tokens":100,"output_tokens":10},
            "content":[{"type":"tool_use","id":format!("toolu_{turn}"),"name":name,"input":args}]})
    } else {
        json!({"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110},"choices":[
            {"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[
                {"type":"function","id":format!("call_{turn}"),"function":{"name":name,"arguments":args.to_string()}}
            ]}}
        ]})
    }
}

fn finish(kind: &str, text: &str) -> Value {
    if kind == "anthropic" {
        json!({"stop_reason":"end_turn","usage":{"input_tokens":100,"output_tokens":10},
            "content":[{"type":"text","text":text}]})
    } else {
        json!({"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110},"choices":[
            {"finish_reason":"stop","message":{"role":"assistant","content":text}}
        ]})
    }
}

fn last_result(kind: &str, body: &Value, turn: usize) -> Value {
    let last = body["messages"].as_array().unwrap().last().unwrap();
    let text = if kind == "anthropic" {
        assert_eq!(last["role"], "user");
        assert_eq!(last["content"][0]["type"], "tool_result");
        assert_eq!(last["content"][0]["tool_use_id"], format!("toolu_{turn}"));
        &last["content"][0]["content"]
    } else {
        assert_eq!(last["role"], "tool");
        assert_eq!(last["tool_call_id"], format!("call_{turn}"));
        &last["content"]
    };
    serde_json::from_str(text.as_str().unwrap()).unwrap()
}

#[test]
fn hosted_agent_reuses_mcp_scratch_workflow_without_approval() {
    fn system_prompt<'a>(kind: &str, body: &'a Value) -> &'a str {
        if kind == "anthropic" {
            body["system"].as_str().unwrap()
        } else {
            assert_eq!(body["messages"][0]["role"], "system");
            body["messages"][0]["content"].as_str().unwrap()
        }
    }

    for kind in ["anthropic", "openai"] {
        let mut t = Test::new();
        let source = fixture(&t, &[1., 2., 3., 4., 5., 6.]);
        let hash = sha(&fs::read(&source).unwrap());
        mock_scratch_worker(&t);
        t.launch();
        let parent = model(&t, &source);
        let (status, assets) = t.request(
            "/api/models",
            "",
            Some(json!({"host":"existing","kind":"assets","path":source.parent().unwrap()})),
        );
        assert_eq!(status, 201, "{assets}");
        assert_eq!(
            t.admin(Some(
                json!({"action":"set_weight_editor","worker":"existing",
            "revision":0,"enabled":true})
            ))
            .0,
            200
        );
        let (base, rx) = mock();
        let mut config = provider(kind, &base);
        config["timeout_seconds"] = json!(15);
        assert_eq!(
            put(&t, SETTINGS, config.clone()),
            (200, json!({"provider":config,"key_configured":false}))
        );
        assert_eq!(
            put(&t, KEY, json!({"key":TOKEN})),
            (200, json!({"provider":config,"key_configured":true}))
        );

        let (_, mcp) = t.request(
            "/mcp",
            "Accept: application/json, text/event-stream\r\n",
            Some(json!({"jsonrpc":"2.0","id":1,"method":"tools/list"})),
        );
        let goal = "Edit the model and request publish.";
        let (status, accepted) = start(&t, &parent, "run-agent", goal);
        assert_eq!(status, 201, "{accepted}");
        let path = format!("{SESSIONS}/{}", accepted["id"].as_str().unwrap());
        let (status, repeat) = start(&t, &parent, "run-agent", goal);
        assert_eq!(status, 200);
        assert_eq!(repeat["id"], accepted["id"]);
        assert_eq!(start(&t, &parent, "run-agent", "Different goal").0, 409);
        assert_eq!(start(&t, &parent, "other-agent", goal).0, 409);
        let first = next(&rx);
        assert_eq!(first.body["model"], "fixture-tools");
        assert!(first.body.to_string().contains(goal));
        assert!(
            first
                .body
                .to_string()
                .contains(parent["id"].as_str().unwrap())
        );
        assert!(!first.body.to_string().contains(TOKEN));
        if kind == "anthropic" {
            assert_eq!(first.path, "/v1/messages");
            assert_eq!(first.headers["x-api-key"], TOKEN);
            assert_eq!(first.headers["anthropic-version"], "2023-06-01");
            assert_eq!(first.body["max_tokens"], 1024);
        } else {
            assert_eq!(first.path, "/v1/chat/completions");
            assert_eq!(first.headers["authorization"], format!("Bearer {TOKEN}"));
            assert_eq!(first.body["max_tokens"], 1024);
        }
        let prompt = system_prompt(kind, &first.body).to_owned();
        for instruction in [
            "Model Brewery",
            "operator-only",
            "`run_checks`",
            "`compute_direction`",
            "`get_direction`",
            "`ablate_direction`",
            "`search_ablation`",
            "`create_direction_intervention`",
            "`request_publish`",
            "not release-qualified",
        ] {
            assert!(
                prompt.contains(instruction),
                "missing {instruction}: {prompt}"
            );
        }
        let projected: Vec<_> = first.body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                if kind == "anthropic" {
                    json!([entry["name"], entry["description"], entry["input_schema"]])
                } else {
                    let function = &entry["function"];
                    json!([
                        function["name"],
                        function["description"],
                        function["parameters"]
                    ])
                }
            })
            .collect();
        let expected: Vec<_> = mcp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| json!([entry["name"], entry["description"], entry["inputSchema"]]))
            .collect();
        assert_eq!(projected, expected);

        let mut request = Some(first);
        let mut turn = 0;
        let mut tool = |name, args| {
            request
                .take()
                .unwrap()
                .reply("200 OK", "", use_tool(kind, turn, name, args));
            let incoming = next(&rx);
            assert_eq!(system_prompt(kind, &incoming.body), prompt);
            let result = last_result(kind, &incoming.body, turn);
            request = Some(incoming);
            turn += 1;
            result
        };
        let scratch = tool(
            "create_scratch",
            json!({"parent_id":parent["id"],"parent_sha256":hash,"idempotency_key":"agent-create"}),
        );
        let edit = tool(
            "apply_edit",
            json!({"id":scratch["id"],"idempotency_key":"agent-edit", "operations":[
                {"op":"add","tensor":"w","origin":[0,1],"shape":[1,1],"values":[6.]}
            ]}),
        );
        for (revision, key) in [(&scratch, "convert-parent"), (&edit, "convert-child")] {
            let queued = tool(
                "convert_scratch_gguf",
                json!({"id":revision["id"],"sha256":revision["sha256"],
                "assets_id":assets["id"],"idempotency_key":key}),
            );
            let job = tool("get_job", json!({"id":queued["id"],"wait_ms":3500}));
            assert_eq!(job["state"], "succeeded", "{job}");
        }
        let queued = tool(
            "run_checks",
            json!({"id":edit["id"],"sha256":edit["sha256"],"idempotency_key":"agent-check"}),
        );
        let job = tool("get_job", json!({"id":queued["id"],"wait_ms":3500}));
        assert_eq!(job["state"], "succeeded", "{job}");
        let checked = tool("get_checks", json!({"id":edit["id"]}));
        for field in ["integrity", "smoke", "behavior"] {
            assert_eq!(checked[field]["status"], "passed", "{checked}");
        }
        assert_eq!(
            checked["measured_report"]["metrics"]
                .as_array()
                .unwrap()
                .len(),
            17
        );
        assert_eq!(checked["qualification"], "not release-qualified");
        assert_eq!(
            tool(
                "request_publish",
                json!({"id":edit["id"],"sha256":edit["sha256"],"idempotency_key":"agent-publish",
                "acknowledge_bundle":checked["id"]})
            )["state"],
            "publish_requested"
        );
        assert_eq!(
            tool("approve", json!({"id":edit["id"],"sha256":edit["sha256"]}))["error"],
            "unknown tool"
        );
        let (_, live) = t.request(&path, "", None);
        assert_eq!(live["status"], "running");
        assert_eq!(live["entries"].as_array().unwrap().len(), 22);
        assert_eq!(live["scratch_ids"], json!([scratch["id"], edit["id"]]));
        request
            .unwrap()
            .reply("200 OK", "", finish(kind, "The operator must approve."));
        let done = terminal(&t, &path);
        assert_eq!(done["status"], "completed", "{done}");
        assert_eq!(done["steps"], 12);
        assert_eq!(done["tokens"], 1320);
        assert!(!done.to_string().contains(TOKEN));
        assert_eq!(
            t.request("/api/models", "", None).1,
            json!([parent, assets])
        );
        assert_eq!(
            t.request(
                &format!("/api/scratch/{}", edit["id"].as_str().unwrap()),
                "",
                None
            )
            .1["state"],
            "publish_requested"
        );
        t.stop();
        t.launch();
        assert_eq!(t.request(&path, "", None), (200, done));
        assert_eq!(t.request(SETTINGS, "", None).1["key_configured"], true);
    }
}

#[test]
fn hosted_agent_connects_only_to_the_confirmed_base_url() {
    let mut t = Test::new();
    let source = fixture(&t, &[1., 2., 3., 4., 5., 6.]);
    t.launch();
    let parent = model(&t, &source);
    let (base, rx) = mock();
    let (other, redirected) = mock();
    let config = provider("openai", &base);
    assert_eq!(
        t.request(SETTINGS, "", None),
        (200, json!({"provider":null,"key_configured":false}))
    );
    assert_eq!(put(&t, KEY, json!({"key":TOKEN})).0, 409);
    assert_eq!(
        put(
            &t,
            SETTINGS,
            provider("openai", "http://example.invalid/v1")
        )
        .0,
        400
    );
    assert_eq!(
        put(
            &t,
            SETTINGS,
            provider("openai", &format!("{base}/chat/completions/"))
        )
        .0,
        400
    );
    assert_eq!(
        put(
            &t,
            SETTINGS,
            provider("openai", "https://user:pass@example.invalid/v1")
        )
        .0,
        400
    );
    let (status, _) = t.request_method(
        "PUT",
        SETTINGS,
        "Origin: https://other.invalid\r\n",
        "application/json",
        Some(config.clone()),
    );
    assert_eq!(status, 403);
    assert_eq!(put(&t, SETTINGS, config.clone()).0, 200);
    let rejected = put(&t, KEY, json!({"key":format!("{TOKEN}\n")}));
    assert_eq!(rejected.0, 400);
    assert!(!rejected.1.to_string().contains(TOKEN));
    assert_eq!(
        put(&t, KEY, json!({"key":"previous-credential"})).1["key_configured"],
        true
    );
    assert_eq!(put(&t, KEY, json!({"key":TOKEN})).1["key_configured"], true);
    let stored = t.root.path().join("state/agent-credentials/key");
    assert_eq!(
        fs::metadata(stored.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(&stored).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(
        t.request(
            "/api/models",
            "",
            Some(json!({"host":"existing","kind":"assets",
        "path":stored.parent().unwrap()}))
        )
        .0,
        400
    );
    let removed = t.request_method("DELETE", KEY, "", "application/json", None);
    assert_eq!(
        removed,
        (200, json!({"provider":config, "key_configured":false}))
    );
    assert!(!stored.exists());
    assert_eq!(
        start(&t, &parent, "no-key", "Try to start without a key").0,
        409
    );
    assert_eq!(put(&t, KEY, json!({"key":TOKEN})).0, 200);
    let invalid = t.request(SESSIONS, "", Some(json!({TOKEN:"unexpected field"})));
    assert_eq!(invalid.0, 422);
    assert!(!invalid.1.to_string().contains(TOKEN));
    let (_, session) = start(&t, &parent, "redirect", "Reject redirects");
    let path = format!("{SESSIONS}/{}", session["id"].as_str().unwrap());
    let req = next(&rx);
    assert_eq!(req.headers["authorization"], format!("Bearer {TOKEN}"));
    req.reply(
        "307 Temporary Redirect",
        &format!("Location: {other}/chat/completions\r\n"),
        json!({"error":TOKEN}),
    );
    let done = terminal(&t, &path);
    assert_eq!(done["status"], "failed", "{done}");
    assert!(!done.to_string().contains(TOKEN));
    for file in ["controller.db", "controller.db-wal"] {
        let bytes = fs::read(t.root.path().join("state").join(file)).unwrap();
        assert!(
            !bytes
                .windows(TOKEN.len())
                .any(|window| window == TOKEN.as_bytes())
        );
    }
    assert!(redirected.recv_timeout(Duration::from_millis(200)).is_err());
    let (status, changed) = put(&t, SETTINGS, provider("openai", &other));
    assert_eq!(status, 200, "{changed}");
    assert_eq!(changed["key_configured"], false);
    assert!(!stored.exists());
}

#[test]
fn hosted_agent_stops_at_cancel_and_budget_limits() {
    let mut t = Test::new();
    let source = fixture(&t, &[1., 2., 3., 4., 5., 6.]);
    t.launch();
    let parent = model(&t, &source);
    let (base, rx) = mock();
    let mut config = provider("anthropic", &base);
    config["max_steps"] = json!(1);
    config["max_tokens"] = json!(100);
    config["max_session_tokens"] = json!(105);
    assert_eq!(put(&t, SETTINGS, config.clone()).0, 200);
    assert_eq!(put(&t, KEY, json!({"key":TOKEN})).0, 200);
    assert_eq!(
        t.request(
            SESSIONS,
            "Idempotency-Key: no-secret-goal\r\n",
            Some(json!({"goal":TOKEN,"model_id":parent["id"]}))
        )
        .0,
        400
    );

    let get = |value: &Value| format!("{SESSIONS}/{}", value["id"].as_str().unwrap());
    let (_, session) = start(&t, &parent, "cancelled", "Cancel my session");
    let request = next(&rx);
    let path = get(&session);
    assert_eq!(put(&t, KEY, json!({"key":"request-blocked"})).0, 409);
    let (status, cancelled) = t.request(&format!("{path}/cancel"), "", Some(json!({})));
    assert_eq!(status, 200);
    assert_eq!(cancelled["status"], "cancelled");
    drop(request);
    assert_eq!(terminal(&t, &path)["entries"], json!([]));

    let new_session = |t: &Test, key, goal| {
        let until = Instant::now() + Duration::from_secs(4);
        loop {
            let (status, session) = start(t, &parent, key, goal);
            if status == 201 {
                return get(&session);
            }
            assert_eq!(status, 409, "{session}");
            assert!(Instant::now() < until, "{session}");
            sleep(Duration::from_millis(20));
        }
    };
    let path = new_session(&t, "token-budget", "Stop when tokens run out");
    let request = next(&rx);
    assert_eq!(request.body["max_tokens"], 100);
    request.reply(
        "200 OK",
        "",
        use_tool("anthropic", 0, "list_models", json!({})),
    );
    let exhausted = terminal(&t, &path);
    assert_eq!(exhausted["status"], "budget_exhausted", "{exhausted}");
    assert_eq!(exhausted["tokens"], 110);
    assert_eq!(exhausted["entries"], json!([]));

    let path = new_session(&t, "echo", "Reject private data");
    next(&rx).reply("200 OK", "", finish("anthropic", &format!("Key: {TOKEN}")));
    let failed = terminal(&t, &path);
    assert_eq!(failed["status"], "failed");
    assert!(failed["entries"].as_array().unwrap().is_empty());
    assert!(!failed.to_string().contains(TOKEN));

    let path = new_session(&t, "step-budget", "Stop before a second turn");
    let mut reply = use_tool("anthropic", 0, "list_models", json!({}));
    reply["usage"]["input_tokens"] = json!(20);
    next(&rx).reply("200 OK", "", reply);
    let exhausted = terminal(&t, &path);
    assert_eq!(exhausted["status"], "budget_exhausted");
    assert_eq!(exhausted["steps"], 1);
    assert_eq!(exhausted["entries"].as_array().unwrap().len(), 2);

    let path = new_session(&t, "controller-stop", "Stop the controller");
    drop(next(&rx));
    t.stop();
    t.launch();
    let failed = terminal(&t, &path);
    assert_eq!(failed["status"], "failed", "{failed}");
    let summaries = t.request(SESSIONS, "", None).1;
    assert_eq!(summaries[0]["id"], failed["id"]);
    assert!(summaries[0].get("entries").is_none());

    config["timeout_seconds"] = json!(1);
    assert_eq!(put(&t, SETTINGS, config).1["key_configured"], true);
    let path = new_session(&t, "wall-budget", "Stop before the provider replies");
    let held = next(&rx);
    let exhausted = terminal(&t, &path);
    assert_eq!(exhausted["status"], "budget_exhausted", "{exhausted}");
    assert_eq!(exhausted["message"], "The session reached its time limit.");
    drop(held);
}
