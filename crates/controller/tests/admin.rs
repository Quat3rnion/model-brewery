use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    os::unix::fs::PermissionsExt,
    process::{Child, Command, Stdio},
    thread::sleep,
    time::{Duration, Instant},
};

#[path = "support/scratch.rs"]
mod scratch;

struct Test {
    root: tempfile::TempDir,
    child: Option<Child>,
    addr: String,
}

impl Test {
    fn new() -> Self {
        let root = tempfile::Builder::new()
            .prefix("brewery-admin-")
            .tempdir()
            .unwrap();
        for (name, value) in [
            ("catalog.json", json!({"workers":{"candidate":{"ssh_alias":"fixture-ssh",
                "runtimes":{"evaluator":{"label":"Evaluator","capabilities":["evaluate"]}}}}}).to_string()),
            ("dispatch.json", json!({"hosts":{"candidate":{"ssh":"fixture-ssh","quantization_pool":true,
                "capabilities":{"evaluate":true}},"existing":{"local":true}}}).to_string()),
            ("dispatch", "#!/bin/sh\nif [ \"$1\" = suggest-paths ]; then cat >/dev/null; printf '{\"items\":[],\"truncated\":false}\\n'; else exit 1; fi\n".into()),
        ] {
            let file = root.path().join(name);
            fs::write(&file, value).unwrap();
            fs::set_permissions(file, fs::Permissions::from_mode(if name=="dispatch" {0o700} else {0o600})).unwrap();
        }
        Self {
            root,
            child: None,
            addr: String::new(),
        }
    }

    fn launch(&mut self) {
        let root = self.root.path();
        let log = root.join("log");
        let mut command = Command::new(env!("CARGO_BIN_EXE_brewery"));
        if root.join("runtime-helper").exists() {
            command.env("BREWERY_ADMIN_RUNTIME_HELPER", root.join("runtime-helper"));
        } else {
            command.env_remove("BREWERY_ADMIN_RUNTIME_HELPER");
        }
        command
            .env_remove("BREWERY_ADMIN_KEY_FILE")
            .env_remove("BREWERY_ADMIN_CATALOG")
            .env("BREWERY_LISTEN", "127.0.0.1:0")
            .env("BREWERY_DATA", root.join("state"))
            .env("BREWERY_DISPATCH", root.join("dispatch"))
            .env("BREWERY_DISPATCH_CONFIG", root.join("dispatch.json"))
            .env("BREWERY_POLL_MS", "25")
            .stdout(Stdio::from(fs::File::create(&log).unwrap()));
        if root.join("hfr").exists() {
            for env in [
                "BREWERY_HF_DISPATCH",
                "BREWERY_HF_WORKER",
                "BREWERY_HF_ENGINE",
                "BREWERY_HF_HFR",
            ] {
                command.env(env, root.join("hfr"));
            }
        } else {
            command.env("BREWERY_HF_HFR", root.join("no-hfr"));
        }
        if root.join("catalog.json").exists() {
            command.env("BREWERY_ADMIN_CATALOG", root.join("catalog.json"));
        }
        for name in ["DISPATCH", "WORKER", "ENGINE", "HELPER", "TUNER"] {
            command.env(
                format!("BREWERY_LOCAL_EDIT_{name}"),
                root.join(format!("editor-{name}")),
            );
        }
        command
            .env("BREWERY_COMPILED_PROGRAM_PYTHON", "/usr/bin/python3")
            .env("BREWERY_COMPILED_PROGRAM_SCRIPT", root.join("program.py"));
        let mut child = command.spawn().unwrap();
        let until = Instant::now() + Duration::from_secs(4);
        self.addr = loop {
            if let Some(addr) = fs::read_to_string(&log)
                .unwrap()
                .lines()
                .find_map(|s| s.strip_prefix("brewery listening on http://"))
            {
                break addr.to_owned();
            }
            assert!(
                Instant::now() < until && child.try_wait().unwrap().is_none(),
                "controller unavailable"
            );
            sleep(Duration::from_millis(10));
        };
        self.child = Some(child);
    }

    fn request_with_type(
        &self,
        path: &str,
        headers: &str,
        media_type: &str,
        body: Option<Value>,
    ) -> (u16, Value) {
        let method = if body.is_some() { "POST" } else { "GET" };
        self.request_method(method, path, headers, media_type, body)
    }
    fn request_method(
        &self,
        method: &str,
        path: &str,
        headers: &str,
        media_type: &str,
        body: Option<Value>,
    ) -> (u16, Value) {
        let body = body.map(|b| b.to_string()).unwrap_or_default();
        let mut stream = TcpStream::connect(&self.addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(12)))
            .unwrap();
        write!(stream,"{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Type: {media_type}\r\nContent-Length: {}\r\n{headers}\r\n{body}",self.addr,body.len()).unwrap();
        let mut raw = String::new();
        stream.read_to_string(&mut raw).unwrap();
        let (head, body) = raw.split_once("\r\n\r\n").unwrap();
        (
            head.split_whitespace().nth(1).unwrap().parse().unwrap(),
            serde_json::from_str(body).unwrap(),
        )
    }
    fn request(&self, path: &str, headers: &str, body: Option<Value>) -> (u16, Value) {
        self.request_with_type(path, headers, "application/json", body)
    }
    fn admin(&self, body: Option<Value>) -> (u16, Value) {
        self.request(
            if body.is_some() {
                "/api/admin/actions"
            } else {
                "/api/admin/state"
            },
            "",
            body,
        )
    }
    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            child.wait().unwrap();
        }
    }
}

impl Drop for Test {
    fn drop(&mut self) {
        self.stop();
    }
}

#[test]
fn infrastructure_uses_normal_request_guards_without_a_second_key() {
    let mut t = Test::new();
    t.launch();
    assert_eq!(t.admin(None).0, 200);
    let path = "/api/admin/actions";
    let action = json!({"action":"enroll","worker":"candidate"});
    for headers in [
        "Origin: https://other.invalid\r\n",
        "Sec-Fetch-Site: cross-site\r\n",
        "Host: other.invalid\r\n",
    ] {
        assert_eq!(t.request("/api/admin/state", headers, None).0, 403);
        assert_eq!(t.request(path, headers, Some(action.clone())).0, 403);
    }
    assert_eq!(
        t.request_with_type(path, "", "text/plain", Some(action.clone()))
            .0,
        415
    );
    let before = t.admin(None).1;
    assert_eq!(before["workers"][0]["status"], "unregistered");
    assert_eq!(before["audit"], json!([]));
    let (status, accepted) = t.admin(Some(action));
    assert_eq!(status, 200, "{accepted}");
    assert_eq!(accepted["workers"][0]["status"], "pending");
    assert_eq!(accepted["audit"][0]["action"], "enroll");
}

#[test]
fn defined_hosts_have_revisioned_policy_without_catalog_workers() {
    let mut t = Test::new();
    fs::remove_file(t.root.path().join("catalog.json")).unwrap();
    t.launch();
    assert_eq!(t.request("/api/state", "", None).0, 200);
    let (status, state) = t.admin(None);
    assert_eq!(status, 200);
    assert_eq!(state["workers"], json!([]));
    assert_eq!(state["hosts"].as_array().unwrap().len(), 2);
    let row = |state: Value| {
        state["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|h| h["id"] == "existing")
            .unwrap()
            .clone()
    };
    assert_eq!(row(state)["revision"], 0);
    assert_eq!(
        t.admin(Some(json!({"action":"enroll","worker":"candidate"})))
            .0,
        404
    );
    let policy = |revision, paused, jobs| {
        json!({"action":"set_host_policy","worker":"existing",
       "revision":revision,"paused":paused,"disabled_jobs":jobs})
    };
    assert_eq!(t.admin(Some(policy(0, false, json!(["evaluate"])))).0, 400);
    assert_eq!(
        t.admin(Some(policy(0, false, json!(["inspect", "inspect"]))))
            .0,
        400
    );
    let (status, updated) = t.admin(Some(policy(0, true, json!(["inspect"]))));
    assert_eq!(status, 200, "{updated}");
    assert_eq!(row(updated.clone())["paused"], true);
    assert_eq!(row(updated.clone())["revision"], 1);
    assert_eq!(row(updated.clone())["disabled_jobs"], json!(["inspect"]));
    assert_eq!(updated["audit"][0]["action"], "set_host_policy");
    assert_eq!(t.admin(Some(policy(0, false, json!([])))).0, 409);
    let resource = |revision, threads| {
        json!({"action":"set_host_resources","worker":"existing",
        "revision":revision,"resources":{"cpu_threads":threads}})
    };
    assert_eq!(t.admin(Some(resource(0, 2))).0, 409);
    assert_eq!(t.admin(Some(resource(1, 65))).0, 400);
    let (status, updated) = t.admin(Some(resource(1, 2)));
    assert_eq!(status, 200, "{updated}");
    assert_eq!(row(updated)["effective_resources"]["cpu_threads"], 2);
    t.stop();
    t.launch();
    let restored = row(t.admin(None).1);
    assert_eq!(restored["paused"], true);
    assert_eq!(restored["resources"]["cpu_threads"], 2);
    let (status, updated) = t.admin(Some(policy(2, false, json!([]))));
    assert_eq!(status, 200, "{updated}");
    let resumed = row(updated);
    assert_eq!(resumed["paused"], false);
    assert_eq!(resumed["revision"], 3);
    assert_eq!(resumed["resources"]["cpu_threads"], 2);
}

#[test]
fn local_weight_editor_requires_an_installed_executor() {
    let mut t = Test::new();
    t.launch();
    let (status, state) = t.admin(None);
    assert_eq!(status, 200);
    let row = state["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|host| host["id"] == "existing")
        .unwrap();
    assert_eq!(row["weight_editor"]["available"], false);
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"existing",
        "revision":0,"enabled":true})
        ))
        .0,
        409
    );
}

#[test]
fn enabled_local_editor_uses_private_route_and_drains_on_disable() {
    const NEW_WORK: &str = "Brewery-Allow-Duplicate: 1\r\n";
    let mut t = Test::new();
    let root = t.root.path();
    let sources = root.join("sources");
    fs::create_dir(&sources).unwrap();
    let config = root.join("local-worker.json");
    let original = json!({"root":root.join("ordinary"),"source_roots":[sources],"engine":"/usr/bin/false",
        "engine_env":{"BREWERY_EXPECTED_SOURCE_SHA256":"qualification-pin"}});
    fs::write(&config, original.to_string()).unwrap();
    let eval_config = root.join("eval-worker.json");
    let index = root.join("index.json");
    fs::write(&index, b"{}").unwrap();
    let pin = format!("{:x}", Sha256::digest(b"{}"));
    fs::write(&eval_config, r#"{"source_roots":[]}"#).unwrap();
    fs::write(
        root.join("dispatch.json"),
        json!({"hosts":{"existing":{
        "local":true,"worker":"/usr/bin/false","config":config,
        "local_edit_chunk_index":{"path":index,"sha256":pin}},
        "local-eval":{"local":true,"worker":"/usr/bin/false","config":eval_config}}})
        .to_string(),
    )
    .unwrap();
    fs::remove_file(root.join("catalog.json")).unwrap();
    for (name, script) in [
        ("editor-DISPATCH", include_str!("support/dispatch.py")),
        ("editor-WORKER", "#!/bin/sh\nexit 1\n"),
        ("editor-ENGINE", "#!/bin/sh\nexit 1\n"),
        (
            "editor-HELPER",
            r#"#!/bin/sh
test "$1" = --capabilities || exit 1
printf '%s\n' '{"schema":"brewery.weight-runtime.v1","jobs":["weight_inspect","weight_propose","weight_apply"],"formats":["gguf","hf","safetensors"],"visual":true,"packed":false,"proof_source_fds":true}'
"#,
        ),
    ] {
        let file = root.join(name);
        fs::write(&file, script).unwrap();
        fs::set_permissions(file, fs::Permissions::from_mode(0o700)).unwrap();
    }
    fs::write(root.join("hold"), "").unwrap();
    fs::write(root.join("weight-fixture"), "").unwrap();
    let source_path = sources.join("checkpoint");
    t.launch();
    let toggle = |revision, enabled| json!({"action":"set_weight_editor","worker":"existing","revision":revision,"enabled":enabled});
    let (status, initial) = t.admin(None);
    assert_eq!(status, 200);
    let editor: Value = serde_json::from_slice(
        &fs::read(t.root.path().join("state/local-edit/existing/worker.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        editor["engine_env"]["BREWERY_WEIGHT_CHUNK_INDEX"],
        json!(index)
    );
    assert_eq!(
        editor["engine_env"]["BREWERY_WEIGHT_CHUNK_INDEX_SHA256"],
        pin
    );
    assert_eq!(initial["hosts"][0]["weight_editor"]["available"], true);
    assert_eq!(initial["hosts"][0]["weight_editor"]["enabled"], false);
    let eval = initial["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["id"] == "local-eval")
        .unwrap();
    assert_eq!(eval["model_directories"]["available"], true);
    let (status, plan) = t.request(
        "/api/admin/model-directories/plan",
        "",
        Some(json!({
            "worker":"local-eval","revision":0,"action":"add","path":sources
        })),
    );
    assert_eq!(status, 200, "{plan}");
    assert_eq!(plan["worker"], "local-eval");
    assert_eq!(plan["path"], json!(sources));
    assert_eq!(eval["weight_editor"]["available"], false);
    let reason = eval["weight_editor"]["reason"].as_str().unwrap();
    assert!(reason.contains("no approved model directory"), "{reason}");
    let (denied, error) = t.admin(Some(
        json!({"action":"set_weight_editor","worker":"local-eval","revision":0,"enabled":true}),
    ));
    assert_eq!(denied, 409);
    assert_eq!(error["error"], reason);
    let (status, source) = t.request(
        "/api/models",
        "",
        Some(json!({"host":"existing","kind":"hf","path":source_path})),
    );
    assert_eq!(status, 201, "{source}");
    let req = json!({"kind":"weight_inspect","input_ids":[source["id"]],"options":{"mode":"inventory","offset":0,"limit":4}});
    assert_eq!(t.request("/api/jobs", "", Some(req.clone())).0, 409);
    let (status, enabled) = t.admin(Some(toggle(0, true)));
    assert_eq!(status, 200, "{enabled}");
    assert_eq!(
        enabled["hosts"][0]["weight_editor"]["capabilities"]["weight_tune"],
        false
    );
    assert_eq!(
        enabled["hosts"][0]["weight_editor"]["capabilities"]["weight_safetensors"],
        true
    );
    assert_eq!(enabled["audit"][0]["action"], "set_weight_editor");
    let (status, job) = t.request("/api/jobs", "", Some(req.clone()));
    assert_eq!(status, 201, "{job}");
    let path = format!("/api/jobs/{}", job["id"].as_str().unwrap());
    let wait = |progress: &str| {
        let until = Instant::now() + Duration::from_secs(4);
        loop {
            let (_, row) = t.request(&path, "", None);
            if row["state"] == progress || row["progress"] == progress {
                return row;
            }
            assert!(Instant::now() < until, "{row}");
            sleep(Duration::from_millis(15));
        }
    };
    wait("worker progress");
    assert_eq!(t.admin(Some(toggle(0, false))).0, 409);
    assert_eq!(t.admin(Some(toggle(1, false))).0, 200);
    assert_eq!(t.request("/api/jobs", NEW_WORK, Some(req.clone())).0, 409);
    fs::remove_file(t.root.path().join("hold")).unwrap();
    assert_eq!(wait("succeeded")["result"]["source_format"], "hf");
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(config).unwrap()).unwrap(),
        original
    );
    t.stop();
    t.launch();
    assert_eq!(
        t.admin(None).1["hosts"][0]["weight_editor"]["enabled"],
        false
    );
    assert_eq!(t.request("/api/jobs", NEW_WORK, Some(req)).0, 409);

    let grant = |action, revision, path: &std::path::Path| {
        t.request(
            "/api/admin/model-directories/plan",
            "",
            Some(json!({
                "worker":"local-eval","revision":revision,"action":action,"path":path
            })),
        )
    };
    let confirm = |plan: &Value| {
        t.request("/api/admin/model-directories/confirm", "", Some(json!({
        "worker":plan["worker"],"revision":plan["revision"],"confirmation":plan["confirmation"]
    })))
    };
    assert_eq!(grant("add", 0, std::path::Path::new("/etc")).0, 400);
    assert_eq!(grant("add", 0, &t.root.path().join("state")).0, 400);
    assert_eq!(confirm(&plan).0, 200);
    assert_eq!(confirm(&plan).0, 409);
    let eval = |state: Value| {
        state["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|h| h["id"] == "local-eval")
            .unwrap()
            .clone()
    };
    let saved = eval(t.admin(None).1);
    assert_eq!(saved["model_directories"]["granted"], json!([sources]));
    assert_eq!(saved["weight_editor"]["enabled"], false);
    assert_eq!(saved["weight_editor"]["available"], true);
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"local-eval",
        "revision":1,"enabled":true})
        ))
        .0,
        200
    );
    let model_path = sources.join("test.gguf");
    fs::write(&model_path, "fixture").unwrap();
    // The regular worker cannot see the directory approved for the editor.
    let dispatch = t.root.path().join("dispatch");
    fs::write(
        &dispatch,
        "#!/bin/sh\ncat >/dev/null\nprintf '{\"state\":\"outside_roots\"}\\n'\n",
    )
    .unwrap();
    let (status, model) = t.request(
        "/api/models",
        "",
        Some(json!({
            "host":"local-eval","kind":"gguf","path":model_path
        })),
    );
    assert_eq!(status, 201, "{model}");
    let request = json!({"kind":"weight_inspect","input_ids":[model["id"]],
        "options":{"mode":"inventory","offset":0,"limit":4}});
    fs::write(t.root.path().join("hold"), "").unwrap();
    let (status, running) = t.request("/api/jobs", "", Some(request.clone()));
    assert_eq!(status, 201, "{running}");
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        let (_, state) = t.request("/api/state", "", None);
        let row = state["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == model["id"])
            .unwrap();
        if row["reason"] != "The controller has not checked this path on the worker." {
            assert_eq!(row["available"], Value::Null, "{row}");
            assert_eq!(
                row["reason"],
                "The standard worker could not find or access this path. Editor access is not checked."
            );
            break;
        }
        assert!(Instant::now() < deadline, "{row}");
        sleep(Duration::from_millis(15));
    }
    let poll = |id: &Value, field, expected: &str| {
        let deadline = Instant::now() + Duration::from_secs(4);
        let path = format!("/api/jobs/{}", id.as_str().unwrap());
        loop {
            let (_, row) = t.request(&path, "", None);
            if row[field] == expected {
                return row;
            }
            assert!(Instant::now() < deadline, "{row}");
            sleep(Duration::from_millis(15));
        }
    };
    poll(&running["id"], "progress", "worker progress");
    let (status, queued) = t.request("/api/jobs", NEW_WORK, Some(request.clone()));
    assert_eq!(status, 201, "{queued}");
    assert_eq!(queued["state"], "queued");
    let (status, revoke) = grant("remove", 2, &sources);
    assert_eq!(status, 200, "{revoke}");
    assert_eq!(confirm(&revoke).0, 200);
    assert_eq!(
        t.request("/api/jobs", NEW_WORK, Some(request.clone())).0,
        409
    );
    fs::remove_file(t.root.path().join("hold")).unwrap();
    poll(&running["id"], "state", "succeeded");
    poll(&queued["id"], "state", "failed");
    t.stop();
    t.launch();
    assert_eq!(
        eval(t.admin(None).1)["model_directories"]["granted"],
        json!([])
    );
    assert_eq!(t.request("/api/jobs", NEW_WORK, Some(request)).0, 409);
    assert_eq!(fs::read_to_string(model_path).unwrap(), "fixture");
}

#[test]
fn backend_download_lane_is_visible_and_can_be_restricted() {
    let mut t = Test::new();
    fs::remove_file(t.root.path().join("catalog.json")).unwrap();
    let hfr = t.root.path().join("hfr");
    fs::write(&hfr, "#!/bin/sh\nexit 1\n").unwrap();
    fs::set_permissions(&hfr, fs::Permissions::from_mode(0o700)).unwrap();
    t.launch();
    let host = |state: Value| {
        state["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|h| h["id"] == "backend-hf")
            .unwrap()
            .clone()
    };
    let initial = host(t.admin(None).1);
    assert!(
        initial["installed_jobs"]
            .as_array()
            .unwrap()
            .contains(&json!("hf_download"))
    );
    let (status, restricted) = t.admin(Some(json!({
        "action":"set_host_policy","worker":"backend-hf","revision":0,"paused":false,
        "disabled_jobs":["hf_download"]
    })));
    assert_eq!(status, 200, "{restricted}");
    assert!(
        !host(restricted)["allowed_jobs"]
            .as_array()
            .unwrap()
            .contains(&json!("hf_download"))
    );
    assert_eq!(
        t.request(
            "/api/jobs",
            "",
            Some(json!({
                "kind":"hf_download","host":"backend-hf","options":{"repo_id":"fixture/model"}
            }))
        )
        .0,
        409
    );
}

#[test]
fn pause_drains_running_job_and_blocks_new_work_bound_to_the_source() {
    let mut t = Test::new();
    fs::remove_file(t.root.path().join("catalog.json")).unwrap();
    fs::write(
        t.root.path().join("dispatch"),
        include_str!("support/dispatch.py"),
    )
    .unwrap();
    fs::write(t.root.path().join("hold"), b"").unwrap();
    t.launch();
    let (status, source) = t.request(
        "/api/models",
        "",
        Some(json!({
            "host":"existing","kind":"gguf","path":"/models/test.gguf"
        })),
    );
    assert_eq!(status, 201, "{source}");
    let inspect = json!({"kind":"inspect","input_ids":[source["id"]]});
    let (status, running) = t.request("/api/jobs", "", Some(inspect.clone()));
    assert_eq!(status, 201, "{running}");
    let path = format!("/api/jobs/{}", running["id"].as_str().unwrap());
    let await_job = |path: &str, condition: &dyn Fn(&Value) -> bool| {
        let deadline = Instant::now() + Duration::from_secs(4);
        loop {
            let (status, row) = t.request(path, "", None);
            assert_eq!(status, 200, "{row}");
            if condition(&row) {
                break row;
            }
            assert!(Instant::now() < deadline, "{row}");
            sleep(Duration::from_millis(15));
        }
    };
    await_job(&path, &|row| row["progress"] == "worker progress");
    let (status, queued) = t.request(
        "/api/jobs",
        "",
        Some(json!({
            "kind":"convert","host":"auto","input_ids":[source["id"]]
        })),
    );
    assert_eq!(status, 201, "{queued}");
    assert_eq!(queued["state"], "queued");
    let (status, state) = t.admin(Some(json!({
        "action":"set_host_policy","worker":"existing","revision":0,"paused":true,
        "disabled_jobs":["inspect"]
    })));
    assert_eq!(status, 200, "{state}");
    let row = |state: Value| {
        state["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|h| h["id"] == "existing")
            .unwrap()
            .clone()
    };
    assert_eq!(row(state)["draining"], true);
    for request in [
        inspect,
        json!({"kind":"transfer_preview","host":"candidate","input_ids":[source["id"]]}),
        json!({"kind":"convert","host":"auto","input_ids":[source["id"]]}),
    ] {
        let (status, response) =
            t.request("/api/jobs", "Brewery-Allow-Duplicate: 1\r\n", Some(request));
        assert_eq!(status, 409, "{response}");
    }
    fs::remove_file(t.root.path().join("hold")).unwrap();
    let terminal = await_job(&path, &|job| job["state"] != "running");
    assert_eq!(terminal["state"], "succeeded", "{terminal}");
    let queued_path = format!("/api/jobs/{}", queued["id"].as_str().unwrap());
    let refused = await_job(&queued_path, &|job| {
        !matches!(job["state"].as_str(), Some("queued" | "running"))
    });
    assert_eq!(refused["state"], "failed", "{refused}");
    assert_eq!(refused["host"], "auto");
    assert_eq!(row(t.admin(None).1)["draining"], false);
}

#[test]
fn catalog_approval_and_resource_changes_apply_and_survive_restart() {
    let mut t = Test::new();
    t.launch();
    let visible = |t: &Test| {
        t.request("/api/state", "", None).1["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|h| h["id"] == "candidate")
    };
    assert!(!visible(&t));
    assert_eq!(
        t.admin(Some(json!({"action":"approve","worker":"candidate"})))
            .0,
        409
    );
    assert_eq!(
        t.admin(Some(json!({"action":"enroll","worker":"unknown"})))
            .0,
        404
    );
    assert_eq!(
        t.admin(Some(
            json!({"action":"enroll","worker":"candidate","ssh":"injected"})
        ))
        .0,
        400
    );
    assert_eq!(
        t.admin(Some(json!({"action":"enroll","worker":"candidate"})))
            .1["workers"][0]["status"],
        "pending"
    );
    assert!(!visible(&t));
    let (status, state) = t.admin(Some(json!({"action":"approve","worker":"candidate"})));
    assert_eq!(status, 200, "{state}");
    assert_eq!(state["workers"][0]["status"], "active");
    assert!(visible(&t));
    let (status, policy) = t.admin(Some(
        json!({"action":"set_resource_policy","worker":"candidate","allow_pool":false}),
    ));
    assert_eq!(status, 200, "{policy}");
    assert_eq!(policy["workers"][0]["allow_pool"], false);
    let normal = t.request("/api/state", "", None).1;
    let candidate = normal["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["id"] == "candidate")
        .unwrap();
    assert_eq!(candidate["quantization_pool"], false);
    assert_eq!(candidate["capabilities"]["evaluate"], false);
    assert_eq!(
        t.admin(Some(
            json!({"action":"install_runtime","worker":"candidate","runtime":"evaluator"})
        ))
        .0,
        409
    );
    t.stop();
    t.launch();
    let (_, state) = t.admin(None);
    assert_eq!(state["workers"][0]["status"], "active");
    assert_eq!(state["workers"][0]["allow_pool"], false);
    assert_eq!(state["audit"][0]["action"], "set_resource_policy");
    assert_eq!(state["audit"][0]["outcome"], "applied");
    assert_eq!(
        t.admin(Some(json!({"action":"deactivate","worker":"candidate"})))
            .0,
        200
    );
    assert!(!visible(&t));
}

#[test]
fn pinned_runtime_requires_verified_install_and_explicit_enable() {
    let mut t = Test::new();
    let root = t.root.path();
    let catalog = root.join("catalog.json");
    let mut data: Value = serde_json::from_slice(&fs::read(&catalog).unwrap()).unwrap();
    data["workers"]["candidate"]["runtimes"]["evaluator"]["sha256"] =
        json!(format!("{:x}", Sha256::digest(b"hello world")));
    fs::write(catalog, data.to_string()).unwrap();
    let helper = root.join("runtime-helper");
    fs::write(
        &helper,
        r#"#!/bin/sh
set -eu
test "$#" = 4
test "$2" = candidate
test "$3" = evaluator
cd -- "$(dirname -- "$0")"
digest() { sha256sum -- "$1" | cut -d' ' -f1; }
case "$1" in
  install-pinned)
    test "$(digest staged)" = "$4"
    cp -- staged installed.new
    test "$(digest installed.new)" = "$4"
    mv -- installed.new installed
    ;;
  probe-installed) ;;
  *) exit 1 ;;
esac
test "$(digest installed)" = "$4"
printf '{"sha256":"%s"}\n' "$4"
"#,
    )
    .unwrap();
    fs::set_permissions(helper, fs::Permissions::from_mode(0o700)).unwrap();
    t.launch();
    let runtime = |verb| {
        t.admin(Some(
            json!({"action":verb,"worker":"candidate","runtime":"evaluator"}),
        ))
    };
    assert_eq!(
        t.admin(Some(json!({"action":"enroll","worker":"candidate"})))
            .0,
        200
    );
    assert_eq!(runtime("enable_runtime").0, 409);
    fs::write(t.root.path().join("staged"), b"bad bytes").unwrap();
    assert_eq!(runtime("install_runtime").0, 409);
    assert!(!t.root.path().join("installed").exists());
    fs::write(t.root.path().join("staged"), b"hello world").unwrap();
    let (status, installed) = runtime("install_runtime");
    assert_eq!(status, 200, "{installed}");
    assert_eq!(installed["workers"][0]["runtimes"][0]["installed"], true);
    assert_eq!(installed["workers"][0]["runtimes"][0]["enabled"], false);
    assert_eq!(runtime("enable_runtime").0, 200);
    assert_eq!(
        t.admin(Some(json!({"action":"approve","worker":"candidate"})))
            .0,
        200
    );
    let state = t.request("/api/state", "", None).1;
    let candidate = state["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["id"] == "candidate")
        .unwrap();
    assert_eq!(candidate["capabilities"]["evaluate"], true);
    assert_eq!(runtime("install_runtime").0, 409);
    assert_eq!(runtime("disable_runtime").0, 200);
}

#[test]
fn explicitly_cataloged_local_worker_can_be_deactivated_and_readmitted() {
    let mut t = Test::new();
    let root = t.root.path();
    let catalog = root.join("catalog.json");
    let dispatch = root.join("dispatch.json");
    let worker_config = root.join("local-worker.json");
    fs::write(&worker_config, b"{}").unwrap();
    fs::write(
        &catalog,
        json!({"workers":{"existing":{"local":true,"initially_active":true}}}).to_string(),
    )
    .unwrap();
    fs::write(
        &dispatch,
        json!({"hosts":{"existing":{"local":true,
        "worker":root.join("dispatch"),"config":worker_config,"capabilities":{"evaluate":true}}}})
        .to_string(),
    )
    .unwrap();
    t.launch();
    assert_eq!(t.admin(None).1["workers"][0]["status"], "active");
    assert_eq!(
        t.request("/api/state", "", None).1["hosts"][0]["capabilities"]["evaluate"],
        true
    );
    assert_eq!(
        t.admin(Some(json!({"action":"deactivate","worker":"existing"})))
            .0,
        200
    );
    assert_eq!(t.request("/api/state", "", None).1["hosts"], json!([]));
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_resource_policy","worker":"existing","allow_pool":true})
        ))
        .0,
        409
    );
    assert_eq!(
        t.admin(Some(json!({"action":"approve","worker":"existing"})))
            .0,
        200
    );
    assert_eq!(
        t.request("/api/state", "", None).1["hosts"][0]["capabilities"]["evaluate"],
        true
    );
}
