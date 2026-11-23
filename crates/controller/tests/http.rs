use serde_json::{Value, json};

#[path = "support/hf_download.rs"]
mod hf_download;

#[path = "support/hf_tokens.rs"]
mod hf_tokens;

#[path = "support/infra_hosts.rs"]
mod infra_hosts;

#[path = "support/interventions.rs"]
mod interventions;

#[path = "support/weights.rs"]
mod weights;

#[path = "support/settings.rs"]
mod settings;
use std::{
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread::sleep,
    time::{Duration, Instant},
};

#[path = "support/conversions.rs"]
mod conversions;

#[path = "support/exl3_layer.rs"]
mod exl3_layer;

struct Fixture {
    root: tempfile::TempDir,
    addr: Option<SocketAddr>,
    child: Option<Child>,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("dispatch.json"),
            json!({
                "hosts": {
                    "alpha":{"local":true},
                    "beta":{"ssh":"beta","capabilities":{"generate":true,"evaluate":true,"baseline":true,"plan":false},"reasons":{"plan":"not installed"},"baseline_sources":[{"id":"provider","label":"Controlled test baseline"}]}
                }
            }).to_string(),
        )
        .unwrap();
        fs::create_dir(root.path().join("dist")).unwrap();
        fs::write(root.path().join("dist/index.html"), "brewery static asset").unwrap();
        let dispatch = root.path().join("dispatch");
        fs::write(&dispatch, include_str!("support/dispatch.py")).unwrap();
        fs::set_permissions(&dispatch, fs::Permissions::from_mode(0o700)).unwrap();
        Self {
            root,
            addr: None,
            child: None,
        }
    }

    fn launch(&mut self) {
        self.launch_with(&[]);
    }

    fn launch_with(&mut self, env: &[(&str, PathBuf)]) {
        let log_path = self.path("server.log");
        let log = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&log_path)
            .unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_brewery"))
            .env("BREWERY_LISTEN", "127.0.0.1:0")
            .env("BREWERY_DATA", self.path("state"))
            .env("BREWERY_DISPATCH", self.path("dispatch"))
            .env("BREWERY_DISPATCH_CONFIG", self.path("dispatch.json"))
            .env("BREWERY_WEB_DIST", self.path("dist"))
            .env("BREWERY_POLL_MS", "25")
            .envs(env.iter().map(|(key, value)| (*key, value)))
            .stdout(Stdio::from(log))
            .spawn()
            .unwrap();
        self.addr = Some(wait_for(|| {
            assert!(
                child.try_wait().unwrap().is_none(),
                "controller exited before binding"
            );
            let text = fs::read_to_string(&log_path).unwrap();
            text.lines().find_map(|line| {
                line.strip_prefix("brewery listening on http://")?
                    .parse()
                    .ok()
            })
        }));
        self.child = Some(child);
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            child.wait().unwrap();
        }
    }

    fn path(&self, key: &str) -> PathBuf {
        self.root.path().join(key)
    }

    fn open_request(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        headers: &str,
    ) -> TcpStream {
        let addr = self.addr.unwrap();
        let mut conn = TcpStream::connect(addr).unwrap();
        conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let body = body.as_ref().map(Value::to_string);
        let host = if headers.contains("Host:") {
            String::new()
        } else {
            format!("Host: {addr}\r\n")
        };
        write!(conn, "{method} {path} HTTP/1.0\r\n{host}{headers}").unwrap();
        if let Some(body) = &body {
            write!(
                conn,
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                body.len()
            )
            .unwrap();
        }
        write!(conn, "\r\n{}", body.as_deref().unwrap_or("")).unwrap();
        conn
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        headers: &str,
    ) -> (u16, String) {
        let mut conn = self.open_request(method, path, body, headers);
        let mut data = String::new();
        conn.read_to_string(&mut data).unwrap();
        let (head, body) = data.split_once("\r\n\r\n").unwrap();
        (
            head.split_whitespace().nth(1).unwrap().parse().unwrap(),
            body.to_owned(),
        )
    }

    fn bytes(&self, method: &str, path: &str) -> (u16, Vec<u8>) {
        let mut conn = self.open_request(method, path, None, "");
        let mut data = Vec::new();
        conn.read_to_end(&mut data).unwrap();
        let split = data.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = std::str::from_utf8(&data[..split]).unwrap();
        (
            head.split_whitespace().nth(1).unwrap().parse().unwrap(),
            data.split_off(split + 4),
        )
    }

    fn json(&self, method: &str, path: &str, body: Option<Value>, expected: u16) -> Value {
        let (status, body) = self.request(method, path, body, "");
        assert_eq!(status, expected, "{method} {path}: {body}");
        if status == 204 {
            Value::Null
        } else {
            serde_json::from_str(&body).unwrap()
        }
    }

    fn register(&self, host: &str, kind: &str, path: &str) -> String {
        id(&self.json(
            "POST",
            "/api/models",
            Some(json!({"host":host,"path":path,"kind":kind})),
            201,
        ))
        .into()
    }

    fn enqueue(&self, body: Value) -> String {
        id(&self.json("POST", "/api/jobs", Some(body), 201)).into()
    }

    fn detail(&self, id: &str) -> Value {
        self.json("GET", &format!("/api/jobs/{id}"), None, 200)
    }

    fn await_job(&self, id: &str, condition: impl Fn(&Value) -> bool) -> Value {
        wait_for(|| {
            let job = self.detail(id);
            condition(&job).then_some(job)
        })
    }

    fn cancel(&self, id: &str) -> Value {
        self.json(
            "POST",
            &format!("/api/jobs/{id}/cancel"),
            Some(json!({})),
            200,
        )
    }

    fn launches(&self) -> Vec<String> {
        fs::read_to_string(self.path("launches"))
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn touch(&self, file: &str) {
        fs::write(self.path(file), "").unwrap();
    }
    fn remove(&self, file: &str) {
        fs::remove_file(self.path(file)).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop();
    }
}

fn wait_for<T>(mut f: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(7);
    loop {
        if let Some(result) = f() {
            return result;
        }
        assert!(Instant::now() < deadline, "timed out");
        sleep(Duration::from_millis(20));
    }
}

fn id(value: &Value) -> &str {
    value["id"].as_str().unwrap()
}

const SUGGEST: &str = "/api/suggestions/paths";

#[path = "support/openapi.rs"]
mod openapi;

#[path = "support/idempotency.rs"]
mod idempotency;

#[test]
fn suggestions_bound_dispatch_and_leave_compute_idle() {
    let mut app = Fixture::new();
    app.launch();
    for prefix in ["/unreachable/", "/malformed/", "/oversized/"] {
        let error = app.json(
            "POST",
            SUGGEST,
            Some(json!({"host":"alpha","prefix":prefix})),
            502,
        );
        assert_eq!(error, json!({"error":"path suggestions unavailable"}));
    }
    assert_eq!(
        app.json(
            "POST",
            SUGGEST,
            Some(json!({"host":"beta","prefix":""})),
            200
        ),
        json!({"items":[{"path":"/model roots","kind":"directory"}],"truncated":false})
    );

    let mut timed = app.open_request(
        "POST",
        SUGGEST,
        Some(json!({"host":"alpha","prefix":"/timeout/"})),
        "",
    );
    let abandoned = app.open_request(
        "POST",
        SUGGEST,
        Some(json!({"host":"beta","prefix":"/disconnect/"})),
        "",
    );
    let pids = |key| {
        wait_for(|| {
            let value = fs::read_to_string(app.path(key)).ok()?;
            let (dispatcher, transport) = value.split_once(' ')?;
            Some([
                dispatcher.parse::<u32>().ok()?,
                transport.parse::<u32>().ok()?,
            ])
        })
    };
    let timed_pids = pids("suggest-timeout-pid");
    let abandoned_pids = pids("suggest-disconnect-pid");
    let finished = |pids: [u32; 2]| {
        pids.into_iter().all(|pid| {
            // An orphan may remain as a zombie until init reaps it.
            fs::read_to_string(format!("/proc/{pid}/stat")).map_or(true, |stat| {
                stat.rsplit_once(") ")
                    .is_some_and(|(_, fields)| fields.starts_with("Z "))
            })
        })
    };
    app.json(
        "POST",
        SUGGEST,
        Some(json!({"host":"alpha","prefix":""})),
        503,
    );
    assert!(
        app.launches().is_empty(),
        "suggestions cannot launch queue jobs"
    );
    drop(abandoned);
    // Prove cancellation before the normal request timeout, not merely eventual reaping.
    let start = Instant::now();
    wait_for(|| finished(abandoned_pids).then_some(()));
    assert!(start.elapsed() < Duration::from_secs(2));
    let mut response = String::new();
    timed.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.0 504"), "{response}");
    wait_for(|| finished(timed_pids).then_some(()));
    assert!(app.launches().is_empty());
    let job = app.enqueue(json!({"kind":"cache_cleanup","host":"alpha"}));
    app.await_job(&job, |job| job["state"] == "succeeded");
}

#[test]
fn suggestions_validate_same_origin_json_and_forward_space_preserving_prefix() {
    let mut app = Fixture::new();
    app.launch();
    let body = json!({"host":"alpha","prefix":"/model roots/"});
    let (code, raw) = app.request(
        "POST",
        SUGGEST,
        Some(body.clone()),
        &format!(
            "Origin: http://{}\r\nSec-Fetch-Site: same-origin\r\n",
            app.addr.unwrap()
        ),
    );
    assert_eq!(code, 200, "{raw}");
    assert_eq!(
        serde_json::from_str::<Value>(&raw).unwrap(),
        json!({
            "items":[{"path":"/model roots/demo model.gguf","kind":"file"},
                     {"path":"/model roots/assets","kind":"directory"}],"truncated":true
        })
    );
    assert_eq!(
        app.request("POST", SUGGEST, Some(body), "Origin: http://other.test\r\n")
            .0,
        403
    );
    assert_eq!(
        app.request("POST", SUGGEST, None, "Content-Type: text/plain\r\n")
            .0,
        415
    );
    for host in ["unknown", "auto", "--option", " alpha"] {
        app.json(
            "POST",
            SUGGEST,
            Some(json!({"host":host,"prefix":"/"})),
            400,
        );
    }
    for prefix in [
        "relative".into(),
        "/a/../b".into(),
        "/a/./b".into(),
        "/a//b".into(),
        "/a\nb".into(),
        format!("/{}", "é".repeat(2048)),
    ] {
        app.json(
            "POST",
            SUGGEST,
            Some(json!({"host":"alpha","prefix":prefix})),
            400,
        );
    }
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["models"], json!([]));
    assert_eq!(state["jobs"], json!([]));
    assert!(app.launches().is_empty());
}

#[test]
fn hy4_gate_encode_requires_the_exact_same_worker_opt_in() {
    let mut app = Fixture::new();
    fs::write(app.path("dispatch.json"), json!({"hosts":{
        "alpha":{"local":true,"capabilities":{"hy4_encode":true}},
        "beta":{"local":true,"capabilities":{"hy4_plan":true}}
    }}).to_string()).unwrap();
    app.launch();
    let model = |host| app.json("POST", "/api/models", Some(json!({
        "name":"Pinned Hy4","host":host,"kind":"hf","path":"/models/hy4"
    })), 201)["id"].clone();
    let source = model("alpha");
    let other = model("beta");
    let body = json!({"kind":"convert","host":"alpha","input_ids":[source],
        "options":{"hy4_overrides":{"blk.1.attn_gate.weight":"Q8_0"}}});
    app.json("POST", "/api/jobs", Some(body.clone()), 201);
    for (key, value, status) in [
        ("kind", json!("plan"), 409),
        ("host", json!("auto"), 400),
        ("input_ids", json!([other]), 400),
        ("options", json!({"hy4_overrides":{"blk.1.ffn_down_exps.weight":"Q8_0"}}), 400),
        ("options", json!({"hy4_overrides":{"blk.1.attn_gate.weight":"Q8_0"},"threads":1}), 400),
        ("options", json!({}), 409),
    ] {
        let mut bad = body.clone();
        bad[key] = value;
        app.json("POST", "/api/jobs", Some(bad), status);
    }
    app.json("POST", "/api/conversions", Some(json!({"host":"alpha","input_ids":[source],
        "options":body["options"]})), 400);
}

#[test]
fn trace_admission_and_reads_are_bounded() {
    let mut app = Fixture::new();
    fs::write(app.path("dispatch.json"), json!({
        "hosts": {"alpha":{"local":true,"capabilities":{"trace":true}}, "beta":{"local":true}},
        "trace":{"default_byte_cap":65536,"max_byte_cap":131072,"retain_jobs":2}
    }).to_string()).unwrap();
    app.launch();
    let model = app.register("alpha", "gguf", "/models/trace.gguf");
    let body = json!({"kind":"trace","host":"alpha","input_ids":[model],"prompt":"one two",
        "options":{"layers":[0],"roles":["l_out"],"positions":[0],"top_k":2}});
    for (key, value) in [
        ("byte_cap", json!(131073)),
        ("roles", json!(["../weights"])),
        ("positions", json!([4096])),
        ("prompts", json!(["conflicts with prompt"])),
        ("layers", json!((0..65).collect::<Vec<_>>())),
        ("retain_jobs", json!(10)),
        ("interventions", json!([{"op":"zero","layer":0,"role":"kq_soft_max","positions":[0]}])),
    ] {
        let mut bad = body.clone();
        bad["options"][key] = value;
        app.json("POST", "/api/jobs", Some(bad), 400);
    }
    let mut off = body.clone();
    off["host"] = json!("beta");
    app.json("POST", "/api/jobs", Some(off), 409);
    assert!(app.launches().is_empty());
    assert_eq!(app.json("GET", "/api/settings", None, 200)["trace"],
        json!({"default_byte_cap":65536,"max_byte_cap":131072,"retain_jobs":2}));
    let id = app.enqueue(body);
    let job = app.await_job(&id, |job| job["state"] == "succeeded");
    assert_eq!(job["result"]["envelope"]["options"]["byte_cap"], 65536);
    assert_eq!(job["result"]["envelope"]["options"]["retain_jobs"], 2);
    let path = format!("/api/jobs/{id}/trace");
    assert_eq!(app.json("GET", &path, None, 200)["schema_version"], 1);
    app.json("GET", &format!("{path}/tensor?layer=0&role=l_out&start=2&end=1"), None, 400);
    assert_eq!(app.json("GET", &format!("{path}/tensor?layer=0&role=l_out&start=0&end=1"), None, 200)
        ["rows"][0]["data_base64"], "AACAPw==");
    app.json("GET", &format!("{path}/tensor?layer=0&role=l_out"), None, 413);
    fs::write(app.path("trace-pruned"), "").unwrap();
    app.json("GET", &path, None, 404);

    let mut large = json!({"kind":"trace","input_ids":[model],
        "options":{"layers":(0..65).collect::<Vec<_>>(),"roles":["l_out"],"positions":[0],"top_k":0,
            "full_dump":true,"byte_cap":33554433,"prompts":["one","two"],
            "interventions":[{"op":"scale","factor":0.5,"layer":0,"role":"l_out","positions":[0]}]}});
    app.json("POST", "/api/jobs", Some(large.clone()), 400);
    large["options"]["byte_cap"] = json!(131072);
    let multi = app.enqueue(large);
    let done = app.await_job(&multi, |job| job["state"] == "succeeded");
    assert_eq!(done["result"]["envelope"]["options"]["prompts"], json!(["one","two"]));
}

#[test]
fn hy4_plan_requires_an_opted_in_source_host_and_exact_override() {
    let mut app = Fixture::new();
    fs::write(app.path("dispatch.json"), json!({"hosts":{
        "alpha":{"local":true,"capabilities":{"hy4_plan":true}},
        "beta":{"local":true}
    }}).to_string()).unwrap();
    app.launch();
    let model = |host| app.json("POST", "/api/models", Some(json!({
        "name":"Pinned Hy4","host":host,"kind":"hf","path":"/models/hy4"
    })), 201)["id"].clone();
    let source = model("alpha");
    let other = model("beta");
    let body = json!({"kind":"plan","host":"alpha","input_ids":[source],
        "options":{"hy4_overrides":{"blk.1.ffn_down_exps.weight":"Q8_0"}}});
    app.json("POST", "/api/jobs", Some(body.clone()), 201);
    for (key, value, status) in [
        ("kind", json!("convert"), 409),
        ("input_ids", json!([other]), 400),
        ("options", json!({"hy4_overrides":{"blk.1.ffn_up_exps.weight":"Q8_0"}}), 400),
        ("options", json!({"hy4_overrides":{"blk.1.ffn_down_exps.weight":"Q8_0"},"threads":1}), 400),
    ] {
        let mut bad = body.clone();
        bad[key] = value;
        app.json("POST", "/api/jobs", Some(bad), status);
    }
    let mut generic = body.clone();
    generic["options"] = json!({});
    app.json("POST", "/api/jobs", Some(generic), 409);
    let mut off = body;
    off["host"] = json!("beta");
    off["input_ids"] = json!([other]);
    app.json("POST", "/api/jobs", Some(off), 409);
}

#[test]
fn benchmark_jobs_only_dispatch_ids_allowlisted_for_an_opted_in_host() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{
                "local":true,"capabilities":{"benchmark":true},
                "benchmark_targets":[{"id":"local-model","label":"Local model"}],
                "benchmarks":[{"id":"tools"},{"id":"better-bench"}]
            },
            "beta":{
                "local":true,"capabilities":{"benchmark":true},
                "benchmark_targets":[{"id":"other-model"}],
                "benchmarks":[{"id":"other-suite"}]
            },
            "off":{
                "local":true,
                "benchmark_targets":[{"id":"local-model"}],
                "benchmarks":[{"id":"tools"}]
            }
        }})
        .to_string(),
    )
    .unwrap();
    app.launch();
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["capabilities"]["benchmark"], true);
    assert_eq!(
        state["hosts"][0]["benchmark_targets"],
        json!([{"id":"local-model","label":"Local model"}])
    );
    assert_eq!(
        state["hosts"][0]["benchmarks"],
        json!([{"id":"tools"},{"id":"better-bench"}])
    );
    assert_eq!(state["hosts"][2]["capabilities"]["benchmark"], false);
    let options = json!({"benchmark_target":"local-model","benchmarks":["better-bench","tools"]});
    let valid = json!({"kind":"benchmark","host":"alpha","input_ids":[],"options":options});
    for invalid in [
        json!({"benchmark_target":"other-model","benchmarks":["tools"]}),
        json!({"benchmark_target":"local-model","benchmarks":["other-suite"]}),
        json!({"benchmark_target":"local-model","benchmarks":["tools","tools"]}),
        json!({"benchmark_target":"local-model","benchmarks":[]}),
        json!({"benchmark_target":"local-model","benchmarks":["a","b","c","d","e","f","g","h","i"]}),
        json!({"benchmark_target":"../local-model","benchmarks":["tools"]}),
        json!({"benchmark_target":"local-model","benchmarks":["https://example.test"]}),
        json!({"benchmark_target":"local-model","benchmarks":["Tools"]}),
        json!({"benchmark_target":"local-model","benchmarks":"tools"}),
        json!({"benchmark_target":"local-model","benchmarks":["tools"],"args":["--anything"]}),
        json!({"benchmark_target":"local-model"}),
    ] {
        let mut job = valid.clone();
        job["options"] = invalid;
        app.json("POST", "/api/jobs", Some(job), 400);
    }
    let mut wrong_host = valid.clone();
    wrong_host["host"] = json!("off");
    app.json("POST", "/api/jobs", Some(wrong_host), 409);
    let mut with_prompt = valid.clone();
    with_prompt["prompt"] = json!("Not allowed");
    app.json("POST", "/api/jobs", Some(with_prompt), 400);
    let input = app.register("alpha", "gguf", "/allowed/a.gguf");
    let mut with_input = valid.clone();
    with_input["input_ids"] = json!([input]);
    app.json("POST", "/api/jobs", Some(with_input), 400);
    let mut without_host = valid.clone();
    without_host.as_object_mut().unwrap().remove("host");
    app.json("POST", "/api/jobs", Some(without_host), 400);
    assert!(app.launches().is_empty());
    let job = app.enqueue(valid);
    let done = app.await_job(&job, |job| job["state"] == "succeeded");
    assert_eq!(
        done["result"],
        json!({
            "action":"start","host":"alpha",
            "envelope":{"kind":"benchmark","inputs":[],"options":options}
        })
    );
}

#[test]
fn malformed_benchmark_allowlists_prevent_controller_startup() {
    let app = Fixture::new();
    for choices in [
        json!([{"id":"good","url":"http://example.test"}]),
        json!([{"id":"good","args":["--unsafe"]}]),
        json!([{"id":"../bad"}]),
        json!([{"id":"same"},{"id":"same"}]),
    ] {
        fs::write(
            app.path("dispatch.json"),
            json!({"hosts":{"alpha":{"local":true,"benchmark_targets":choices}}}).to_string(),
        )
        .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_brewery"))
            .env("BREWERY_LISTEN", "127.0.0.1:0")
            .env("BREWERY_DATA", app.path("state"))
            .env("BREWERY_DISPATCH_CONFIG", app.path("dispatch.json"))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("benchmark_targets"));
    }
}

#[test]
fn in_flight_staging_progress_is_advisory_and_cancellable() {
    let mut app = Fixture::new();
    app.launch();
    let source = app.register("alpha", "gguf", "/registered/source.gguf");
    app.touch("sample_launch");
    let transfer = app.enqueue(json!({"kind":"transfer","host":"beta","input_ids":[source]}));
    let next = app.enqueue(json!({"kind":"cache_cleanup","host":"beta"}));

    let copy = app.await_job(&transfer, |job| job["progress"] == "Copying: 120 / 360 GB");
    assert_eq!(copy["state"], "running");
    assert_eq!(copy["has_result"], false);
    assert!(copy.get("result").is_none());
    assert_eq!(copy["events"].as_array().unwrap().len(), 2);
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["jobs"][0]["progress"], copy["progress"]);
    assert_eq!(app.detail(&next)["state"], "queued");
    assert_eq!(app.launches(), [format!("{transfer} transfer beta")]);

    let status_pid_file = app.path(&format!("{transfer}.status-pid"));
    let pid = |file| wait_for(|| fs::read_to_string(&file).ok().filter(|pid| !pid.is_empty()));
    let status_pid = pid(status_pid_file);
    let launch_pid = pid(app.path(&format!("{transfer}.sampling")));
    app.cancel(&transfer);
    let cancelled = app.await_job(&transfer, |job| job["state"] == "cancelled");
    assert!(cancelled.get("result").is_none());
    wait_for(|| {
        (!PathBuf::from(format!("/proc/{status_pid}")).exists()
            && !PathBuf::from(format!("/proc/{launch_pid}")).exists())
        .then_some(())
    });
    app.await_job(&next, |job| job["state"] == "succeeded");
    assert_eq!(
        app.launches(),
        [
            format!("{transfer} transfer beta"),
            format!("{next} cleanup beta")
        ]
    );
}

#[test]
fn same_origin_library_survives_restart() {
    let mut app = Fixture::new();
    app.launch();
    assert_eq!(
        app.request("GET", "/", None, ""),
        (200, "brewery static asset".into())
    );
    assert_eq!(
        app.request("GET", "/api/state", None, "Host: rebound.example\r\n")
            .0,
        403
    );
    assert_eq!(
        app.request("GET", "/api/state", None, "Sec-Fetch-Site: cross-site\r\n")
            .0,
        403
    );
    assert_eq!(
        app.request(
            "POST",
            "/api/prompts",
            Some(json!({"name":"x","text":"y"})),
            "Origin: https://different.example\r\n"
        )
        .0,
        403
    );
    assert_eq!(
        app.request("DELETE", "/api/models/unknown", None, "Origin: null\r\n")
            .0,
        403
    );
    assert_eq!(
        app.request("POST", "/api/prompts", None, "Content-Type: text/plain\r\n")
            .0,
        415
    );
    let created = app.json(
        "POST",
        "/api/prompts",
        Some(json!({"name":"p","text":"Which is longer?\n"})),
        201,
    );
    app.json(
        "POST",
        "/api/models",
        Some(json!({"host":"ghost","path":"/a.gguf","kind":"gguf"})),
        400,
    );
    app.json(
        "POST",
        "/api/models",
        Some(json!({"host":"alpha","path":"/models/../a.gguf","kind":"gguf"})),
        400,
    );
    let model = app.json(
        "POST",
        "/api/models",
        Some(json!({"host":"alpha","path":"/models/a file.gguf","kind":"gguf"})),
        201,
    );
    app.stop();
    app.launch();
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["hosts"][1]["reasons"]["plan"], "not installed");
    assert_eq!(state["models"][0], model);
    assert_eq!(state["prompts"][0], created);
    app.json(
        "DELETE",
        &format!("/api/models/{}", model["id"].as_str().unwrap()),
        None,
        204,
    );
    assert_eq!(
        app.json("GET", "/api/state", None, 200)["models"],
        json!([])
    );
}

#[test]
fn fifo_holds_slot_during_transport_loss_and_reconciles_restart() {
    let mut app = Fixture::new();
    app.launch();
    let a = app.register("alpha", "gguf", "/model roots/a'.gguf");
    let b = app.register("beta", "gguf", "/models/b.gguf");
    let saved = app.json(
        "POST",
        "/api/prompts",
        Some(json!({"name":"unicode","text":"<< Café? >>\n"})),
        201,
    );
    app.touch("hold");
    let compare = app.enqueue(
        json!({"kind":"compare","input_ids":[a,b],"prompt_id":id(&saved),"options":{"n":0}}),
    );
    let active = app.await_job(&compare, |job| job["progress"] == "worker progress");
    assert!(
        active.get("result").is_none(),
        "must not publish staging data"
    );
    assert_eq!(active["has_result"], false);
    let next = app.enqueue(json!({"kind":"inspect","input_ids":[b]}));
    let queued_cancel = app.enqueue(json!({"kind":"cache_cleanup","host":"beta"}));
    assert_eq!(app.cancel(&queued_cancel)["state"], "cancelled");
    app.json("DELETE", &format!("/api/models/{b}"), None, 204);
    app.touch("offline");
    let offline = app.await_job(&compare, |job| {
        job["error"]
            .as_str()
            .is_some_and(|error| error.contains("transport failure"))
    });
    assert_eq!(offline["state"], "running");
    app.stop();
    app.launch();
    app.await_job(&compare, |job| {
        job["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| event["progress"] == "Controller was interrupted; reconciling with worker")
    });
    assert_eq!(app.detail(&next)["state"], "queued");
    assert_eq!(app.launches(), [format!("{compare} start alpha")]);
    app.remove("offline");
    app.remove("hold");
    let completed = app.await_job(&compare, |job| job["state"] == "succeeded");
    assert_eq!(completed["result"]["host"], "alpha");
    assert_eq!(
        completed["result"]["envelope"]["inputs"][0]["path"],
        "/model roots/a'.gguf"
    );
    assert_eq!(completed["result"]["envelope"]["inputs"][1]["host"], "beta");
    assert_eq!(completed["result"]["envelope"]["prompt"], "<< Café? >>\n");
    let completed_next = app.await_job(&next, |job| job["state"] == "succeeded");
    assert_eq!(
        completed_next["result"]["envelope"]["inputs"][0]["path"],
        "/models/b.gguf"
    );
    assert_eq!(
        app.launches(),
        [
            format!("{compare} start alpha"),
            format!("{next} start beta")
        ]
    );
    assert_eq!(app.cancel(&compare)["state"], "succeeded");
    assert_eq!(
        app.detail(&compare),
        completed,
        "terminal result and log are immutable"
    );
    let state = app.json("GET", "/api/state", None, 200);
    assert_eq!(state["jobs"][0]["has_result"], true);
    assert!(state["jobs"][0].get("result").is_none());
}

#[test]
fn restart_missing_fails_without_launching_again() {
    let mut app = Fixture::new();
    app.launch();
    let a = app.register("alpha", "hf", "/allowed/hf");
    app.touch("hold");
    let running = app.enqueue(json!({"kind":"inspect","input_ids":[a]}));
    app.await_job(&running, |job| job["progress"] == "worker progress");
    app.stop();
    app.touch(&format!("{running}.forget"));
    app.launch();
    let failed = app.await_job(&running, |job| job["state"] == "failed");
    assert!(failed["error"].as_str().unwrap().contains("not restarted"));
    assert!(failed.get("result").is_none());
    assert_eq!(app.launches(), [format!("{running} start alpha")]);
}

#[test]
fn live_reconnect_reuses_id_and_running_cancel_discards_partials() {
    let mut app = Fixture::new();
    app.launch();
    let a = app.register("alpha", "gguf", "/allowed/model.gguf");
    app.touch("failstart");
    app.touch("hold");
    let running = app.enqueue(json!({"kind":"inspect","input_ids":[a]}));
    app.await_job(&running, |job| job["progress"] == "worker progress");
    assert_eq!(
        app.launches(),
        [
            format!("{running} start alpha"),
            format!("{running} start alpha")
        ]
    );
    app.touch("offline");
    assert_eq!(app.cancel(&running)["cancel_requested"], true);
    let cancelled = app.await_job(&running, |job| {
        job["error"]
            .as_str()
            .is_some_and(|error| error.contains("transport failure"))
    });
    assert_eq!(
        cancelled["state"], "running",
        "unreachable cannot confirm cancellation"
    );
    app.remove("offline");
    let cancelled = app.await_job(&running, |job| job["state"] == "cancelled");
    assert!(cancelled.get("result").is_none());

    app.touch("blocking_start");
    let blocked = app.enqueue(json!({"kind":"inspect","input_ids":[a]}));
    wait_for(|| app.path(&format!("{blocked}.json")).exists().then_some(()));
    app.cancel(&blocked);
    app.await_job(&blocked, |job| job["state"] == "cancelled");

    app.remove("blocking_start");
    app.remove("hold");
    app.touch("fail");
    let bad = app.enqueue(json!({"kind":"inspect","input_ids":[a]}));
    let bad = app.await_job(&bad, |job| job["state"] == "failed");
    assert_eq!(bad["error"], "validation failed");
    assert!(bad.get("result").is_none());
}

#[test]
fn transfer_and_evaluation_workflows_use_registered_paths_and_typed_options() {
    let mut app = Fixture::new();
    app.launch();
    let a = app.register("alpha", "gguf", "/purported/model.gguf");
    let hgn = app.register("beta", "hgn", "/verified/bundle");
    let preview = app.enqueue(json!({"kind":"transfer_preview","host":"beta","input_ids":[a]}));
    let preview = app.await_job(&preview, |job| job["state"] == "succeeded");
    assert_eq!(preview["result"]["action"], "preview_transfer");
    assert_eq!(preview["result"]["missing_bytes"], 7);
    let transfer = app.enqueue(json!({"kind":"transfer","host":"beta","input_ids":[a]}));
    let transfer = app.await_job(&transfer, |job| job["state"] == "succeeded");
    let artifact = app.register(
        "beta",
        "gguf",
        transfer["result"]["inputs"][0]["path"].as_str().unwrap(),
    );
    let cleanup = app.enqueue(
        json!({"kind":"cache_cleanup","host":"beta","options":{"bundle_sha256":"b".repeat(64)}}),
    );
    let cleanup = app.await_job(&cleanup, |job| job["state"] == "succeeded");
    assert_eq!(cleanup["result"]["action"], "cleanup");
    assert_eq!(cleanup["result"]["envelope"]["kind"], "cache-cleanup");
    let generated = app.enqueue(
        json!({"kind":"generate","input_ids":[hgn],"prompt":"Hello","options":{"max_tokens":64}}),
    );
    let generated = app.await_job(&generated, |job| job["state"] == "succeeded");
    assert_eq!(
        generated["result"]["envelope"]["options"],
        json!({"max_tokens":64})
    );
    let baseline = app
        .enqueue(json!({"kind":"baseline","host":"beta","options":{"baseline_source":"provider"}}));
    assert_eq!(
        app.await_job(&baseline, |job| job["state"] == "succeeded")["result"]["envelope"]["options"]
            ["baseline_source"],
        "provider"
    );
    let eval = app.enqueue(json!({"kind":"evaluate","input_ids":[hgn]}));
    app.await_job(&eval, |job| job["state"] == "succeeded");
    let reference = app.register("beta", "gsm8k-evaluation", "/prior/predictions.json");
    let paired = app.enqueue(json!({"kind":"evaluate","input_ids":[hgn,reference]}));
    let paired = app.await_job(&paired, |job| job["state"] == "succeeded");
    assert_eq!(
        paired["result"]["envelope"]["inputs"][1]["path"],
        "/prior/predictions.json"
    );
    for body in [
        json!({"kind":"baseline","host":"beta","options":{"baseline_source":"https://unconfigured.example"}}),
        json!({"kind":"generate","input_ids":[hgn],"prompt":"Hello","options":{"max_tokens":513}}),
        json!({"kind":"evaluate","input_ids":[hgn],"options":{"synthetic_fixture":true}}),
        json!({"kind":"evaluate","input_ids":[hgn,reference],"options":{"baseline_source":"provider"}}),
        json!({"kind":"convert","input_ids":[a],"options":{"output_path":"/unregistered"}}),
        json!({"kind":"compare","input_ids":[a,artifact],"options":{"n":1}}),
    ] {
        app.json("POST", "/api/jobs", Some(body), 400);
    }
    app.json(
        "POST",
        "/api/jobs",
        Some(json!({"kind":"plan","host":"beta","input_ids":[artifact]})),
        409,
    );
}

#[test]
fn conversion_slots_follow_host_policy_without_interrupting_active_jobs() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"quantization_pool":true,
                "resource_bounds":{"concurrency":{"min":1,"max":2},"cpu_threads":{"min":1,"max":4}}}
        }})
        .to_string(),
    )
    .unwrap();
    app.launch();
    let source = app.register("alpha", "hf", "/registered/model");
    app.touch("hold");
    let policy = |revision, concurrency| {
        json!({
            "action":"set_host_resources","worker":"alpha","revision":revision,
            "resources":{"concurrency":concurrency,"cpu_threads":1}
        })
    };
    app.json("POST", "/api/admin/actions", Some(policy(0, 3)), 400);
    app.json("POST", "/api/admin/actions", Some(policy(0, 2)), 200);
    let submit = || json!({"kind":"convert","host":"alpha","input_ids":[source]});
    let first = app.enqueue(submit());
    let second = app.enqueue(submit());
    for id in [&first, &second] {
        let running = app.await_job(id, |job| job["progress"] == "worker progress");
        assert_eq!(
            running["dispatch_resources"],
            json!({"concurrency":2,"cpu_threads":1})
        );
    }
    app.json("POST", "/api/admin/actions", Some(policy(1, 1)), 200);
    let next = app.enqueue(submit());
    let barrier = app.enqueue(json!({"kind":"inspect","input_ids":[source]}));
    let behind = app.enqueue(submit());
    assert_eq!(app.detail(&first)["state"], "running");
    assert_eq!(app.detail(&second)["state"], "running");
    for id in [&next, &barrier, &behind] {
        assert_eq!(app.detail(id)["state"], "queued");
    }
    fs::remove_file(app.path("hold")).unwrap();
    for id in [&first, &second, &next, &barrier, &behind] {
        app.await_job(id, |job| job["state"] == "succeeded");
    }
    assert_eq!(app.detail(&next)["dispatch_resources"]["concurrency"], 1);
}

#[test]
fn quantization_pool_overlaps_workers_but_preserves_pins_barriers_and_restart() {
    let mut app = Fixture::new();
    fs::write(app.path("dispatch.json"), json!({"hosts":{
        "alpha":{"local":true,"quantization_pool":true,"quantization_backends":{"ht":["cpu"]}},
        "beta":{"ssh":"beta","quantization_pool":true,"quantization_backends":{"ht":["cuda","cpu"]}}
    }}).to_string()).unwrap();
    app.launch();
    let source = app.register("alpha", "hf", "/registered/model");
    app.touch("hold");
    let one = app.enqueue(
        json!({"kind":"convert","host":"auto","input_ids":[source],"options":{"ht_backend":"cpu"}}),
    );
    app.await_job(&one, |job| {
        job["progress"] == "worker progress" && job["host"] == "alpha"
    });
    let pin = app.enqueue(json!({"kind":"convert","host":"alpha","input_ids":[source]}));
    let two = app.enqueue(json!({"kind":"convert","host":"auto","input_ids":[source],"options":{"ht_backend":"cuda"}}));
    app.await_job(&two, |job| {
        job["progress"] == "worker progress" && job["host"] == "beta"
    });
    let barrier = app.enqueue(json!({"kind":"inspect","input_ids":[source]}));
    let after_barrier = app.enqueue(json!({"kind":"convert","host":"beta","input_ids":[source]}));
    assert_eq!(app.detail(&pin)["state"], "queued");
    assert_eq!(app.detail(&barrier)["state"], "queued");
    app.stop();
    app.launch();
    app.await_job(&two, |job| {
        job["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| event["progress"] == "Controller was interrupted; reconciling with worker")
    });
    assert_eq!(app.detail(&two)["host"], "beta");
    app.cancel(&two);
    app.await_job(&two, |job| job["state"] == "cancelled");
    assert_eq!(app.detail(&after_barrier)["state"], "queued");
    assert_eq!(
        app.launches(),
        [format!("{one} start alpha"), format!("{two} start beta")]
    );
    app.cancel(&one);
    app.await_job(&pin, |job| {
        job["progress"] == "worker progress" && job["host"] == "alpha"
    });
    assert_eq!(app.detail(&barrier)["state"], "queued");
    app.cancel(&pin);
    app.await_job(&barrier, |job| job["progress"] == "worker progress");
    assert_eq!(app.detail(&after_barrier)["state"], "queued");
    app.cancel(&barrier);
    app.await_job(&after_barrier, |job| {
        job["progress"] == "worker progress" && job["host"] == "beta"
    });
}

#[test]
fn conversion_auto_falls_back_to_serial_on_a_single_source_without_pool() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{"alpha":{"local":true}}}).to_string(),
    )
    .unwrap();
    app.launch();
    let source = app.register("alpha", "hf", "/registered/model");
    app.touch("hold");
    let first = app.enqueue(json!({"kind":"convert","host":"auto","input_ids":[source]}));
    app.await_job(&first, |job| {
        job["progress"] == "worker progress" && job["host"] == "alpha"
    });
    let second = app.enqueue(json!({"kind":"convert","host":"auto","input_ids":[source]}));
    assert_eq!(app.detail(&second)["state"], "queued");
    assert_eq!(app.launches(), [format!("{first} start alpha")]);
    app.cancel(&first);
    app.await_job(&second, |job| {
        job["progress"] == "worker progress" && job["host"] == "alpha"
    });
}
