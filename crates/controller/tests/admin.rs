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

const KEY: &str = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";

struct Test {
    root: tempfile::TempDir,
    child: Option<Child>,
    addr: String,
}

impl Test {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        for (name, value) in [
            ("key", format!("{:x}\n", Sha256::digest(KEY))),
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
        let mut child = command
            .env("BREWERY_ADMIN_KEY_FILE", root.join("key"))
            .env("BREWERY_ADMIN_CATALOG", root.join("catalog.json"))
            .env("BREWERY_LISTEN", "127.0.0.1:0")
            .env("BREWERY_DATA", root.join("state"))
            .env("BREWERY_DISPATCH", root.join("dispatch"))
            .env("BREWERY_DISPATCH_CONFIG", root.join("dispatch.json"))
            .stdout(Stdio::from(fs::File::create(&log).unwrap()))
            .spawn()
            .unwrap();
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

    fn request(&self, path: &str, headers: &str, body: Option<Value>) -> (u16, Value) {
        let method = if body.is_some() { "POST" } else { "GET" };
        let body = body.map(|b| b.to_string()).unwrap_or_default();
        let mut stream = TcpStream::connect(&self.addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(12)))
            .unwrap();
        write!(stream,"{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{headers}\r\n{body}",self.addr,body.len()).unwrap();
        let mut raw = String::new();
        stream.read_to_string(&mut raw).unwrap();
        let (head, body) = raw.split_once("\r\n\r\n").unwrap();
        (
            head.split_whitespace().nth(1).unwrap().parse().unwrap(),
            serde_json::from_str(body).unwrap(),
        )
    }
    fn admin(&self, body: Option<Value>) -> (u16, Value) {
        self.request(
            if body.is_some() {
                "/api/admin/actions"
            } else {
                "/api/admin/state"
            },
            &format!("X-Brewery-Admin-Key: {KEY}\r\n"),
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
fn distinct_admin_auth_cannot_be_forged_with_operator_role_or_query() {
    let mut t = Test::new();
    t.launch();
    for (path, headers) in [
        ("/api/admin/state", "X-Brewery-Role: admin\r\n".into()),
        ("/api/admin/state?role=admin", String::new()),
        (
            "/api/admin/state",
            format!("X-Brewery-Admin-Key: {KEY}\r\nX-Brewery-Admin-Key: {KEY}\r\n"),
        ),
        (
            "/api/admin/state",
            format!("Authorization: Bearer {KEY}\r\n"),
        ),
    ] {
        assert_eq!(t.request(path, &headers, None).0, 401);
    }
    assert_eq!(t.admin(None).0, 200);
    assert_eq!(t.request("/api/state", "", None).0, 200);
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
    assert!(!state.to_string().contains(KEY));
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
