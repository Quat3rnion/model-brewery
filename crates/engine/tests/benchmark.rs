use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

fn adapter(root: &Path, name: &str, script: &str) -> PathBuf {
    let path = root.join(name);
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}

fn run(root: &Path, config: Value, job: Value) -> Output {
    let config_file = root.join("adapters.json");
    fs::write(&config_file, serde_json::to_vec(&config).unwrap()).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(root)
        .env("BREWERY_BENCHMARK_CONFIG", &config_file)
        .env("BREWERY_TEST_SECRET", "secret-from-engine-environment")
        .env("BREWERY_CPU_THREADS", "1")
        .env("OMP_NUM_THREADS", "999")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&job).unwrap())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn job(benchmarks: Value) -> Value {
    json!({"kind":"benchmark","inputs":[],
           "options":{"benchmark_target":"local-model","benchmarks":benchmarks}})
}

fn config(adapters: Value) -> Value {
    json!({"targets":{"local-model":{"endpoint":"http://127.0.0.1:9000/v1","model":"fixture-model"}},
           "adapters":adapters})
}

#[test]
fn runs_selected_adapters_in_order_and_publishes_the_report() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let first = adapter(
        root,
        "first.py",
        r#"#!/usr/bin/python3
import json, os, pathlib, sys
assert len(sys.argv) == 1 and not os.listdir()
assert 'BREWERY_TEST_SECRET' not in os.environ
for key in ('BREWERY_CPU_THREADS', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    assert os.environ[key] == '1'
progress = json.loads(pathlib.Path('../progress.json').read_text())
assert (progress['phase'], progress['unit'], progress['completed'], progress['total']) == ('benchmark', 'suites', 0, 2)
assert json.load(sys.stdin) == dict(benchmark='first', target='local-model',
    endpoint='http://127.0.0.1:9000/v1', model='fixture-model', version='v1', protocol='fixture-v1')
pathlib.Path('scratch').write_text('not an artifact')
import subprocess
child = subprocess.Popen(['/usr/bin/sleep', '2'], start_new_session=True)
pathlib.Path(sys.argv[0]).with_name('order').write_text(str(child.pid))
print(json.dumps(dict(metrics=[dict(name='accuracy', value=0.75, unit='ratio', direction='higher')], samples=4)))
"#,
    );
    let second = adapter(
        root,
        "second.py",
        r#"#!/usr/bin/python3
import json, os, pathlib, sys
assert len(sys.argv) == 1 and not os.listdir()
assert os.environ['BREWERY_CPU_THREADS'] == '1'
progress = json.loads(pathlib.Path('../progress.json').read_text())
assert (progress['phase'], progress['completed'], progress['total']) == ('benchmark', 1, 2)
previous_child = pathlib.Path(sys.argv[0]).with_name('order').read_text()
assert not pathlib.Path('/proc', previous_child).exists()
assert json.load(sys.stdin)['benchmark'] == 'second'
print(json.dumps(dict(metrics=[dict(name='latency', value=42, unit='ms', direction='lower')])))
"#,
    );
    let out = run(
        root,
        config(json!({
            "first":{"executable":first,"version":"v1","protocol":"fixture-v1"},
            "second":{"executable":second,"version":"v2","protocol":"fixture-v1"}
        })),
        job(json!(["first", "second"])),
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["kind"], "benchmark");
    assert_eq!(report["schema"], "brewery-benchmark-report-v1");
    assert_eq!(report["target"], "local-model");
    assert_eq!(report["model"], "fixture-model");
    assert_eq!(report["status"], "complete");
    assert_eq!(report["release_qualified"], false);
    assert_eq!(report["runs"].as_array().unwrap().len(), 2);
    let first = &report["runs"][0];
    assert_eq!(first["benchmark"], "first");
    assert_eq!(first["status"], "succeeded");
    assert_eq!(first["version"], "v1");
    assert_eq!(first["protocol"], "fixture-v1");
    assert_eq!(first["samples"], 4);
    assert_eq!(first["metrics"][0]["value"], 0.75);
    let second = &report["runs"][1];
    assert_eq!(second["benchmark"], "second");
    assert_eq!(second["status"], "succeeded");
    assert_eq!(second["metrics"][0]["direction"], "lower");
    assert!(second.get("samples").is_none());
    let artifact = report.as_object_mut().unwrap().remove("artifact").unwrap();
    assert_eq!(artifact["kind"], "report");
    assert_eq!(
        artifact["path"],
        root.join("benchmark-report.json").to_str().unwrap()
    );
    let bytes = fs::read(artifact["path"].as_str().unwrap()).unwrap();
    assert_eq!(artifact["sha256"], format!("{:x}", Sha256::digest(&bytes)));
    assert_eq!(artifact["bytes"], bytes.len() as u64);
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), report);
    assert!(fs::read_dir(root)
        .unwrap()
        .all(|entry| !entry.unwrap().path().is_dir()));
}

#[test]
fn keeps_valid_raw_data_private_and_reports_adapter_failures_separately() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let passed = adapter(
        root,
        "passed.py",
        r#"#!/usr/bin/python3
import pathlib
pathlib.Path('raw.json').write_text('{"questions":4}')
print('{"metrics":[{"name":"accuracy","value":1,"unit":"ratio","direction":"higher"}]}')
"#,
    );
    let failing = adapter(
        root,
        "failing.py",
        r#"#!/usr/bin/python3
import pathlib, sys
pathlib.Path('raw.json').symlink_to(pathlib.Path(sys.argv[0]).with_name('adapters.json'))
print('secret-from-adapter-stdout')
print('secret-from-adapter-stderr', file=sys.stderr)
raise SystemExit(7)
"#,
    );
    let out = run(
        root,
        config(json!({
            "passed":{"executable":passed,"version":"1","protocol":"fixture"},
            "failing":{"executable":failing,"version":"2","protocol":"fixture"}
        })),
        job(json!(["failing", "passed"])),
    );
    assert!(out.status.success());
    assert!(out.stderr.is_empty());
    assert!(!String::from_utf8_lossy(&out.stdout).contains("secret-from-adapter"));
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["status"], "partial");
    let failed = &report["runs"][0];
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["metrics"], json!([]));
    assert!(failed["error"].is_string() && failed["raw_error"].is_string());
    assert!(failed.get("raw_artifact").is_none());
    let succeeded = &report["runs"][1];
    assert_eq!(succeeded["status"], "succeeded");
    assert_eq!(
        succeeded["executable_sha256"],
        format!("{:x}", Sha256::digest(fs::read(&passed).unwrap()))
    );
    let raw = &succeeded["raw_artifact"];
    let bytes = fs::read(raw["path"].as_str().unwrap()).unwrap();
    assert_eq!(bytes, b"{\"questions\":4}");
    assert_eq!(raw["sha256"], format!("{:x}", Sha256::digest(&bytes)));
    assert_eq!(raw["bytes"], bytes.len() as u64);
    assert!(!root.join("benchmark-raw-failing.json").exists());
}

#[test]
fn rejects_untrusted_options_and_non_loopback_or_credentialed_endpoints() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let script = adapter(
        root,
        "must-not-start.py",
        r#"#!/usr/bin/python3
import pathlib, sys
pathlib.Path(sys.argv[0]).with_name('started').touch()
"#,
    );
    let cfg = config(json!({"fixture":{"executable":script,"version":"1","protocol":"fixture"}}));
    let mut args = job(json!(["fixture"]));
    args["options"]["args"] = json!(["--untrusted"]);
    let mut with_input = job(json!(["fixture"]));
    with_input["inputs"] = json!([{"kind":"gguf","host":"fixture","path":"/fake"}]);
    let mut with_prompt = job(json!(["fixture"]));
    with_prompt["prompt"] = json!("not accepted");
    for invalid in [
        args,
        with_input,
        with_prompt,
        job(json!([])),
        job(json!(["fixture", "fixture"])),
        job(json!(["../fixture"])),
        job(json!(["unknown"])),
    ] {
        let result = run(root, cfg.clone(), invalid);
        assert!(!result.status.success() && result.stdout.is_empty());
    }
    for endpoint in [
        "http://192.0.2.25",
        "http://localhost.example.org",
        "http://127.1",
        "http://[::ffff:127.0.0.1]",
        "http://secret@127.0.0.1",
        "http://127.0.0.1?token=secret",
        r"http://127.0.0.1\localhost.example.org",
        "https://127.0.0.1/v1#secret",
        "http://127.0.0.1:1234/bad%xxpath",
    ] {
        let mut cfg = cfg.clone();
        cfg["targets"]["local-model"]["endpoint"] = json!(endpoint);
        let result = run(root, cfg, job(json!(["fixture"])));
        assert!(
            !result.status.success() && result.stdout.is_empty(),
            "{endpoint}"
        );
        assert!(!String::from_utf8_lossy(&result.stderr).contains("secret"));
    }
    let mut relative = cfg;
    relative["adapters"]["fixture"]["executable"] = json!("fixture.py");
    let result = run(root, relative, job(json!(["fixture"])));
    assert!(!result.status.success() && result.stdout.is_empty());
    assert!(!root.join("started").exists());
    assert!(!root.join("benchmark-report.json").exists());
}

#[test]
fn oversized_and_invalid_metric_output_produce_a_durable_failed_report() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let huge = adapter(root, "huge.py", "#!/usr/bin/python3\nprint('x'*65537)\n");
    let invalid = adapter(
        root,
        "invalid.py",
        r#"#!/usr/bin/python3
print('{"metrics":[{"name":"latency","value":10,"unit":"ms","direction":"faster"}]}')
"#,
    );
    let out = run(
        root,
        config(json!({
            "huge":{"executable":huge,"version":"1","protocol":"fixture"},
            "invalid":{"executable":invalid,"version":"1","protocol":"fixture"}
        })),
        job(json!(["huge", "invalid"])),
    );
    assert!(out.status.success());
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["status"], "failed");
    assert_eq!(report["runs"][0]["error"], "adapter stdout exceeds limit");
    assert_eq!(
        report["runs"][1]["error"],
        "adapter did not return valid metric JSON"
    );
    assert!(root.join("benchmark-report.json").is_file());
}
