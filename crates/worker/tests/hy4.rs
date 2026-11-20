use serde_json::{Value, json};
use std::{fs, io::Write, path::Path, process::{Command, Stdio}, thread, time::{Duration, Instant}};

fn dispatch(config: &Path, action: &str, id: &str, body: Option<&Value>) -> Value {
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-dispatch"))
        .env("BREWERY_DISPATCH_CONFIG", config).args([action, "frame", id])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    if let Some(body) = body {
        write!(input, "{body}").unwrap();
    }
    drop(input);
    let out = child.wait_with_output().unwrap();
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn hy4_worker_dispatches_only_a_direct_metadata_plan() {
    let tmp = tempfile::tempdir_in("/arboretum").unwrap();
    let root = tmp.path();
    let source = root.join("source");
    fs::create_dir(&source).unwrap();
    let engine = root.join("hy4-fixture.py");
    fs::write(&engine, r#"import json, sys
job = json.load(sys.stdin)
host, source, tools = sys.argv[1:]
assert host == "frame" and source == tools
assert len(job["inputs"]) == 1 and "files" not in job["inputs"][0]
assert job["inputs"][0]["integrity_basis"] == "hy4_metadata_only"
print(json.dumps({"kind":"hy4_plan","payload_verified":False,"runtime_ready":False}))
"#).unwrap();
    let worker = root.join("worker.json");
    fs::write(&worker, json!({"root":root.join("worker"), "engine":engine,
        "hy4_plan":{"host":"frame","source":source,"halo_tools":source},
        "disk_reserve_bytes":0}).to_string()).unwrap();
    let config = root.join("dispatch.json");
    fs::write(&config,json!({"hosts":{"frame":{"local":true,
        "worker":env!("CARGO_BIN_EXE_brewery-worker"),"config":worker}}}).to_string()).unwrap();
    let job = json!({"kind":"plan","inputs":[{"host":"frame","path":source,"kind":"hf"}],
        "options":{"hy4_overrides":{"blk.1.ffn_down_exps.weight":"Q8_0"}}});
    let start = dispatch(&config, "start", "hy4", Some(&job));
    assert_eq!(start["state"], "queued", "{start}");
    let until = Instant::now() + Duration::from_secs(8);
    loop {
        let current = dispatch(&config, "status", "hy4", None);
        if current["state"] == "succeeded" {
            assert_eq!(current["result"]["payload_verified"], false);
            break;
        }
        assert!(current["state"] != "failed" && Instant::now() < until, "{current}");
        thread::sleep(Duration::from_millis(30));
    }
    for (i, bad) in [
        json!({"kind":"convert","inputs":job["inputs"],"options":job["options"]}),
        json!({"kind":"plan","inputs":job["inputs"],"options":{}}),
        json!({"kind":"plan","inputs":[{"host":"other","path":source,"kind":"hf"}],"options":job["options"]}),
        json!({"kind":"plan","inputs":[{"host":"frame","path":root,"kind":"hf"}],"options":job["options"]}),
    ].iter().enumerate() {
        assert_eq!(dispatch(&config, "start", &format!("bad-{i}"), Some(bad))["state"], "failed");
    }
    assert_eq!(dispatch(&config, "preview_transfer", "no-transfer", Some(&job))["state"], "failed");
}
