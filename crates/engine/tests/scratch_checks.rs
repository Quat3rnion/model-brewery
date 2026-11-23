use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Stdio},
};

#[test]
fn scratch_checks_run_offline_and_bind_the_native_report() {
    let tmp = tempfile::tempdir().unwrap();
    let runner = tmp.path().join("runner");
    fs::write(&runner, b"#!/bin/sh\nexit 1\n").unwrap();
    fs::set_permissions(&runner, fs::Permissions::from_mode(0o700)).unwrap();
    let script = tmp.path().join("check.py");
    fs::write(
        &script,
        r#"
import argparse, hashlib, json, os, pathlib
p = argparse.ArgumentParser()
for arg in ("runner", "cache", "parent-gguf", "child-gguf", "parent-safetensors", "child-safetensors"):
    p.add_argument("--" + arg, required=True)
p.add_argument("--threads", type=int, required=True)
a = p.parse_args()
assert a.threads == 2
assert pathlib.Path(os.environ["BREWERY_NATIVE_PROGRESS"]).name == "progress.json"
pins = json.loads(pathlib.Path(a.cache, "manifest.json").read_text())
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
digest = sha(a.child_safetensors) if not pathlib.Path(a.cache, "bad").exists() else "0" * 64
print(json.dumps(dict(schema="brewery.scratch-behavior.v1", qualification="not release-qualified",
    runner_sha256=sha(a.runner), revision_sha256=digest, parent_sha256=sha(a.parent_safetensors),
    parent_gguf_sha256=sha(a.parent_gguf), revision_gguf_sha256=sha(a.child_gguf),
    fixture_sha256=pins, parent_generation=" parent", child_generation=" child",
    metrics=[dict(name="perplexity", parent=1., child=2., better="lower")])))
"#,
    )
    .unwrap();
    fs::write(
        tmp.path().join("manifest.json"),
        include_str!("../../eval/data/edit-checks/manifest.json"),
    )
    .unwrap();
    let digest = |path: &Path| format!("{:x}", Sha256::digest(fs::read(path).unwrap()));
    let inputs: Vec<_> = ["gguf", "gguf", "safetensors", "safetensors"]
        .into_iter()
        .enumerate()
        .map(|(i, kind)| {
            let path = tmp.path().join(format!("model-{i}.{kind}"));
            fs::write(&path, format!("GGUF{i}")).unwrap();
            json!({"host":"local","kind":kind,"path":path,"sha256":digest(&path)})
        })
        .collect();
    let run = |name, runner_sha| {
        let work = tmp.path().join(name);
        fs::create_dir(&work).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
            .env_clear()
            .env("BREWERY_CPU_THREADS", "2")
            .env("BREWERY_SCRATCH_CHECKS_SCRIPT", &script)
            .env("BREWERY_SCRATCH_CHECKS_RUNNER", &runner)
            .env("BREWERY_SCRATCH_CHECKS_RUNNER_SHA256", runner_sha)
            .env("BREWERY_SCRATCH_CHECKS_CACHE", tmp.path())
            .current_dir(work)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        write!(
            child.stdin.take().unwrap(),
            "{}",
            json!({"kind":"scratch_checks","inputs":inputs,"options":{}})
        )
        .unwrap();
        child.wait_with_output().unwrap()
    };
    let output = run("wrong-runner", "0".repeat(64));
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("scratch checks runner SHA-256 mismatch"),
        "{stderr}"
    );
    let output = run("verified", digest(&runner));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["revision_sha256"], inputs[3]["sha256"]);
    assert_eq!(report["metrics"][0]["child"], 2.0);
    fs::write(tmp.path().join("bad"), b"").unwrap();
    let output = run("wrong-report", digest(&runner));
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("scratch checks report does not match"),
        "{stderr}"
    );
}
