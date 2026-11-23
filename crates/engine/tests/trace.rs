use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

fn trace(cwd: &Path, binary: &Path, model: &Path, sha: &str, multiple: bool) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(cwd)
        .env("BREWERY_LANTERNFISH_TRACE", binary)
        .env("BREWERY_CPU_THREADS", "2")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut job = json!({
        "kind":"trace","inputs":[{"kind":"gguf","host":"test","path":model,
            "id":"qwen-bf16","sha256":sha}],
        "options":{"layers":[0],"roles":["l_out"],"positions":[0,1],"top_k":3,
            "byte_cap":65536,"retain_jobs":2}
    });
    if multiple {
        job["options"]["prompts"] = json!(["The capital of France is", "The capital of Italy is"]);
        job["options"]["interventions"] =
            json!([{"layer":0,"role":"l_out","positions":[1],"op":"scale","factor":0.5}]);
    } else {
        job["prompt"] = json!("The capital of France is");
    }
    writeln!(child.stdin.take().unwrap(), "{job}").unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn trace_rejects_an_unpinned_executable_before_launch() {
    use std::os::unix::fs::PermissionsExt;
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().canonicalize().unwrap();
    let fake = root.join("fake");
    let marker = root.join("was-run");
    fs::write(&fake, format!("#!/bin/sh\ntouch '{}'\n", marker.display())).unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o700)).unwrap();
    let model = root.join("demo.gguf");
    fs::write(&model, b"GGUF").unwrap();
    let out = trace(
        &root,
        &fake,
        &model,
        &format!("{:x}", Sha256::digest(b"GGUF")),
        false,
    );
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Lanternfish") && stderr.contains("SHA-256"),
        "{stderr}"
    );
    assert!(!marker.exists());
}

#[test]
#[ignore = "requires the provisioned CPU Lanternfish, trace library directory and Qwen BF16 GGUF"]
fn trace_real_qwen_bf16() {
    let root = tempfile::Builder::new()
        .prefix("brewery-w1-trace-")
        .tempdir()
        .unwrap()
        .keep()
        .canonicalize()
        .unwrap();
    let out = trace(
        &root,
        Path::new(&std::env::var("BREWERY_LANTERNFISH_TRACE").unwrap()),
        Path::new(&std::env::var("BREWERY_TEST_TRACE_GGUF").unwrap()),
        "a33250ab73bbef0d876bf04e01517f5f43292dd8ede34e12d3a3a886e49f36f0",
        true,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let receipt: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(receipt["kind"], "trace");
    let data = fs::read(root.join("trace/0/manifest.json")).unwrap();
    let manifest: Value = serde_json::from_slice(&data).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["tensors"].as_array().unwrap().len(), 2);
    assert_eq!(manifest["logit_lens"].as_array().unwrap().len(), 2);
    assert_eq!(manifest["model_logits"].as_array().unwrap().len(), 2);
    assert_eq!(
        manifest["interventions"],
        json!([{"layer":0,"role":"l_out","position":1}])
    );
    let next: Value =
        serde_json::from_slice(&fs::read(root.join("trace/1/manifest.json")).unwrap()).unwrap();
    assert_eq!(
        next["spec"]["byte_cap"],
        65536 - receipt["prompts"][0]["bytes"].as_u64().unwrap()
    );
    fs::write(root.join("receipt.json"), out.stdout).unwrap();
    println!("Trace evidence: {}", root.display());
}
