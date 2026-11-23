use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn gpu_exl3_requires_admission_but_cpu_does_not() {
    let tmp = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let root = tmp.path();
    let config = root.join("worker.json");
    let sources = root.join("sources");
    fs::create_dir(&sources).unwrap();
    let source = sources.join("layer.safetensors");
    fs::write(&source, b"fixture: preparation must not read the weights").unwrap();
    fs::write(
        &config,
        json!({
            "root": root.join("worker"),
            "source_roots": [sources],
            "engine": "/unused-engine",
            "disk_reserve_bytes": 0,
            "engine_env": {
                "BREWERY_WEIGHT_SAFETENSORS": "1",
                "BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES": "4194304",
                "BREWERY_EXL3_CHECKPOINT": "/unused-helper"
            }
        })
        .to_string(),
    )
    .unwrap();
    for (backend, expected) in [("cuda", "failed"), ("rocm", "failed"), ("cpu", "queued")] {
        let request = json!({
            "op": "prepare",
            "id": backend,
            "action": "start",
            "job": {
                "kind": "exl3_layer",
                "inputs": [{"host": "test", "path": source, "kind": "safetensors"}],
                "options": {
                    "backend": backend,
                    "source_sha256": "a".repeat(64),
                    "tensor": "model.proj.weight",
                    "shape": [128, 128],
                    "codebook": "mul1",
                    "twice_bits": 7
                }
            }
        });
        let mut worker = Command::new(env!("CARGO_BIN_EXE_brewery-worker"))
            .env("BREWERY_WORKER_CONFIG", &config)
            .arg("rpc")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        writeln!(worker.stdin.take().unwrap(), "{request}").unwrap();
        let output = worker.wait_with_output().unwrap();
        assert!(output.status.success(), "{output:?}");
        let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
        let reply = &envelope["ok"];
        assert_eq!(reply["state"], expected, "{backend}: {envelope}");
        if expected == "failed" {
            assert_eq!(
                reply["error"], "EXL3 GPU admission is not configured on this worker",
                "{backend}: {envelope}"
            );
        }
    }
}
