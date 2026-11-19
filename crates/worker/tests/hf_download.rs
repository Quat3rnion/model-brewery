use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn ordinary_workers_do_not_accept_network_downloads() {
    let rig = tempfile::tempdir_in("/arboretum").unwrap();
    let config = rig.path().join("worker.json");
    fs::write(
        &config,
        json!({
            "root": rig.path().join("state"),
            "engine": "/usr/bin/false"
        })
        .to_string(),
    )
    .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-worker"))
        .arg("rpc")
        .env("BREWERY_WORKER_CONFIG", config)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(
            json!({
                "op": "prepare", "id": "download-1", "action": "start",
                "job": {"kind": "hf_download", "inputs": [], "options": {"repo_id": "owner/model"}}
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
    let response: Value =
        serde_json::from_slice(&child.wait_with_output().unwrap().stdout).unwrap();
    assert_eq!(response["ok"]["state"], "failed", "{response}");
    assert_eq!(
        response["ok"]["error"],
        "Hugging Face downloads are not configured on this worker"
    );
}
