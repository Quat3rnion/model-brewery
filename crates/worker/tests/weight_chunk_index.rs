use brewery_worker::{Entry, tree_digest};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    process::{Command, Stdio},
};

#[test]
fn explicit_builder_authenticates_members_and_never_overwrites_the_source() {
    let scratch = tempfile::tempdir().unwrap();
    fs::set_permissions(scratch.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let root = scratch.path().join("model");
    fs::create_dir(&root).unwrap();
    let shard = vec![42; (1 << 20) + 1];
    let index = b"fixture index";
    let files: Vec<_> = [
        ("model.safetensors.index.json", index.as_slice()),
        ("weights.safetensors", shard.as_slice()),
    ]
    .into_iter()
    .map(|(name, data)| {
        fs::write(root.join(name), data).unwrap();
        Entry {
            path: name.into(),
            bytes: data.len() as u64,
            sha256: format!("{:x}", Sha256::digest(data)),
        }
    })
    .collect();
    let digest = tree_digest(&files);
    let mut members = serde_json::Map::new();
    for f in &files {
        members.insert(f.path.clone(), json!({"size":f.bytes,"sha256":f.sha256}));
    }
    let receipt = scratch.path().join("published.json");
    fs::write(
        &receipt,
        json!({
            "schema":"model-brewery.source-publication.v1",
            "model_dir":root,"repo_id":"fixture/model","revision":"a".repeat(40),
            "shard_count":1,"weight_bytes":shard.len(),"files":members
        })
        .to_string(),
    )
    .unwrap();
    let output = scratch.path().join("chunks.json");
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_brewery-worker"))
            .arg("build-weight-chunk-index")
            .arg(&receipt)
            .arg(&digest)
            .arg(&output)
            .output()
            .unwrap()
    };
    let first = run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let bytes = fs::read(&output).unwrap();
    let receipt_out: Value = serde_json::from_slice(&first.stdout).unwrap();
    let pin = receipt_out["ok"]["sha256"].as_str().unwrap();
    assert_eq!(pin, format!("{:x}", Sha256::digest(&bytes)));
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["bundle_sha256"], digest);
    assert_eq!(
        parsed["files"][1]["chunk_sha256"],
        json!([
            format!("{:x}", Sha256::digest(&shard[..1 << 20])),
            format!("{:x}", Sha256::digest([42]))
        ])
    );
    assert!(!run().status.success());
    assert_eq!(fs::read(&output).unwrap(), bytes);
    assert_eq!(fs::read(root.join("weights.safetensors")).unwrap(), shard);

    let config = scratch.path().join("worker.json");
    fs::write(
        &config,
        json!({
            "root":scratch.path().join("worker"),
            "source_roots":[root],
            "engine":"/usr/bin/false",
            "engine_env":{
                "BREWERY_WEIGHT_EDIT":"/usr/bin/false",
                "BREWERY_WEIGHT_SAFETENSORS":"1",
                "BREWERY_WEIGHT_CHUNK_INDEX":output,
                "BREWERY_WEIGHT_CHUNK_INDEX_SHA256":pin
            }
        })
        .to_string(),
    )
    .unwrap();
    let rpc = |request: Value| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-worker"))
            .arg("rpc")
            .env("BREWERY_WORKER_CONFIG", &config)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        serde_json::to_writer(child.stdin.take().unwrap(), &request).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        serde_json::from_slice::<Value>(&output.stdout).unwrap()
    };
    let input = json!({"host":"local","kind":"hf","path":root});
    let resolve = |id| {
        assert_eq!(
            rpc(json!({"op":"prepare","id":id,"action":"start",
              "job":{"kind":"weight_inspect","inputs":[input]}}))["ok"]["state"],
            "queued"
        );
        rpc(json!({"op":"resolve","id":id,"index":0,"input":input}))
    };
    fs::write(root.join("weights.safetensors"), vec![41; shard.len()]).unwrap();
    // Resolve authenticates the sidecar, not source payload; the engine must
    // reject modified chunks when it actually reads them.
    assert_eq!(resolve("verified")["ok"]["bundle_sha256"], digest);
    fs::write(&output, b"{}").unwrap();
    assert!(
        resolve("tampered")["error"]
            .as_str()
            .unwrap()
            .contains("chunk index SHA-256 mismatch")
    );
    fs::remove_file(&output).unwrap();
    assert!(!run().status.success());
    assert!(!output.exists());
}
