use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    process::{Command, Stdio},
};

#[test]
fn rocm_edit_refuses_to_launch_an_unpinned_editor() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().canonicalize().unwrap();
    let fake = root.join("fake");
    let marker = root.join("was-run");
    fs::write(&fake, format!("#!/bin/sh\ntouch '{}'\n", marker.display())).unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o700)).unwrap();

    let model = root.join("model.gguf");
    let safe = root.join("model.safetensors");
    fs::write(&model, b"GGUF").unwrap();
    fs::write(&safe, b"safetensors").unwrap();
    let job = json!({
        "kind":"scratch_locate_edit",
        "inputs":[
            {"kind":"gguf","host":"test","path":model,"sha256":format!("{:x}", Sha256::digest(b"GGUF"))},
            {"kind":"safetensors","host":"test","path":safe,"sha256":format!("{:x}", Sha256::digest(b"safetensors"))}
        ],
        "options":{"method":"rome","backend":"rocm","layers":[0],
            "facts":[{"subject":"Alice","prompt":"{} lives in","target":"Paris"}]}
    });
    for (permission, diagnostic) in [
        ("0", "ROCm edit requires GPU load permission"),
        ("1", "Lanternfish SHA-256 mismatch"),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
            .current_dir(&root)
            .env("BREWERY_LANTERNFISH_EDIT_ROCM", &fake)
            .env("BREWERY_ALLOW_GPU_LOAD", permission)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        writeln!(child.stdin.take().unwrap(), "{job}").unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(!out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(diagnostic), "{stderr}");
        assert!(!marker.exists());
    }
}
