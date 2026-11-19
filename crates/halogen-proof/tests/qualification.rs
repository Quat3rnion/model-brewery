#![cfg(target_os = "linux")]

use std::{
    fs,
    process::{Command, Output},
};

fn qualify_fixture(format: &str, digest: &str) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = [0u8; 352];
    fixture[..8].copy_from_slice(b"HGN1\x02\x00\x00\x00");
    fixture[8] = 1;
    fixture[16] = 104;
    fixture[24..32].copy_from_slice(&320u64.to_le_bytes());
    fixture[32..40].copy_from_slice(&352u64.to_le_bytes());
    let model_id = b"qwen3.8-flash-next-ht43";
    fixture[40..40 + model_id.len()].copy_from_slice(model_id);
    fs::write(dir.path().join("model.hgn"), fixture).unwrap();
    fs::write(
        dir.path().join("bundle.json"),
        format!(
            r#"{{"format":"{format}","files":[{{"role":"checkpoint","path":"model.hgn","sha256":"{digest}"}}]}}"#
        ),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_halogen-proof"))
        .args(["qualify", dir.path().join("bundle.json").to_str().unwrap()])
        .output()
        .unwrap()
}

#[test]
fn qualification_refuses_a_digest_mismatch_without_starting_a_runtime() {
    let result = qualify_fixture("ht43", &"a".repeat(64));
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("SHA-256 mismatch"));
}

#[test]
fn ht43_is_qualified_by_header_and_digest_not_filename_or_claimed_format() {
    const DIGEST: &str = "831e219571648ce7994bfeecde2aac80601a687ac7d7241bb934a47e58689e3a";
    let accepted = qualify_fixture("ht43", DIGEST);
    assert!(accepted.status.success(), "{:?}", accepted);
    let json: serde_json::Value = serde_json::from_slice(&accepted.stdout).unwrap();
    assert_eq!(json["files"][0]["sha256"], DIGEST);
    assert_eq!(json["files"][0]["hgn_model_id"], "qwen3.8-flash-next-ht43");
    let refused = qualify_fixture("hgn-v2", DIGEST);
    assert!(!refused.status.success());
    assert!(refused.stdout.is_empty());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("does not match"));
}
