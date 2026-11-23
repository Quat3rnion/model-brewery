use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    process::{Command, Stdio},
};

fn sha(data: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn gguf(ty: u32, payload: &[u8]) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    string(&mut out, "general.architecture");
    out.extend(8u32.to_le_bytes());
    string(&mut out, "qwen2");
    string(&mut out, "blk.0.attn_q.weight");
    out.extend(2u32.to_le_bytes());
    out.extend(32u64.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    out.extend(ty.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.resize(out.len().next_multiple_of(32), 0);
    out.extend(payload);
    out
}

fn damage(options: Value, fixture_mode: &str) -> std::process::Output {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    let script = include_str!("damage_fixture.py").replace("__FIXTURE_MODE__", fixture_mode);
    let bin = dir.join("lanternfish");
    fs::write(&bin, &script).unwrap();
    fs::set_permissions(&bin, fs::Permissions::from_mode(0o700)).unwrap();
    fs::create_dir_all(dir.join("lib/cpu")).unwrap();
    fs::write(dir.join("lib/cpu/libfixture.so"), "fixture").unwrap();
    let manifest = format!(
        "{}  lanternfish\n{}  lib/cpu/libfixture.so\n",
        sha(&script),
        sha("fixture")
    );
    fs::write(dir.join("SHA256SUMS"), &manifest).unwrap();
    let input = |name: &str, data: Vec<u8>| {
        fs::write(dir.join(name), &data).unwrap();
        let digest = sha(&data);
        let bundle = sha(format!("{name}\0{}\0{digest}\n", data.len()));
        json!({"id":name,"path":dir.join(name),"kind":"gguf","host":"local","sha256":digest,"bundle_sha256":bundle,
               "files":[{"path":name,"sha256":digest,"bytes":data.len()}]})
    };
    let reference = input("reference.gguf", gguf(30, &[0x00, 0x40].repeat(32))); // BF16 2
    let quant = input(
        "quant.gguf",
        gguf(8, &[vec![0, 0x38], vec![2; 32]].concat()),
    ); // .5 * 2 = 1
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(dir)
        .env_clear()
        .env("BREWERY_LANTERNFISH_SENSITIVITY_PIN", dir)
        .env(
            "BREWERY_LANTERNFISH_SENSITIVITY_MANIFEST_SHA256",
            sha(manifest),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut job_options = json!({"reference_sha256":reference["sha256"],"quantized_sha256":quant["sha256"],
                                "swaps":1,"gguf_type":"Q4_K_M"});
    job_options
        .as_object_mut()
        .unwrap()
        .extend(options.as_object().unwrap().clone());
    writeln!(
        child.stdin.take().unwrap(),
        "{}",
        json!({
            "kind":"quantization_damage","inputs":[reference,quant],
            "options":job_options
        })
    )
    .unwrap();
    child.wait_with_output().unwrap()
}

fn success(out: std::process::Output) -> Value {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn attributes_dequantized_error_and_checks_a_real_file_swap() {
    let receipt = success(damage(json!({}), ""));
    assert_eq!(receipt["tensors"][0]["predicted_recovery"], json!([8.0]));
    assert_eq!(receipt["tensors"][0]["measured_recovery"], json!([8.0]));
    assert_eq!(receipt["layers"][0]["layer"], 0);
    assert_eq!(receipt["observations"][0]["measured_drift"], 8.0);
    assert_eq!(receipt["recipe_proposal"]["applied"], false);
}

#[test]
fn corpus_loss_attributes_quantized_minus_reference_without_logit_swaps() {
    let receipt = success(damage(json!({"objective":"corpus_nll"}), ""));
    assert_eq!(receipt["objective"], "corpus_nll");
    assert_eq!(receipt["corpus"]["reference_mean_nll"], 1.25);
    assert_eq!(receipt["corpus"]["observation_count"], 222);
    assert_eq!(receipt["corpus"]["predicted_nll_increase"], -8.0);
    assert_eq!(receipt["tensors"][0]["predicted_nll_increase"], -8.0);
    assert_eq!(receipt["tensors"][0]["mean_absolute_damage"], 8.0);
    assert_eq!(receipt["layers"][0]["predicted_nll_increase"], -8.0);
    assert!(receipt["tensors"][0]["measured_recovery"].is_null());
    assert_eq!(receipt["observations"], json!([]));
}

#[test]
fn corpus_loss_rejects_corrupt_or_unexpected_gradient_entries() {
    for (mode, error) in [
        ("bad-hash", "gradient payload size or SHA-256 mismatch"),
        ("bad-path", "invalid corpus gradient file"),
        ("bad-shape", "invalid shape for corpus gradient"),
    ] {
        let out = damage(json!({"objective":"corpus_nll"}), mode);
        assert!(!out.status.success(), "accepted {mode}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(error),
            "{mode}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
