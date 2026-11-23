use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output, Stdio},
};

const REV: &str = "fb08d7cdb67082ddda15a25e64dc7ec413afc41c";

fn gguf(path: &Path, file_type: u32, ty: u32, payload: usize) {
    fn string(bytes: &mut Vec<u8>, s: &str) {
        bytes.extend((s.len() as u64).to_le_bytes());
        bytes.extend(s.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    bytes.extend(2u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8u32.to_le_bytes());
    string(&mut bytes, "llama");
    string(&mut bytes, "general.file_type");
    bytes.extend(4u32.to_le_bytes());
    bytes.extend(file_type.to_le_bytes());
    string(&mut bytes, "blk.0.attn_q.weight");
    bytes.extend(2u32.to_le_bytes());
    bytes.extend(64u64.to_le_bytes());
    bytes.extend(32u64.to_le_bytes());
    bytes.extend(ty.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.resize(bytes.len().next_multiple_of(32) + payload, 0);
    fs::write(path, bytes).unwrap();
}

fn input(path: &Path) -> Value {
    json!({"kind":"gguf","host":"fixture","path":path})
}

fn stage(
    work: &Path,
    kind: &str,
    inputs: Vec<Value>,
    options: Value,
    exe: &Path,
    sha: &str,
) -> Output {
    fs::create_dir(work).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .env_remove("BREWERY_EXPECTED_SOURCE_SHA256")
        .env("BREWERY_GGUF_QUANTIZER", exe.with_file_name("stock"))
        .env("BREWERY_GGUF_ROCMFPX_QUANTIZER", exe)
        .env("BREWERY_GGUF_ROCMFPX_SHA256", sha)
        .current_dir(work)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    write!(
        child.stdin.take().unwrap(),
        "{}",
        json!({"kind":kind,"inputs":inputs,"options":options})
    )
    .unwrap();
    child.wait_with_output().unwrap()
}

fn report(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn failure(output: Output, reason: &str) {
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(reason), "{stderr}");
}

#[test]
fn fork_requires_operator_pin_and_verifies_private_type_layout_independently() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let source = root.join("source.gguf");
    let template = root.join("template.gguf");
    let marker = root.join("executed");
    let exe = root.join("fork");
    let stock_exe = exe.with_file_name("stock");
    fs::write(&stock_exe, "#!/bin/sh\nexit 1\n").unwrap();
    fs::set_permissions(&stock_exe, fs::Permissions::from_mode(0o700)).unwrap();
    // File type 103 is the FAST preset, but tensor 103 means FP8, not FAST FP4.
    gguf(&source, 0, 0, 64 * 32 * 4);
    gguf(&template, 103, 101, 2 * 32 * 17);
    let script = format!(
        r#"#!/bin/sh
set -eu
touch '{marker}'
dry=no
if [ "$1" = --dry-run ]; then dry=yes; shift; fi
test "$1" = --tensor-type
test "$2" = '^blk\.0\.attn_q\.weight$=q4_0_rocmfp4_fast'
test "$5" = Q4_0_ROCMFP4_FAST
if [ "$dry" = yes ]; then
    echo '[   1/   1] blk.0.attn_q.weight - [64, 32], type = f32, size = 0.01 MiB -> 0.00 MiB (q4_0_rocmfp4_fast)'
else
    cp '{template}' "$4"
fi
"#,
        marker = marker.display(),
        template = template.display()
    );
    let sha = format!("{:x}", Sha256::digest(script.as_bytes()));
    fs::write(&exe, script).unwrap();
    fs::set_permissions(&exe, fs::Permissions::from_mode(0o700)).unwrap();
    let opts = json!({"output_format":"gguf", "gguf_type":"Q4_0_ROCMFP4_FAST", "encoder_family":"rocmfpx",
        "encoder_revision":REV, "tensor_types":{"blk.0.attn_q.weight":"Q4_0_ROCMFP4_FAST"}, "threads":1});
    let run = |label: &str, kind, options, inputs, pin: &str| {
        stage(&root.join(label), kind, inputs, options, &exe, pin)
    };
    failure(
        run(
            "bad-pin",
            "inspect",
            opts.clone(),
            vec![input(&source)],
            &"0".repeat(64),
        ),
        "ROCmFPX quantizer SHA-256 mismatch",
    );
    assert!(!marker.exists());
    let mut stock = opts.clone();
    stock.as_object_mut().unwrap().remove("encoder_family");
    stock.as_object_mut().unwrap().remove("encoder_revision");
    failure(
        run("stock", "inspect", stock, vec![input(&source)], &sha),
        "unsupported GGUF preset",
    );
    assert!(!marker.exists());
    let mut wrong_revision = opts.clone();
    wrong_revision["encoder_revision"] = json!("other");
    failure(
        run(
            "wrong-revision",
            "inspect",
            wrong_revision,
            vec![input(&source)],
            &sha,
        ),
        "unsupported GGUF encoder family or revision",
    );
    assert!(!marker.exists());
    gguf(&template, 103, 0, 64 * 32 * 4);
    failure(
        run(
            "fork-source",
            "inspect",
            opts.clone(),
            vec![input(&template)],
            &sha,
        ),
        "source GGUF cannot use a private ROCmFPX file type",
    );
    assert!(!marker.exists());
    gguf(&template, 103, 101, 2 * 32 * 17);
    let inspect = report(run(
        "inspect",
        "inspect",
        opts.clone(),
        vec![input(&source)],
        &sha,
    ));
    assert_eq!(inspect["backend_receipt"]["encoder_family"], "rocmfpx");
    assert_eq!(inspect["backend_receipt"]["encoder_revision"], REV);
    assert_eq!(inspect["backend_receipt"]["executable_sha256"], sha);
    assert_eq!(
        inspect["backend_receipt"]["stock_runtime_compatible"],
        false
    );
    assert_eq!(
        inspect["report"]["gguf_tensors"][0]["planned_type"],
        "Q4_0_ROCMFP4_FAST"
    );
    report(run(
        "plan",
        "plan",
        opts.clone(),
        vec![input(&source)],
        &sha,
    ));
    let converted = report(run(
        "convert",
        "convert",
        opts.clone(),
        vec![input(&source)],
        &sha,
    ));
    assert_eq!(
        converted["report"]["gguf_tensors"][0]["current_type"],
        "Q4_0_ROCMFP4_FAST"
    );
    assert_eq!(converted["release_qualified"], false);
    report(run(
        "verify",
        "verify",
        opts.clone(),
        vec![
            input(&source),
            input(Path::new(converted["artifact"]["path"].as_str().unwrap())),
        ],
        &sha,
    ));
    gguf(&template, 101, 101, 2 * 32 * 17);
    failure(
        run(
            "wrong-file-type",
            "verify",
            opts.clone(),
            vec![input(&source), input(&template)],
            &sha,
        ),
        "GGUF output preset metadata mismatch",
    );
    gguf(&template, 103, 103, 2 * 32 * 33);
    failure(
        run(
            "wrong-type",
            "verify",
            opts.clone(),
            vec![input(&source), input(&template)],
            &sha,
        ),
        "GGUF tensor type mismatch",
    );
    gguf(&template, 103, 101, 2 * 32 * 17 - 1);
    failure(
        run(
            "truncated",
            "verify",
            opts,
            vec![input(&source), input(&template)],
            &sha,
        ),
        "GGUF tensor payload overlaps or is truncated",
    );
}
