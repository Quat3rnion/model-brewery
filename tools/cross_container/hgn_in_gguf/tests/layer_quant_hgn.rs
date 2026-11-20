use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

const HELPER: &str = env!("CARGO_BIN_EXE_brewery-layer-quant-hgn");

fn write(model: &Path, dtype: &str, target: &str, payload: &[u8]) -> Output {
    let mut child = Command::new(HELPER)
        .arg("write")
        .arg(model)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let descriptor =
        json!([{"name": "test.weight", "shape": [1, 2], "dtype": dtype, "target": target}]);
    writeln!(input, "{descriptor}").unwrap();
    input.write_all(payload).unwrap();
    drop(input);
    child.wait_with_output().unwrap()
}

#[test]
fn f32_copy_preserves_nan_payload_and_negative_zero() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.hgn");
    let payload = [0x42, 0, 0xc0, 0x7f, 0, 0, 0, 0x80];
    let result = write(&model, "F32", "F32", &payload);
    assert!(result.status.success(), "{:?}", result.stderr);
    let result = Command::new(HELPER)
        .arg("inspect")
        .arg(&model)
        .output()
        .unwrap();
    assert!(result.status.success(), "{:?}", result.stderr);
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    let tensor = &report["tensors"][0];
    assert_eq!(report["version"], 2);
    assert_eq!(tensor["dtype"], "F32");
    assert_eq!(tensor["size"], 8);
    let start = tensor["offset"].as_u64().unwrap() as usize;
    assert_eq!(fs::read(model).unwrap()[start..start + 8], payload);
}

#[test]
fn unrepresentable_f16_and_widening_do_not_publish_a_file() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.hgn");
    for (dtype, target, bytes, reason) in [
        (
            "BF16",
            "F16",
            [0x80, 0x47, 0, 0],
            "cannot be represented as a finite f16",
        ),
        ("F16", "F32", [0, 0x3c, 0, 0], "widening is not supported"),
    ] {
        let result = write(&model, dtype, target, &bytes);
        assert!(!result.status.success());
        assert!(String::from_utf8(result.stderr).unwrap().contains(reason));
        assert!(!model.exists());
    }
}
