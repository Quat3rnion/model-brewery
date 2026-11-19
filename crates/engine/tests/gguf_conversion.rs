use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

fn stage(root: &Path, kind: &str, options: Value, inputs: Vec<Value>, exe: &Path) -> Output {
    let work = root.join(kind);
    fs::create_dir(&work).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .env("BREWERY_GGUF_QUANTIZER", exe)
        .env_remove("BREWERY_EXPECTED_SOURCE_SHA256")
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

#[test]
fn gguf_route_requires_a_dedicated_quantizer_before_reading_the_model() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("source.gguf");
    fs::write(&source, b"GGUF").unwrap();
    let out = stage(
        tmp.path(),
        "inspect",
        json!({"output_format":"gguf","gguf_type":"Q8_0"}),
        vec![json!({"kind":"gguf","host":"fixture","path":source})],
        Path::new("inherited-command"),
    );
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("BREWERY_GGUF_QUANTIZER must be absolute")
    );
}

#[test]
#[ignore = "requires an explicitly selected tiny local fixture and CPU llama-quantize"]
fn real_cpu_quantizer_plan_convert_and_verify_exact_tensor_types() {
    let executable = std::env::var("BREWERY_TEST_LLAMA_QUANTIZE").unwrap();
    let source_path = std::env::var("BREWERY_TEST_TINY_GGUF").unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let input = json!({"kind":"gguf","host":"fixture","path":source_path});
    let options = json!({"output_format":"gguf","gguf_type":"Q8_0",
        "tensor_types":{"token_embd.weight":"F16"},"threads":1});
    let mut converted = Value::Null;
    for kind in ["inspect", "plan", "convert", "verify"] {
        let mut inputs = vec![input.clone()];
        if kind == "verify" {
            inputs.push(
                json!({"kind":"gguf","host":"fixture","path":converted["artifact"]["path"],
                "sha256":converted["artifact"]["sha256"]}),
            );
        }
        let out = stage(
            tmp.path(),
            kind,
            options.clone(),
            inputs,
            Path::new(&executable),
        );
        assert!(
            out.status.success(),
            "{kind}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let body: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(body["output_format"], "gguf");
        assert_eq!(body["release_qualified"], false);
        if kind == "convert" {
            assert_eq!(body["artifact"]["kind"], "gguf");
            converted = body;
        } else if kind == "verify" {
            assert_eq!(body["report"]["passed"], true);
            assert_eq!(body["source_sha256"], converted["source_sha256"]);
        }
    }
    let wrong = tmp.path().join("wrong");
    fs::create_dir(&wrong).unwrap();
    let different = json!({"output_format":"gguf","gguf_type":"Q8_0","tensor_types":{"token_embd.weight":"Q8_0"}});
    let out = stage(
        &wrong,
        "verify",
        different,
        vec![
            input,
            json!({"kind":"gguf","host":"fixture","path":converted["artifact"]["path"]}),
        ],
        Path::new(&executable),
    );
    assert!(!out.status.success() && out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("tensor type mismatch"));
}
