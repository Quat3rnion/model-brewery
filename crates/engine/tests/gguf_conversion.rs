use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output, Stdio},
};

fn stage(root: &Path, kind: &str, options: Value, inputs: Vec<Value>, exe: &Path) -> Output {
    let work = root.join(kind);
    fs::create_dir(&work).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .env("BREWERY_GGUF_QUANTIZER", exe)
        .env("BREWERY_GGUF_HF_EXPORTER", root.join("exporter"))
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
fn scratch_conversion_routes_to_the_standalone_gguf_converter() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = tmp.path().join("scratch.safetensors");
    let assets = tmp.path().join("assets");
    fs::write(&weights, b"weights").unwrap();
    fs::create_dir(&assets).unwrap();
    let out = stage(
        tmp.path(),
        "scratch_gguf",
        json!({"output_format":"gguf","gguf_type":"BF16"}),
        vec![
            json!({"kind":"safetensors","host":"fixture","path":weights,"sha256":"0".repeat(64)}),
            json!({"kind":"assets","host":"fixture","path":assets}),
        ],
        Path::new("/usr/bin/false"),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("SHA-256 mismatch"), "{stderr}");
}

#[test]
fn scratch_conversion_enables_private_exporter_mode() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = tmp.path().join("scratch.safetensors");
    let assets = tmp.path().join("assets");
    fs::write(&weights, b"abc").unwrap();
    fs::create_dir(&assets).unwrap();
    fs::write(assets.join("config.json"), b"{}").unwrap();
    let exporter = tmp.path().join("exporter");
    fs::write(
        &exporter,
        "#!/bin/sh\nfor arg; do\n if [ \"$arg\" = --scratch-private ]; then\n echo 'private mode selected' >&2\n fi\ndone\nexit 1\n",
    ).unwrap();
    fs::set_permissions(&exporter, fs::Permissions::from_mode(0o700)).unwrap();
    let out = stage(
        tmp.path(),
        "scratch_gguf",
        json!({"output_format":"gguf","gguf_type":"BF16"}),
        vec![
            json!({"kind":"safetensors","host":"fixture","path":weights,
                "sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}),
            json!({"kind":"assets","host":"fixture","path":assets}),
        ],
        Path::new("/usr/bin/false"),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("private mode selected"), "{stderr}");
}

#[test]
fn standalone_inspect_passes_explicit_assets_and_attests_their_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let weights = root.join("weights");
    let assets = root.join("assets");
    fs::create_dir(&weights).unwrap();
    fs::create_dir(&assets).unwrap();
    let source = weights.join("explicit.safetensors");
    fs::write(&source, b"fixture weights").unwrap();
    fs::write(weights.join("config.json"), b"not selected").unwrap();
    fs::write(assets.join("config.json"), b"selected").unwrap();
    let exporter = root.join("exporter");
    fs::write(
        &exporter,
        r#"#!/bin/sh
set -eu
test "$#" = 7
test "$1" = inspect
test "$2" = --source
test "${3##*/}" = explicit.safetensors
test "$(cat "$3")" = "fixture weights"
test "$4 $5 $6" = "--outtype bf16 --assets"
test "${7##*/}" = assets
test -f "$7/config.json"
printf '%s\n' '{"source_format":"safetensors","export_supported":true,"expected_gguf_tensors":{"output.weight":[2,2]}}'
"#,
    )
    .unwrap();
    fs::set_permissions(&exporter, fs::Permissions::from_mode(0o700)).unwrap();
    let inputs = vec![
        json!({"kind":"safetensors","host":"fixture","path":source}),
        json!({"kind":"assets","host":"fixture","path":assets}),
    ];
    let inspect = |inputs: Vec<Value>| {
        let output = stage(
            root,
            "inspect",
            json!({"output_format":"gguf","gguf_type":"Q8_0"}),
            inputs,
            Path::new("/usr/bin/false"),
        );
        fs::remove_dir_all(root.join("inspect")).unwrap();
        output
    };
    let digest = || {
        let output = inspect(inputs.clone());
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["report"]["source_format"], "safetensors");
        assert_eq!(result["inputs"][1]["kind"], "assets");
        result["source_sha256"].clone()
    };
    let initial = digest();
    fs::write(weights.join("config.json"), b"changed unselected sibling").unwrap();
    assert_eq!(digest(), initial);
    fs::write(assets.join("config.json"), b"changed explicit metadata").unwrap();
    assert_ne!(digest(), initial);
    let mut bad_pin = inputs;
    bad_pin[1]["bundle_sha256"] = json!("a".repeat(64));
    let output = inspect(bad_pin);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("bundle SHA-256 mismatch"));
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
