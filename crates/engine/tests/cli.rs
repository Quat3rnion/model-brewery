use serde_json::json;
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

fn job(root: &Path, value: serde_json::Value) -> Output {
    run(
        root,
        &value.to_string(),
        "BREWERY_CONVERTER",
        Path::new("/native-must-not-start"),
    )
}

fn run(root: &Path, json: &str, var: &str, executable: &Path) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(root)
        .env("BREWERY_LANTERNFISH", "/native-must-not-start")
        .env("BREWERY_CONVERTER", "/native-must-not-start")
        .env(var, executable)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn executable(root: &Path, name: &str, script: &str) -> std::path::PathBuf {
    let path = root.join(name);
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}

#[test]
fn hf_roles_use_separate_converter_and_fail_closed_on_pin_or_accelerator() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let base = root.join("base");
    fs::create_dir(&base).unwrap();
    fs::write(base.join("model.safetensors.index.json"), "{}").unwrap();
    let vision = root.join("vision.safetensors");
    fs::write(&vision, "small sidecar fixture").unwrap();
    let standard = executable(
        root,
        "standard",
        r#"#!/bin/sh
echo '{"source_format":"hf","source_tensors":1,"output_hgn_bytes":16,"tensor_payloads_read":false,"outputs":[{}]}'
"#,
    );
    let component = executable(
        root,
        "component",
        r#"#!/bin/sh
printf '%s\n' "$@" > args
echo '{"source_format":"hf","source_tensors":1,"output_hgn_bytes":16,"tensor_payloads_read":false,"outputs":[{}],"source_selection":{"mode":"hf-role-overlays-v1","roles":{"assets":"embedded","ngram":"embedded","vision":"selected","mtp":"embedded"}}}'
"#,
    );
    let run_case = |name: &str, request: serde_json::Value, pin: bool, cpu_only: bool| {
        let work = root.join(name);
        fs::create_dir(&work).unwrap();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_brewery-engine"));
        cmd.current_dir(&work)
            .env("BREWERY_CONVERTER", &standard)
            .env("BREWERY_COMPONENT_CONVERTER", &component)
            .env(
                "BREWERY_COMPONENT_CPU_ONLY",
                if cpu_only { "1" } else { "0" },
            )
            .env_remove("BREWERY_CONVERTER_ADMISSION")
            .env_remove("BREWERY_EXPECTED_SOURCE_SHA256")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if pin {
            cmd.env("BREWERY_EXPECTED_SOURCE_SHA256", "a".repeat(64));
        }
        let mut child = cmd.spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(request.to_string().as_bytes())
            .unwrap();
        child.wait_with_output().unwrap()
    };
    let request = json!({"kind":"inspect","inputs":[
        {"path":base,"kind":"hf","host":"fixture"},
        {"id":"chosen-vision","path":vision,"kind":"vision","host":"fixture"}
    ],"options":{"vision":1}});
    let selected = run_case("selected", request.clone(), false, true);
    assert!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&selected.stdout).unwrap();
    assert_eq!(result["components"]["vision"]["source"], "input");
    assert_eq!(result["components"]["vision"]["input_id"], "chosen-vision");
    assert_eq!(result["components"]["vision"]["format"], "safetensors");
    assert_eq!(result["components"]["mtp"]["source"], "embedded");
    let args = fs::read_to_string(root.join("selected/args")).unwrap();
    assert!(args.contains("--vision\n"));
    assert!(args.contains("--ht-backend\ncpu\n"));
    let pristine = json!({"kind":"inspect","inputs":[request["inputs"][0]]});
    assert!(run_case("pristine", pristine, false, true).status.success());
    assert!(!root.join("pristine/args").exists());
    assert!(!run_case("pinned", request.clone(), true, false)
        .status
        .success());
    assert!(!root.join("pinned/args").exists());
    let mut gpu = request.clone();
    gpu["options"]["ht_backend"] = json!("cuda");
    assert!(!run_case("gpu", gpu, false, false).status.success());
    let output = root.join("output");
    fs::create_dir(&output).unwrap();
    let verify = json!({"kind":"verify","inputs":[request["inputs"][0],
        {"path":output,"kind":"hgn","host":"fixture"}, request["inputs"][1]],
        "options":{"vision":2,"decode_backend":"hip"}});
    let result = run_case("cpu_verify", verify, false, true);
    assert!(String::from_utf8_lossy(&result.stderr).contains("CPU-only"));
    assert!(!root.join("gpu/args").exists());
    assert!(!root.join("cpu_verify/args").exists());
}

#[test]
fn eval_subprocess_receives_the_host_cpu_budget() {
    let dir = tempfile::tempdir().unwrap();
    let probe = executable(dir.path(), "eval-env", "#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' \"$BREWERY_CPU_THREADS,$OMP_NUM_THREADS,$OPENBLAS_NUM_THREADS,$RAYON_NUM_THREADS\"\n");
    let limit = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(3);
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(dir.path())
        .env("BREWERY_EVAL", probe)
        .env("BREWERY_CPU_THREADS", "3")
        .env("OMP_NUM_THREADS", "77")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"kind":"generate"}"#)
        .unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(result.status.success());
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        format!("{limit},{limit},{limit},{limit}\n")
    );
}

#[test]
fn inspect_verifies_all_bundle_members_and_preserves_canonical_identity() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("checkpoint");
    fs::create_dir_all(source.join("a")).unwrap();
    fs::write(source.join("z"), "omega\n").unwrap();
    fs::write(source.join("middle"), "").unwrap();
    fs::write(source.join("a/child"), "alpha").unwrap();

    let members = json!([
        {"path":"a/child","bytes":5,"sha256":"8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"},
        {"path":"middle","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"path":"z","bytes":6,"sha256":"3eeb0cea8bf176427633a47a62ee8c813844d574d48554a0d715e12dcbbaeda6"}
    ]);
    let digest = "a9fb64baa189fa087abe3a0dce782d08be914685a36db9bd6ca27730ebaf3093";
    let request = json!({"kind":"inspect","inputs":[{
        "path":source,"kind":"hf","host":"fixture","bundle_sha256":digest,"files":members
    }]});
    let work = tmp.path().join("invalid-settings");
    fs::create_dir(&work).unwrap();
    let out = run(
        &work,
        &request.to_string(),
        "BREWERY_HASH_THREADS",
        Path::new("0"),
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("BREWERY_HASH_THREADS must be 1..16"));

    let converter = executable(
        tmp.path(),
        "inspect-fixture",
        r#"#!/bin/sh
printf '%s\n' '{"source_format":"hf","tensor_payloads_read":false,"outputs":[],"output_hgn_bytes":1}'
"#,
    );
    let work = tmp.path().join("valid");
    fs::create_dir(&work).unwrap();
    let out = run(&work, &request.to_string(), "BREWERY_CONVERTER", &converter);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["inputs"][0]["files"], members);
    assert_eq!(report["inputs"][0]["bytes"], 11);
    assert_eq!(report["inputs"][0]["sha256"], digest);
    assert_eq!(report["inputs"][0]["bundle_sha256"], digest);

    // A same-size change to a later member must still invalidate the input.
    fs::write(source.join("z"), "OMEGA\n").unwrap();
    let work = tmp.path().join("stale");
    fs::create_dir(&work).unwrap();
    let out = job(&work, request);
    assert!(!out.status.success() && out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("bundle SHA-256 mismatch"));
    assert!(!work.join("converter-evidence").exists());
    let progress: serde_json::Value =
        serde_json::from_slice(&fs::read(work.join("progress.json")).unwrap()).unwrap();
    assert_eq!(progress["phase"], "verify_input");
    assert_eq!(progress["unit"], "bytes");
    assert_eq!(progress["completed"], 11);
    assert_eq!(progress["total"], 11);
    assert_eq!(progress["files_completed"], 3);
    assert_eq!(progress["files_total"], 3);
    assert!(progress.get("threads").is_none());
}

#[test]
fn pinned_source_reports_its_basis_but_rejects_same_size_corruption_before_converter() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let source = root.join("checkpoint");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("weights"), "ALPHA").unwrap();
    let native = executable(
        root,
        "must-not-start",
        "#!/bin/sh\ntouch converter-started\n",
    );
    for (change, payload, error) in [
        (None, "ALPHA", "bundle SHA-256 mismatch"),
        (Some("files"), "alpha", "requires a nonempty file manifest"),
        (
            Some("bundle_sha256"),
            "alpha",
            "requires a valid bundle SHA-256",
        ),
        (
            Some("bytes"),
            "alpha",
            "pinned publication bundle size mismatch",
        ),
    ] {
        fs::write(source.join("weights"), payload).unwrap();
        let work = tempfile::tempdir_in(root).unwrap();
        let mut input = json!({
            "path":source,"kind":"hf","host":"fixture","source_host":"fixture",
            "integrity_basis":"pinned_publication",
            "bundle_sha256":"abc79d435a3dce1f7cfbb3e361f5df6dda8718a5d7cec3ba95a14d0466458c5a",
            "files":[{"path":"weights","bytes":5,
                "sha256":"8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"}]
        });
        if change == Some("bytes") {
            input["bytes"] = json!(6);
        } else if let Some(field) = change {
            input.as_object_mut().unwrap().remove(field);
        }
        let output = run(
            work.path(),
            &json!({"kind":"inspect","inputs":[input]}).to_string(),
            "BREWERY_CONVERTER",
            &native,
        );
        assert!(!output.status.success() && output.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(error), "{stderr}");
        assert!(!work.path().join("converter-started").exists());
        assert!(!work.path().join("converter-evidence").exists());
        if change.is_none() {
            let progress: serde_json::Value =
                serde_json::from_slice(&fs::read(work.path().join("progress.json")).unwrap())
                    .unwrap();
            assert_eq!(progress["phase"], "verify_pinned_input");
            assert_eq!(progress["completed"], 5);
            assert_eq!(progress["files_completed"], 1);
        }
    }
}

#[test]
fn inspect_reports_the_active_hash_slot_without_paths_or_stale_assignments() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("checkpoint");
    fs::create_dir_all(source.join("private-subdir")).unwrap();
    fs::File::create(source.join("private-subdir/model.safetensors"))
        .unwrap()
        .set_len(96 << 20)
        .unwrap();
    let converter = executable(
        tmp.path(),
        "inspect-fixture",
        r#"#!/bin/sh
printf '%s\n' '{"source_format":"hf","tensor_payloads_read":false,"outputs":[],"output_hgn_bytes":1}'
"#,
    );
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(tmp.path())
        .env("BREWERY_CONVERTER", converter)
        .env("BREWERY_HASH_THREADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let request = json!({"kind":"inspect","inputs":[{"path":source,"kind":"hf","host":"fixture"}]});
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();

    let path = tmp.path().join("progress.json");
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut observed = None;
    while Instant::now() < deadline && child.try_wait().unwrap().is_none() {
        if let Ok(bytes) = fs::read(&path) {
            assert!(bytes.len() <= 4096);
            let snapshot: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            if snapshot["phase"] == "verify_input" && snapshot.get("threads").is_some() {
                observed = Some(snapshot["threads"].clone());
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    if Instant::now() >= deadline {
        let _ = child.kill();
    }
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        observed,
        Some(json!([{"slot":1,"task":"model.safetensors"}]))
    );
    let final_progress: serde_json::Value =
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(final_progress["phase"], "identify_output");
    assert!(final_progress.get("threads").is_none());
}

#[test]
fn compare_rejects_digest_mismatch_before_creating_traces() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model.gguf");
    std::fs::write(&model, b"GGUFtest").unwrap();
    let a = json!({"path": model, "host": "test", "kind": "gguf",
                   "sha256": "0000000000000000000000000000000000000000000000000000000000000000"});
    let b = json!({"path": model, "host": "test", "kind": "gguf"});
    let result = job(
        tmp.path(),
        json!({"kind":"compare","inputs":[a,b],"options":{},"prompt":"x"}),
    );
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("SHA-256 mismatch"));
    assert!(!tmp.path().join("comparison").exists());
    let path = tmp.path().join("progress.json");
    let snapshot: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(snapshot["phase"], "verify_input");
    assert_eq!(snapshot["unit"], "bytes");
    assert_eq!(snapshot["completed"], 8);
    assert_eq!(snapshot["total"], 8);
    assert_eq!(snapshot["files_completed"], 1);
    assert_eq!(snapshot["files_total"], 1);
    assert_eq!(snapshot["input_index"], 1);
    assert_eq!(snapshot["input_count"], 2);
    assert_eq!(snapshot["current_file"], "model.gguf");
    assert_eq!(
        fs::metadata(path).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

#[test]
fn compare_rejects_shards_not_in_the_staged_file_set() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("unnumbered.gguf");
    let mut header = Vec::from(b"GGUF");
    header.extend(3u32.to_le_bytes());
    header.extend(1u64.to_le_bytes());
    header.extend(2u64.to_le_bytes());
    for (key, value) in [
        ("general.architecture", b"llama".as_slice()),
        ("split.count", &[2, 0]),
    ] {
        header.extend((key.len() as u64).to_le_bytes());
        header.extend(key.as_bytes());
        if key == "split.count" {
            header.extend(2u32.to_le_bytes()); // UINT16
        } else {
            header.extend(8u32.to_le_bytes()); // STRING
            header.extend((value.len() as u64).to_le_bytes());
        }
        header.extend(value);
    }
    fs::write(&model, header).unwrap();
    let input = json!({"path":model,"host":"test","kind":"gguf"});
    let result = job(
        tmp.path(),
        json!({"kind":"compare","inputs":[input,input],"options":{},"prompt":"x"}),
    );
    assert!(!result.status.success() && result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("split metadata disagrees"));
    assert!(!tmp.path().join("comparison").exists());
}

#[test]
fn compare_exports_bounded_native_signals_and_decode_text() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let model = root.join("test.gguf");
    let mut header = Vec::from(b"GGUF");
    header.extend(3u32.to_le_bytes());
    header.extend(1u64.to_le_bytes());
    header.extend(1u64.to_le_bytes());
    header.extend(20u64.to_le_bytes());
    header.extend(b"general.architecture");
    header.extend(8u32.to_le_bytes());
    header.extend(4u64.to_le_bytes());
    header.extend(b"test");
    fs::write(&model, header).unwrap();
    let tokens: Vec<_> = (0..132)
        .map(|i| {
            let (phase, piece) = if i < 130 {
                ("prefill", "prompt")
            } else if i == 130 {
                ("decode", " Hello")
            } else {
                ("decode", "!")
            };
            json!({"i":i,"id":i,"piece":piece,"phase":phase})
        })
        .collect();
    let layers: Vec<_> = (0..40).map(|layer| {
        let cells: serde_json::Map<_, _> = (0..132).filter(|i| !(layer == 0 && *i == 131))
            .map(|i| (i.to_string(), if layer == 0 && i == 130 {
                json!({"write":{"attn":0.125,"ffn":null},"experts":[9,3],"weights":[0.67,0.33],"lens":[["not included",1.0]]})
            } else { json!({"write":{"attn":0.0,"ffn":0.5}}) })).collect();
        json!({"layer":layer,"kind":"attention","tokens":cells})
    }).collect();
    fs::write(
        root.join("fixture.json"),
        json!({"tool":"lanternfish fixture","prompt":"prompt",
        "n_layers":40,"n_experts":16,"layers":layers,"tokens":tokens,
        "steps":[{"first":0,"n":130,"phase":"prefill"},{"first":130,"n":1,"phase":"decode"}],
        "next_candidates":[[],[]]})
        .to_string(),
    )
    .unwrap();
    let native = executable(
        root,
        "trace-fixture",
        r#"#!/bin/sh
set -eu
d="$2/traces/trace"
phase=$(basename "$2")
grep -q "\"phase\":\"compare_$phase\"" progress.json
grep -q '"completed":0' progress.json
! grep -q '"total"' progress.json
mkdir -p "$d"
cp fixture.json "$d/viz.json"
printf '{"tokens":132}\n' > "$d/meta.json"
printf '{"tokens":132,"path":"%s/viz.json"}\n' "$d"
"#,
    );
    let input = json!({"path":model,"host":"test","kind":"gguf"});
    let request =
        json!({"kind":"compare","inputs":[input,input],"options":{"n":2},"prompt":"prompt"});
    let out = run(root, &request.to_string(), "BREWERY_LANTERNFISH", &native);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.stdout.len() < 16 << 20);
    let result: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let snapshot: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("progress.json")).unwrap()).unwrap();
    assert_eq!(snapshot["phase"], "identify_output");
    assert_eq!(snapshot["completed"], result["artifact"]["bytes"]);
    assert_eq!(snapshot["completed"], snapshot["total"]);
    assert!(result["artifact"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .all(|file| !file["path"].as_str().unwrap().contains("progress.json")));
    for trace in result["traces"].as_array().unwrap() {
        assert_eq!(trace["output_text"], " Hello!");
        let signals = &trace["signals"];
        assert_eq!(signals["coverage"]["captured_cells"], 5279);
        assert!(signals["coverage"]["included_cells"].as_u64().unwrap() < 5279);
        assert_eq!(signals["coverage"]["truncated"], true);
        assert_eq!(signals["coverage"]["layer_span"], json!([0, 39]));
        assert_eq!(signals["coverage"]["token_span"], json!([0, 131]));
        assert_eq!(signals["omissions"]["missing_cells"], 1);
        let cells = signals["layers"][0]["cells"].as_array().unwrap();
        let decode = cells.iter().find(|c| c["token"] == 130).unwrap();
        assert_eq!(
            decode,
            &json!({"token":130,"attn":0.125,"experts":[9,3],"weights":[0.67,0.33]})
        );
        assert!(!cells.iter().any(|c| c["token"] == 131));
        assert_eq!(cells[0], json!({"token":0,"attn":0.0,"ffn":0.5}));
    }
}

#[test]
fn convert_publishes_bundle_with_distinct_encoder_and_executable_digests() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let source_dir = root.join("source");
    fs::create_dir(&source_dir).unwrap();
    fs::write(source_dir.join("payload"), "input fixture").unwrap();
    let source = json!({"path":source_dir,"host":"test","kind":"hf"});
    let encoder = "a".repeat(64);
    let fixture = root.join("fixture");
    fs::create_dir_all(fixture.join("tokenizer")).unwrap();
    fs::write(fixture.join("tokenizer/tokenizer.json"), "{}").unwrap();
    let containers = [
        ("model.hgn", "qwen3.8-flash-next-v2"),
        ("qwen38-flash-next-ngram.hgn", "qwen3.8-flash-next-ngram"),
        ("qwen38-flash-next-vision.hgn", "qwen3.8-flash-next-vision"),
    ];
    for (filename, identity) in containers {
        let mut header = vec![0; 321];
        header[..8].copy_from_slice(b"HGN1\x02\0\0\0");
        for (offset, value) in [(8, 1u64), (16, 104), (24, 320), (32, 321)] {
            header[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        header[40..40 + identity.len()].copy_from_slice(identity.as_bytes());
        fs::write(fixture.join(filename), header).unwrap();
    }
    let plan_files: Vec<_> = containers
        .iter()
        .map(|(filename, identity)| json!({"filename":filename,"identity":identity}))
        .collect();
    fs::write(
        fixture.join("conversion-manifest.json"),
        json!({"schema":2,"source_kind":"hf",
        "source_provenance":{},"encoder_sha256":encoder,"options":{"profile":"v2"},
        "plan":{"architecture":"qwen4_exp","files":plan_files}})
        .to_string(),
    )
    .unwrap();
    // Adapter contract test: this is not evidence of real encoding or of
    // successful device execution. The smoke tests use the pinned native CLI.
    let backend = executable(
        root,
        "fake-converter",
        r#"#!/bin/sh
set -eu
kind="$1"
while [ "$1" != '--output' ]; do shift; done
root=$(dirname "$0")
if [ "$kind" = 'convert' ]; then
  [ "$BREWERY_NATIVE_PROGRESS" = "$PWD/progress.json" ]
  printf '{"phase":"convert_passes_values","unit":"items","completed":2,"total":32,"updated_at_ms":%s}\n' "$(date +%s%3N)" > "$BREWERY_NATIVE_PROGRESS"
  tries=0
  until grep -q '"phase":"convert_passes_values","unit":"items","completed":2,"total":32' "$root/progress.json"; do
    tries=$((tries+1)); [ "$tries" -lt 80 ]; sleep .05
  done
  cp -r "$root/fixture" "$2"
  printf '%s\n' '{"published":true,"source_format":"hf","profile":"v2","manifest":"conversion-manifest.json"}'
else
  [ "$kind" = 'verify' ] && [ -f "$2/conversion-manifest.json" ] && [ ! -e "$2/manifest.json" ]
  printf '%s\n' '{"manifest_schema":2,"provenance_attested":true,"source_tensors":1,"output_tensors":3,"payload_bytes":321,"errors":{}}'
fi
"#,
    );
    let convert = json!({"kind":"convert","inputs":[source],"options":{}}).to_string();
    let result = run(root, &convert, "BREWERY_CONVERTER", &backend);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(result["report"]["encoder_source_sha256"], encoder);
    assert_ne!(
        result["backend_receipt"]["processes"][0]["executable_sha256"],
        encoder
    );
    assert_eq!(result["hardware_proof"]["status"], "not_proved");
    assert_eq!(result["file_verification"]["status"], "not_performed");
    let output = root.join("converted");
    assert_eq!(result["artifact"]["path"], json!(output));
    let snapshot: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("progress.json")).unwrap()).unwrap();
    assert_eq!(snapshot["phase"], "publish_output");
    assert_eq!(snapshot["completed"], 1);
    assert!(!output.join("progress.json").exists());
    let runtime: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(runtime["files"][0]["path"], "native/model.hgn");
    assert_eq!(
        runtime["tokenizer"][0]["path"],
        "native/tokenizer/tokenizer.json"
    );
    assert_eq!(runtime["source_sha256"], result["source_sha256"]);
    assert!(output.join("native/conversion-manifest.json").is_file());
    assert!(!output.join("native/manifest.json").exists());

    let retry = run(root, &convert, "BREWERY_CONVERTER", &backend);
    assert!(!retry.status.success() && retry.stdout.is_empty());
    assert!(String::from_utf8_lossy(&retry.stderr).contains("path already exists"));
    let verify_dir = root.join("verify");
    fs::create_dir(&verify_dir).unwrap();
    let verify =
        json!({"kind":"verify","inputs":[source,{"path":output,"host":"test","kind":"hgn"}],
                        "options":{"decode_backend":"cuda"}})
        .to_string();
    let verified = run(&verify_dir, &verify, "BREWERY_CONVERTER", &backend);
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let verified: serde_json::Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(verified["file_verification"]["status"], "passed");
    assert_eq!(verified["hardware_proof"]["status"], "not_proved");
}

#[test]
fn verify_rejects_contradictory_status_but_accepts_native_report_shape() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let source = root.join("source");
    let output = root.join("converted");
    fs::create_dir(&source).unwrap();
    fs::create_dir(&output).unwrap();
    fs::write(source.join("config.json"), "{}").unwrap();
    fs::write(output.join("conversion-manifest.json"), "{}").unwrap();
    let verify = json!({"kind":"verify","inputs":[
        {"path": source,"host":"test","kind":"hf"},
        {"path": output,"host":"test","kind":"hgn"}],"options":{"decode_backend":"cpu"}})
    .to_string();
    for (label, passed, code, error) in [
        ("absent", "", 0, None),
        ("true", r#","passed":true"#, 0, None),
        ("failed-report", r#","passed":false"#, 0, Some("passed")),
        ("null-report", r#","passed":null"#, 0, Some("passed")),
        ("string-report", r#","passed":"true""#, 0, Some("passed")),
        ("nonzero", "", 2, Some("exit status: 2")),
    ] {
        let cwd = root.join(label);
        fs::create_dir(&cwd).unwrap();
        let native = executable(
            &cwd, "verify-fixture", &format!(
                "#!/bin/sh\nprintf '%s\\n' '{{\"manifest_schema\":2,\"provenance_attested\":true,\"source_tensors\":1,\"output_tensors\":1,\"payload_bytes\":321,\"errors\":{{}}{passed}}}'\nexit {code}\n"
            ),
        );
        let actual = run(&cwd, &verify, "BREWERY_CONVERTER", &native);
        let stderr = String::from_utf8_lossy(&actual.stderr);
        if let Some(expected) = error {
            assert!(!actual.status.success(), "{label} unexpectedly succeeded");
            assert!(
                actual.stdout.is_empty(),
                "{label} produced a success result"
            );
            assert!(stderr.contains(expected), "{label}: {stderr}");
            assert!(!cwd.join("converter-evidence/verify.json").exists());
        } else {
            assert!(actual.status.success(), "{label}: {stderr}");
            let valid: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
            assert_eq!(valid["report"]["passed"], true);
            assert_eq!(valid["file_verification"]["status"], "passed");
        }
    }
}

#[test]
fn delegate_keeps_original_json_bytes_and_workdir() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("expected-cwd"), "").unwrap();
    let backend = executable(
        tmp.path(),
        "eval",
        "#!/bin/sh\nset -eu\ntest -f expected-cwd\ncat\n",
    );
    let message = " { \"kind\": \"baseline\", \"extra\": [ 1 ] }\n";
    let out = run(tmp.path(), message, "BREWERY_EVAL", &backend);
    assert!(out.status.success());
    assert_eq!(out.stdout, message.as_bytes());
}
