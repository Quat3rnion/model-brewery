use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    process::{Command, Output, Stdio},
};

fn sha(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn gguf() -> Vec<u8> {
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
    string(&mut out, "token_embd.weight");
    out.extend(2u32.to_le_bytes());
    out.extend(2u64.to_le_bytes());
    out.extend(6u64.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.resize(out.len().next_multiple_of(32) + 48, 0);
    out
}

fn run(mode: &str) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let bin = root.join("lanternfish");
    let script = format!(
        r#"#!/usr/bin/python3
import hashlib, json, pathlib, struct, sys
mode = {mode:?}
cmd, *args = sys.argv[1:]
opts = dict(a[2:].split("=",1) if "=" in a else (a[2:], next(it)) for it in [iter(args)] for a in it)
def sha(payload):
    return hashlib.sha256(payload).hexdigest()
def pack(*values):
    return struct.pack("<" + "f" * len(values), *values)
if cmd == "tokenize":
    text = opts["text"]
    print(json.dumps({{"tokens": [1 if text == "clean" else 2] + ([] if mode == "length" and text == "clean" else [3])}}))
    sys.exit(0)
out = pathlib.Path(opts["out"])
out.mkdir()
spec = json.loads(pathlib.Path(opts["spec"]).read_text())
if cmd == "trace":
    clean = opts["prompt"] == "clean"
    final = []
    tensors = []
    edits = []
    if spec["top_k"]:
        margin = 10 if clean else 0
        if "intervention" in opts:
            [rule] = json.loads(pathlib.Path(opts["intervention"]).read_text())
            assert rule["op"] == "replace" and rule["vector"] == [3, 1]
            role = rule["role"]
            margin = {{"l_out": 2, "attn_out": -4}}[role]
            if mode != "not-patched":
                edits.append(dict(layer=rule["layer"], role=role, position=rule["positions"][0]))
        final.append(dict(position=1, top=[dict(token_id=i, logit=margin if i == 4 else 0) for i in range(6)]))
    else:
        for i, role in enumerate(spec["roles"]):
            file = f"tensor-{{i:06}}.f32"
            data = pack(3 if clean else 1, 1)
            (out / file).write_bytes(data)
            tensors.append(dict(layer=0, role=role, position=1, file=file, node=role, shape=[2],
                                bytes=len(data), dtype="f32_le", sha256=sha(data)))
    tokens = [dict(position=0, token_id=1 if clean else 2), dict(position=1, token_id=3)]
    (out / "manifest.json").write_text(json.dumps(dict(schema_version=1, kind="trace", model={{}}, spec=spec,
                    tokens=tokens, tensors=tensors, missing=[], interventions=edits, model_logits=final, logit_lens=[])))
else:
    assert cmd == "grad-select"
    assert spec["tokens"] == [2,3] and spec["objective"] == "target_logit"
    gradient = {{"l_out":2, "attn_out":-1, "ffn_out":1}}[spec["site"]["role"]]
    value = gradient if spec["target_id"] == 4 else 1
    data = pack(value, 100)
    (out / "gradient.f32").write_bytes(data)
    if mode == "objective":
        spec["objective"] = "next_token_cross_entropy"
    (out / "gradient.json").write_text(json.dumps(dict(**spec, schema_version=1, backend="cpu",
                placement=dict(strict=True,device="CPU"), shape=[2], logit=0, dtype="f32_le",
                sha256=sha(data) if mode != "digest" else "a"*64)))
"#
    );
    fs::write(&bin, &script).unwrap();
    fs::set_permissions(&bin, fs::Permissions::from_mode(0o700)).unwrap();
    let manifest = format!(
        "{}  lanternfish\n{}  lib/cpu/libfixture.so\n",
        sha(&script),
        sha("fixture")
    );
    fs::create_dir_all(root.join("lib/cpu")).unwrap();
    fs::write(root.join("lib/cpu/libfixture.so"), "fixture").unwrap();
    fs::write(root.join("SHA256SUMS"), &manifest).unwrap();
    let model = root.join("tiny.gguf");
    let data = gguf();
    fs::write(&model, &data).unwrap();
    let digest = sha(&data);
    let bundle = sha(format!("tiny.gguf\0{}\0{digest}\n", data.len()));
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(root)
        .env_clear()
        .env("BREWERY_LANTERNFISH_SENSITIVITY_PIN", root)
        .env(
            "BREWERY_LANTERNFISH_SENSITIVITY_MANIFEST_SHA256",
            sha(manifest),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut options = json!({"model_sha256":digest,
        "clean_prompt":"clean", "corrupt_prompt":"corrupt", "target_id":4, "contrast_id":5,
        "layers":[0], "positions":[1], "roles":["l_out","ffn_out","attn_out"], "top_k":2});
    if mode == "raw" {
        options.as_object_mut().unwrap().remove("contrast_id");
        options["top_k"] = json!(1);
    }
    if mode == "sample" {
        options["top_k"] = json!(1);
        options["sample_lower"] = json!(1);
    }
    writeln!(
        child.stdin.take().unwrap(),
        "{}",
        json!({
            "kind":"circuit", "options":options,
            "inputs":[{"path":model,"kind":"gguf","host":"local","id":"model",
                "sha256":digest,"bundle_sha256":bundle,
                "files":[{"path":"tiny.gguf","sha256":digest,"bytes":data.len()}]}]
        })
    )
    .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn supports_a_raw_logit_and_leaves_single_patch_correlation_undefined() {
    let output = run("raw");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["sites"][0]["role"], "l_out");
    assert_eq!(result["sites"][0]["attribution"], 4.);
    assert_eq!(result["sites"][0]["patched_delta"], 2.);
    assert!(result["sites"][1]["patched_delta"].is_null());
    assert!(result["contrast_id"].is_null());
    assert!(result["pearson_r"].is_null());
    assert!(result["spearman_rho"].is_null());
}

#[test]
fn samples_lower_ranks_without_requiring_a_patch_at_each_rank() {
    let output = run("sample");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["top_k"], 1);
    assert_eq!(result["sample_lower"], 1);
    let sites = result["sites"].as_array().unwrap();
    assert_eq!(sites[0]["patched_delta"], -4.);
    assert_eq!(sites[1]["patched_delta"], 2.);
    assert!(sites[2]["patched_delta"].is_null());
    assert!((result["spearman_rho"].as_f64().unwrap() - 1.).abs() < 1e-12);
}

#[test]
fn ranks_signed_margins_and_confirms_only_the_top_sites() {
    let output = run("ok");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["clean_metric"], 10.);
    assert_eq!(result["corrupt_metric"], 0.);
    let sites = result["sites"].as_array().unwrap();
    assert_eq!(sites.len(), 3);
    assert_eq!(sites[0]["role"], "attn_out");
    assert_eq!(sites[0]["attribution"], -4.);
    assert_eq!(sites[0]["patched_delta"], -4.);
    assert_eq!(sites[1]["role"], "l_out");
    assert_eq!(sites[1]["patched_delta"], 2.);
    assert_eq!(sites[2]["role"], "ffn_out");
    assert!(sites[2]["patched_delta"].is_null());
    assert_eq!(result["pearson_r"], 1.);
    for mode in ["length", "objective", "digest", "not-patched"] {
        let output = run(mode);
        assert!(!output.status.success(), "{mode} accepted");
        assert!(output.stdout.is_empty(), "{mode} published a result");
    }
}
