use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output, Stdio},
};

const CORPUS: &str = "9678f9afdc31b2a9be7fec2045997e2360693ee72b893287c0d539983bff4923";

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
    out.extend(3u64.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    string(&mut out, "general.architecture");
    out.extend(8u32.to_le_bytes());
    string(&mut out, "qwen2");
    for (i, (name, shape)) in [
        ("blk.0.attn_q.weight", &[2u64, 2][..]),
        ("blk.0.attn_k.weight", &[2u64, 1][..]),
        ("blk.0.attn_q.bias", &[2u64][..]),
    ]
    .into_iter()
    .enumerate()
    {
        string(&mut out, name);
        out.extend((shape.len() as u32).to_le_bytes());
        for n in shape {
            out.extend(n.to_le_bytes());
        }
        out.extend(0u32.to_le_bytes());
        out.extend((32 * i as u64).to_le_bytes());
    }
    out.resize(out.len().next_multiple_of(32) + 96, 0);
    out
}

fn run(root: &Path, mode: &str, options: Value) -> Output {
    let bin = root.join("lanternfish");
    let script = format!(
        r#"#!/usr/bin/python3
import hashlib, json, pathlib, struct, sys
mode = {mode:?}
cmd, *args = sys.argv[1:]
opts = dict(zip(args[::2], args[1::2]))
if cmd == "tokenize":
    print(json.dumps({{"tokens": [1, 2, 3] if " " in opts["--text"] else [1]}}))
else:
    spec = json.loads(pathlib.Path(opts["--spec"]).read_text())
    is_q = spec["site"]["name"] == "blk.0.attn_q.weight"
    dim = 2 if is_q else 1
    value = spec["target_id"] - (1 if is_q else 0) + (spec["target_id"] == 3)
    data = struct.pack("<" + "f" * (2 * dim), *[float("nan") if mode == "nan" else value] * (2 * dim))
    out = pathlib.Path(opts["--out"])
    out.mkdir()
    (out / "gradient.f32").write_bytes(data[:-4] if mode == "short" else data)
    (out / "gradient.json").write_text(json.dumps({{
        **spec, "schema_version": 1, "shape": [2, dim], "loss": 0.5,
        "backend": "cuda" if mode == "backend" else opts["--backend"],
        "placement": {{"strict": mode != "fallback", "device": "CPU"}},
        "sha256": hashlib.sha256(data).hexdigest() if mode != "digest" else "a" * 64
    }}))
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
    fs::write(
        root.join("lib/cpu/libfixture.so"),
        if mode == "lib" { "wrong" } else { "fixture" },
    )
    .unwrap();
    fs::write(root.join("SHA256SUMS"), &manifest).unwrap();
    if mode == "binary" {
        fs::write(&bin, "#!/bin/sh\nexit 0\n").unwrap();
    }
    let model = root.join("tiny.gguf");
    let data = gguf();
    fs::write(&model, &data).unwrap();
    let digest = sha(&data);
    let bundle = sha(format!("tiny.gguf\0{}\0{digest}\n", data.len()));
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(tempfile::tempdir_in(root).unwrap().keep())
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
    writeln!(
        child.stdin.take().unwrap(),
        "{}",
        json!({
            "kind":"sensitivity", "options":options,
            "inputs":[{"path":model,"kind":"gguf","host":"local","id":"model",
                "sha256":digest,"bundle_sha256":bundle,
                "files":[{"path":"tiny.gguf","sha256":digest,"bytes":data.len()}]}]
        })
    )
    .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn measures_individual_observations_and_rejects_invalid_runtime_output() {
    let dir = tempfile::tempdir().unwrap();
    let options =
        json!({"layers":[0],"observations":2,"corpus_sha256":CORPUS,"model_sha256":sha(gguf())});
    let output = run(dir.path(), "ok", options.clone());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let artifact: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(artifact["schema_version"], 1);
    assert_eq!(artifact["qualification"], "not release-qualified");
    assert_eq!(artifact["revision_sha256"], options["model_sha256"]);
    assert_eq!(artifact["observation_count"], 2);
    let rows = artifact["tensors"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["name"], "blk.0.attn_k.weight");
    assert_eq!(rows[0]["fisher"], 10.);
    assert_eq!(rows[1]["fisher"], 5.);
    assert!((artifact["layers"][0]["fisher"].as_f64().unwrap() - 20. / 3.).abs() < 1e-14);
    for mode in [
        "binary", "lib", "short", "nan", "digest", "backend", "fallback",
    ] {
        let output = run(dir.path(), mode, options.clone());
        assert!(!output.status.success(), "{mode} accepted");
        assert!(output.stdout.is_empty(), "{mode} published a result");
    }
    let mut bad = options.clone();
    bad["model_sha256"] = json!("a".repeat(64));
    assert!(!run(dir.path(), "ok", bad).status.success());
    let mut bad = options;
    bad["corpus_sha256"] = json!("a".repeat(64));
    assert!(!run(dir.path(), "ok", bad).status.success());
}
