use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output, Stdio},
};

const TENSOR: &str = "layers.0.mlp.experts.down_proj.weight";
const CONVERTER: &str = r#"#!/usr/bin/python3
import hashlib, json, os, pathlib, stat, sys
kind, *args = sys.argv[1:]
arg = lambda key: args[args.index('--' + key) + 1]
recipe_file = pathlib.Path(arg('recipe'))
assert stat.S_IMODE(recipe_file.stat().st_mode) == 0o600
assert recipe_file.parent == pathlib.Path(arg('source')).parent / kind
recipe = json.loads(recipe_file.read_bytes())
assert recipe == dict(schema='bf16-hgn-tensor-recipe-v1', base_profile='v2',
                     alternate_tensors=['layers.0.mlp.experts.down_proj.weight'])
for backend in (['decode-backend'] if kind == 'verify' else ['ht-backend','i4r-backend','q4cp-backend']):
    assert arg(backend) == 'cpu'
h = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
recipe['sha256'] = h
policy = f'bf16-hgn-tensor-recipe-v1:{h}'
summary = dict(source_format='hf', tensor_payloads_read=False, outputs=[], output_hgn_bytes=320,
               recipe=recipe,policy=policy,profile='v2')
identity = f'qwen3.8-flash-next-recipe-{h[:32]}'
if kind == 'plan':
    out = dict(summary=summary,plan=dict(architecture='qwen4_exp',files=[
        dict(filename='model.hgn',identity=identity)]),layouts=[])
elif kind == 'convert':
    output = pathlib.Path(arg('output'))
    output.mkdir()
    manifest = dict(schema=2, source_kind='hf', source_provenance={}, encoder_sha256='a'*64,
        recipe=recipe, policy=policy, options=dict(profile='v2'),
        plan=dict(architecture='qwen4_exp',files=[dict(filename='model.hgn',identity=identity)]))
    (output/'conversion-manifest.json').write_text(json.dumps(manifest))
    header = bytearray(321)
    header[:8] = b'HGN1\x02\x00\x00\x00'
    for i,n in [(8,1),(16,104),(24,320),(32,321)]: header[i:i+8]=n.to_bytes(8,'little')
    header[40:40+len(identity)] = identity.encode()
    (output/'model.hgn').write_bytes(header)
    out = dict(published=True,manifest='conversion-manifest.json',source_format='hf',profile='v2',recipe=recipe,policy=policy)
elif kind == 'verify':
    out = dict(manifest_schema=2, provenance_attested=True, source_tensors=1, output_tensors=1,
               payload_bytes=320, errors={}, recipe=recipe, policy=policy)
else:
    out = summary
if os.getenv('BREWERY_TEST_WRONG_RECIPE'):
    out['recipe'] = dict(recipe, sha256='b'*64)
print(json.dumps(out))
"#;

fn run(root: &Path, work: &Path, kind: &str, extra: &[(&str, &str)]) -> Output {
    run_with(root, work, kind, extra, |_| {})
}

fn run_with(
    root: &Path,
    work: &Path,
    kind: &str,
    extra: &[(&str, &str)],
    options_patch: impl FnOnce(&mut Value),
) -> Output {
    let mut recipe = json!({"schema":1,"name":"Fixture","catalog_sha256":"0".repeat(64),
        "architecture":"qwen4_exp","base":"v2","overrides":{(TENSOR):"ht43"}});
    let revision = format!("{:x}", Sha256::digest(serde_json::to_vec(&recipe).unwrap()));
    recipe["revision"] = json!(revision);
    let mut options = json!({"profile_recipe":recipe});
    let mut inputs = vec![json!({"kind":"hf","host":"fixture","path":root.join("source")})];
    if kind == "verify" {
        inputs.push(json!({"kind":"hgn","host":"fixture","path":root.join("convert/converted")}));
    } else {
        options["profile"] = json!("v2");
    }
    options_patch(&mut options);
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(work)
        .env_remove("BREWERY_EXPECTED_SOURCE_SHA256")
        .env_remove("BREWERY_CONVERTER_ADMISSION")
        .env("BREWERY_CUSTOM_CONVERTER", root.join("fixture-converter"))
        .env("BREWERY_CONVERTER", "/native-must-not-start")
        .envs(extra.iter().copied())
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
fn applies_the_same_attested_recipe_at_each_cpu_stage_without_a_runtime_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let fixture = root.join("fixture-converter");
    fs::write(&fixture, CONVERTER).unwrap();
    fs::set_permissions(&fixture, fs::Permissions::from_mode(0o700)).unwrap();
    fs::create_dir(root.join("source")).unwrap();
    fs::write(root.join("source/weights"), b"CPU fixture").unwrap();
    let mut recipe = None;
    for kind in ["inspect", "plan", "convert", "verify"] {
        let work = root.join(kind);
        fs::create_dir(&work).unwrap();
        let out = run(root, &work, kind, &[]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let report: Value = serde_json::from_slice(&out.stdout).unwrap();
        let selected = &report["custom_profile"];
        assert_eq!(selected["name"], "Fixture");
        assert_eq!(selected["release_qualified"], false);
        assert_eq!(report["release_qualified"], false);
        assert!(fs::read_dir(&work).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".profile-recipe-")));
        if let Some(previous) = &recipe {
            assert_eq!(previous, selected);
        }
        recipe = Some(selected.clone());
        if kind == "convert" {
            assert_eq!(report["inference_bundle"]["status"], "not_created");
            assert!(!work.join("converted/manifest.json").exists());
        }
        if kind == "verify" {
            assert_eq!(report["file_verification"]["status"], "passed");
        }
    }
    for (case, env, reason) in [
        (
            "wrong",
            ("BREWERY_TEST_WRONG_RECIPE", "1"),
            "did not attest the selected recipe",
        ),
        (
            "pinned",
            ("BREWERY_EXPECTED_SOURCE_SHA256", "pin"),
            "qualification workers",
        ),
        (
            "unconfigured",
            ("BREWERY_CUSTOM_CONVERTER", ""),
            "BREWERY_CUSTOM_CONVERTER",
        ),
    ] {
        let root = root.join(case);
        fs::create_dir_all(root.join("inspect")).unwrap();
        fs::create_dir(root.join("source")).unwrap();
        fs::write(root.join("source/weights"), b"fixture").unwrap();
        fs::copy(&fixture, root.join("fixture-converter")).unwrap();
        let out = run(&root, &root.join("inspect"), "inspect", &[env]);
        assert!(!out.status.success() && out.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(reason), "{stderr}");
    }
    for (case, change, reason) in [
        ("verify-gpu", ("decode_backend", json!("cuda")), "CPU-only"),
        (
            "verify-contradictory-profile",
            ("profile", json!("ht43")),
            "unsupported verify option: profile",
        ),
    ] {
        let work = root.join(case);
        fs::create_dir(&work).unwrap();
        let out = run_with(root, &work, "verify", &[], |opts| opts[change.0] = change.1);
        assert!(!out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(reason), "{stderr}");
        assert!(!work.join("converter-evidence").exists());
    }
    let work = root.join("altered-revision");
    fs::create_dir(&work).unwrap();
    let out = run_with(root, &work, "inspect", &[], |opts| {
        opts["profile_recipe"]["name"] = json!("Incorrect revision")
    });
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("revision does not match"));
    assert!(!work.join("converter-evidence").exists());
}
