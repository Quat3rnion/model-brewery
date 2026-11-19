use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output, Stdio},
};

fn input(kind: &str, path: &Path) -> Value {
    let bytes = fs::read(path).unwrap();
    let sha = format!("{:x}", Sha256::digest(&bytes));
    let name = path.file_name().unwrap().to_str().unwrap();
    let tree = format!(
        "{:x}",
        Sha256::digest(format!("{name}\0{}\0{sha}\n", bytes.len()))
    );
    json!({"kind":kind,"host":"fixture","path":path,"sha256":sha,"bundle_sha256":tree,
        "files":[{"path":name,"bytes":bytes.len(),"sha256":sha}]})
}

fn run(root: &Path, kind: &str, inputs: &[Value], options: Value) -> Output {
    run_with_hgn(root, kind, inputs, options, true)
}

fn run_with_hgn(root: &Path, kind: &str, inputs: &[Value], options: Value, hgn: bool) -> Output {
    run_gated(root, kind, inputs, options, hgn, false)
}

fn run_gated(
    root: &Path,
    kind: &str,
    inputs: &[Value],
    options: Value,
    hgn: bool,
    packed: bool,
) -> Output {
    run_configured(root, kind, inputs, options, hgn, packed, &[])
}

fn run_configured(
    root: &Path,
    kind: &str,
    inputs: &[Value],
    options: Value,
    hgn: bool,
    packed: bool,
    env: &[(&str, &str)],
) -> Output {
    let cwd = tempfile::tempdir_in(root).unwrap().keep();
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(cwd)
        .env("BREWERY_WEIGHT_EDIT", root.join("edit"))
        .env("BREWERY_WEIGHT_TUNE", root.join("tune"))
        .env("BREWERY_WEIGHT_VISUAL", "1")
        .env("BREWERY_WEIGHT_HGN", if hgn { "1" } else { "0" })
        .env("BREWERY_WEIGHT_PACKED", if packed { "1" } else { "0" })
        .env_remove("BREWERY_EXPECTED_SOURCE_SHA256")
        .env_remove("BREWERY_WEIGHT_SAFETENSORS_PUBLICATION")
        .env_remove("BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES")
        .env_remove("BREWERY_WEIGHT_SAFETENSORS_DISK_RESERVE_BYTES")
        .envs(env.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(
        child.stdin.take().unwrap(),
        "{}",
        json!({"kind":kind,"inputs":inputs,"options":options})
    )
    .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
#[ignore = "requires local NumPy and the pinned gguf-py checkout"]
fn real_safetensors_and_indexed_hf_apply_publish_complete_independent_outputs() {
    use brewery_worker::{tree_digest, Entry};
    let root = tempfile::tempdir().unwrap();
    python_helpers(root.path());
    let tensor = "layer.weight";
    let config_bytes = b"{\"model_type\":\"fixture\"}\n";
    let index_name = "model.safetensors.index.json";
    let shard_name = "model-00001-of-00002.safetensors";
    let neighbor_name = "model-00002-of-00002.safetensors";
    let encode = |name: &str, values: [f32; 4]| {
        let mut header = json!({name:{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})
            .to_string()
            .into_bytes();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(values.into_iter().flat_map(f32::to_le_bytes));
        bytes
    };
    for kind in ["safetensors", "hf"] {
        let original = encode(tensor, [1., 2., 3., 4.]);
        let neighbor = encode("neighbor.weight", [5., 6., 7., 8.]);
        let path = root.path().join(format!("fixture.{kind}"));
        let (source, original_path) = if kind == "hf" {
            fs::create_dir(&path).unwrap();
            fs::write(path.join("config.json"), config_bytes).unwrap();
            fs::write(
                path.join(index_name),
                json!({
                    "metadata":{"total_size":32},
                    "weight_map":{tensor:shard_name,"neighbor.weight":neighbor_name}
                })
                .to_string(),
            )
            .unwrap();
            fs::write(path.join(shard_name), &original).unwrap();
            fs::write(path.join(neighbor_name), &neighbor).unwrap();
            let files = ["config.json", shard_name, neighbor_name, index_name]
                .into_iter()
                .map(|name| {
                    let data = fs::read(path.join(name)).unwrap();
                    Entry {
                        path: name.into(),
                        bytes: data.len() as u64,
                        sha256: format!("{:x}", Sha256::digest(data)),
                    }
                })
                .collect::<Vec<_>>();
            let tree = tree_digest(&files);
            let bytes = files.iter().map(|i| i.bytes).sum::<u64>();
            (
                json!({"kind":"hf","host":"fixture","path":path,"sha256":tree,"bundle_sha256":tree,
                "bytes":bytes,"files":files}),
                path.join(shard_name),
            )
        } else {
            fs::write(&path, &original).unwrap();
            let mut source = input(kind, &path);
            source["bytes"] = json!(original.len());
            (source, path.clone())
        };
        let source_only = std::slice::from_ref(&source);
        let safe_env = &[
            ("BREWERY_WEIGHT_SAFETENSORS", "1"),
            ("BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES", "67108864"),
        ];
        let execute = |stage, inputs: &[Value], opts, env: &[(&str, &str)]| {
            run_configured(root.path(), stage, inputs, opts, false, false, env)
        };
        let success = |stage, inputs: &[Value], opts| {
            let out = execute(stage, inputs, opts, safe_env);
            assert!(
                out.status.success(),
                "{kind} {stage}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let v: Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(v["source_format"], kind);
            assert_eq!(v["source_sha256"], source["sha256"]);
            v
        };
        let inspect = json!({"tensor":tensor,"index":[0,1]});
        let inspected = success("weight_inspect", source_only, inspect.clone());
        assert_eq!(inspected["report"]["value"], 2.0);
        let denied = execute(
            "weight_inspect",
            source_only,
            inspect,
            &[("BREWERY_WEIGHT_SAFETENSORS", "1")],
        );
        assert!(String::from_utf8_lossy(&denied.stderr).contains("immutable publication"));
        let inventory = success(
            "weight_inspect",
            source_only,
            json!({"mode":"inventory","offset":0,"limit":2,"source_sha256":source["sha256"]}),
        );
        assert_eq!(
            inventory["report"]["tensors"].as_array().unwrap().len(),
            if kind == "hf" { 2 } else { 1 }
        );
        let tile = success(
            "weight_inspect",
            source_only,
            json!({"mode":"tile","tensor":tensor,"origin":[0,0],"shape":[2,2],"source_sha256":source["sha256"]}),
        );
        assert_eq!(tile["report"]["values"], json!([[1., 2.], [3., 4.]]));
        let proposal = success(
            "weight_propose",
            source_only,
            json!({"edits":[{"tensor":tensor,"index":[0,1],"op":"set","value":6.0}],"source_sha256":source["sha256"]}),
        );
        assert_eq!(proposal["proposal"]["edits"][0]["before"], 2.0);
        let cal = root.path().join("cal.json");
        let held = root.path().join("held.json");
        fs::write(&cal, br#"{"activations":[[0,1]],"targets":[[3,4]]}"#).unwrap();
        fs::write(&held, br#"{"activations":[[1,0]],"targets":[[1,3]]}"#).unwrap();
        let tuned = success(
            "weight_tune",
            &[
                source.clone(),
                input("weight-calibration", &cal),
                input("weight-calibration", &held),
            ],
            json!({"tensor":tensor,"indices":[[0,1]],"max_update":1.0,"source_sha256":source["sha256"]}),
        );
        assert_eq!(tuned["proposal"]["edits"][0]["after"], 3.0);
        let artifact = input(
            "weight-edit-proposal",
            Path::new(proposal["artifact"]["path"].as_str().unwrap()),
        );
        let approved =
            json!({"source_sha256":source["sha256"],"proposal_sha256":artifact["sha256"]});
        let apply_inputs = [source.clone(), artifact];
        let mut stale = approved.clone();
        stale["proposal_sha256"] = json!("f".repeat(64));
        let rejected = execute("weight_apply", &apply_inputs, stale, safe_env);
        assert!(!rejected.status.success() && rejected.stdout.is_empty());
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("approved source or proposal"));
        let no_space = execute(
            "weight_apply",
            &apply_inputs,
            approved.clone(),
            &[
                safe_env[0],
                safe_env[1],
                (
                    "BREWERY_WEIGHT_SAFETENSORS_DISK_RESERVE_BYTES",
                    "1000000000000000",
                ),
            ],
        );
        assert!(!no_space.status.success() && no_space.stdout.is_empty());
        assert!(String::from_utf8_lossy(&no_space.stderr).contains("insufficient disk"));
        let applied = success("weight_apply", &apply_inputs, approved);
        assert_eq!(applied["verified"], true);
        assert_eq!(applied["proposal_sha256"], apply_inputs[1]["sha256"]);
        assert_eq!(applied["inputs"][0]["files"], source["files"]);
        let output = Path::new(applied["artifact"]["path"].as_str().unwrap());
        assert_eq!(applied["artifact"]["kind"], kind);
        let leaf = if kind == "hf" {
            assert_eq!(output.file_name().unwrap(), "edited.hf");
            assert_eq!(applied["artifact"]["files"].as_array().unwrap().len(), 4);
            assert_eq!(
                applied["artifact"]["bundle_sha256"],
                applied["report"]["output_sha256"]
            );
            assert_eq!(fs::read(output.join(neighbor_name)).unwrap(), neighbor);
            for name in ["config.json", index_name] {
                assert_eq!(
                    fs::read(output.join(name)).unwrap(),
                    fs::read(path.join(name)).unwrap()
                );
            }
            output.join(shard_name)
        } else {
            assert_eq!(output.file_name().unwrap(), "edited.safetensors");
            output.into()
        };
        assert_eq!(fs::read(&leaf).unwrap(), encode(tensor, [1., 6., 3., 4.]));
        assert_eq!(fs::read(original_path).unwrap(), original);
    }
}

#[test]
fn packed_intent_requires_worker_opt_in() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source.gguf");
    fs::write(&source, b"GGUF").unwrap();
    let out = run(
        root.path(),
        "weight_inspect",
        &[input("gguf", &source)],
        json!({"tensor":"layer.weight","index":[0,0],"packed":true}),
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("packed weight editing is not enabled"));
}

#[test]
fn hgn_requires_its_own_opt_in_and_a_single_file() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source.hgn");
    fs::write(&source, b"HGN1").unwrap();
    let options = json!({"tensor":"layer.weight","index":[0,0]});
    let out = run_with_hgn(
        root.path(),
        "weight_inspect",
        &[input("hgn", &source)],
        options.clone(),
        false,
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("HGN weight editing is not enabled"));
    let dir = root.path().join("bundle");
    fs::create_dir(&dir).unwrap();
    let out = run(
        root.path(),
        "weight_inspect",
        &[json!({"kind":"hgn","host":"fixture","path":dir})],
        options,
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("bounded file limit"));
}

#[test]
fn rejects_unverified_source_before_helper() {
    let root = tempfile::tempdir().unwrap();
    let tool = root.path().join("edit");
    fs::write(&tool, "#!/bin/sh\nexit 91\n").unwrap();
    fs::set_permissions(tool, fs::Permissions::from_mode(0o700)).unwrap();
    let source = root.path().join("unverified.gguf");
    fs::write(&source, b"GGUF").unwrap();
    let out = run(
        root.path(),
        "weight_inspect",
        &[json!({"kind":"gguf","path":source,"host":"fixture"})],
        json!({"tensor":"layer.weight","index":[0,0]}),
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("verified single-file"));
}

fn python_helpers(root: &Path) {
    let gguf = std::env::var("BREWERY_TEST_GGUF_PY").unwrap();
    let tools = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools")
        .canonicalize()
        .unwrap();
    for name in ["edit", "tune"] {
        let path = root.join(name);
        fs::write(&path, format!("#!/bin/sh\nexport PYTHONPATH='{}'\nexec /usr/bin/python3 -B '{}/weight_{name}.py' \"$@\"\n",gguf,tools.display())).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    }
}

#[test]
#[ignore = "requires local NumPy and the pinned gguf-py checkout"]
fn real_packed_helpers_bind_format_and_approved_bytes_and_leave_neighbors_unchanged() {
    let root = tempfile::tempdir().unwrap();
    python_helpers(root.path());
    let tensor = "layer.weight";
    for (kind, dtype, store, width, data) in [
        (
            "gguf",
            "Q4_0",
            2u32,
            32u64,
            [&[0, 60][..], &[0x88; 16]].concat(),
        ),
        ("gguf", "Q8_0", 8, 32, [&[0, 60][..], &[0; 32]].concat()),
        (
            "hgn",
            "Q8G64",
            7,
            64,
            [&[0; 64][..], &[0, 60, 0, 0]].concat(),
        ),
    ] {
        let path = root.path().join(format!("{dtype}.{kind}"));
        let offset = if kind == "hgn" { 320 } else { 96 };
        let mut bytes = if kind == "gguf" {
            let mut bytes = b"GGUF\x03\0\0\0".to_vec();
            for n in [1u64, 0, tensor.len() as u64] {
                bytes.extend_from_slice(&n.to_le_bytes());
            }
            bytes.extend_from_slice(tensor.as_bytes());
            bytes.extend_from_slice(&2u32.to_le_bytes());
            bytes.extend_from_slice(&width.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&store.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.resize(offset, 0);
            bytes
        } else {
            let mut bytes = vec![0; offset];
            bytes[..8].copy_from_slice(b"HGN1\x02\0\0\0");
            bytes[40..44].copy_from_slice(b"test");
            bytes[104..104 + tensor.len()].copy_from_slice(tensor.as_bytes());
            for (i, n) in [
                (8, 1u64),
                (16, 104),
                (24, 320),
                (32, 388),
                (208, 1),
                (216, width),
                (240, 320),
                (248, 68),
            ] {
                bytes[i..i + 8].copy_from_slice(&n.to_le_bytes());
            }
            for (i, n) in [(200, store), (204, 2), (256, 0x3c00)] {
                bytes[i..i + 4].copy_from_slice(&n.to_le_bytes());
            }
            bytes
        };
        bytes.extend_from_slice(&data);
        fs::write(&path, &bytes).unwrap();
        let source = input(kind, &path);
        let execute = |stage: &str, inputs: &[Value], opts: Value| {
            run_gated(root.path(), stage, inputs, opts, true, true)
        };
        let success = |stage, inputs: &[Value], mut opts: Value| {
            opts["packed"] = json!(true);
            let out = execute(stage, inputs, opts);
            assert!(
                out.status.success(),
                "{dtype} {stage}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let result: Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(result["packed"], true);
            result
        };
        let selection = std::slice::from_ref(&source);
        let inventory = success(
            "weight_inspect",
            selection,
            json!({"mode":"inventory","offset":0,"limit":1}),
        );
        assert_eq!(inventory["report"]["tensors"][0]["editable"], true);
        let legacy = execute(
            "weight_inspect",
            selection,
            json!({"mode":"inventory","offset":0,"limit":1}),
        );
        let legacy: Value = serde_json::from_slice(&legacy.stdout).unwrap();
        assert_eq!(legacy["report"]["tensors"][0]["editable"], false);
        let tile = success(
            "weight_inspect",
            selection,
            json!({"mode":"tile","tensor":tensor,"origin":[0,0],"shape":[1,1],"source_sha256":source["sha256"]}),
        );
        assert_eq!(tile["report"]["dtype"], dtype);
        assert_eq!(tile["report"]["packed_steps"][0][0]["upper"], 1.0);
        let propose_options =
            json!({"edits":[{"tensor":tensor,"index":[0,0],"op":"set","value":1.0}]});
        let proposal = success("weight_propose", selection, propose_options.clone());
        assert_eq!(proposal["proposal"]["edits"][0]["dtype"], dtype);
        assert_eq!(proposal["proposal"]["edits"][0]["after"], 1.0);
        let calibration = root.path().join("packed-calibration.json");
        let held_out = root.path().join("packed-held-out.json");
        let mut cal = vec![0; width as usize];
        cal[0] = 1;
        let mut held = vec![0; width as usize];
        held[1] = 1;
        fs::write(
            &calibration,
            json!({"activations":[cal],"targets":[[1]]}).to_string(),
        )
        .unwrap();
        fs::write(
            &held_out,
            json!({"activations":[held],"targets":[[0]]}).to_string(),
        )
        .unwrap();
        let tuned = success(
            "weight_tune",
            &[
                source.clone(),
                input("weight-calibration", &calibration),
                input("weight-calibration", &held_out),
            ],
            json!({"tensor":tensor,"indices":[[0,0]],"max_update":1.0}),
        );
        assert_eq!(tuned["proposal"], proposal["proposal"]);
        assert_eq!(tuned["report"]["tune"]["calibration"]["after_mse"], 0.0);
        assert!(!execute("weight_propose", selection, propose_options)
            .status
            .success());
        let proposal_input = input(
            "weight-edit-proposal",
            Path::new(proposal["artifact"]["path"].as_str().unwrap()),
        );
        let inputs = [source.clone(), proposal_input.clone()];
        let mut options =
            json!({"source_sha256":source["sha256"],"proposal_sha256":proposal_input["sha256"]});
        let apply = success("weight_apply", &inputs, options.clone());
        assert_eq!(apply["verified"], true);
        assert_eq!(apply["proposal_sha256"], proposal_input["sha256"]);
        assert_eq!(apply["artifact"]["kind"], kind);
        let mut expected = bytes.clone();
        if kind == "hgn" {
            expected[offset] = 1;
            expected[256] = 1;
        } else {
            expected[offset + 2] = if dtype == "Q4_0" { 0x89 } else { 1 };
        }
        assert_eq!(
            fs::read(apply["artifact"]["path"].as_str().unwrap()).unwrap(),
            expected
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
        options["packed"] = json!(true);
        options["proposal_sha256"] = json!("0".repeat(64));
        let stale = execute("weight_apply", &inputs, options);
        assert!(!stale.status.success() && stale.stdout.is_empty());
        assert!(String::from_utf8_lossy(&stale.stderr).contains("approved source or proposal"));
        let wrong_kind = if kind == "hgn" { "gguf" } else { "hgn" };
        let wrong_path = root.path().join(format!("disguised.{wrong_kind}"));
        fs::write(&wrong_path, &bytes).unwrap();
        let disguised = input(wrong_kind, &wrong_path);
        let false_format = execute(
            "weight_inspect",
            &[disguised],
            json!({"packed":true,"tensor":tensor,"index":[0,0]}),
        );
        assert!(!false_format.status.success() && false_format.stdout.is_empty());
        assert!(String::from_utf8_lossy(&false_format.stderr).contains("magic"));
    }
}

#[test]
#[ignore = "requires local NumPy and the pinned gguf-py checkout"]
fn real_hgn_helpers_keep_format_across_all_four_weight_modes() {
    let root = tempfile::tempdir().unwrap();
    python_helpers(root.path());
    let path = root.path().join("source.hgn");
    let mut original = vec![0; 328];
    original[..8].copy_from_slice(b"HGN1\x02\0\0\0");
    for (offset, value) in [(8, 1u64), (16, 104), (24, 320), (32, 328)] {
        original[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    let identity = b"qwen3.8-flash-next-v2";
    original[40..40 + identity.len()].copy_from_slice(identity);
    let tensor = "layers.0.attn_q.weight";
    original[104..104 + tensor.len()].copy_from_slice(tensor.as_bytes());
    for (offset, value) in [(200, 2u32), (204, 2), (256, 0x0400_7e00)] {
        original[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [(208, 2u64), (216, 2), (240, 320), (248, 8)] {
        original[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    original[320..].copy_from_slice(&[0, 60, 0, 64, 0, 66, 0, 68]);
    fs::write(&path, &original).unwrap();
    let source = input("hgn", &path);
    let parsed = |kind, inputs: &[Value], options| {
        let out = run(root.path(), kind, inputs, options);
        assert!(
            out.status.success(),
            "{kind}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let report: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(report["source_format"], "hgn");
        report
    };
    let inspected = parsed(
        "weight_inspect",
        std::slice::from_ref(&source),
        json!({"tensor":tensor,"index":[0,1]}),
    );
    assert_eq!(inspected["report"]["value"], 2.0);
    let page = parsed(
        "weight_inspect",
        std::slice::from_ref(&source),
        json!({"mode":"inventory","offset":0,"limit":1}),
    );
    assert_eq!(page["report"]["tensors"][0]["editable"], true);
    let tile = parsed(
        "weight_inspect",
        std::slice::from_ref(&source),
        json!({"mode":"tile","tensor":tensor,"origin":[0,1],"shape":[2,1],"source_sha256":source["sha256"]}),
    );
    assert_eq!(tile["report"]["values"], json!([[2.0], [4.0]]));
    let proposed = parsed(
        "weight_propose",
        std::slice::from_ref(&source),
        json!({"edits":[{"tensor":tensor,"index":[0,0],"op":"set","value":2}]}),
    );
    assert_eq!(proposed["proposal"]["edits"][0]["after"], 2.0);
    let calibration = root.path().join("calibration.json");
    let held = root.path().join("held-out.json");
    fs::write(
        &calibration,
        br#"{"activations":[[1,0],[0,1]],"targets":[[2,3],[2,3]]}"#,
    )
    .unwrap();
    fs::write(
        &held,
        br#"{"activations":[[1,1],[2,-1]],"targets":[[3,7],[0,2]]}"#,
    )
    .unwrap();
    let tuned = parsed(
        "weight_tune",
        &[
            source.clone(),
            input("weight-calibration", &calibration),
            input("weight-calibration", &held),
        ],
        json!({"tensor":tensor,"indices":[[0,0],[1,1]],"max_update":1}),
    );
    let proposal = input(
        "weight-edit-proposal",
        Path::new(tuned["artifact"]["path"].as_str().unwrap()),
    );
    let applied = parsed(
        "weight_apply",
        &[source.clone(), proposal.clone()],
        json!({"source_sha256":source["sha256"],"proposal_sha256":proposal["sha256"]}),
    );
    assert_eq!(applied["verified"], true);
    assert_eq!(applied["artifact"]["kind"], "hgn");
    let edited = Path::new(applied["artifact"]["path"].as_str().unwrap());
    assert_eq!(edited.file_name().unwrap(), "edited.hgn");
    assert_eq!(
        &fs::read(edited).unwrap()[320..],
        &[0, 64, 0, 64, 0, 66, 0, 66]
    );
    assert_eq!(fs::read(path).unwrap(), original);
}

#[test]
#[ignore = "requires local NumPy and the pinned gguf-py checkout"]
fn real_helpers_inspect_propose_tune_and_apply() {
    let root = tempfile::tempdir().unwrap();
    python_helpers(root.path());
    let p = root.path().join("source.gguf");
    let mut bytes = b"GGUF\x03\0\0\0".to_vec();
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&12u64.to_le_bytes());
    bytes.extend_from_slice(b"layer.weight");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.resize(96, 0);
    for half in [0x3c00u16, 0x4000, 0x4200, 0x4400] {
        bytes.extend_from_slice(&half.to_le_bytes());
    }
    fs::write(&p, &bytes).unwrap();
    let source = input("gguf", &p);
    let parsed = |kind, inputs: &[Value], options| {
        let out = run(root.path(), kind, inputs, options);
        assert!(
            out.status.success(),
            "{kind}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice::<Value>(&out.stdout).unwrap()
    };
    let inspected = parsed(
        "weight_inspect",
        std::slice::from_ref(&source),
        json!({"tensor":"layer.weight","index":[0,0]}),
    );
    assert_eq!(inspected["report"]["value"], 1.0);
    let inventory = parsed(
        "weight_inspect",
        std::slice::from_ref(&source),
        json!({"mode":"inventory","offset":0,"limit":1}),
    );
    assert_eq!(
        inventory["report"],
        json!({"mode":"inventory","source_sha256":source["sha256"],
        "offset":0,"total":1,"next_offset":null,"tensors":[{"name":"layer.weight","shape":[2,2],"dtype":"F16","elements":4}]})
    );
    let tile = parsed(
        "weight_inspect",
        std::slice::from_ref(&source),
        json!({"mode":"tile","tensor":"layer.weight","origin":[0,1],"shape":[2,1],"source_sha256":source["sha256"]}),
    );
    assert_eq!(
        tile["report"],
        json!({"mode":"tile","source_sha256":source["sha256"],
        "tensor":"layer.weight","dtype":"F16","shape":[2,2],"origin":[0,1],"tile_shape":[2,1],"values":[[2.0],[4.0]]})
    );
    for options in [
        json!({"mode":"tile","tensor":"layer.weight","origin":[0,1],"shape":[2,2],"source_sha256":source["sha256"]}),
        json!({"mode":"tile","tensor":"layer.weight","origin":[0],"shape":[1,1],"source_sha256":source["sha256"]}),
        json!({"mode":"tile","tensor":"layer.weight","origin":[0,0],"shape":[33,1],"source_sha256":source["sha256"]}),
        json!({"mode":"tile","tensor":"layer.weight","origin":[0,0],"shape":[1,1],"source_sha256":"0".repeat(64)}),
        json!({"mode":"inventory","offset":0,"limit":129}),
    ] {
        let out = run(
            root.path(),
            "weight_inspect",
            std::slice::from_ref(&source),
            options,
        );
        assert!(!out.status.success() && out.stdout.is_empty());
    }
    let proposed = parsed(
        "weight_propose",
        std::slice::from_ref(&source),
        json!({"edits":[{"tensor":"layer.weight","index":[0,0],"op":"set","value":2}]}),
    );
    assert_eq!(proposed["proposal"]["edits"][0]["after"], 2.0);
    let cal = root.path().join("cal.json");
    let held = root.path().join("held.json");
    fs::write(
        &cal,
        br#"{"activations":[[1,0],[0,1]],"targets":[[2,3],[2,3]]}"#,
    )
    .unwrap();
    fs::write(
        &held,
        br#"{"activations":[[1,1],[2,-1]],"targets":[[3,7],[0,2]]}"#,
    )
    .unwrap();
    let tuned = parsed(
        "weight_tune",
        &[
            source.clone(),
            input("weight-calibration", &cal),
            input("weight-calibration", &held),
        ],
        json!({"tensor":"layer.weight","indices":[[0,0],[1,1]],"max_update":1}),
    );
    assert_eq!(tuned["report"]["tune"]["held_out"]["regression"], true);
    let proposal = input(
        "weight-edit-proposal",
        Path::new(tuned["artifact"]["path"].as_str().unwrap()),
    );
    let opts = json!({"source_sha256":source["sha256"],"proposal_sha256":proposal["sha256"]});
    let applied = parsed("weight_apply", &[source.clone(), proposal.clone()], opts);
    assert_eq!(applied["verified"], true);
    assert_eq!(applied["report"]["edit_count"], 2);
    assert_eq!(
        applied["artifact"]["sha256"],
        applied["report"]["output_sha256"]
    );
    assert_eq!(
        &fs::read(applied["artifact"]["path"].as_str().unwrap()).unwrap()[96..],
        &[0, 64, 0, 64, 0, 66, 0, 66]
    );
    assert_eq!(fs::read(p).unwrap(), bytes);
    let out = run(
        root.path(),
        "weight_apply",
        &[source, proposal],
        json!({"source_sha256":"0".repeat(64),"proposal_sha256":tuned["artifact"]["sha256"]}),
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("approved source or proposal"));
}
