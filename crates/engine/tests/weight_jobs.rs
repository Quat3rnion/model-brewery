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
    let cwd = tempfile::tempdir_in(root).unwrap().keep();
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-engine"))
        .current_dir(cwd)
        .env("BREWERY_WEIGHT_EDIT", root.join("edit"))
        .env("BREWERY_WEIGHT_TUNE", root.join("tune"))
        .env("BREWERY_WEIGHT_VISUAL", "1")
        .env("BREWERY_WEIGHT_HGN", if hgn { "1" } else { "0" })
        .env("BREWERY_WEIGHT_PACKED", if packed { "1" } else { "0" })
        .env_remove("BREWERY_EXPECTED_SOURCE_SHA256")
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
