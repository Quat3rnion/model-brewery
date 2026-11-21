use super::*;
use sha2::{Digest, Sha256};

fn fixture() -> (Fixture, PathBuf, PathBuf, PathBuf, Vec<u8>) {
    let bin = PathBuf::from(std::env::var_os("BREWERY_TEST_BIN_DIR").unwrap())
        .canonicalize()
        .unwrap();
    let helper = PathBuf::from(std::env::var_os("BREWERY_TEST_EXL3_HELPER").unwrap())
        .canonicalize()
        .unwrap();
    let app = Fixture::new();
    let sources = app.path("source");
    fs::create_dir(&sources).unwrap();
    let path = sources.join("model.safetensors");
    let tensor = "model.proj.weight";
    let raw =
        json!({tensor:{"dtype":"F32","shape":[128,128],"data_offsets":[0,65536]}}).to_string();
    let mut header = raw.into_bytes();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    // Non-symmetric [out,in] source with a distinct zero output row.
    for c in 0..128 {
        for r in 0..128 {
            let v = if c == 11 {
                0.
            } else {
                ((r * 13 + c * 7 + r * c * 11) % 19 - 9) as f32 / 8.
            };
            bytes.extend(v.to_le_bytes());
        }
    }
    fs::write(&path, &bytes).unwrap();
    let worker_config = app.path("worker.json");
    fs::write(&worker_config, json!({
        "root":app.path("worker"), "source_roots":[sources], "disk_reserve_bytes":0,
        "engine":bin.join("brewery-engine"),
        "engine_env":{
            "BREWERY_WEIGHT_SAFETENSORS":"1", "BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES":"4194304",
            "BREWERY_EXL3_CHECKPOINT":helper
        }
    }).to_string()).unwrap();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"worker":bin.join("brewery-worker"),"config":worker_config,
                "capabilities":{"exl3_layer":true}}
        }})
        .to_string(),
    )
    .unwrap();
    (app, bin, helper, path, bytes)
}

fn job(app: &Fixture, path: &std::path::Path, bytes: &[u8]) -> Value {
    let model = app.register("alpha", "safetensors", path.to_str().unwrap());
    json!({"kind":"exl3_layer","input_ids":[model],"host":"alpha",
        "options":{"tensor":"model.proj.weight","shape":[128,128],
            "source_sha256":format!("{:x}", Sha256::digest(bytes)),
            "codebook":"mul1","twice_bits":7,"backend":"cpu"}})
}

#[test]
#[ignore = "requires real local worker/engine and a CPU native EXL3 checkpoint CLI"]
fn managed_exl3_native_round_trip() {
    let (mut app, bin, helper, path, bytes) = fixture();
    app.launch_with(&[("BREWERY_DISPATCH", bin.join("brewery-dispatch"))]);
    let start = job(&app, &path, &bytes);
    let digest = start["options"]["source_sha256"].clone();
    for (field, value) in [
        ("twice_bits", json!(9)),
        ("shape", json!([128, 256])),
        ("backend", json!("auto")),
        ("source_sha256", json!("bad hash")),
        ("output_path", json!(path)),
    ] {
        let mut invalid = start.clone();
        invalid["options"][field] = value;
        app.json("POST", "/api/jobs", Some(invalid), 400);
    }
    let mut wrong_host = start.clone();
    wrong_host["host"] = json!("auto");
    app.json("POST", "/api/jobs", Some(wrong_host), 400);

    let terminal = |job: &str| {
        app.await_job(job, |row| {
            matches!(row["state"].as_str(), Some("failed" | "succeeded"))
        })
    };
    for (field, value) in [
        ("source_sha256", json!("0".repeat(64))),
        ("backend", json!("cuda")),
        ("tensor", json!("absent.weight")),
    ] {
        let mut invalid = start.clone();
        invalid["options"][field] = value;
        let id = app.enqueue(invalid);
        let failed = terminal(&id);
        assert_eq!(failed["state"], "failed", "{failed}");
        assert!(failed.get("result").is_none());
        assert!(
            !app.path(&format!("worker/jobs/{id}/work/exl3-layer.safetensors"))
                .exists()
        );
    }
    let id = app.enqueue(start);
    let done = terminal(&id);
    assert_eq!(done["state"], "succeeded", "{done}");
    let result = &done["result"];
    assert_eq!(result["report"]["schema"], "brewery.exl3-layer.v1");
    assert_eq!(result["report"]["backend"], "cpu");
    assert_eq!(result["report"]["source_sha256"], digest);
    assert_eq!(result["artifact"]["kind"], "exl3-layer");
    assert!(
        result["report"]["relative_l2_error"].as_f64().unwrap() < 0.4,
        "{done}"
    );
    assert_eq!(fs::read(&path).unwrap(), bytes);
    let packed = fs::read(result["artifact"]["path"].as_str().unwrap()).unwrap();
    assert_eq!(
        result["artifact"]["sha256"],
        format!("{:x}", Sha256::digest(&packed))
    );
    let x = app.path("test-x.f32");
    let y = app.path("test-y.f32");
    let mut one_hot = [0; 128 * 4];
    one_hot[17 * 4..18 * 4].copy_from_slice(&1.0f32.to_le_bytes());
    fs::write(&x, one_hot).unwrap();
    assert!(
        Command::new(helper)
            .args([
                "forward",
                "cpu",
                result["artifact"]["path"].as_str().unwrap(),
                "layer"
            ])
            .arg(&x)
            .arg(&y)
            .status()
            .unwrap()
            .success()
    );
    let y = fs::read(y).unwrap();
    assert_eq!(y.len(), 128 * 4);
    let mut error = 0.0;
    let mut energy = 0.0;
    for (c, chunk) in y.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes(chunk.try_into().unwrap());
        let pos =
            8 + (u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize) + (c * 128 + 17) * 4;
        let want = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        if c == 11 {
            assert_eq!(got, 0.);
        }
        error += (got - want).powi(2);
        energy += want.powi(2);
    }
    assert!((error / energy).sqrt() < 0.4);
    let models = app.json("GET", "/api/state", None, 200);
    assert_eq!(models["models"].as_array().unwrap().len(), 1);
    app.stop();
    app.launch_with(&[("BREWERY_DISPATCH", bin.join("brewery-dispatch"))]);
    assert_eq!(
        app.json("GET", &format!("/api/jobs/{id}"), None, 200)["result"],
        *result
    );
}

fn dispatcher(app: &Fixture, bin: &std::path::Path, hooks: &str) {
    script(
        &app.path("dispatch"),
        &format!(
            "root='{}'\ndispatch='{}'\n{hooks}\nexec \"$dispatch\" \"$@\"",
            app.root.path().display(),
            bin.join("brewery-dispatch").display()
        ),
    );
}

fn script(path: &std::path::Path, body: &str) {
    fs::write(path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

fn worker_status(app: &Fixture, bin: &std::path::Path, id: &str) -> Value {
    let output = Command::new(bin.join("brewery-dispatch"))
        .env("BREWERY_DISPATCH_CONFIG", app.path("dispatch.json"))
        .args(["status", "alpha", id])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
#[ignore = "requires real local worker/engine and a CPU native EXL3 checkpoint CLI"]
fn managed_exl3_rejects_preexisting_work_without_touching_it() {
    let (mut app, bin, _, source, bytes) = fixture();
    dispatcher(
        &app,
        &bin,
        r#"
if [ "$1" = start ]; then
    work="$root/worker/jobs/$3/work"
    if [ ! -e "$root/preexisting-created" ]; then
        mkdir -m 700 -p "$root/worker"
        mkdir -p "$work/.exl3-layer-existing"
        printf existing-layer > "$work/exl3-layer.safetensors"
        printf existing-scratch > "$work/.exl3-layer-existing/keep"
        touch "$root/preexisting-created"
    fi
fi
"#,
    );
    app.launch();
    let id = app.enqueue(job(&app, &source, &bytes));
    let failed = app.await_job(&id, |row| {
        matches!(
            row["state"].as_str(),
            Some("failed" | "succeeded" | "cancelled")
        )
    });
    assert_eq!(failed["state"], "failed", "{failed}");
    assert!(failed.get("result").is_none());
    let work = app.path(&format!("worker/jobs/{id}/work"));
    let unchanged = || {
        assert_eq!(
            fs::read(work.join("exl3-layer.safetensors"))
                .ok()
                .as_deref(),
            Some(b"existing-layer".as_slice()),
            "rejected pre-existing artifact changed"
        );
        assert_eq!(
            fs::read(work.join(".exl3-layer-existing/keep")).unwrap(),
            b"existing-scratch"
        );
        assert_eq!(fs::read(&source).unwrap(), bytes);
    };
    unchanged();
    app.stop();
    app.launch();
    assert_eq!(app.detail(&id)["state"], "failed");
    unchanged();
}

#[test]
#[ignore = "requires real local worker/engine and a CPU native EXL3 checkpoint CLI"]
fn managed_exl3_worker_success_wins_late_cancel_after_controller_restart() {
    let (mut app, bin, _, source, bytes) = fixture();
    // Hide status replies only from the controller. Direct worker reads still
    // observe the real terminal state before submitting HTTP cancellation.
    dispatcher(
        &app,
        &bin,
        r#"
if [ "$1" = status ] || { [ "$1" = cancel ] && [ ! -e "$root/after-restart" ]; }; then
    printf '{"state":"running","progress":"test gate"}\n'
    exit 0
fi
"#,
    );
    app.launch();
    let id = app.enqueue(job(&app, &source, &bytes));
    let result = wait_for(|| {
        let status = worker_status(&app, &bin, &id);
        (status["state"] == "succeeded").then(|| status["result"].clone())
    });
    let path = result["artifact"]["path"].as_str().unwrap();
    let artifact = fs::read(path).unwrap();
    assert_eq!(app.cancel(&id)["cancel_requested"], true);
    assert_eq!(app.detail(&id)["state"], "running");
    app.stop();
    app.touch("after-restart");
    app.launch();
    let terminal = app.await_job(&id, |r| {
        matches!(
            r["state"].as_str(),
            Some("succeeded" | "cancelled" | "failed")
        )
    });
    assert_eq!(terminal["state"], "succeeded", "{terminal}");
    assert_eq!(terminal["result"], result);
    assert_eq!(worker_status(&app, &bin, &id)["state"], "succeeded");
    assert_eq!(fs::read(path).unwrap(), artifact);
    assert_eq!(fs::read(source).unwrap(), bytes);
    app.stop();
    app.launch();
    assert_eq!(app.detail(&id)["result"], result);
}

#[test]
#[ignore = "requires real local worker/engine and a CPU native EXL3 checkpoint CLI"]
fn managed_exl3_cancel_before_worker_commit_discards_owned_outputs() {
    let (mut app, bin, _, source, bytes) = fixture();
    let wrapper = app.path("blocked-engine");
    script(
        &wrapper,
        &format!(
            r#"
'{}'
mkdir .exl3-layer-fixture
printf partial > .exl3-layer-fixture/partial
touch ../before-commit
while :; do sleep 1; done
"#,
            bin.join("brewery-engine").display()
        ),
    );
    let worker_path = app.path("worker.json");
    let mut cfg: Value = serde_json::from_slice(&fs::read(&worker_path).unwrap()).unwrap();
    cfg["engine"] = json!(wrapper);
    fs::write(worker_path, cfg.to_string()).unwrap();
    app.launch_with(&[("BREWERY_DISPATCH", bin.join("brewery-dispatch"))]);
    let id = app.enqueue(job(&app, &source, &bytes));
    let base = app.path(&format!("worker/jobs/{id}"));
    wait_for(|| base.join("before-commit").is_file().then_some(()));
    assert!(base.join("work/exl3-layer.safetensors").is_file());
    assert_eq!(app.cancel(&id)["cancel_requested"], true);
    let terminal = app.await_job(&id, |r| {
        matches!(
            r["state"].as_str(),
            Some("cancelled" | "succeeded" | "failed")
        )
    });
    assert_eq!(terminal["state"], "cancelled", "{terminal}");
    assert!(terminal.get("result").is_none());
    let clean = || {
        assert!(!base.join("work").exists());
        assert_eq!(fs::read(&source).unwrap(), bytes);
    };
    clean();
    app.stop();
    app.launch_with(&[("BREWERY_DISPATCH", bin.join("brewery-dispatch"))]);
    let after = app.detail(&id);
    assert_eq!(after["state"], "cancelled");
    assert!(after.get("result").is_none());
    clean();
}
