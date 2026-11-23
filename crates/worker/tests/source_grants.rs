use brewery_worker::source_grants::{revalidate, validate};
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::symlink,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

fn fixture() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("brewery-grant-")
        .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
        .unwrap()
}

#[test]
fn grants_pin_the_exact_directory_on_the_selected_worker() {
    let fixture = fixture();
    let root = fixture.path().join("models");
    fs::create_dir(&root).unwrap();
    let grant = validate("local-eval", &root, &[]).unwrap();
    assert_eq!(grant.path, root);
    assert_eq!(grant.host, "local-eval");
    let grant = serde_json::from_value(serde_json::to_value(grant).unwrap()).unwrap();
    revalidate("local-eval", &grant, &[]).unwrap();
    assert!(revalidate("other-worker", &grant, &[]).is_err());
    fs::rename(&root, fixture.path().join("old-models")).unwrap();
    fs::create_dir(&root).unwrap();
    assert!(revalidate("local-eval", &grant, &[]).is_err());
}

#[test]
fn grants_reject_aliases_missing_directories_and_protected_overlaps() {
    let fixture = fixture();
    let root = fixture.path().join("models");
    let child = root.join("checkpoint");
    fs::create_dir_all(&child).unwrap();
    let file = root.join("weights.gguf");
    fs::write(&file, b"GGUF").unwrap();
    let link = fixture.path().join("link");
    symlink(&root, &link).unwrap();
    let protection_alias = fixture.path().join("app-config");
    symlink(&root, &protection_alias).unwrap();
    for bad in [
        link.clone(),
        link.join("checkpoint"),
        root.join("..").join("models"),
        root.join(".").join("checkpoint"),
        root.join("missing"),
        file,
        "/".into(),
        "/etc".into(),
        "/proc/self".into(),
        fixture.path().join(".ssh"),
    ] {
        assert!(validate("local-eval", &bad, &[]).is_err(), "{bad:?}");
    }
    assert!(validate("local-eval", &root, &[child]).is_err());
    assert!(validate("local-eval", &root, &[fixture.path().into()]).is_err());
    assert!(validate("local-eval", &root, &[protection_alias.join("new-state")]).is_err());
    assert!(validate("local-eval", &root, &[fixture.path().join("models-other")]).is_ok());
}

#[test]
fn local_worker_grants_authorize_only_direct_weight_inputs_and_recheck_on_launch() {
    let fixture = fixture();
    let root = fixture.path().join("models");
    fs::create_dir(&root).unwrap();
    let model = root.join("source.gguf");
    fs::write(&model, b"GGUF synthetic model input").unwrap();
    let grant = validate("local-eval", &root, &[]).unwrap();
    let config = fixture.path().join("worker.json");
    fs::write(
        &config,
        json!({
            "root":fixture.path().join("state"),
            "engine":"/bin/false",
            "engine_env":{"BREWERY_WEIGHT_EDIT":"/bin/false"},
            "disk_reserve_bytes":0,
            "source_grants":[grant],
            "source_grant_host":"local-eval",
        })
        .to_string(),
    )
    .unwrap();
    let input = |host: &str, path: &Path| json!({"host":host,"path":path,"kind":"gguf"});
    let check = |id: &str, kind: &str, host: &str, path: &Path| {
        let input = input(host, path);
        assert_eq!(
            rpc(
                &config,
                json!({"op":"prepare","id":id,"action":"start","job":{"kind":kind,"inputs":[input]}}),
            )["ok"]["state"],
            "queued"
        );
        rpc(
            &config,
            json!({"op":"resolve","id":id,"index":0,"input":input,"bundle":null}),
        )
    };
    assert!(
        rpc(
            &config,
            json!({"op":"describe","input":input("local-eval",&model)}),
        )["error"]
            .is_string()
    );
    assert!(
        rpc(
            &config,
            json!({"op":"read","root":root,
            "entry":{"path":"source.gguf","bytes":26,"sha256":"0".repeat(64)},"offset":0}),
        )["error"]
            .is_string()
    );
    let prefix = format!("{}/", root.display());
    assert_eq!(
        rpc(&config, json!({"op":"suggest_paths","prefix":prefix}))["ok"]["items"],
        json!([])
    );
    assert!(check("inspect", "inspect", "local-eval", &model)["error"].is_string());
    assert!(check("wrong-host", "weight_inspect", "other-worker", &model)["error"].is_string());
    let link = root.join("symlink.gguf");
    symlink(&model, &link).unwrap();
    assert!(check("symlink", "weight_inspect", "local-eval", &link)["error"].is_string());
    let private_model = root.join(".ssh/model.gguf");
    fs::create_dir(private_model.parent().unwrap()).unwrap();
    fs::write(&private_model, b"GGUF").unwrap();
    assert!(check("private", "weight_inspect", "local-eval", &private_model)["error"].is_string());
    assert_eq!(
        check("edit", "weight_inspect", "local-eval", &model)["ok"]["path"],
        json!(model)
    );
    fs::rename(&root, fixture.path().join("old-models")).unwrap();
    fs::create_dir(&root).unwrap();
    fs::write(&model, b"GGUF synthetic model input").unwrap();
    assert!(check("replacement", "weight_inspect", "local-eval", &model)["error"].is_string());
    assert_eq!(
        rpc(&config, json!({"op":"start","id":"edit"}))["ok"]["state"],
        "queued"
    );
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        let state = rpc(&config, json!({"op":"status","id":"edit"}));
        if state["ok"]["state"] == "failed" {
            assert_eq!(
                state["ok"]["error"],
                "Source input unavailable; check the model directory grant"
            );
            break;
        }
        assert!(Instant::now() < until, "{state}");
        thread::sleep(Duration::from_millis(20));
    }
}

fn rpc(config: &Path, request: Value) -> Value {
    let mut worker = Command::new(env!("CARGO_BIN_EXE_brewery-worker"))
        .arg("rpc")
        .env("BREWERY_WORKER_CONFIG", config)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    serde_json::to_writer(worker.stdin.take().unwrap(), &request).unwrap();
    let output = worker.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}
