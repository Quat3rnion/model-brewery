use base64::{Engine, engine::general_purpose::STANDARD};
use brewery_worker::{Bundle, CHUNK, Description, Plan, Rpc};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

struct Rig {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    dispatch: PathBuf,
    b: PathBuf,
    src: PathBuf,
}
impl Rig {
    fn new() -> Self {
        let tmp = tempfile::Builder::new()
            .prefix("brewery-worker-")
            .tempdir_in("/arboretum")
            .unwrap();
        let root = tmp.path().to_path_buf();
        let engine = root.join("engine");
        fs::write(
            &engine,
            r#"#!/bin/sh
set -eu
cat > input.json
if grep -q '"kind":"cancel-test"' input.json; then
  setsid sh -c 'trap "" TERM; exec sleep 60' &
  echo "$!" > descendant
  wait
elif grep -q '"kind":"failure-test"' input.json; then
  printf 'CUDA error: out of memory (secret-token=fixture-private-token)\n' >&2
  exit 19
elif grep -q '"kind":"progress-test"' input.json; then
  printf 'brewery-engine: tracing reference\n' >&2
  printf 'prompt: fixture-private-token\n' >&2
  touch progress-ready
  while [ ! -f progress-done ]; do sleep .05; done
  printf '{"completed":true}\n'
else
  mkdir only-once
  sleep .2
  printf '{"completed":true}\n'
fi
"#,
        )
        .unwrap();
        fs::set_permissions(&engine, fs::Permissions::from_mode(0o700)).unwrap();
        let src = root.join("source");
        fs::create_dir(&src).unwrap();
        let mut hosts = serde_json::Map::new();
        let mut b = PathBuf::new();
        for name in ["a", "b"] {
            let config = root.join(format!("{name}.json"));
            fs::write(
                &config,
                json!({"root":root.join(name),"source_roots":[src],
                "engine":engine,"disk_reserve_bytes":0})
                .to_string(),
            )
            .unwrap();
            hosts.insert(
                name.into(),
                json!({"local":true,"worker":env!("CARGO_BIN_EXE_brewery-worker"),"config":config}),
            );
            if name == "b" {
                b = config;
            }
        }
        let dispatch = root.join("dispatch.json");
        fs::write(&dispatch, json!({"hosts":hosts}).to_string()).unwrap();
        Self {
            _tmp: tmp,
            root,
            dispatch,
            b,
            src,
        }
    }
    fn dispatch(&self, action: &str, id: &str, body: Option<&Value>) -> Value {
        let mut c = Command::new(env!("CARGO_BIN_EXE_brewery-dispatch"));
        c.env("BREWERY_DISPATCH_CONFIG", &self.dispatch)
            .args([action, "b", id]);
        output(c, body)
    }
    fn rpc(&self, r: Rpc) -> Value {
        let v = self.rpc_raw(&serde_json::to_value(r).unwrap());
        assert!(v.get("ok").is_some(), "{v}");
        v["ok"].clone()
    }
    fn rpc_raw(&self, request: &Value) -> Value {
        let mut c = Command::new(env!("CARGO_BIN_EXE_brewery-worker"));
        c.env("BREWERY_WORKER_CONFIG", &self.b).arg("rpc");
        output(c, Some(request))
    }
    fn suggest(&self, prefix: impl serde::Serialize) -> Value {
        let mut c = Command::new(env!("CARGO_BIN_EXE_brewery-dispatch"));
        c.env("BREWERY_DISPATCH_CONFIG", &self.dispatch)
            .args(["suggest-paths", "b"]);
        output(c, Some(&json!({"prefix":prefix})))
    }
    fn wait(&self, id: &str, expected: &str) -> Value {
        let until = Instant::now() + Duration::from_secs(8);
        loop {
            let r = self.dispatch("status", id, None);
            if r["state"] == expected {
                return r;
            }
            assert!(Instant::now() < until, "{r}");
            thread::sleep(Duration::from_millis(50));
        }
    }
}

#[test]
fn pinned_hf_resolution_passes_expected_manifest_without_reading_model_payloads() {
    let t = Rig::new();
    let (_, _) = pinned_fixture(&t);
    // Same-size substitution: this is not proof of integrity. The worker
    // hands off the expected manifest, and the engine must verify the bytes.
    fs::write(t.src.join("model.safetensors"), b"wrong").unwrap();
    use std::os::{
        fd::{AsRawFd, FromRawFd},
        unix::ffi::OsStrExt,
    };
    let fd = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
    assert!(fd >= 0);
    let watch = unsafe { fs::File::from_raw_fd(fd) };
    for name in ["config.json", "model.safetensors"] {
        let path = std::ffi::CString::new(t.src.join(name).as_os_str().as_bytes()).unwrap();
        assert!(
            unsafe { libc::inotify_add_watch(watch.as_raw_fd(), path.as_ptr(), libc::IN_ACCESS) }
                >= 0
        );
    }

    let actual = resolve_input(&t, "pinned", "start", "inspect", 0, pinned_input(&t, "hf"));
    assert_eq!(
        actual["ok"]["integrity_basis"], "pinned_publication",
        "{actual}"
    );
    assert_eq!(actual["ok"]["bundle_sha256"], PINNED_FIXTURE_TREE);
    assert_eq!(actual["ok"]["bytes"], 10);
    assert_eq!(
        actual["ok"]["files"],
        json!([
            {"path":"config.json","bytes":5,"sha256":PINNED_CONFIG_HASH},
            {"path":"model.safetensors","bytes":5,"sha256":PINNED_WEIGHT_HASH},
        ])
    );
    let mut events = [0_u8; 4096];
    use std::io::Read as _;
    assert_eq!(
        (&watch).read(&mut events).unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn pinned_hf_resolution_rejects_tampered_and_malformed_receipts() {
    let t = Rig::new();
    let (path, original) = pinned_fixture(&t);
    let mut digest = original.clone();
    digest["files"]["config.json"]["sha256"] = json!(PINNED_WEIGHT_HASH);
    let json = original.to_string();
    let duplicates = json.replacen(
        "\"config.json\":",
        &format!(
            "\"config.json\":{},\"config.json\":",
            original["files"]["config.json"]
        ),
        1,
    );
    let mut receipts = vec![
        digest.to_string(),
        duplicates,
        "{".into(),
        format!("{json}{}", " ".repeat(128 * 1024)),
    ];
    for (field, value) in [
        ("schema", json!("model-brewery.source-publication.v2")),
        ("repo_id", json!("other/model")),
        ("model_dir", json!(t.root)),
        (
            "revision",
            json!("abcdef0123456789abcdef0123456789abcdef01"),
        ),
        ("weight_bytes", json!(100)),
    ] {
        let mut receipt = original.clone();
        receipt[field] = value;
        receipts.push(receipt.to_string());
    }
    for (index, receipt) in receipts.into_iter().enumerate() {
        fs::write(&path, receipt).unwrap();
        let actual = resolve_input(
            &t,
            &format!("receipt-{index}"),
            "start",
            "plan",
            0,
            pinned_input(&t, "hf"),
        );
        assert!(
            actual["error"]
                .as_str()
                .is_some_and(|e| e.contains("Pinned publication manifest unavailable")),
            "{actual}"
        );
    }
    fs::remove_file(&path).unwrap();
    let target = t.root.join("valid.json");
    fs::write(&target, json).unwrap();
    symlink(target, path).unwrap();
    let actual = resolve_input(
        &t,
        "receipt-link",
        "start",
        "plan",
        0,
        pinned_input(&t, "hf"),
    );
    assert!(actual["error"].is_string(), "{actual}");
}

#[test]
fn pinned_hf_resolution_requires_exact_safe_inventory_and_source_root() {
    let t = Rig::new();
    pinned_fixture(&t);
    let model = t.src.join("model.safetensors");
    let extra = t.src.join("extra");
    fs::write(&extra, []).unwrap();
    let resolve = |id| resolve_input(&t, id, "start", "verify", 0, pinned_input(&t, "hf"));
    assert!(resolve("extra")["error"].is_string());
    fs::remove_file(extra).unwrap();
    fs::remove_file(&model).unwrap();
    assert!(resolve("missing")["error"].is_string());
    fs::write(&model, b"too long").unwrap();
    assert!(resolve("wrong-size")["error"].is_string());
    fs::remove_file(&model).unwrap();
    fs::create_dir(&model).unwrap();
    assert!(resolve("directory")["error"].is_string());
    fs::remove_dir(&model).unwrap();
    let target = t.root.join("weights");
    fs::write(&target, b"other").unwrap();
    symlink(&target, &model).unwrap();
    assert!(resolve("symlink")["error"].is_string());
    fs::remove_file(&model).unwrap();
    fs::hard_link(&target, &model).unwrap();
    assert!(resolve("hardlink")["error"].is_string());
    fs::remove_file(&model).unwrap();
    fs::write(&model, b"other").unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["source_roots"] = json!([]);
    fs::write(&t.b, config.to_string()).unwrap();
    assert!(resolve("outside")["error"].is_string());
}

#[test]
fn pinned_hf_resolution_stays_scoped_and_strips_client_markers() {
    let t = Rig::new();
    let (receipt, _) = pinned_fixture(&t);
    fs::write(receipt, b"not JSON").unwrap();
    fs::write(t.src.join("model.safetensors"), b"wrong").unwrap();
    let input = pinned_input(&t, "hf");
    let expect_hashed = |resolved: &Value| {
        assert!(resolved["bundle_sha256"].is_string(), "{resolved}");
        assert_ne!(resolved["bundle_sha256"], PINNED_FIXTURE_TREE, "{resolved}");
        assert!(resolved.get("integrity_basis").is_none(), "{resolved}");
    };
    for (id, job, index, input) in [
        ("later-input", "inspect", 1, input.clone()),
        ("wrong-kind", "inspect", 0, pinned_input(&t, "hgn")),
        ("wrong-job", "generate", 0, input.clone()),
        (
            "other-path",
            "inspect",
            0,
            json!({"path":t.src.join("config.json"),"host":"b","kind":"hf","integrity_basis":"pinned_publication"}),
        ),
    ] {
        let result = resolve_input(&t, id, "start", job, index, input);
        expect_hashed(&result["ok"]);
    }
    t.rpc_raw(
        &json!({"op":"prepare","id":"path-substitution","action":"start",
        "job":{"kind":"inspect","inputs":[input]}}),
    );
    let mut alias = input.clone();
    alias["path"] = json!(format!("{}/", t.src.display()));
    let reply =
        t.rpc_raw(&json!({"op":"resolve","id":"path-substitution","index":0,"input":alias}));
    assert!(reply["error"].is_string(), "{reply}");
    for action in ["transfer", "preview_transfer"] {
        let job = json!({"kind":"inspect","inputs":[input]});
        let reply = t.dispatch(action, action, Some(&job));
        assert_eq!(reply["state"], "succeeded", "{reply}");
        expect_hashed(&reply["result"]["inputs"][0]);
    }
}

const PINNED_CONFIG_HASH: &str = "a7937b64b8caa58f03721bb6bacf5c78cb235febe0e70b1b84cd99541461a08e";
const PINNED_WEIGHT_HASH: &str = "d9298a10d1b0735837dc4bd85dac641b0f3cef27a47e5d53a54f2f3f5b2fcffa";
const PINNED_FIXTURE_TREE: &str =
    "14573baa9ec70c91e9ffe211207fa9a4df919b628cc7da36f928ecfaa23023ac";
const PINNED_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn pinned_fixture(t: &Rig) -> (PathBuf, Value) {
    let receipt = t.root.join("publication.json");
    let data = json!({
        "schema":"model-brewery.source-publication.v1",
        "model_dir":t.src,
        "revision":PINNED_REVISION,
        "repo_id":"publisher/model",
        "shard_count":1,
        "weight_bytes":5,
        "files":{
            "config.json":{"size":5,"sha256":PINNED_CONFIG_HASH},
            "model.safetensors":{"size":5,"sha256":PINNED_WEIGHT_HASH}
        }
    });
    fs::write(&receipt, data.to_string()).unwrap();
    fs::write(t.src.join("config.json"), b"first").unwrap();
    fs::write(t.src.join("model.safetensors"), b"other").unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["pinned_publication"] = json!({"source":t.src,"receipt":receipt,
        "repo_id":"publisher/model","revision":PINNED_REVISION,"bundle_sha256":PINNED_FIXTURE_TREE});
    fs::write(&t.b, config.to_string()).unwrap();
    (receipt, data)
}

fn pinned_input(t: &Rig, kind: &str) -> Value {
    json!({"path":t.src,"host":"b","kind":kind,"integrity_basis":"forged",
        "bundle_sha256":"not-a-digest","files":[]})
}

fn resolve_input(t: &Rig, id: &str, action: &str, kind: &str, index: usize, input: Value) -> Value {
    let inputs = vec![input.clone(); index + 1];
    let prepared = t.rpc_raw(&json!({"op":"prepare","id":id,"action":action,
        "job":{"kind":kind,"inputs":inputs}}));
    if prepared["error"].is_string() {
        return prepared;
    }
    t.rpc_raw(&json!({"op":"resolve","id":id,"index":index,"input":input}))
}

#[test]
fn rocmfpx_admission_requires_the_current_revision_and_separate_pinned_tool() {
    let t = Rig::new();
    let mut opts = json!({
        "output_format":"gguf", "gguf_type":"Q4_0_ROCMFP4",
        "encoder_family":"rocmfpx", "encoder_revision":brewery_worker::ROCMFPX_REVISION
    });
    let prepare = |id, kind, options: &Value| {
        t.rpc_raw(&json!({"op":"prepare","id":id,"action":"start","job":{
            "kind":kind,"inputs":[{"host":"b","path":t.src.join("source.gguf"),"kind":"gguf"}],
            "options":options
        }}))["ok"]
            .clone()
    };
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["engine_env"] = json!({"BREWERY_GGUF_QUANTIZER":"/usr/bin/true"});
    fs::write(&t.b, config.to_string()).unwrap();
    assert_eq!(prepare("stock-only", "inspect", &opts)["state"], "failed");
    config["engine_env"]["BREWERY_GGUF_ROCMFPX_QUANTIZER"] = json!("/usr/bin/true");
    config["engine_env"]["BREWERY_GGUF_ROCMFPX_SHA256"] = json!("a".repeat(64));
    fs::write(&t.b, config.to_string()).unwrap();
    assert_eq!(prepare("same-path", "inspect", &opts)["state"], "failed");
    config["engine_env"]["BREWERY_GGUF_ROCMFPX_QUANTIZER"] = json!("/usr/bin/false");
    fs::write(&t.b, config.to_string()).unwrap();
    assert_eq!(
        prepare("wrong-role", "weight_inspect", &opts)["state"],
        "failed"
    );
    assert_eq!(prepare("fork-enabled", "inspect", &opts)["state"], "queued");
    opts["encoder_revision"] = json!("stale");
    assert_eq!(
        prepare("wrong-revision", "inspect", &opts)["state"],
        "failed"
    );
}

#[test]
fn standalone_gguf_admission_requires_config_and_direct_pinned_assets() {
    let t = Rig::new();
    let weights = t.src.join("weights.safetensors");
    fs::write(&weights, b"weights").unwrap();
    let metadata = t.src.join("assets");
    fs::create_dir(&metadata).unwrap();
    fs::write(metadata.join("config.json"), b"{}").unwrap();
    let inputs = vec![
        json!({"host":"b","path":weights,"kind":"safetensors"}),
        json!({"host":"b","path":metadata,"kind":"assets"}),
    ];
    let prepare = |id, inputs: &[Value]| {
        t.rpc_raw(&json!({"op":"prepare","id":id,"action":"start","job":{
            "kind":"inspect","inputs":inputs,"options":{"output_format":"gguf","gguf_type":"Q8_0"}
        }}))["ok"]
            .clone()
    };
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["engine_env"] = json!({"BREWERY_GGUF_SAFETENSORS":"1"});
    fs::write(&t.b, config.to_string()).unwrap();
    assert_eq!(
        prepare("flag-only", &inputs)["error"],
        "standalone SafeTensors GGUF is not configured on this worker"
    );
    config["engine_env"]["BREWERY_GGUF_QUANTIZER"] = json!("/usr/bin/false");
    config["engine_env"]["BREWERY_GGUF_HF_EXPORTER"] = json!("/usr/bin/false");
    fs::write(&t.b, config.to_string()).unwrap();
    let mut remote = inputs.clone();
    remote[1]["host"] = json!("a");
    assert_eq!(prepare("remote-assets", &remote)["state"], "failed");
    assert_eq!(prepare("direct-assets", &inputs)["state"], "queued");
    let reply =
        t.rpc_raw(&json!({"op":"resolve","id":"direct-assets","index":1,"input":inputs[1]}));
    let pinned = &reply["ok"];
    assert_eq!(pinned["source_host"], "b");
    assert_eq!(pinned["path"], metadata.to_str().unwrap());
    assert_eq!(pinned["bundle_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(pinned["files"][0]["path"], "config.json");
}

#[test]
fn requested_disk_reserve_denies_launch_before_engine_work() {
    let t = Rig::new();
    let id = "disk-reserve";
    t.dispatch(
        "start",
        id,
        Some(&json!({
            "kind":"inspect","inputs":[],"resources":{"disk_reserve_bytes":1_u64 << 50}
        })),
    );
    let result = t.wait(id, "failed");
    assert!(result["error"].as_str().unwrap().contains("disk reserve"));
    assert!(!t.root.join("b/jobs/disk-reserve/work/input.json").exists());
}

#[test]
fn resource_limits_reach_the_engine_and_worker_floors_cannot_be_lowered() {
    let t = Rig::new();
    fs::write(
        t.root.join("engine"),
        r#"#!/bin/sh
set -eu
cat >/dev/null
mask=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
printf '{"threads":"%s","mask":"%s"}\n' "$BREWERY_CPU_THREADS" "$mask"
"#,
    )
    .unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["engine_env"] = json!({"BREWERY_CPU_THREADS":"1"});
    fs::write(&t.b, config.to_string()).unwrap();
    t.dispatch(
        "start",
        "cpu-clamped",
        Some(&json!({
            "kind":"inspect","inputs":[],"resources":{"cpu_threads":4}
        })),
    );
    let result = t.wait("cpu-clamped", "succeeded");
    assert_eq!(result["result"]["threads"], "1");
    assert!(
        result["result"]["mask"]
            .as_str()
            .unwrap()
            .parse::<u32>()
            .is_ok()
    );

    config["disk_reserve_bytes"] = json!(1_u64 << 50);
    fs::write(&t.b, config.to_string()).unwrap();
    t.dispatch(
        "start",
        "disk-floor",
        Some(&json!({
            "kind":"inspect","inputs":[],"resources":{"disk_reserve_bytes":0}
        })),
    );
    assert_eq!(
        t.wait("disk-floor", "failed")["error"],
        "Worker disk reserve not met"
    );

    config["disk_reserve_bytes"] = json!(0);
    config["resource_bounds"] = json!({"min_free_ram_bytes":{"min":1_u64 << 44,"max":1_u64 << 44}});
    fs::write(&t.b, config.to_string()).unwrap();
    t.dispatch(
        "start",
        "ram-floor",
        Some(&json!({
            "kind":"inspect","inputs":[],"resources":{"min_free_ram_bytes":0}
        })),
    );
    assert_eq!(
        t.wait("ram-floor", "failed")["error"],
        "Worker minimum free RAM not met"
    );
}

#[test]
fn worker_launch_concurrency_never_exceeds_configured_ceiling() {
    let t = Rig::new();
    fs::write(
        t.root.join("engine"),
        "#!/bin/sh\ncat >/dev/null\nexec sleep 60\n",
    )
    .unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["resource_bounds"] = json!({"concurrency":{"min":1,"max":2}});
    fs::write(&t.b, config.to_string()).unwrap();
    let request = json!({"kind":"convert","inputs":[],"resources":{"concurrency":2}});
    for id in ["slot-one", "slot-two"] {
        t.dispatch("start", id, Some(&request));
        t.wait(id, "running");
    }
    let raised = json!({"kind":"convert","inputs":[],"resources":{"concurrency":16}});
    let denied = t.dispatch("start", "slot-three", Some(&raised));
    assert_eq!(denied["state"], "failed");
    t.dispatch("cancel", "slot-one", None);
    t.wait("slot-one", "cancelled");
    assert_eq!(t.dispatch("status", "slot-two", None)["state"], "running");
    t.dispatch("cancel", "slot-two", None);
    t.wait("slot-two", "cancelled");
}

#[test]
fn safetensors_weight_jobs_require_their_own_worker_opt_in() {
    let t = Rig::new();
    let source = t.src.join("model.safetensors");
    fs::write(&source, b"not parsed without opt-in").unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["engine_env"] = json!({"BREWERY_WEIGHT_EDIT":t.root.join("engine")});
    fs::write(&t.b, config.to_string()).unwrap();
    let job = json!({
        "kind":"weight_inspect",
        "inputs":[{"kind":"safetensors","host":"b","path":source}],
        "options":{"tensor":"layer.weight","index":[0,0]}
    });
    let denied = t.dispatch("start", "st-disabled", Some(&job));
    assert_eq!(denied["state"], "failed");
    assert_eq!(
        denied["error"],
        "SafeTensors weight editing is not configured on this worker"
    );
    assert_eq!(t.dispatch("status", "st-disabled", None), denied);
    assert!(!t.root.join("b/jobs/st-disabled/work").exists());
}

#[test]
fn unproven_large_safetensors_is_refused_before_engine_start() {
    let t = Rig::new();
    let source = t.src.join("model.safetensors");
    fs::File::create(&source)
        .unwrap()
        .set_len(360_000_000_000)
        .unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["engine_env"] = json!({
        "BREWERY_WEIGHT_EDIT":t.root.join("engine"),
        "BREWERY_WEIGHT_SAFETENSORS":"1",
        "BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES":"67108864",
    });
    fs::write(&t.b, config.to_string()).unwrap();
    let job = json!({
        "kind":"weight_inspect",
        "inputs":[{"kind":"safetensors","host":"b","path":source}],
        "options":{"tensor":"layer.weight","index":[0,0]}
    });
    let denied = t.dispatch("start", "st-unproven", Some(&job));
    assert_eq!(denied["state"], "failed");
    assert!(
        denied["error"]
            .as_str()
            .unwrap()
            .contains("valid immutable publication")
    );
    assert!(denied.get("result").is_none());
    assert!(!t.root.join("b/jobs/st-unproven/work").exists());
}

#[test]
fn safetensors_rejects_a_manifest_without_kernel_verity_even_with_explicit_fallback() {
    use brewery_worker::{Entry, hash, tree_digest};
    let t = Rig::new();
    let source = t.src.join("model.safetensors");
    fs::write(&source, b"ordinary mutable source").unwrap();
    let entry = Entry {
        path: "model.safetensors".into(),
        bytes: fs::metadata(&source).unwrap().len(),
        sha256: hash(&source).unwrap(),
    };
    let receipt = t.root.join("weight-publication.json");
    fs::write(
        &receipt,
        json!({
            "schema":"brewery.weight-publication.v1",
            "source":source,
            "bundle_sha256":tree_digest(std::slice::from_ref(&entry)),
            "files":[{"path":entry.path,"bytes":entry.bytes,"sha256":entry.sha256,
                "fsverity_sha256":entry.sha256}]
        })
        .to_string(),
    )
    .unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["engine_env"] = json!({
        "BREWERY_WEIGHT_EDIT":t.root.join("engine"),
        "BREWERY_WEIGHT_SAFETENSORS":"1",
        "BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES":"67108864",
        "BREWERY_WEIGHT_SAFETENSORS_PUBLICATION":receipt,
    });
    fs::write(&t.b, config.to_string()).unwrap();
    let job = json!({
        "kind":"weight_inspect",
        "inputs":[{"kind":"safetensors","host":"b","path":source}],
        "options":{"tensor":"layer.weight","index":[0,0]}
    });
    let denied = t.dispatch("start", "st-false-proof", Some(&job));
    assert_eq!(denied["state"], "failed");
    assert_eq!(
        denied["error"],
        "SafeTensors immutable publication could not be verified"
    );
    assert_eq!(t.dispatch("status", "st-false-proof", None), denied);
    assert!(!t.root.join("b/jobs/st-false-proof/work").exists());
}

#[test]
fn cancelled_safetensors_job_removes_unpublished_weight_output() {
    let t = Rig::new();
    let source = t.src.join("model.safetensors");
    fs::write(&source, b"fixture").unwrap();
    let engine = t.root.join("engine");
    fs::write(
        &engine,
        "#!/bin/sh\nset -eu\ncat >/dev/null\nmkdir .weight-safetensors-fixture\n\
         printf partial > .weight-safetensors-fixture/output.safetensors\n\
         printf edited > edited.safetensors\nexec sleep 60\n",
    )
    .unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["engine_env"] = json!({
        "BREWERY_WEIGHT_EDIT":engine,
        "BREWERY_WEIGHT_SAFETENSORS":"1",
        "BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES":"67108864",
    });
    fs::write(&t.b, config.to_string()).unwrap();
    let id = "st-cancelled";
    t.dispatch(
        "start",
        id,
        Some(&json!({
            "kind":"weight_apply",
            "inputs":[{"kind":"safetensors","host":"b","path":source}]
        })),
    );
    let work = t.root.join("b/jobs/st-cancelled/work");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !work.join("edited.safetensors").exists() {
        assert!(Instant::now() < deadline, "engine did not start");
        thread::sleep(Duration::from_millis(20));
    }
    t.dispatch("cancel", id, None);
    assert!(t.wait(id, "cancelled").get("result").is_none());
    assert!(!work.join("edited.safetensors").exists());
    assert!(!work.join(".weight-safetensors-fixture").exists());
    assert_eq!(fs::read(source).unwrap(), b"fixture");
}

#[test]
fn packed_gate_denial_is_terminal_for_the_same_job_id() {
    let t = Rig::new();
    let source = t.src.join("packed.gguf");
    fs::write(&source, b"GGUF fixture").unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["engine_env"] = json!({
        "BREWERY_WEIGHT_EDIT":t.root.join("engine"),
        "BREWERY_WEIGHT_PACKED":"0"
    });
    fs::write(&t.b, config.to_string()).unwrap();
    let job = json!({
        "kind":"weight_inspect",
        "inputs":[{"kind":"gguf","host":"b","path":source}],
        "options":{"packed":true,"tensor":"layer.weight","index":[0,0]}
    });
    let id = "packed-denied";

    let denied = t.dispatch("start", id, Some(&job));
    assert_eq!(denied["state"], "failed");
    assert_eq!(
        denied["error"],
        "packed weight editing is not configured on this worker"
    );
    assert!(denied.get("result").is_none());
    assert_eq!(t.dispatch("status", id, None), denied);

    config["engine_env"]["BREWERY_WEIGHT_PACKED"] = json!("1");
    fs::write(&t.b, config.to_string()).unwrap();
    assert_eq!(t.dispatch("start", id, Some(&job)), denied);
    assert_eq!(t.dispatch("status", id, None), denied);
    assert!(!t.root.join("b/jobs/packed-denied/work").exists());
    assert_eq!(fs::read(source).unwrap(), b"GGUF fixture");
}

#[test]
fn weight_hgn_requires_explicit_opt_in_and_resolves_only_single_files() {
    let t = Rig::new();
    let hgn = t.src.join("one.hgn");
    fs::write(&hgn, b"HGN1 fixture").unwrap();
    let bundle = t.src.join("bundle.hgn");
    fs::create_dir(&bundle).unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["engine_env"] = json!({"BREWERY_WEIGHT_EDIT":t.root.join("engine")});
    fs::write(&t.b, config.to_string()).unwrap();
    let input = json!({"kind":"hgn","host":"b","path":hgn});
    assert!(
        resolve_input(
            &t,
            "unconfigured",
            "start",
            "weight_inspect",
            0,
            input.clone()
        )["error"]
            .as_str()
            .unwrap()
            .contains("HGN weight editing is not configured")
    );
    config["engine_env"]["BREWERY_WEIGHT_HGN"] = json!("1");
    fs::write(&t.b, config.to_string()).unwrap();
    let result = resolve_input(&t, "single", "start", "weight_inspect", 0, input);
    assert_eq!(result["ok"]["files"].as_array().unwrap().len(), 1);
    assert!(result["ok"]["sha256"].is_string());
    let result = resolve_input(
        &t,
        "bundle",
        "start",
        "weight_inspect",
        0,
        json!({"kind":"hgn","host":"b","path":bundle}),
    );
    assert!(result["error"].as_str().unwrap().contains("not a bundle"));
}

#[test]
fn describe_hashes_multiple_fixture_files_concurrently_in_manifest_order() {
    let t = Rig::new();
    let files = ["a shard.gguf", "z shard.gguf"].map(|name| t.src.join(name));
    for path in files.iter().rev() {
        fs::File::create(path)
            .unwrap()
            .set_len(64 * 1024 * 1024)
            .unwrap();
    }
    let mut child = Command::new(env!("CARGO_BIN_EXE_brewery-worker"))
        .arg("rpc")
        .env("BREWERY_WORKER_CONFIG", &t.b)
        .env("BREWERY_HASH_THREADS", "2")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    serde_json::to_writer(
        child.stdin.take().unwrap(),
        &json!({"op":"describe","input":{"path":t.src,"host":"b","kind":"hf"}}),
    )
    .unwrap();
    if std::thread::available_parallelism().is_ok_and(|cpus| cpus.get() >= 2) {
        let descriptors = format!("/proc/{}/fd", child.id());
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let open: Vec<_> = fs::read_dir(&descriptors)
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|entry| fs::read_link(entry.path()).ok())
                .collect();
            if files.iter().all(|path| open.contains(path)) {
                break;
            }
            if Instant::now() >= deadline || child.try_wait().unwrap().is_some() {
                let _ = child.kill();
                child.wait().unwrap();
                panic!("worker never opened both fixture shards concurrently");
            }
            thread::sleep(Duration::from_millis(1));
        }
    }
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let reply: Value = serde_json::from_slice(&out.stdout).unwrap();
    let manifest: Description = serde_json::from_value(reply["ok"].clone()).unwrap();
    let expected = "3b6a07d0d404fab4e23b6d34bc6696a6a312dd92821332385e5af7c01c421351";
    assert_eq!(manifest.bundle.files.len(), 2);
    for (entry, path) in manifest.bundle.files.iter().zip(files) {
        assert_eq!(entry.path, path.file_name().unwrap().to_str().unwrap());
        assert_eq!(entry.bytes, 64 * 1024 * 1024);
        assert_eq!(entry.sha256, expected);
    }
    assert_eq!(
        manifest.bundle.digest,
        "19ccecd6e7cc3d6ab59fff4b4710862dab197f245db6a70fd8d77557542b83b4"
    );
    manifest.bundle.validate().unwrap();
}

#[test]
fn queued_input_hash_progress_is_measured_and_clears_on_cancel() {
    assert_hash_progress("b");
}

#[test]
fn remote_source_hash_progress_reaches_the_destination_job() {
    assert_hash_progress("a");
}

fn assert_hash_progress(host: &str) {
    let t = Rig::new();
    // Sparse fixture: long enough to observe the throttled update even in
    // optimized builds, without allocating disk blocks. Cancel after a tick.
    let bytes = 2_u64 * 1024 * 1024 * 1024;
    for name in ["first weights.gguf", "second weights.gguf"] {
        fs::File::create(t.src.join(name))
            .unwrap()
            .set_len(bytes)
            .unwrap();
    }
    let input = json!({"host":host,"kind":"hf","path":t.src});
    let job = json!({"kind":"inspect","inputs":[input],"options":{}});
    let mut c = Command::new(env!("CARGO_BIN_EXE_brewery-dispatch"));
    let mut child = c
        .env("BREWERY_DISPATCH_CONFIG", &t.dispatch)
        .env("BREWERY_HASH_THREADS", "2")
        .args(["start", "b", "tracked"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    serde_json::to_writer(child.stdin.take().unwrap(), &job).unwrap();
    let until = Instant::now() + Duration::from_secs(3);
    let found = loop {
        let reply = t.dispatch("status", "tracked", None);
        let detail = &reply["progress_detail"];
        if detail["phase"] == "hash_input"
            && detail["completed"]
                .as_u64()
                .is_some_and(|n| n > 0 && n < 2 * bytes)
        {
            if thread::available_parallelism().is_ok_and(|n| n.get() >= 2) {
                let Some(slots) = detail["threads"].as_array().filter(|s| s.len() == 2) else {
                    if Instant::now() >= until {
                        break false;
                    }
                    thread::sleep(Duration::from_millis(10));
                    continue;
                };
                assert_eq!(slots[0]["slot"], 1);
                assert_eq!(slots[1]["slot"], 2);
                assert_ne!(slots[0]["task"], slots[1]["task"]);
                for slot in slots {
                    assert!(matches!(
                        slot["task"].as_str(),
                        Some("first weights.gguf" | "second weights.gguf")
                    ));
                }
            }
            assert_eq!(reply["state"], "queued");
            assert_eq!(detail["unit"], "bytes");
            assert_eq!(detail["total"], 2 * bytes);
            assert!(matches!(
                detail["current_file"].as_str(),
                Some("first weights.gguf" | "second weights.gguf")
            ));
            assert_eq!(detail["files_total"], 2);
            assert_eq!(detail["input_index"], 1);
            assert_eq!(detail["input_count"], 1);
            break true;
        }
        if Instant::now() >= until || child.try_wait().unwrap().is_some() {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let cancel_started = Instant::now();
    let cancelled = t.dispatch("cancel", "tracked", None);
    child.wait_with_output().unwrap();
    assert!(found, "did not see mid-hash byte progress");
    assert!(cancel_started.elapsed() < Duration::from_secs(3));
    assert_eq!(cancelled["state"], "cancelled");
    assert!(cancelled.get("progress_detail").is_none());
    assert!(
        t.dispatch("status", "tracked", None)
            .get("progress_detail")
            .is_none()
    );
    assert!(!t.root.join("b/jobs/tracked/work").exists());
}

#[test]
fn running_job_accepts_only_bounded_valid_engine_progress() {
    let t = Rig::new();
    let job = json!({"kind":"progress-test","inputs":[],"options":{}});
    t.dispatch("start", "engine-progress", Some(&job));
    t.wait("engine-progress", "running");
    let work = t.root.join("b/jobs/engine-progress/work");
    let until = Instant::now() + Duration::from_secs(3);
    while !work.join("progress-ready").exists() {
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(10));
    }
    let updated_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let valid = json!({"phase":"verify_input","unit":"bytes","completed":2,"total":8,
        "current_file":"weights α.gguf","updated_at_ms":updated_at_ms});
    let path = work.join("progress.json");
    for invalid in [
        json!({"phase":"not_allowlisted","unit":"bytes","completed":2,"updated_at_ms":updated_at_ms}),
        json!({"phase":"verify_input","unit":"bytes","completed":9,"total":1,"updated_at_ms":updated_at_ms}),
        json!({"phase":"verify_input","unit":"bytes","completed":2,"current_file":"../private",
            "updated_at_ms":updated_at_ms}),
        json!({"phase":"verify_input","unit":"bytes","completed":2,"updated_at_ms":1}),
    ] {
        fs::write(&path, invalid.to_string()).unwrap();
        assert!(
            t.dispatch("status", "engine-progress", None)
                .get("progress_detail")
                .is_none()
        );
    }
    fs::remove_file(&path).unwrap();
    let elsewhere = t.root.join("forged-progress");
    fs::write(&elsewhere, valid.to_string()).unwrap();
    symlink(elsewhere, &path).unwrap();
    assert!(
        t.dispatch("status", "engine-progress", None)
            .get("progress_detail")
            .is_none()
    );
    fs::remove_file(&path).unwrap();
    fs::write(work.join("progress.tmp"), valid.to_string()).unwrap();
    fs::rename(work.join("progress.tmp"), path).unwrap();
    let running = t.dispatch("status", "engine-progress", None);
    assert_eq!(running["progress_detail"], valid);
    assert_eq!(running["state"], "running");
    for rows in [
        json!([{"slot":1,"task":"../private"}]),
        json!([{"slot":1,"task":"\u{202e}concealed"}]),
        json!([{"slot":1,"task":"é".repeat(49)}]),
        json!([{"slot":1,"task":"first"},{"slot":1,"task":"second"}]),
        json!([{"slot":2,"task":"first"},{"slot":1,"task":"second"}]),
        json!([{"slot":65,"task":"outside"}]),
    ] {
        let mut snapshot = valid.clone();
        snapshot["threads"] = rows;
        fs::write(work.join("progress.tmp"), snapshot.to_string()).unwrap();
        fs::rename(work.join("progress.tmp"), work.join("progress.json")).unwrap();
        assert_eq!(
            t.dispatch("status", "engine-progress", None)["progress_detail"],
            valid
        );
    }
    let mut snapshot = valid.clone();
    snapshot["threads"] =
        json!([{"slot":1,"task":"weights α.gguf"},{"slot":64,"task":"Verify result"}]);
    fs::write(work.join("progress.tmp"), snapshot.to_string()).unwrap();
    fs::rename(work.join("progress.tmp"), work.join("progress.json")).unwrap();
    assert_eq!(
        t.dispatch("status", "engine-progress", None)["progress_detail"],
        snapshot
    );
    fs::write(work.join("progress-done"), []).unwrap();
    assert!(
        t.wait("engine-progress", "succeeded")
            .get("progress_detail")
            .is_none()
    );
}

#[test]
fn suggest_paths_lists_immediate_matches_without_initializing_worker_state() {
    let t = Rig::new();
    let dir = t.src.join("model files");
    let file = t.src.join("model α.gguf");
    fs::create_dir(&dir).unwrap();
    fs::write(dir.join("model nested.gguf"), []).unwrap();
    fs::write(&file, []).unwrap();

    assert_eq!(
        t.suggest(t.src.join("model ")),
        json!({"items":[{"path":dir,"kind":"directory"},{"path":file,"kind":"file"}],
               "truncated":false})
    );
    assert!(!t.root.join("b").exists());
}

#[test]
fn suggest_paths_never_lists_ancestors_hidden_entries_or_symlink_targets() {
    let t = Rig::new();
    let outside = t.root.join("unrelated");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("private.gguf"), []).unwrap();
    fs::write(t.src.join("visible.gguf"), []).unwrap();
    fs::write(t.src.join("control\n.gguf"), []).unwrap();
    fs::create_dir(t.src.join(".hidden")).unwrap();
    fs::write(t.src.join(".hidden/model.gguf"), []).unwrap();
    symlink(&outside, t.src.join("escape")).unwrap();
    symlink(outside.join("private.gguf"), t.src.join("leak.gguf")).unwrap();
    let bad_root = t.root.join("symlink-root");
    symlink(&outside, &bad_root).unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.b).unwrap()).unwrap();
    config["source_roots"] = json!([t.src, bad_root, t.src.join(".hidden")]);
    fs::write(&t.b, config.to_string()).unwrap();

    let root_only = json!({"items":[{"path":t.src,"kind":"directory"}],"truncated":false});
    assert_eq!(t.suggest(""), root_only);
    assert_eq!(t.suggest(format!("{}/", t.root.display())), root_only);
    assert_eq!(t.suggest(t.root.join("sou")), root_only);
    let empty = json!({"items":[],"truncated":false});
    assert_eq!(t.suggest(outside), empty);
    assert_eq!(t.suggest(format!("{}/escape/", t.src.display())), empty);
    assert_eq!(
        t.suggest(format!("{}/", t.src.display())),
        json!({"items":[{"path":t.src.join("visible.gguf"),"kind":"file"}],"truncated":false})
    );
}

#[test]
fn suggest_paths_rpc_rejects_unsafe_or_non_path_requests() {
    let t = Rig::new();
    for prefix in [
        "relative".into(),
        format!("{}/../unrelated/", t.src.display()),
        format!("{}/.hidden/", t.src.display()),
        format!("{}/\n", t.src.display()),
        format!("/{}", "a".repeat(4096)),
    ] {
        let result = t.rpc_raw(&json!({"op":"suggest_paths","prefix":prefix}));
        assert!(result["error"].is_string(), "{result}");
    }
    let result = t.rpc_raw(&json!({"op":"suggest_paths","prefix":"","id":"not-a-job"}));
    assert!(result["error"].is_string(), "{result}");
    assert!(!t.root.join("b").exists());
}

#[test]
fn suggest_paths_bounds_matching_items_and_path_bytes() {
    let t = Rig::new();
    for i in 0..33 {
        fs::write(t.src.join(format!("model {i}.gguf")), []).unwrap();
    }
    let result = t.rpc_raw(&json!({"op":"suggest_paths","prefix":t.src.join("model ")}));
    assert_eq!(result["ok"]["items"].as_array().unwrap().len(), 32);
    assert_eq!(result["ok"]["truncated"], true);

    let deep = (0..12).fold(t.src.clone(), |dir, _| dir.join("a".repeat(210)));
    fs::create_dir_all(&deep).unwrap();
    for i in 0..8 {
        fs::write(deep.join(format!("model {i}.gguf")), []).unwrap();
    }
    let result = t.suggest(deep.join("model "));
    assert_eq!(result["items"].as_array().unwrap().len(), 6);
    assert_eq!(result["truncated"], true);
}

#[test]
fn benchmark_ids_pass_unchanged_through_dispatcher_to_engine() {
    let t = Rig::new();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.dispatch).unwrap()).unwrap();
    let host = &mut config["hosts"]["b"];
    host["capabilities"] = json!({"benchmark":true});
    host["benchmark_targets"] = json!([{"id":"local-model","label":"Local model"}]);
    host["benchmarks"] = json!([{"id":"tool-eval-bench"},{"id":"betterbench"}]);
    fs::write(&t.dispatch, config.to_string()).unwrap();
    let benchmark = json!({
        "kind":"benchmark","inputs":[],
        "options":{"benchmark_target":"local-model","benchmarks":["tool-eval-bench","betterbench"]}
    });
    let started = t.dispatch("start", "benchmark-run", Some(&benchmark));
    assert!(matches!(
        started["state"].as_str(),
        Some("queued" | "running")
    ));
    t.wait("benchmark-run", "succeeded");
    let engine_input: Value = serde_json::from_slice(
        &fs::read(t.root.join("b/jobs/benchmark-run/work/input.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(engine_input, benchmark);
}

fn output(mut c: Command, body: Option<&Value>) -> Value {
    let mut child = c
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    if let Some(value) = body {
        serde_json::to_writer(&mut stdin, value).unwrap();
    }
    stdin.flush().unwrap();
    drop(stdin);
    let out = child.wait_with_output().unwrap();
    let v = serde_json::from_slice(&out.stdout).unwrap();
    assert!(out.status.success(), "{v}");
    v
}

#[test]
fn managed_runtime_applies_new_config_for_each_job() {
    use brewery_worker::operational::{Settings, Target};
    let rig = Rig::new();
    let root = rig.root.join("b");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let catalog = rig.root.join("catalog.json");
    fs::write(
        &catalog,
        json!({"adapters":{"suite":{"executable":"/bin/true","version":"1","protocol":"fixture"}}})
            .to_string(),
    )
    .unwrap();
    let engine = rig.root.join("engine");
    fs::write(
        &engine,
        "#!/bin/sh\nset -eu\ncat \"$BREWERY_BENCHMARK_CONFIG\"\n",
    )
    .unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&rig.b).unwrap()).unwrap();
    config["managed_settings"] =
        json!({"catalog":catalog,"allowed_ports":[9101],"evaluation_allowed":false});
    fs::write(&rig.b, config.to_string()).unwrap();
    let job = json!({"kind":"benchmark","inputs":[],"options":{"benchmark_target":"target","benchmarks":["suite"]}});
    for (revision, model) in [(1, "first"), (2, "second")] {
        Settings {
            revision,
            benchmark_targets: vec![Target {
                id: "target".into(),
                label: None,
                endpoint: "http://127.0.0.1:9101/v1".into(),
                model: model.into(),
            }],
            enabled_suites: vec!["suite".into()],
            ..Default::default()
        }
        .save(&root)
        .unwrap();
        rig.dispatch("start", model, Some(&job));
        let result = rig.wait(model, "succeeded");
        assert_eq!(result["result"]["targets"]["target"]["model"], model);
        assert_eq!(
            result["result"]["targets"]["target"]["endpoint"],
            "http://127.0.0.1:9101/v1"
        );
    }
}

#[test]
fn local_hosts_preview_resume_reuse_and_cleanup_bundles() {
    let t = Rig::new();
    let source = t.src.join("nested-model");
    fs::create_dir_all(source.join("nested")).unwrap();
    let bytes = vec![23; CHUNK + 347];
    fs::write(source.join("nested/weights"), &bytes).unwrap();
    fs::write(source.join("manifest.json"), b"{}").unwrap();
    let job =
        json!({"kind":"transfer","inputs":[{"host":"a","path":source,"kind":"hgn"}],"options":{}});
    let preview = t.dispatch("preview_transfer", "preview", Some(&job));
    assert_eq!(preview["state"], "succeeded");
    assert_eq!(preview["result"]["missing_bytes"], (bytes.len() + 2) as u64);
    assert!(!Path::new(preview["result"]["inputs"][0]["path"].as_str().unwrap()).exists());

    // Simulate an interrupted relay after the first durable chunk.
    let input = serde_json::from_value(job["inputs"][0].clone()).unwrap();
    let desc: Description = serde_json::from_value(t.rpc(Rpc::Describe { input })).unwrap();
    let bundle: Bundle = desc.bundle;
    t.rpc(Rpc::Prepare {
        id: "move".into(),
        action: "transfer".into(),
        job: serde_json::from_value(job.clone()).unwrap(),
    });
    let before: Plan = serde_json::from_value(t.rpc(Rpc::Plan {
        id: "move".into(),
        bundle: bundle.clone(),
        preview: false,
    }))
    .unwrap();
    assert_eq!(before.offsets, [0, 0]);
    assert_eq!(
        t.rpc(Rpc::Write {
            id: "move".into(),
            digest: bundle.digest,
            entry: bundle.files[1].clone(),
            offset: 0,
            data: STANDARD.encode(&bytes[..CHUNK])
        }),
        CHUNK as u64
    );
    let status = t.dispatch("status", "move", None);
    assert_eq!(status["state"], "queued");
    assert_eq!(
        status["progress"],
        "Copying bundle: 8388608 / 8388957 bytes staged"
    );

    let done = t.dispatch("transfer", "move", Some(&job));
    assert_eq!(done["state"], "succeeded", "{done}");
    assert_eq!(done["result"]["missing_bytes"], 349);
    let path = Path::new(done["result"]["inputs"][0]["path"].as_str().unwrap());
    assert_eq!(fs::read(path.join("nested/weights")).unwrap(), bytes);
    let reuse = t.dispatch("transfer", "reuse", Some(&job));
    assert_eq!(reuse["result"]["missing_bytes"], 0);
    assert_eq!(reuse["result"]["inputs"][0]["cached"], true);
    assert_eq!(t.dispatch("status", "move", None)["state"], "succeeded");
    let cleanup = t.dispatch("cleanup", "cleanup", None);
    assert_eq!(cleanup["state"], "succeeded");
    assert_eq!(cleanup["result"]["removed"].as_array().unwrap().len(), 1);
    assert!(!path.exists());
    assert!(source.join("nested/weights").exists());
}

#[test]
fn cross_worker_transfer_overlaps_file_reads_and_publishes_verified_bytes() {
    if thread::available_parallelism().is_ok_and(|n| n.get() < 2) {
        return;
    }
    let t = Rig::new();
    let wrapper = t.root.join("source-worker");
    fs::write(
        &wrapper,
        format!(
            r#"#!/usr/bin/python3
import json, pathlib, subprocess, sys, time
body = sys.stdin.buffer.read()
request = json.loads(body)
if request["op"] == "read":
    directory = pathlib.Path(__file__).parent
    (directory / (request["entry"]["path"] + ".started")).touch()
    deadline = time.monotonic() + 3
    while not (directory / "finish-reads").exists():
        if time.monotonic() > deadline:
            print(json.dumps({{"error": "independent read never started"}}))
            sys.exit(0)
        time.sleep(.01)
sys.exit(subprocess.run([{worker:?}, "rpc"], input=body).returncode)
"#,
            worker = env!("CARGO_BIN_EXE_brewery-worker"),
        ),
    )
    .unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o700)).unwrap();
    let mut config: Value = serde_json::from_slice(&fs::read(&t.dispatch).unwrap()).unwrap();
    config["hosts"]["a"]["worker"] = json!(wrapper);
    fs::write(&t.dispatch, config.to_string()).unwrap();
    fs::write(t.src.join("a"), b"first").unwrap();
    fs::write(t.src.join("z"), b"second").unwrap();
    let job = json!({"kind":"transfer","inputs":[{"host":"a","path":t.src,"kind":"hf"}]});
    let mut c = Command::new(env!("CARGO_BIN_EXE_brewery-dispatch"));
    c.env("BREWERY_DISPATCH_CONFIG", &t.dispatch)
        .env("BREWERY_CPU_THREADS", "2")
        .args(["transfer", "b", "parallel-relay"]);
    let mut child = c
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    serde_json::to_writer(child.stdin.take().unwrap(), &job).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let slots = loop {
        let status = t.dispatch("status", "parallel-relay", None);
        let detail = &status["progress_detail"];
        if detail["phase"] == "copy_input"
            && let Some(slots) = detail["threads"].as_array().filter(|s| s.len() == 2)
            && ["a.started", "z.started"]
                .iter()
                .all(|name| t.root.join(name).exists())
        {
            break Some(slots.clone());
        }
        if Instant::now() >= deadline {
            break None;
        }
        thread::sleep(Duration::from_millis(10));
    };
    fs::write(t.root.join("finish-reads"), []).unwrap();
    let out = child.wait_with_output().unwrap();
    let slots =
        slots.expect("both actual relay tasks must be displayed while their reads are blocked");
    assert_eq!(slots[0]["slot"], 1);
    assert_eq!(slots[1]["slot"], 2);
    assert_ne!(slots[0]["task"], slots[1]["task"]);
    assert!(out.status.success());
    let reply: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(reply["state"], "succeeded", "{reply}");
    let dest = Path::new(reply["result"]["inputs"][0]["path"].as_str().unwrap());
    assert_eq!(fs::read(dest.join("a")).unwrap(), b"first");
    assert_eq!(fs::read(dest.join("z")).unwrap(), b"second");
    assert_eq!(reply["result"]["missing_bytes"], 11);
    assert!(
        t.dispatch("status", "parallel-relay", None)
            .get("progress_detail")
            .is_none()
    );
}

#[test]
fn seal_checks_all_partials_and_refuses_an_unverified_existing_destination() {
    let t = Rig::new();
    for (name, contents) in [("a", b"first"), ("z", b"other")] {
        fs::write(t.src.join(name), contents).unwrap();
    }
    let input = json!({"host":"a","path":t.src,"kind":"hf"});
    let desc: Description = serde_json::from_value(t.rpc(Rpc::Describe {
        input: serde_json::from_value(input.clone()).unwrap(),
    }))
    .unwrap();
    let bundle = desc.bundle;
    t.rpc(Rpc::Prepare {
        id: "seal".into(),
        action: "transfer".into(),
        job: serde_json::from_value(json!({"kind":"transfer","inputs":[input]})).unwrap(),
    });
    let plan: Plan = serde_json::from_value(t.rpc(Rpc::Plan {
        id: "seal".into(),
        bundle: bundle.clone(),
        preview: false,
    }))
    .unwrap();
    let write = |index: usize, data: &[u8]| {
        t.rpc(Rpc::Write {
            id: "seal".into(),
            digest: bundle.digest.clone(),
            entry: bundle.files[index].clone(),
            offset: 0,
            data: STANDARD.encode(data),
        })
    };
    let seal = || t.rpc_raw(&json!({"op":"seal","id":"seal","digest":bundle.digest}));

    assert_eq!(write(0, b"first"), 5);
    assert_eq!(write(1, b"wrong"), 5);
    assert!(seal()["error"].as_str().unwrap().contains("hash mismatch"));
    assert!(!plan.path.exists());
    let retry: Plan = serde_json::from_value(t.rpc(Rpc::Plan {
        id: "seal".into(),
        bundle: bundle.clone(),
        preview: false,
    }))
    .unwrap();
    assert_eq!(retry.offsets, [5, 0]);
    assert_eq!(write(1, b"other"), 5);

    let published = plan.path.parent().unwrap();
    fs::create_dir(published).unwrap();
    fs::write(published.join("untouched"), b"existing").unwrap();
    assert!(
        seal()["error"].is_string(),
        "must not trust or clobber an existing directory"
    );
    assert_eq!(fs::read(published.join("untouched")).unwrap(), b"existing");
    assert!(!plan.path.exists());
    fs::remove_dir_all(published).unwrap();

    assert_eq!(seal(), json!({"ok":null}));
    assert_eq!(fs::read(plan.path.join("a")).unwrap(), b"first");
    assert_eq!(fs::read(plan.path.join("z")).unwrap(), b"other");
    assert_eq!(seal(), json!({"ok":null}));
    assert!(!t.root.join("b/partial").join(bundle.digest).exists());
}

#[test]
fn cache_reuse_rejects_size_preserving_corruption_in_one_of_multiple_files() {
    let t = Rig::new();
    for file in ["a.gguf", "z.gguf"] {
        fs::write(t.src.join(file), b"good").unwrap();
    }
    let job = json!({"kind":"transfer","inputs":[{"host":"a","path":t.src,"kind":"hf"}]});
    let copied = t.dispatch("transfer", "publish", Some(&job));
    assert_eq!(copied["state"], "succeeded", "{copied}");
    let dest = Path::new(copied["result"]["inputs"][0]["path"].as_str().unwrap());
    fs::write(dest.join("z.gguf"), b"evil").unwrap();
    let rejected = t.dispatch("transfer", "reuse", Some(&job));
    assert_eq!(rejected["state"], "failed", "{rejected}");
    assert!(rejected.get("result").is_none());
    assert_eq!(fs::read(t.src.join("z.gguf")).unwrap(), b"good");
}

#[test]
fn public_status_reports_allowlisted_stage_and_private_errors() {
    let t = Rig::new();
    let job = json!({"kind":"progress-test","inputs":[],"prompt":"fixture-private-token"});
    t.dispatch("start", "watch", Some(&job));
    let work = t.root.join("b/jobs/watch/work");
    let until = Instant::now() + Duration::from_secs(4);
    while !work.join("progress-ready").exists() {
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(20));
    }
    let reply = t.dispatch("status", "watch", None);
    fs::write(work.join("progress-done"), b"").unwrap();
    t.wait("watch", "succeeded");
    assert_eq!(reply["state"], "running");
    assert!(
        reply["progress"]
            .as_str()
            .unwrap()
            .starts_with("Tracing reference model (")
    );
    assert!(!reply.to_string().contains("fixture-private-token"));
    let job = json!({"kind":"failure-test","inputs":[]});
    t.dispatch("start", "bad", Some(&job));
    let reply = t.wait("bad", "failed");
    assert!(
        reply["error"]
            .as_str()
            .unwrap()
            .contains("CUDA out of memory")
    );
    assert!(!reply.to_string().contains("fixture-private-token"));
    let log = t.root.join("b/jobs/bad/engine-stderr.log");
    assert!(
        fs::read_to_string(&log)
            .unwrap()
            .contains("fixture-private-token")
    );
    assert_eq!(
        fs::metadata(log).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

#[test]
fn local_jobs_survive_clients_cancel_descendants_and_report_errors() {
    let t = Rig::new();
    let input = t.src.join("model weird ; name.gguf");
    fs::write(&input, b"a model fixture").unwrap();
    let job = json!({"kind":"compare","inputs":[{"host":"a","path":input,"kind":"gguf"}],
        "options":{},"prompt":"quotes; $(not a command)"});
    assert_eq!(t.dispatch("status", "one", None)["state"], "missing");
    let first = t.dispatch("start", "one", Some(&job));
    assert!(matches!(
        first["state"].as_str(),
        Some("queued" | "running")
    ));
    t.dispatch("start", "one", Some(&job));
    assert_eq!(
        t.wait("one", "succeeded")["result"],
        json!({"completed":true})
    );
    let jobdir = t.root.join("b/jobs/one/work");
    let received: Value =
        serde_json::from_slice(&fs::read(jobdir.join("input.json")).unwrap()).unwrap();
    assert_eq!(received["prompt"], job["prompt"]);
    assert_eq!(received["inputs"][0]["host"], "b");
    assert_eq!(received["inputs"][0]["source_host"], "a");
    assert_eq!(
        fs::read(received["inputs"][0]["path"].as_str().unwrap()).unwrap(),
        b"a model fixture"
    );

    let job = json!({"kind":"cancel-test","inputs":[],"options":{}});
    assert_eq!(t.dispatch("cancel", "never", None)["state"], "cancelled");
    assert_eq!(
        t.dispatch("start", "never", Some(&job))["state"],
        "cancelled"
    );
    t.dispatch("start", "kill", Some(&job));
    let pid_path = t.root.join("b/jobs/kill/work/descendant");
    let until = Instant::now() + Duration::from_secs(4);
    let pid = loop {
        if let Ok(v) = fs::read_to_string(&pid_path)
            && !v.trim().is_empty()
        {
            break v.trim().to_owned();
        }
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(25));
    };
    t.dispatch("cancel", "kill", None);
    t.wait("kill", "cancelled");
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
    t.dispatch(
        "start",
        "error",
        Some(&json!({"kind":"failure-test","inputs":[],"options":{}})),
    );
    assert!(
        t.wait("error", "failed")["error"]
            .as_str()
            .unwrap()
            .contains("19")
    );
}

#[test]
fn source_symlinks_and_paths_outside_operator_roots_fail() {
    let t = Rig::new();
    let secret = t.root.join("private");
    fs::write(&secret, b"no").unwrap();
    let link = t.src.join("link.gguf");
    std::os::unix::fs::symlink(&secret, &link).unwrap();
    for (id, path) in [("link", link), ("outside", secret)] {
        let r = t.dispatch(
            "transfer",
            id,
            Some(&json!({"kind":"transfer","inputs":[{"host":"a","kind":"gguf","path":path}]})),
        );
        assert_eq!(r["state"], "failed");
    }
    let missing = t.src.join("fixture-private-missing.gguf");
    let r = t.dispatch(
        "transfer",
        "missing-src",
        Some(&json!({"kind":"transfer","inputs":[{"host":"a","kind":"gguf","path":missing}]})),
    );
    assert_eq!(r["state"], "failed");
    assert!(
        r["error"]
            .as_str()
            .unwrap()
            .contains("Source input unavailable")
    );
    assert!(!r.to_string().contains("fixture-private-missing"));
}
