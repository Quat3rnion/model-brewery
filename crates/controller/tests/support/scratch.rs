use super::*;

fn fixture(t: &Test, values: &[f32]) -> std::path::PathBuf {
    let dir = t.root.path().join("scratch-fixtures");
    fs::create_dir(&dir).unwrap();
    let path = dir.join("parent.safetensors");
    // Include metadata and trailing header padding to check byte-exact restoration.
    let header = br#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}        "#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    fs::write(&path, bytes).unwrap();
    let config = t.root.path().join("scratch-worker.json");
    fs::write(
        &config,
        json!({"source_roots":[dir],"disk_reserve_bytes":0}).to_string(),
    )
    .unwrap();
    fs::write(
        t.root.path().join("dispatch.json"),
        json!({"hosts":{"existing":{"local":true,"config":config,
          "capabilities":{"benchmark":true,"trace":true},
          "benchmark_targets":[{"id":"cpu","label":"CPU"}],
          "benchmarks":[{"id":"gsm8k","label":"GSM8K"}]}}})
        .to_string(),
    )
    .unwrap();
    fs::remove_file(t.root.path().join("catalog.json")).unwrap();
    fs::write(
        t.root.path().join("dispatch"),
        r#"#!/bin/sh
case "$1" in
  start)
    python3 -c 'import json,sys
j=json.load(sys.stdin)
result=dict(kind="trace",schema_version=1,model_id=j["inputs"][0]["id"],
    model_sha256="a"*64,prompts=[{}]) if j["kind"]=="trace" else {"mock":True}
print(json.dumps(dict(state="succeeded",progress="mock done",result=result)))' > "${0%/*}/$3.result"
    printf '%s\n' '{"state":"running","progress":"mock running"}'
    ;;
  status) cat "${0%/*}/$3.result" ;;
  *) exit 1 ;;
esac
"#,
    )
    .unwrap();
    for (name, script) in [
        ("editor-DISPATCH", "#!/bin/sh\nexit 1\n"),
        ("editor-WORKER", "#!/bin/sh\nexit 1\n"),
        ("editor-ENGINE", "#!/bin/sh\nexit 1\n"),
        (
            "editor-HELPER",
            r#"#!/bin/sh
test "$1" = --capabilities || exit 1
printf '%s\n' '{"schema":"brewery.weight-runtime.v1","jobs":["weight_inspect","weight_propose","weight_apply"],"formats":["gguf","hf","safetensors"],"visual":true,"packed":false}'
"#,
        ),
    ] {
        let file = t.root.path().join(name);
        fs::write(&file, script).unwrap();
        fs::set_permissions(file, fs::Permissions::from_mode(0o700)).unwrap();
    }
    path
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn mcp_compiled_root_checks_and_approval_boundary() {
    let mut t = Test::new();
    fixture(&t, &[1., 2., 3., 4., 5., 6.]);
    fs::write(
        t.root.path().join("program.py"),
        r#"import hashlib, json, pathlib, struct, sys
action = sys.argv[1]
directory = pathlib.Path(sys.argv[3 if action == "compile" else 2])
weights = directory / "model.safetensors"
config = directory / "config.json"
if action == "compile":
    header = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    weights.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1))
    config.write_text('{"schema":"brewery.compiled-program.v1","program":"sort","compiler_commit":"9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd","architecture":"tracr","runtime":"mock"}')
    print(json.dumps({"sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}))
elif action == "run":
    print(json.dumps([0, 1, 2, 2]))
elif action == "check":
    print(json.dumps({"cases":340}))
"#,
    )
    .unwrap();
    t.launch();
    let call = |name, args| {
        let (status, response) = t.request(
            "/mcp",
            "Accept: application/json, text/event-stream\r\n",
            Some(json!({"jsonrpc":"2.0","id":2,"method":"tools/call",
                "params":{"name":name,"arguments":args}})),
        );
        assert_eq!(status, 200, "{name}: {response}");
        assert!(response["error"].is_null(), "{name}: {response}");
        response["result"].clone()
    };
    let ok = |name, args| {
        let result = call(name, args);
        assert_eq!(result["isError"], false, "{name}: {result}");
        result["structuredContent"].clone()
    };
    let compile = json!({"worker":"existing", "program":"sort", "idempotency_key":"compile-sort"});
    assert_eq!(
        call("compile_program", compile.clone())["structuredContent"]["status"],
        409
    );
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"existing",
        "revision":0,"enabled":true})
        ))
        .0,
        200
    );
    let row = ok("compile_program", compile.clone());
    assert_eq!(
        call(
            "compile_program",
            json!({"worker":"existing", "program":"histogram",
            "idempotency_key":"compile-sort"})
        )["structuredContent"]["status"],
        409
    );
    assert_eq!(row["kind"], "compiled_program");
    assert_eq!(row["provenance"]["method"], "compile_program");
    assert_eq!(ok("compile_program", compile)["id"], row["id"]);
    assert!(
        ok("list_models", json!({}))["models"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let (id, pin) = (&row["id"], &row["sha256"]);
    let run = json!({"id":id, "sha256":pin, "tokens":[2,0,2,1], "idempotency_key":"run-sort"});
    assert_eq!(
        ok("run_program", run.clone())["output"],
        json!([0, 1, 2, 2])
    );
    let invalid = json!({"id":id,"sha256":pin,"tokens":[4],"idempotency_key":"run-invalid"});
    assert_eq!(
        call("run_program", invalid)["structuredContent"]["status"],
        400
    );
    let checks = |key| {
        ok(
            "run_checks",
            json!({"id":id,"sha256":pin,"idempotency_key":key}),
        )
    };
    let bundle = checks("check-sort")["check_bundle"].clone();
    assert_eq!(bundle["integrity"]["status"], "passed");
    assert_eq!(bundle["smoke"]["status"], "passed");
    assert_eq!(bundle["program_correctness"]["status"], "passed");
    assert_eq!(bundle["program_cases"], 340);
    assert_eq!(bundle["behavior"]["status"], "not_run");
    let approve = format!("/api/admin/scratch/{}/approve", id.as_str().unwrap());
    assert_eq!(t.request(&approve, "", Some(json!({"sha256":pin}))).0, 409);

    let directory = t
        .root
        .path()
        .join("state/scratch")
        .join(id.as_str().unwrap());
    fs::write(directory.join("config.json"), "{}").unwrap();
    assert_eq!(call("run_program", run)["structuredContent"]["status"], 409);
    let failed = checks("check-tampered-sort");
    assert_eq!(failed["check_bundle"]["integrity"]["status"], "failed");
    let publish = json!({"id":id,"sha256":pin,"idempotency_key":"publish-sort",
                        "acknowledge_bundle":failed["check_bundle"]["id"]});
    assert_eq!(
        call("request_publish", publish)["structuredContent"]["status"],
        409
    );
}

#[test]
fn merge_pins_inputs_and_reuses_scratch_admission() {
    let mut t = Test::new();
    let source = fixture(&t, &[1., 2., 3., 4., 5., 6.]);
    let digest = sha(&fs::read(&source).unwrap());
    t.launch();
    let (201, library) = t.request(
        "/api/models",
        "",
        Some(json!({
            "kind":"safetensors","path":source,"host":"existing"
        })),
    ) else {
        panic!("parent registration")
    };
    let mut body = json!({
        "base_id":library["id"],"base_sha256":digest,
        "algorithm":{"method":"task_arithmetic"},
        "inputs":[{"id":library["id"],"sha256":digest,"weight":1.}]
    });
    let merge = |body: &Value| {
        t.request(
            "/api/scratch/merge",
            "Idempotency-Key: merge-1\r\n",
            Some(body.clone()),
        )
    };
    assert_eq!(merge(&body).0, 409, "disabled editor");
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"existing",
        "revision":0,"enabled":true})
        ))
        .0,
        200
    );
    let (201, root) = t.request(
        "/api/scratch",
        "",
        Some(json!({
            "parent_id":library["id"],"parent_sha256":digest
        })),
    ) else {
        panic!("scratch creation")
    };
    let (201, child) = t.request(&format!("/api/scratch/{}/edits", root["id"].as_str().unwrap()),
        "", Some(json!({"operations":[{"op":"add","tensor":"w","origin":[0,1],"shape":[1,1],"values":[4.]}]})))
    else { panic!("scratch edit") };
    body["base_id"] = root["id"].clone();
    body["inputs"].as_array_mut().unwrap().push(json!({
        "id":child["id"],"sha256":"0".repeat(64),"weight":0.5
    }));
    assert_eq!(merge(&body).0, 400, "wrong merge input pin");
    body["inputs"][1]["sha256"] = child["sha256"].clone();
    let (status, output) = merge(&body);
    assert_eq!(status, 201, "{output}");
    assert_eq!(output["provenance"]["parent_id"], root["id"]);
    assert_eq!(output["provenance"]["method"], "task_arithmetic");
    assert_eq!(output["provenance"]["merge"]["inputs"], body["inputs"]);
    assert_eq!(output["delta"]["changed_tensors"], 1);
    assert_eq!(merge(&body), (200, output.clone()));
    let mut args = body.clone();
    args["idempotency_key"] = json!("merge-1");
    let (status, rpc) = t.request(
        "/mcp",
        "Accept: application/json, text/event-stream\r\nMCP-Protocol-Version: 2025-06-18\r\n",
        Some(json!({"jsonrpc":"2.0","id":1,"method":"tools/call",
            "params":{"name":"merge_models","arguments":args}})),
    );
    assert_eq!(status, 200, "{rpc}");
    assert_eq!(rpc["result"]["isError"], false, "{rpc}");
    assert_eq!(rpc["result"]["structuredContent"], output);
    body["inputs"][1]["weight"] = json!(1.);
    assert_eq!(merge(&body).0, 409, "key cannot be rebound");
    let (status, diff) = t.request(
        &format!(
            "/api/diff/tensors?a={}&b={}",
            root["id"].as_str().unwrap(),
            output["id"].as_str().unwrap()
        ),
        "",
        None,
    );
    assert_eq!(status, 200, "{diff}");
    assert_eq!(diff["tensors"][0]["changed_elements"], 1);
    assert_eq!(diff["tensors"][0]["l2"], 2.);
}

#[test]
fn private_gguf_is_bound_to_scratch_and_removed_on_discard() {
    let mut t = Test::new();
    let source = fixture(&t, &[1., 2., 3., 4., 5., 6.]);
    let digest = sha(&fs::read(&source).unwrap());
    let mock = t.root.path().join("editor-DISPATCH");
    fs::write(
        t.root.path().join("mock-check-protocol.json"),
        include_str!("../../../../api/openapi.json"),
    )
    .unwrap();
    fs::write(&mock, r#"#!/bin/sh
set -eu
case "$1" in
  start)
    body=$(cat)
    dir=${BREWERY_DISPATCH_CONFIG%/*}/worker/jobs/$3/work
    mkdir -p "$dir"
    if echo "$body" | grep -q '"kind":"scratch_checks"'; then
        printf '%s' "$body" | python3 -c 'import hashlib,json,pathlib,sys
job=json.load(sys.stdin); models=job["inputs"]
cfg=json.load(open(sys.argv[1]))["hosts"][models[0]["host"]]["config"]
grant=json.load(open(cfg))["scratch_access"]
paths=[m["path"] for m in models]
pins={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}
assert grant == {"job_id":sys.argv[2],"paths":pins}
root=pathlib.Path(sys.argv[3]); protocol=json.load((root/"mock-check-protocol.json").open())
schema=protocol["components"]["schemas"]["ScratchOfflineReport"]["properties"]
metrics=[]
for name in schema["metrics"]["items"]["properties"]["name"]["enum"]:
    numeric=name.endswith("_rank") or name=="perplexity"
    metrics.append(dict(name=name,parent=10 if numeric else 0.5,child=11 if numeric else 0.6,
                        better="neither" if "_locality_" in name else "lower" if numeric else "higher"))
result=dict(schema="brewery.scratch-behavior.v1",qualification="not release-qualified",
    fixture_sha256={k:v["const"] for k,v in schema["fixture_sha256"]["properties"].items()},
    runner_sha256="c"*64,metrics=metrics,parent_generation="Before",child_generation="After")
result.update(zip(["parent_gguf_sha256","revision_gguf_sha256","parent_sha256","revision_sha256"],
                  map(pins.__getitem__,paths),strict=True))
if (root/"reject-check-receipt").exists(): result["revision_gguf_sha256"]="0"*64
print(json.dumps(dict(state="succeeded",progress="done",result=result)))' "$BREWERY_DISPATCH_CONFIG" "$3" "${0%/*}" > "$dir/result"
        printf '%s\n' '{"state":"running","progress":"mock scratch checks"}'
        exit
    fi
    if echo "$body" | grep -q '"kind":"trace"'; then
        printf '%s' "$body" | python3 -c 'import hashlib,json,pathlib,sys
job=json.load(sys.stdin); model=job["inputs"][0]
cfg=json.load(open(sys.argv[1]))["hosts"][model["host"]]["config"]
grant=json.load(open(cfg))["scratch_access"]
sha=hashlib.sha256(pathlib.Path(model["path"]).read_bytes()).hexdigest()
assert grant == {"job_id":sys.argv[2],"paths":{model["path"]:sha}}
assert job["options"]["expected_model_sha256"] == sha
print(json.dumps({"state":"succeeded","progress":"done","result":{"kind":"trace","schema_version":1,"model_id":model["id"],"model_sha256":sha}}))' "$BREWERY_DISPATCH_CONFIG" "$3" > "$dir/result"
        printf '%s\n' '{"state":"running","progress":"mock scratch trace"}'
        exit
    fi
    echo "$body" | grep -q '"kind":"scratch_gguf"'
    printf GGUFmock > "$dir/converted.gguf"
    rev=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["inputs"][0]["path"])')
    printf '%s\n' "$(sha256sum "$rev" | cut -d' ' -f1)" > "$dir/input.sha"
    basename "$(dirname "$rev")" > "$dir/input.id"
    printf '%s\n' '{"state":"running","progress":"mock conversion"}'
    ;;
  status)
    dir=${BREWERY_DISPATCH_CONFIG%/*}/worker/jobs/$3/work
    if [ -f "$dir/result" ]; then cat "$dir/result"; exit; fi
    hash=$(sha256sum "$dir/converted.gguf" | cut -d' ' -f1)
    tool=$(printf '%064d' 1)
    if [ -f "${0%/*}/bad-conversion-pin" ]; then
        printf '%064d\n' 0 > "$dir/input.sha"
    fi
    printf '{"state":"succeeded","progress":"done","result":{"kind":"convert","gguf_type":"BF16","file_verification":{"status":"passed"},"artifact":{"kind":"gguf","path":"%s","sha256":"%s","bytes":8},"inputs":[{"sha256":"%s","id":"%s"}],"backend_receipt":{"executable_sha256":"%s","hf_exporter_sha256":"%s"}}}\n' "$dir/converted.gguf" "$hash" "$(cat "$dir/input.sha")" "$(cat "$dir/input.id")" "$tool" "$tool"
    ;;
  *) exit 1 ;;
esac
"#).unwrap();
    t.launch();
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"existing",
        "revision":0,"enabled":true})
        ))
        .0,
        200
    );
    let register = |kind, path: &std::path::Path| {
        t.request(
            "/api/models",
            "",
            Some(json!({
                "kind":kind,"path":path,"host":"existing"
            })),
        )
    };
    let (201, parent) = register("safetensors", &source) else {
        panic!("parent registration")
    };
    let (201, assets) = register("assets", source.parent().unwrap()) else {
        panic!("asset registration")
    };
    let (201, scratch) = t.request(
        "/api/scratch",
        "",
        Some(json!({
            "parent_id":parent["id"],"parent_sha256":digest
        })),
    ) else {
        panic!("scratch creation")
    };
    let id = scratch["id"].as_str().unwrap();
    let path = format!("/api/scratch/{id}");
    let convert = format!("{path}/gguf");
    let body = json!({"sha256":scratch["sha256"],"assets_id":assets["id"]});
    let bad_pin = t.root.path().join("bad-conversion-pin");
    fs::write(&bad_pin, "").unwrap();
    let (202, bad_job) = t.request(&convert, "", Some(body.clone())) else {
        panic!("conversion admission")
    };
    let bad_job_path = format!("/api/jobs/{}", bad_job["id"].as_str().unwrap());
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let (_, detail) = t.request(&bad_job_path, "", None);
        if !matches!(detail["state"].as_str(), Some("queued" | "running")) {
            assert_eq!(detail["state"], "failed", "{detail}");
            assert!(
                detail["error"]
                    .as_str()
                    .unwrap()
                    .contains("pinned revision")
            );
            break;
        }
        assert!(Instant::now() < deadline, "{detail}");
        sleep(Duration::from_millis(25));
    }
    assert!(t.request(&path, "", None).1.get("gguf").is_none());
    assert!(
        !t.root
            .path()
            .join("state/local-edit/existing/worker/jobs")
            .join(bad_job["id"].as_str().unwrap())
            .join("work/converted.gguf")
            .exists()
    );
    fs::remove_file(bad_pin).unwrap();
    let (status, job) = t.request(
        &convert,
        "Idempotency-Key: private-gguf\r\n",
        Some(body.clone()),
    );
    assert_eq!(status, 202, "{job}");
    let retry = t.request(&convert, "Idempotency-Key: private-gguf\r\n", Some(body));
    assert_eq!(retry.1["id"], job["id"]);
    let until = Instant::now() + Duration::from_secs(5);
    let job = loop {
        let (_, job) = t.request(
            &format!("/api/jobs/{}", job["id"].as_str().unwrap()),
            "",
            None,
        );
        if !matches!(job["state"].as_str(), Some("queued" | "running")) {
            break job;
        }
        assert!(Instant::now() < until, "{job}");
        sleep(Duration::from_millis(25));
    };
    assert_eq!(job["state"], "succeeded", "{job}");
    let (_, revision) = t.request(&path, "", None);
    assert_eq!(revision["gguf"]["revision_sha256"], scratch["sha256"]);
    assert_eq!(revision["gguf"]["sha256"], sha(b"GGUFmock"));
    assert_eq!(
        revision["gguf"]["converter_version"],
        format!(
            "llama-quantize sha256:{:064}; HF exporter sha256:{:064}",
            1, 1
        )
    );
    let private = t
        .root
        .path()
        .join("state/scratch")
        .join(id)
        .join("model.gguf");
    assert_eq!(fs::read(&private).unwrap(), b"GGUFmock");
    assert!(!job.to_string().contains(".gguf"));
    assert_eq!(
        t.request("/api/models", "", None)
            .1
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(register("gguf", &private).0, 400);
    let staged = t
        .root
        .path()
        .join("state/local-edit/existing/worker/jobs")
        .join(job["id"].as_str().unwrap())
        .join("work/converted.gguf");
    assert!(!staged.exists());
    assert_eq!(register("gguf", &staged).0, 400);
    assert_eq!(
        register(
            "hf",
            &t.root.path().join("state/local-edit/existing/worker/jobs")
        )
        .0,
        400
    );
    for kind in ["convert", "evaluate"] {
        assert_eq!(
            t.request(
                "/api/jobs",
                "",
                Some(json!({
                    "kind":kind,"input_ids":[id],"host":"existing"
                }))
            )
            .0,
            400
        );
    }
    let (_, library_gguf) = register("gguf", &source.with_extension("gguf"));
    let (201, art) = t.request("/api/interventions", "", Some(json!({
        "name":"Library binding","model_id":library_gguf["id"],"model_sha256":revision["gguf"]["sha256"],
        "interventions":[{"layer":0,"role":"l_out","positions":[0],"op":"zero"}]
    }))) else { panic!("Library intervention creation") };
    let trace = |options: Value| {
        t.request(
            "/api/jobs",
            "",
            Some(json!({
                "kind":"trace","input_ids":[id],"prompt":"One","options":options
            })),
        )
    };
    let mut options = json!({"layers":[0],"roles":["l_out"],"positions":[0],"top_k":3,
        "intervention_artifact":{"id":art["id"],"version":1}});
    let (status, error) = trace(options.clone());
    assert_eq!(status, 400);
    assert!(
        error["error"].as_str().unwrap().contains("different model"),
        "{error}"
    );
    let (201, scratch_art) = t.request(
        "/api/interventions",
        "",
        Some(json!({
            "name":"Scratch binding","model_id":id,"model_sha256":revision["gguf"]["sha256"],
            "interventions":art["interventions"]
        })),
    ) else {
        panic!("scratch intervention creation")
    };
    let (201, baseline) = t.request(
        "/api/jobs",
        "",
        Some(json!({
            "kind":"trace","input_ids":[library_gguf["id"]],"prompt":"One",
            "options":{"layers":[0],"roles":["l_out"],"positions":[0],"top_k":3}
        })),
    ) else {
        panic!("Library trace admission")
    };
    let baseline_path = format!("/api/jobs/{}", baseline["id"].as_str().unwrap());
    loop {
        let (_, prior) = t.request(&baseline_path, "", None);
        if prior["state"] == "succeeded" {
            break;
        }
        assert!(Instant::now() < until, "{prior}");
        sleep(Duration::from_millis(25));
    }
    let (status, error) = trace(
        json!({"layers":[0],"roles":["l_out"],"positions":[0],"top_k":3,
        "activation_patches":[{"source_job_id":baseline["id"],"source_prompt":0,
                               "layer":0,"role":"l_out","positions":[0]}]}),
    );
    assert_eq!(status, 400);
    assert!(
        error["error"].as_str().unwrap().contains("different model"),
        "{error}"
    );
    options["intervention_artifact"]["id"] = scratch_art["id"].clone();
    let (status, trace_job) = trace(options);
    assert_eq!(status, 201, "{trace_job}");
    loop {
        let (_, job) = t.request(
            &format!("/api/jobs/{}", trace_job["id"].as_str().unwrap()),
            "",
            None,
        );
        if !matches!(job["state"].as_str(), Some("queued" | "running")) {
            assert_eq!(job["state"], "succeeded", "{job}");
            break;
        }
        assert!(Instant::now() < until, "{job}");
        sleep(Duration::from_millis(25));
    }
    let checks = format!("{path}/checks");
    let (status, check_job) = t.request(&checks, "", Some(json!({"sha256":scratch["sha256"]})));
    assert_eq!(status, 202, "{check_job}");
    loop {
        let (_, job) = t.request(
            &format!("/api/jobs/{}", check_job["id"].as_str().unwrap()),
            "",
            None,
        );
        if !matches!(job["state"].as_str(), Some("queued" | "running")) {
            assert_eq!(job["state"], "succeeded", "{job}");
            break;
        }
        assert!(Instant::now() < until, "{job}");
        sleep(Duration::from_millis(25));
    }
    let (200, bundle) = t.request(&checks, "", None) else {
        panic!("check bundle")
    };
    assert_eq!(bundle["integrity"]["status"], "passed");
    assert_eq!(bundle["smoke"]["status"], "passed");
    assert_eq!(bundle["behavior"]["status"], "passed");
    assert_eq!(bundle["measured_report"]["runner_sha256"], "c".repeat(64));
    assert_eq!(
        bundle["measured_report"]["revision_gguf_sha256"],
        revision["gguf"]["sha256"]
    );
    assert_eq!(
        bundle["measured_report"]["revision_sha256"],
        scratch["sha256"]
    );
    assert_eq!(
        bundle["measured_report"]["parent_sha256"],
        scratch["provenance"]["parent_sha256"]
    );
    let (200, diff) = t.request(
        &format!(
            "/api/diff/behavior?a={}&b={id}",
            parent["id"].as_str().unwrap()
        ),
        "",
        None,
    ) else {
        panic!("measured behavior diff")
    };
    assert_eq!(diff["metrics"].as_array().unwrap().len(), 17);
    assert_eq!(
        t.request(
            &checks,
            "",
            Some(json!({"sha256":scratch["sha256"],"offline_report":bundle["measured_report"]}))
        )
        .0,
        422
    );
    fs::write(t.root.path().join("reject-check-receipt"), "").unwrap();
    let (202, mismatched) = t.request(&checks, "", Some(json!({"sha256":scratch["sha256"]})))
    else {
        panic!("check job")
    };
    loop {
        let (_, job) = t.request(
            &format!("/api/jobs/{}", mismatched["id"].as_str().unwrap()),
            "",
            None,
        );
        if !matches!(job["state"].as_str(), Some("queued" | "running")) {
            assert_eq!(job["state"], "failed", "{job}");
            assert!(
                job["error"].as_str().unwrap().contains("pinned inputs"),
                "{job}"
            );
            break;
        }
        assert!(Instant::now() < until, "{job}");
        sleep(Duration::from_millis(25));
    }
    assert_eq!(t.request(&checks, "", None), (200, bundle));
    assert_eq!(
        t.request(&format!("{path}/discard"), "", Some(json!({}))).0,
        200
    );
    assert!(!private.exists());
}

#[test]
fn mcp_scratch_workflow_stops_at_publish_request() {
    let mut t = Test::new();
    let source = fixture(&t, &[1., 2., 3., 4., 5., 6.]);
    let hash = sha(&fs::read(&source).unwrap());
    t.launch();
    let (status, parent) = t.request(
        "/api/models",
        "",
        Some(json!({"host":"existing","kind":"safetensors","path":source})),
    );
    assert_eq!(status, 201, "{parent}");

    let rpc = |method, params| {
        let (status, response) = t.request(
            "/mcp",
            "Accept: application/json, text/event-stream\r\n",
            Some(json!({"jsonrpc":"2.0","id":1,"method":method,"params":params})),
        );
        assert_eq!(status, 200, "{method}: {response}");
        assert_eq!(response["id"], 1);
        response
    };
    let init = rpc(
        "initialize",
        json!({"protocolVersion":"2025-06-18","capabilities":{},
        "clientInfo":{"name":"mock test","version":"1"}}),
    );
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert!(init["result"]["capabilities"]["tools"].is_object());
    let list = rpc("tools/list", json!({}));
    let listed = list["result"]["tools"].as_array().unwrap();
    let mut names: Vec<_> = listed
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "apply_edit",
            "compile_program",
            "convert_scratch_gguf",
            "create_intervention",
            "create_scratch",
            "diff_behavior",
            "diff_tensors",
            "diff_traces",
            "discard_scratch",
            "export_control_vector",
            "get_checks",
            "get_intervention",
            "get_job",
            "get_scratch",
            "list_interventions",
            "list_models",
            "list_scratch",
            "merge_models",
            "new_intervention_version",
            "read_feature_trace",
            "read_trace",
            "register_model",
            "request_publish",
            "run_checks",
            "run_program",
            "start_trace",
            "steer_feature",
        ]
    );
    for tool in listed {
        assert_eq!(tool["inputSchema"]["type"], "object", "{tool}");
    }
    let apply = listed
        .iter()
        .find(|tool| tool["name"] == "apply_edit")
        .unwrap();
    assert!(apply["inputSchema"]["properties"]["operations"]["items"]["oneOf"].is_array());
    assert!(
        apply["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .contains(&json!("idempotency_key"))
    );
    assert!(!list.to_string().contains("#/components/"));

    let call = |name, arguments| rpc("tools/call", json!({"name":name,"arguments":arguments}));
    let ok = |name, arguments| {
        let reply = call(name, arguments);
        assert_eq!(reply["result"]["isError"], false, "{reply}");
        let result = &reply["result"];
        let text: Value =
            serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, result["structuredContent"]);
        text
    };
    assert_eq!(ok("list_models", json!({}))["models"], json!([parent]));
    let registration = json!({"idempotency_key":"mcp-model","host":"existing","kind":"assets","path":source.parent()});
    let registered = ok("register_model", registration.clone());
    assert_eq!(
        ok("register_model", registration.clone())["id"],
        registered["id"]
    );
    let mut different = registration;
    different["kind"] = json!("gguf");
    assert_eq!(
        call("register_model", different)["result"]["structuredContent"]["status"],
        409
    );
    let mut create = json!({"parent_id":parent["id"],"parent_sha256":hash});
    assert_eq!(
        call("create_scratch", create.clone())["error"]["code"],
        -32602
    );
    create["idempotency_key"] = json!("mcp-create");
    assert_eq!(
        call("create_scratch", create.clone())["result"]["structuredContent"]["status"],
        409
    );
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"existing",
            "revision":0,"enabled":true})
        ))
        .0,
        200
    );
    let scratch = ok("create_scratch", create.clone());
    assert_eq!(ok("create_scratch", create.clone()), scratch);
    create["parent_sha256"] = json!("0".repeat(64));
    assert_eq!(
        call("create_scratch", create)["result"]["structuredContent"]["status"],
        409
    );

    let edit = ok(
        "apply_edit",
        json!({"id":scratch["id"],"idempotency_key":"mcp-edit",
        "operations":[{"op":"add","tensor":"w","origin":[0,1],"shape":[1,1],"values":[6.]}]}),
    );
    let check = ok(
        "run_checks",
        json!({"id":edit["id"],"idempotency_key":"mcp-check","sha256":edit["sha256"]}),
    );
    assert_eq!(check["check_bundle"]["integrity"]["status"], "passed");
    assert_eq!(
        ok("get_checks", json!({"id":edit["id"]})),
        check["check_bundle"]
    );
    let detail = ok("get_scratch", json!({"id":edit["id"]}));
    assert_eq!(detail["delta"]["changed_tensors"], 1);
    assert_eq!(detail["provenance"]["parent_id"], scratch["id"]);
    assert_eq!(detail["check_bundle"]["id"], check["check_bundle"]["id"]);

    let pair = json!({"a":scratch["id"],"b":edit["id"]});
    let diff = ok("diff_tensors", pair.clone());
    assert_eq!(diff["tensors"][0]["changed_elements"], 1);
    assert_eq!(diff["qualification"], "not release-qualified");
    assert_eq!(
        ok("diff_behavior", pair)["qualification"],
        "not release-qualified"
    );
    let request = ok(
        "request_publish",
        json!({"id":edit["id"],"idempotency_key":"mcp-publish",
        "sha256":edit["sha256"],"acknowledge_bundle":check["check_bundle"]["id"]}),
    );
    assert_eq!(request["state"], "publish_requested");
    for name in [
        "approve",
        "approve_scratch",
        "apply_admin_action",
        "set_settings",
        "deploy",
        "unknown",
    ] {
        assert_eq!(
            call(name, json!({"id":edit["id"],"sha256":edit["sha256"]}))["error"]["code"],
            -32602
        );
    }
    assert_eq!(
        ok("list_models", json!({}))["models"],
        json!([parent, registered])
    );

    let leaf = ok(
        "create_scratch",
        json!({"parent_id":scratch["id"],"parent_sha256":scratch["sha256"],"idempotency_key":"mcp-leaf"}),
    );
    assert_eq!(
        ok("list_scratch", json!({}))["revisions"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    let discard = json!({"id":leaf["id"],"idempotency_key":"mcp-discard"});
    assert_eq!(ok("discard_scratch", discard.clone())["state"], "discarded");
    assert_eq!(ok("discard_scratch", discard)["state"], "discarded");
    assert_eq!(
        ok("list_scratch", json!({}))["revisions"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn diff_scratch_tensors_and_behavior() {
    let mut t = Test::new();
    let source = fixture(&t, &[1., 2., 3., 4., 5., 6.]);
    t.launch();
    let (_, parent) = t.request(
        "/api/models",
        "",
        Some(json!({"host":"existing","kind":"safetensors","path":source})),
    );
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"existing",
        "revision":0,"enabled":true})
        ))
        .0,
        200
    );
    let library = parent["id"].as_str().unwrap();
    let (status, base) = t.request(
        "/api/scratch",
        "",
        Some(json!({
            "parent_id":library,"parent_sha256":sha(&fs::read(&source).unwrap())
        })),
    );
    assert_eq!(status, 201, "{base}");
    let base = base["id"].as_str().unwrap();
    let edit = |row, col, add| {
        let (status, child) = t.request(&format!("/api/scratch/{base}/edits"), "", Some(json!({
            "operations":[{"op":"add","tensor":"w","origin":[row,col],"shape":[1,1],"values":[add]}]
        })));
        assert_eq!(status, 201, "{child}");
        child["id"].as_str().unwrap().to_owned()
    };
    let child = edit(0, 1, 6.);
    let sibling = edit(1, 0, 2.);
    let report = |a: &str, b: &str| {
        let (status, diff) = t.request(&format!("/api/diff/tensors?a={a}&b={b}"), "", None);
        assert_eq!(status, 200, "{diff}");
        diff
    };
    let direct = report(base, &child);
    assert_eq!(direct["tensors"], report(library, &child)["tensors"]);
    let w = &direct["tensors"][0];
    for (field, value) in [
        ("name", json!("w")),
        ("shape", json!([2, 3])),
        ("dtype", json!("F32")),
        ("changed_elements", json!(1)),
        ("bounds", json!({"start":[0,1],"end":[1,2]})),
        ("l2", json!(6.)),
        ("linf", json!(6.)),
        (
            "low_rank",
            json!({"sample_shape":[1,1],"singular_values":[6.],"sample_rank":1}),
        ),
    ] {
        assert_eq!(w[field], value, "{field}");
    }
    let ratio = |diff: &Value| diff["tensors"][0]["relative_frobenius"].as_f64().unwrap();
    assert!((ratio(&direct) - 6. / 91f64.sqrt()).abs() < 1e-12);
    assert!((ratio(&report(&child, base)) - 6. / 151f64.sqrt()).abs() < 1e-12);
    assert_eq!(report(base, base)["tensors"], json!([]));
    let cross = report(&child, &sibling);
    assert_eq!(cross["tensors"][0]["changed_elements"], 2);
    assert_eq!(
        cross["tensors"][0]["bounds"],
        json!({"start":[0,0],"end":[2,2]})
    );
    assert_eq!(
        cross["tensors"][0]["low_rank"]["singular_values"],
        json!([6., 2.])
    );

    let artifact = t.root.path().join("state/scratch").join(&child);
    let path = format!("/api/diff/behavior?a={base}&b={child}");
    assert_eq!(t.request(&path, "", None).0, 404);
    let file = artifact.join("behavior.v1.json");
    let mut input = json!({"schema_version":1,"parent_id":base,"child_id":child,"metrics":[
        {"name":"locality","parent":1.0,"child":0.75,"better":"higher"},
        {"name":"ppl","parent":12.0,"child":12.1,"better":"lower","threshold":0.2},
        {"name":"output","parent":"Paris","child":"Rome"},
        {"name":"gsm8k","parent":null,"child":0.5}
    ]});
    fs::write(&file, input.to_string()).unwrap();
    let (status, behavior) = t.request(&path, "", None);
    assert_eq!(status, 200, "{behavior}");
    assert_eq!(behavior["qualification"], "not release-qualified");
    let metrics = behavior["metrics"].as_array().unwrap();
    for (name, delta, flag) in [
        ("locality", Some(-0.25), "regressed"),
        ("output", None, "changed"),
        ("ppl", Some(0.1), "within_threshold"),
        ("gsm8k", None, "missing"),
    ] {
        let m = metrics.iter().find(|m| m["name"] == name).unwrap();
        assert_eq!(m["flag"], flag);
        if let Some(delta) = delta {
            assert!((m["delta"].as_f64().unwrap() - delta).abs() < 1e-12);
        } else {
            assert!(m["delta"].is_null());
        }
    }
    input["schema_version"] = json!(2);
    fs::write(&file, input.to_string()).unwrap();
    assert_eq!(t.request(&path, "", None).0, 422);
    fs::write(&file, vec![b' '; (256 << 10) + 1]).unwrap();
    assert_eq!(t.request(&path, "", None).0, 413);

    // An earlier cached report does not hide damage to the delta or a mutable Library parent.
    let delta = artifact.join("delta.bin");
    let mut bytes = fs::read(&delta).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::set_permissions(&delta, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&delta, bytes).unwrap();
    assert_eq!(
        t.request(&format!("/api/diff/tensors?a={base}&b={child}"), "", None)
            .0,
        422
    );
    let mut bytes = fs::read(&source).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&source, bytes).unwrap();
    assert_eq!(
        t.request(&format!("/api/diff/tensors?a={library}&b={base}"), "", None)
            .0,
        422
    );
}

#[test]
fn scratch_integrity_cannot_be_waived_for_nonfinite_tensors() {
    let mut t = Test::new();
    let source = fixture(&t, &[1., 2., 3., 4., 5., f32::NAN]);
    let hash = sha(&fs::read(&source).unwrap());
    t.launch();
    let (status, parent) = t.request(
        "/api/models",
        "",
        Some(json!({"host":"existing","kind":"safetensors","path":source})),
    );
    assert_eq!(status, 201, "{parent}");
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"existing",
            "revision":0,"enabled":true})
        ))
        .0,
        200
    );
    let (status, scratch) = t.request(
        "/api/scratch",
        "",
        Some(json!({"parent_id":parent["id"],"parent_sha256":hash})),
    );
    assert_eq!(status, 201, "{scratch}");
    let id = scratch["id"].as_str().unwrap();
    let checks = format!("/api/scratch/{id}/checks");
    let (status, checked) = t.request(&checks, "", Some(json!({"sha256":hash})));
    assert_eq!(status, 200, "{checked}");
    assert_eq!(checked["check_bundle"]["integrity"]["status"], "failed");
    assert!(
        checked["check_bundle"]["integrity"]["reason"]
            .as_str()
            .unwrap()
            .contains("non-finite")
    );
    assert_eq!(
        t.request(
            &format!("/api/scratch/{id}/publish-request"),
            "",
            Some(json!({"sha256":hash,"acknowledge_bundle":checked["check_bundle"]["id"]})),
        )
        .0,
        409
    );
}

#[test]
fn scratch_edits_keep_parents_and_require_separate_publish_approval() {
    let mut t = Test::new();
    let source = fixture(&t, &[1., 2., 3., 4., 5., 6.]);
    let original = fs::read(&source).unwrap();
    let parent_hash = sha(&original);
    t.launch();
    let (status, parent) = t.request(
        "/api/models",
        "",
        Some(json!({"host":"existing","kind":"safetensors","path":source})),
    );
    assert_eq!(status, 201, "{parent}");
    let create = json!({"parent_id":parent["id"],"parent_sha256":parent_hash});
    let path = "/api/scratch";
    assert_eq!(t.request(path, "", Some(create.clone())).0, 409);
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"existing",
            "revision":0,"enabled":true})
        ))
        .0,
        200
    );
    let mut invalid = create.clone();
    invalid["parent_sha256"] = json!("0".repeat(64));
    assert_eq!(t.request(path, "", Some(invalid.clone())).0, 400);
    let key = "Idempotency-Key: scratch.create\r\n";
    let (status, scratch) = t.request(path, key, Some(create.clone()));
    assert_eq!(status, 201, "{scratch}");
    assert_eq!(scratch["kind"], "scratch");
    assert_eq!(scratch["sha256"], parent_hash);
    assert_eq!(t.request(path, key, Some(create)), (200, scratch.clone()));
    assert_eq!(t.request(path, key, Some(invalid)).0, 409);

    let id = scratch["id"].as_str().unwrap();
    let root = t.root.path().join("state/scratch");
    let model = |id: &str| root.join(id).join("model.safetensors");
    assert_eq!(fs::read(model(id)).unwrap(), original);
    let edit = format!("{path}/{id}/edits");
    let mut request = json!({"operations":[
        {"op":"set","tensor":"w","origin":[0,1],"shape":[1,2],"values":[8.,9.]},
        {"op":"add","tensor":"w","origin":[1,0],"shape":[1,2],"values":[1.,2.]},
        {"op":"low_rank","tensor":"w","u":[2.,1.],"v":[3.,0.,2.],"scale":0.5},
        {"op":"orthogonalize","tensor":"w","axis":"rows","direction":[1.,0.,0.]}
    ]});
    request["operations"][3]["direction"] = json!([0., 0., 0.]);
    assert_eq!(t.request(&edit, "", Some(request.clone())).0, 400);
    request["operations"][3]["direction"] = json!([1., 0., 0.]);
    let key = "Idempotency-Key: scratch.edit\r\n";
    let (status, child) = t.request(&edit, key, Some(request.clone()));
    assert_eq!(status, 201, "{child}");
    assert_eq!(t.request(&edit, key, Some(request)), (200, child.clone()));
    assert_eq!(child["provenance"]["parent_id"], id);
    assert_eq!(child["provenance"]["parent_sha256"], parent_hash);
    assert_eq!(
        child["provenance"]["tensors"],
        json!([{"name":"w","shape":[2,3],"dtype":"F32"}])
    );
    assert_eq!(child["qualification"], "not release-qualified");
    assert_eq!(sha(&fs::read(&source).unwrap()), parent_hash);
    assert_eq!(sha(&fs::read(model(id)).unwrap()), parent_hash);
    let child_id = child["id"].as_str().unwrap();
    let output = fs::read(model(child_id)).unwrap();
    let values: Vec<_> = output[output.len() - 24..]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(values, [0., 8., 11., 0., 7., 7.]);
    assert_eq!(sha(&output), child["sha256"]);
    let delta_file = root.join(child_id).join("delta.bin");
    let restored = t.root.path().join("reverted.safetensors");
    brewery_worker::weight_source::scratch::revert(&model(child_id), &delta_file, &restored)
        .unwrap();
    assert_eq!(fs::read(&restored).unwrap(), original);
    assert!(
        brewery_worker::weight_source::scratch::revert(&model(child_id), &delta_file, &source)
            .is_err()
    );
    assert_eq!(sha(&fs::read(&source).unwrap()), parent_hash);
    let corrupt = t.root.path().join("corrupt.bin");
    let mut damaged = fs::read(&delta_file).unwrap();
    *damaged.last_mut().unwrap() ^= 1;
    fs::write(&corrupt, damaged).unwrap();
    let rejected = t.root.path().join("rejected.safetensors");
    assert!(
        brewery_worker::weight_source::scratch::revert(&model(child_id), &corrupt, &rejected)
            .is_err()
    );
    assert!(!rejected.exists());
    let summary = format!("{path}/{child_id}/delta");
    assert_eq!(t.request(&summary, "", None).1["changed_tensors"], 1);
    assert_eq!(
        t.request("/api/state", "", None).1["models"],
        json!([parent])
    );
    let alias = json!({"host":"existing","kind":"safetensors","path":model(child_id)});
    assert_eq!(t.request("/api/models", "", Some(alias)).0, 400);
    let link = t.root.path().join("alias.safetensors");
    std::os::unix::fs::symlink(model(child_id), &link).unwrap();
    assert_eq!(
        t.request(
            "/api/models",
            "",
            Some(json!({"host":"existing","kind":"safetensors","path":link}))
        )
        .0,
        400
    );
    let approval = format!("/api/admin/scratch/{child_id}/approve");
    let review = json!({"sha256":child["sha256"]});
    assert_eq!(t.request(&approval, "", Some(review.clone())).0, 409);
    assert_eq!(
        t.request(&format!("{path}/{id}/discard"), "", Some(json!({})))
            .0,
        409
    );
    let requested = format!("{path}/{child_id}/publish-request");
    assert_eq!(t.request(&requested, "", Some(review.clone())).0, 409);
    let checks = format!("{path}/{child_id}/checks");
    assert_eq!(
        t.request(&checks, "", Some(json!({"sha256": "0".repeat(64)})))
            .0,
        409
    );
    let key = "Idempotency-Key: scratch.checks\r\n";
    let (status, checked) = t.request(&checks, key, Some(review.clone()));
    assert_eq!(status, 200, "{checked}");
    let bundle = &checked["check_bundle"];
    assert_eq!(bundle["schema"], "brewery.scratch-checks.v1");
    assert_eq!(bundle["revision_sha256"], child["sha256"]);
    assert_eq!(bundle["parent_sha256"], parent_hash);
    assert_eq!(bundle["integrity"]["status"], "passed");
    assert_eq!(bundle["smoke"]["status"], "not_run");
    assert_eq!(t.request(&checks, "", None), (200, bundle.clone()));
    assert_eq!(
        t.request(&checks, key, Some(review.clone())),
        (200, checked.clone())
    );
    let sidecar: Value =
        serde_json::from_slice(&fs::read(root.join(child_id).join("behavior.v1.json")).unwrap())
            .unwrap();
    assert_eq!(
        sidecar,
        json!({"schema_version":1,"parent_id":id,"child_id":child_id,"metrics":[]})
    );
    assert_eq!(
        t.request(
            &checks,
            "",
            Some(
                json!({"sha256":child["sha256"],"job_refs":[{"job_id":"missing","role":"child"}]})
            )
        )
        .0,
        422
    );
    assert_eq!(t.request(&requested, "", Some(review.clone())).0, 409);
    let (status, checked_again) = t.request(&checks, "", Some(review.clone()));
    assert_eq!(status, 200, "{checked_again}");
    let latest = &checked_again["check_bundle"];
    assert_eq!(latest["behavior"]["status"], "not_run");
    assert_eq!(
        t.request(
            &requested,
            "",
            Some(json!({
                "sha256":child["sha256"], "acknowledge_bundle":bundle["id"]
            }))
        )
        .0,
        409
    );
    assert_eq!(t.request(&requested, "", Some(review.clone())).0, 409);
    let ack = json!({"sha256":child["sha256"],"acknowledge_bundle":latest["id"]});
    // A bundle for the original hash cannot permit publication of changed bytes.
    fs::set_permissions(model(child_id), fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(model(child_id), &original).unwrap();
    assert_eq!(t.request(&requested, "", Some(ack.clone())).0, 409);
    fs::write(model(child_id), &output).unwrap();
    assert_eq!(
        t.request(&requested, "", Some(ack)).1["state"],
        "publish_requested"
    );
    assert_eq!(
        t.request(
            &approval,
            "Origin: https://other.invalid\r\n",
            Some(review.clone())
        )
        .0,
        403
    );
    t.stop();
    t.launch();
    assert_eq!(t.request(path, "", None).1.as_array().unwrap().len(), 2);
    let (status, approved) = t.request(&approval, "Idempotency-Key: approve\r\n", Some(review));
    assert_eq!(status, 200, "{approved}");
    assert_eq!(approved["state"], "approved");
    assert_eq!(approved["check_bundle"]["id"], latest["id"]);
    assert!(approved["check_bundle"]["acknowledged_at_ms"].is_number());
    let library = t.request("/api/state", "", None).1["models"].clone();
    assert!(
        library
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == approved["library_model_id"] && m["kind"] == "safetensors")
    );

    let (_, leaf) = t.request(
        path,
        "",
        Some(json!({"parent_id":child_id,"parent_sha256":child["sha256"]})),
    );
    let leaf_id = leaf["id"].as_str().unwrap();
    assert!(model(leaf_id).is_file());
    let discard = format!("{path}/{leaf_id}/discard");
    let key = "Idempotency-Key: scratch.discard\r\n";
    let (status, discarded) = t.request(&discard, key, Some(json!({})));
    assert_eq!(status, 200, "{discarded}");
    assert_eq!(t.request(&discard, key, Some(json!({}))), (200, discarded));
    assert!(!model(leaf_id).exists());
    assert_eq!(
        t.request(&format!("{path}/{leaf_id}/delta"), "", None).0,
        404
    );
    assert_eq!(t.request(path, "", None).1.as_array().unwrap().len(), 2);
    assert_eq!(fs::read(model(child_id)).unwrap(), output);
}
