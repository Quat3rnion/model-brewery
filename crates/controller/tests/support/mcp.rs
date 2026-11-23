use super::*;

const ACCEPT: &str = "Accept: application/json, text/event-stream\r\n";
const VERSION: &str = "MCP-Protocol-Version: 2025-06-18\r\n";

#[test]
fn interp_benchmark_reads_the_versioned_report_without_driving_compute() {
    let mut t = Fixture::new();
    t.launch();
    let report = t.json("GET", "/api/interp/benchmark", None, 200);
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["qualification"], "not release-qualified");
    for method in ["x5", "w11"] {
        for program in ["sort", "histogram"] {
            assert!(
                report["methods"][method]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|row| row["program"] == program)
            );
        }
    }
    let reply = call(&t, "interp_benchmark", json!({}));
    assert_eq!(reply["result"]["structuredContent"], report);
    assert_eq!(
        call(&t, "interp_benchmark", json!({"program":"sort"}))["error"]["code"],
        -32602
    );
    assert!(!t.path("launches").exists());
}

pub(super) fn call(t: &Fixture, name: &str, arguments: Value) -> Value {
    let (status, body) = t.request(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":name,"arguments":arguments}})),
        &format!("{ACCEPT}{VERSION}"),
    );
    assert_eq!(status, 200, "{name}: {body}");
    serde_json::from_str(&body).unwrap()
}

#[test]
fn mcp_queue_duplicate_requires_explicit_opt_out() {
    let mut t = Fixture::new();
    fs::write(
        t.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"trace":true}}}}"#,
    )
    .unwrap();
    t.touch("hold");
    t.launch();
    let model = t.register("alpha", "gguf", "/models/trace.gguf");
    let mut args = json!({"idempotency_key":"trace-first","input_ids":[model],"prompt":"one two",
        "options":{"layers":[0],"roles":["l_out"],"positions":[0],"top_k":3}});
    let queue = |args: Value| {
        let reply = call(&t, "start_trace", args);
        assert_eq!(reply["result"]["isError"], false, "{reply}");
        reply["result"]["structuredContent"].clone()
    };
    let original = queue(args.clone());
    assert_eq!(original["kind"], "trace");
    args["idempotency_key"] = json!("trace-match");
    let duplicate = json!({"id":original["id"],"duplicate_of":true});
    assert_eq!(queue(args.clone()), duplicate);

    args["idempotency_key"] = json!("trace-false");
    args["allow_duplicate"] = json!(false);
    assert_eq!(queue(args.clone()), duplicate);
    args["idempotency_key"] = json!("trace-repeat");
    args["allow_duplicate"] = json!(true);
    let repeat = queue(args.clone());
    assert_eq!(repeat["kind"], "trace");
    assert_ne!(repeat["id"], original["id"]);
    assert_eq!(queue(args.clone()), json!({"id":repeat["id"]}));
    assert_eq!(
        t.json("GET", "/api/state", None, 200)["jobs"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    args["allow_duplicate"] = json!("true");
    assert_eq!(call(&t, "start_trace", args)["error"]["code"], -32602);
}

#[test]
fn mcp_stateless_transport_uses_workbench_guards() {
    let mut t = Fixture::new();
    t.launch();
    let ping = json!({"jsonrpc":"2.0","method":"ping","id":"probe"});
    let headers = format!(
        "{ACCEPT}{VERSION}Origin: http://{}\r\nSec-Fetch-Site: same-origin\r\n",
        t.addr.unwrap()
    );
    let (status, body) = t.request("POST", "/mcp", Some(ping.clone()), &headers);
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap(),
        json!({"jsonrpc":"2.0","id":"probe","result":{}})
    );
    for bad in [
        "Origin: http://other.invalid\r\n",
        "Origin: null\r\n",
        "Host: other.invalid\r\n",
        "Sec-Fetch-Site: cross-site\r\n",
        "Sec-Fetch-Site: same-site\r\n",
        "Sec-Fetch-Site: same-origin\r\nSec-Fetch-Site: none\r\n",
    ] {
        assert_eq!(
            t.request(
                "POST",
                "/mcp",
                Some(ping.clone()),
                &format!("{ACCEPT}{bad}")
            )
            .0,
            403,
            "{bad}"
        );
        assert_eq!(t.request("GET", "/mcp", None, bad).0, 403, "{bad}");
    }
    assert_eq!(
        t.request("GET", "/mcp", None, "Accept: text/event-stream\r\n")
            .0,
        405
    );
    assert_eq!(t.request("POST", "/mcp", None, ACCEPT).0, 415);
    assert_eq!(
        t.request("POST", "/mcp", Some(ping.clone()), "Accept: text/plain\r\n")
            .0,
        406
    );
    let bad_version = format!("{ACCEPT}MCP-Protocol-Version: 1900-01-01\r\n");
    assert_eq!(t.request("POST", "/mcp", Some(ping), &bad_version).0, 400);
    let notification = json!({"jsonrpc":"2.0","method":"notifications/initialized"});
    assert_eq!(
        t.request("POST", "/mcp", Some(notification), ACCEPT),
        (202, String::new())
    );
    let (status, body) = t.request("POST", "/mcp", Some(json!([{}])), ACCEPT);
    assert_eq!(status, 400);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
        -32600
    );
}

#[test]
fn mcp_routes_trace_and_intervention_tools_without_allowing_route_selection() {
    let mut t = Fixture::new();
    fs::write(
        t.path("dispatch.json"),
        json!({
            "hosts":{"alpha":{"local":true,"capabilities":{"trace":true}}}
        })
        .to_string(),
    )
    .unwrap();
    t.launch();
    let model = t.register("alpha", "gguf", "/models/trace.gguf");
    let ok = |name, args| {
        let reply = call(&t, name, args);
        assert_eq!(reply["result"]["isError"], false, "{name}: {reply}");
        reply["result"]["structuredContent"].clone()
    };
    let saved = ok(
        "create_intervention",
        json!({"idempotency_key":"mcp-steer","name":"Direction",
        "model_id":model,"model_sha256":"a".repeat(64),
        "interventions":[{"op":"add","layer":1,"role":"l_out","positions":[0],"vector":[1.,2.]}]}),
    );
    let second = ok(
        "new_intervention_version",
        json!({"id":saved["id"],"idempotency_key":"mcp-steer-2",
        "interventions":[{"op":"zero","layer":0,"role":"l_out","positions":[0]}]}),
    );
    assert_eq!(second["version"], 2);
    assert_eq!(
        ok("list_interventions", json!({}))["interventions"][0]["latest_version"],
        2
    );
    let first = json!({"id":saved["id"],"version":1});
    assert_eq!(ok("get_intervention", first.clone()), saved);
    let url = ok("export_control_vector", first);
    let (status, gguf) = t.bytes("GET", url["url"].as_str().unwrap());
    assert_eq!(status, 200);
    assert!(gguf.starts_with(b"GGUF"));
    let invalid = call(
        &t,
        "export_control_vector",
        json!({"id":saved["id"],"version":2}),
    );
    assert_eq!(invalid["result"]["isError"], true, "{invalid}");
    assert_eq!(invalid["result"]["structuredContent"]["status"], 400);

    let mut start = json!({"idempotency_key":"mcp-trace","input_ids":[model],"prompt":"one two",
        "options":{"layers":[0],"roles":["l_out"],"positions":[0],"top_k":3}});
    start["kind"] = json!("convert");
    assert_eq!(
        call(&t, "start_trace", start.clone())["error"]["code"],
        -32602
    );
    start.as_object_mut().unwrap().remove("kind");
    let queued = ok("start_trace", start.clone());
    assert_eq!(queued["kind"], "trace");
    assert_eq!(ok("start_trace", start.clone())["id"], queued["id"]);
    let wait = |id: &Value| ok("get_job", json!({"id":id,"wait_ms":4000}));
    assert_eq!(wait(&queued["id"])["state"], "succeeded");
    let manifest = json!({"id":queued["id"],"view":"manifest","prompt":0});
    assert_eq!(ok("read_trace", manifest)["schema_version"], 1);
    let lens = json!({"id":queued["id"],"view":"logit_lens","layer":0,"start":0,"end":1});
    assert_eq!(ok("read_trace", lens)["rows"][0]["layer"], 0);
    let logits = json!({"id":queued["id"],"view":"model_logits","start":0,"end":1});
    assert_eq!(ok("read_trace", logits)["rows"][0]["top"][0]["token_id"], 1);
    let mut slice = json!({"id":queued["id"],"view":"tensor","layer":0,"role":"l_out"});
    let too_large = call(&t, "read_trace", slice.clone());
    assert_eq!(too_large["result"]["isError"], true, "{too_large}");
    assert_eq!(too_large["result"]["structuredContent"]["status"], 413);
    slice["start"] = json!(0);
    slice["end"] = json!(1);
    assert_eq!(
        ok("read_trace", slice.clone())["rows"][0]["data_base64"],
        "AACAPw=="
    );
    slice["role"] = json!("l_out&prompt=1");
    assert_eq!(call(&t, "read_trace", slice)["error"]["code"], -32602);
    assert_eq!(
        call(&t, "get_job", json!({"id":"../admin/state"}))["error"]["code"],
        -32602
    );

    start["idempotency_key"] = json!("mcp-trace-edit");
    start["options"]["interventions"] = second["interventions"].clone();
    let edited = ok("start_trace", start);
    assert_eq!(wait(&edited["id"])["state"], "succeeded");
    let diff = ok(
        "diff_traces",
        json!({"a":queued["id"],"b":edited["id"],"layer":0,"start":0,"end":1}),
    );
    assert_eq!(diff["activations"][0]["l2"], 1.);
    assert_eq!(diff["qualification"], "not release-qualified");
    assert_eq!(t.launches().len(), 2);
}
