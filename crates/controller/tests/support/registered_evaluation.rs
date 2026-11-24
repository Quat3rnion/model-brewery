use super::*;

#[test]
fn registered_evaluation_requires_operator_and_lists_only_metadata() {
    let mut app = Fixture::new();
    let credentials = app.path("evaluation-operator.json");
    let token = "a".repeat(64);
    fs::write(
        &credentials,
        json!({"operator":"test-operator","token":token}).to_string(),
    )
    .unwrap();
    fs::set_permissions(&credentials, fs::Permissions::from_mode(0o600)).unwrap();
    app.launch_with(&[("BREWERY_EVALUATION_OPERATOR_FILE", credentials)]);
    let source = |name: &str| {
        json!({
            "url": format!("https://example.test/repo/{}/{name}.jsonl", "a".repeat(40)),
            "sha256":"d".repeat(64), "size":128, "license":"CC0-1.0", "format":"jsonl",
            "field":"question", "system":"Name two garden plants."
        })
    };
    let data = json!({
        "name":"garden", "positive":source("positive"), "negative":source("negative"),
        "sizes":{"tune":1,"final_gate":1},
        "pins":{"tune":"b".repeat(64),"final_gate":"c".repeat(64)},
        "classifier":{"max_tokens":12,"classes":[
            {"name":"context","kind":"excluded","prefixes":["About:"]},
            {"name":"color","kind":"counts","substrings":["green"]}
        ]}
    });
    let register =
        |body, header| app.request("POST", "/api/admin/evaluation-sets", Some(body), header);
    assert_eq!(register(data.clone(), "").0, 403);
    assert_eq!(
        register(data.clone(), "Authorization: Bearer wrong\r\n").0,
        403
    );
    let auth = format!("Authorization: Bearer {token}\r\n");
    let mut bad = data.clone();
    bad["positive"]["url"] = json!("https://example.test/repo/main/positive.jsonl");
    assert_eq!(register(bad, &auth).0, 400);
    let mut bad = data.clone();
    bad["classifier"]["max_tokens"] = json!(129);
    assert_eq!(register(bad, &auth).0, 400);
    assert_eq!(register(data.clone(), &auth).0, 201);
    assert_eq!(register(data, &auth).0, 409);
    let (status, listed) = app.request("GET", "/api/evaluation-sets", None, "");
    assert_eq!(status, 200);
    let listed: Value = serde_json::from_str(&listed).unwrap();
    let item = &listed["sets"][0];
    assert_eq!(item["name"], "garden");
    assert_eq!(item["operator"], "test-operator");
    assert!(item["registered_at_ms"].as_u64().unwrap() > 0);
    assert_eq!(item["sizes"]["final_gate"], 1);
    assert!(item.get("pins").is_none());
    assert!(item.get("classifier").is_none());
    assert!(item.get("positive").is_none());
    let mcp = |name: &str| {
        let (status, reply) = app.request(
            "POST",
            "/mcp",
            Some(json!({"jsonrpc":"2.0","id":1,"method":"tools/call",
                "params":{"name":name,"arguments":{}}})),
            "Accept: application/json, text/event-stream\r\n",
        );
        assert_eq!(status, 200, "{reply}");
        serde_json::from_str::<Value>(&reply).unwrap()
    };
    assert_eq!(
        mcp("list_evaluation_sets")["result"]["structuredContent"],
        listed
    );
    assert_eq!(mcp("register_evaluation_set")["error"]["message"], "unknown tool");
    let (status, body) = app.request(
        "POST",
        "/api/scratch/00000000-0000-0000-0000-000000000000/search_ablation",
        Some(json!({
            "sha256":"a".repeat(64), "direction_id":"unused", "assets_id":"unused",
            "seed":1, "trial_budget":2, "max_children":1, "disk_budget_bytes":1073741824,
            "behavior_direction":{"prompt_sets":{"slop":"a".repeat(64),
                "sycophancy":"b".repeat(64),"response_length":"c".repeat(64)}},
            "registered_marker":{"name":"garden","partition":"final_gate"}
        })),
        "",
    );
    assert_eq!(status, 422, "{body}");
}
