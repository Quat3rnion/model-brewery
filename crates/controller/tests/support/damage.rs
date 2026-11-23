use super::*;

#[test]
fn quantization_damage_rejects_prompts_and_returns_only_an_unapplied_proposal() {
    let mut app = Fixture::new();
    fs::write(
        app.path("dispatch.json"),
        r#"{"hosts":{"alpha":{"local":true,"capabilities":{"quantization_damage":true}}}}"#,
    )
    .unwrap();
    app.launch();
    let a = app.register("alpha", "gguf", "/models/a.gguf");
    let b = app.register("alpha", "gguf", "/models/b.gguf");
    let options = json!({"reference_sha256":"a".repeat(64),"quantized_sha256":"b".repeat(64),"gguf_type":"Q4_K_M"});
    app.json(
        "POST",
        "/api/jobs",
        Some(json!({"kind":"quantization_damage","input_ids":[a,b],
        "options":options,"prompt":"ignored?"})),
        400,
    );
    let result = mcp::call(
        &app,
        "quantization_damage_map",
        json!({"input_ids":[a,b],
        "options":options,"idempotency_key":"damage-map"}),
    );
    assert_eq!(result["result"]["isError"], false, "{result}");
    let id = result["result"]["structuredContent"]["id"]
        .as_str()
        .unwrap();
    let job = app.await_job(id, |j| j["state"] == "succeeded");
    let path = format!("/api/jobs/{id}/quantization-damage");
    let artifact = app.json("GET", &path, None, 200);
    assert_eq!(artifact, job["result"]);
    assert_eq!(artifact["recipe_proposal"]["applied"], false);
    assert_eq!(artifact["recipe_proposal"]["profile"]["preset"], "Q4_K_M");
    app.touch("damage-bad-receipt");
    let mut options = options;
    options["swaps"] = json!(1);
    let id = app.enqueue(json!({"kind":"quantization_damage","input_ids":[a,b],
        "options":options}));
    app.await_job(&id, |j| j["state"] == "failed");
    app.json(
        "GET",
        &format!("/api/jobs/{id}/quantization-damage"),
        None,
        404,
    );
}
