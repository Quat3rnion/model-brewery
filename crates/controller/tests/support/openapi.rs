use super::{Fixture, Read, Value};
use std::collections::BTreeSet;

const CONTRACT: &str = include_str!("../../../../api/openapi.json");

#[test]
fn discovery_serves_json_under_the_loopback_guard() {
    let mut app = Fixture::new();
    app.launch();
    let mut conn = app.open_request("GET", "/api/openapi.json", None, "");
    let mut response = String::new();
    conn.read_to_string(&mut response).unwrap();
    let (head, body) = response.split_once("\r\n\r\n").unwrap();
    assert_eq!(head.split_whitespace().nth(1), Some("200"), "{head}");
    let headers = head.to_ascii_lowercase();
    assert!(
        headers
            .lines()
            .any(|line| line == "content-type: application/json")
    );
    assert!(
        headers
            .lines()
            .any(|line| line == "cache-control: no-store")
    );
    assert_eq!(body, CONTRACT);
    let (status, _) = app.request(
        "GET",
        "/api/openapi.json",
        None,
        "Origin: http://foreign.invalid\r\n",
    );
    assert_eq!(status, 403);
}

#[test]
fn contract_covers_the_controller_operations_and_resolves_local_refs() {
    let contract: Value = serde_json::from_str(CONTRACT).unwrap();
    assert_eq!(contract["openapi"], "3.1.0");
    let operations = contract["paths"]
        .as_object()
        .unwrap()
        .iter()
        .flat_map(|(path, item)| {
            item.as_object()
                .unwrap()
                .iter()
                .map(move |(method, operation)| {
                    let id = operation["operationId"].as_str().unwrap();
                    assert!(operation["responses"].is_object(), "{id}");
                    (method.as_str(), path.as_str(), id)
                })
        })
        .collect::<BTreeSet<_>>();
    let expected = BTreeSet::from([
        ("get", "/api/openapi.json", "getOpenApi"),
        ("get", "/api/state", "getState"),
        ("get", "/api/events", "streamEvents"),
        ("post", "/api/suggestions/paths", "suggestPaths"),
        ("get", "/mcp", "getMcpStream"),
        ("post", "/mcp", "sendMcpMessage"),
        ("get", "/api/models", "listModels"),
        ("post", "/api/models", "createModel"),
        (
            "post",
            "/api/models/weight-calibration",
            "registerWeightCalibration",
        ),
        ("delete", "/api/models/{id}", "deleteModel"),
        ("get", "/api/scratch", "listScratch"),
        ("get", "/api/diff/tensors", "diffTensors"),
        ("get", "/api/diff/traces", "diffTraces"),
        ("get", "/api/diff/behavior", "diffBehavior"),
        ("post", "/api/scratch", "createScratch"),
        ("get", "/api/scratch/{id}", "getScratch"),
        ("post", "/api/scratch/{id}/edits", "editScratch"),
        ("get", "/api/scratch/{id}/delta", "getScratchDelta"),
        ("post", "/api/scratch/{id}/gguf", "convertScratchGguf"),
        ("get", "/api/scratch/{id}/checks", "getScratchChecks"),
        ("post", "/api/scratch/{id}/checks", "runScratchChecks"),
        ("post", "/api/scratch/{id}/discard", "discardScratch"),
        (
            "post",
            "/api/scratch/{id}/publish-request",
            "requestScratchPublish",
        ),
        ("post", "/api/admin/scratch/{id}/approve", "approveScratch"),
        ("post", "/api/prompts", "createPrompt"),
        ("post", "/api/jobs", "createJob"),
        ("get", "/api/jobs/{id}", "getJob"),
        ("get", "/api/jobs/{id}/trace", "getTraceManifest"),
        ("get", "/api/jobs/{id}/trace/tensor", "getTraceTensor"),
        ("get", "/api/jobs/{id}/trace/logit-lens", "getTraceLogitLens"),
        ("get", "/api/jobs/{id}/trace/model-logits", "getTraceModelLogits"),
        ("get", "/api/interventions", "listInterventions"),
        ("post", "/api/interventions", "createIntervention"),
        (
            "post",
            "/api/interventions/{id}/versions",
            "createInterventionVersion",
        ),
        (
            "get",
            "/api/interventions/{id}/versions/{version}",
            "getInterventionVersion",
        ),
        (
            "get",
            "/api/interventions/{id}/versions/{version}/gguf",
            "exportInterventionGguf",
        ),
        ("post", "/api/jobs/{id}/cancel", "cancelJob"),
        ("post", "/api/conversions", "createConversion"),
        ("get", "/api/conversions", "listConversions"),
        ("get", "/api/conversions/{id}", "getConversion"),
        ("post", "/api/conversions/{id}/cancel", "cancelConversion"),
        ("get", "/api/settings", "getSettings"),
        ("put", "/api/settings/{host}", "putSettings"),
        ("get", "/api/settings/hf-tokens", "getHfTokenStatus"),
        ("put", "/api/settings/{host}/hf-token", "setHfToken"),
        ("delete", "/api/settings/{host}/hf-token", "removeHfToken"),
        ("get", "/api/admin/state", "getAdminState"),
        (
            "post",
            "/api/admin/model-directories/plan",
            "planModelDirectory",
        ),
        (
            "post",
            "/api/admin/model-directories/confirm",
            "confirmModelDirectory",
        ),
        ("post", "/api/admin/actions", "applyAdminAction"),
        ("get", "/api/admin/ssh-identities", "listSshIdentities"),
        ("post", "/api/admin/ssh-host-key", "probeSshHostKey"),
        ("get", "/api/admin/ssh-hosts", "listSshHosts"),
        ("post", "/api/admin/ssh-hosts", "createSshHost"),
        ("put", "/api/admin/ssh-hosts/{id}", "updateSshHost"),
        ("delete", "/api/admin/ssh-hosts/{id}", "removeSshHost"),
        (
            "post",
            "/api/admin/ssh-hosts/{id}/action",
            "provisionSshHost",
        ),
        ("get", "/api/profiles/catalog", "getProfileCatalog"),
        ("get", "/api/profiles/gguf-catalog", "getGgufCatalog"),
        (
            "get",
            "/api/profiles/{revision}/export-options",
            "getProfileExportOptions",
        ),
        ("get", "/api/profiles", "listProfiles"),
        ("post", "/api/profiles", "saveProfile"),
    ]);
    assert_eq!(operations, expected);
    let ids: BTreeSet<_> = operations.iter().map(|entry| entry.2).collect();
    assert_eq!(ids.len(), operations.len());
    walk_refs(&contract, &contract);
}

fn walk_refs(document: &Value, node: &Value) {
    match node {
        Value::Object(object) => {
            if let Some(reference) = object.get("$ref") {
                let target = resolve(document, reference);
                assert!(target.is_object(), "invalid reference: {reference}");
            }
            for value in object.values() {
                walk_refs(document, value);
            }
        }
        Value::Array(values) => {
            for value in values {
                walk_refs(document, value);
            }
        }
        _ => {}
    }
}

fn resolve<'a>(document: &'a Value, reference: &Value) -> &'a Value {
    let pointer = reference
        .as_str()
        .and_then(|reference| reference.strip_prefix("#/components/"))
        .expect("references must use local components");
    document
        .pointer(&format!("/components/{pointer}"))
        .unwrap_or_else(|| panic!("unresolved component: {reference}"))
}

#[test]
fn every_live_job_kind_has_one_typed_create_variant() {
    let mut app = Fixture::new();
    app.launch();
    let snapshot = app.json("GET", "/api/state", None, 200);
    assert!(snapshot["event_cursor"].is_u64());
    let capabilities: BTreeSet<_> = snapshot["capabilities"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();

    let contract: Value = serde_json::from_str(CONTRACT).unwrap();
    let schemas = &contract["components"]["schemas"];
    let job_create = &contract["paths"]["/api/jobs"]["post"];
    let request = resolve(
        &contract,
        &job_create["requestBody"]["content"]["application/json"]["schema"]["$ref"],
    );
    let variants = request["oneOf"].as_array().unwrap();
    let mapping = request["discriminator"]["mapping"].as_object().unwrap();
    let documented: BTreeSet<_> = variants
        .iter()
        .map(|variant| {
            let schema = resolve(&contract, &variant["$ref"]);
            let kind = schema["properties"]["kind"]["const"].as_str().unwrap();
            assert_eq!(mapping[kind], variant["$ref"]);
            let options = resolve(&contract, &schema["properties"]["options"]["$ref"]);
            assert_eq!(options["type"], "object");
            assert!(
                options["additionalProperties"] == false
                    || options["maxProperties"] == 0
                    || options["oneOf"]
                        .as_array()
                        .is_some_and(|variants| !variants.is_empty()
                            && variants
                                .iter()
                                .all(|variant| {
                                    let variant = variant.get("$ref").map_or(variant, |r| resolve(&contract, r));
                                    variant["additionalProperties"] == false
                                }))
                    || options["allOf"].is_array(),
                "{kind} options are not constrained"
            );
            kind
        })
        .collect();
    let kinds: BTreeSet<_> = schemas["JobKind"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|kind| kind.as_str().unwrap())
        .collect();
    assert_eq!(documented.len(), variants.len());
    assert_eq!(documented.len(), mapping.len());
    assert_eq!(capabilities, documented);
    assert_eq!(capabilities, kinds);
}

#[test]
fn ssh_editor_setup_is_published_as_a_revisioned_admin_action_and_readback() {
    let mut app = Fixture::new();
    app.launch();
    let contract: Value = serde_json::from_str(CONTRACT).unwrap();
    let operation = &contract["paths"]["/api/admin/actions"]["post"];
    let action = resolve(
        &contract,
        &operation["requestBody"]["content"]["application/json"]["schema"]["$ref"],
    )["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .map(|variant| resolve(&contract, &variant["$ref"]))
        .find(|variant| variant["properties"]["action"]["const"] == "setup_weight_editor")
        .expect("setup_weight_editor is not published");
    let state = app.json("GET", "/api/admin/state", None, 200);
    let hosts = state["hosts"].as_array().unwrap();
    let local = hosts.iter().find(|host| host["id"] == "alpha").unwrap();
    let ssh = hosts.iter().find(|host| host["id"] == "beta").unwrap();
    let request = serde_json::json!({
        "action":"setup_weight_editor", "worker":"beta", "revision":ssh["revision"],
        "model_directory":"/models/editor", "credential":"fixture"
    });
    assert_closed_fields(action, &request);
    assert!(local["weight_editor"].get("setup").is_none());
    let setup = &ssh["weight_editor"]["setup"];
    let host_policy = &contract["components"]["schemas"]["AdminHostPolicy"];
    let setup_schema = resolve(
        &contract,
        &host_policy["properties"]["weight_editor"]["properties"]["setup"]["$ref"],
    );
    assert_closed_fields(setup_schema, setup);
    assert_eq!(setup["status"], "needs_setup");
    assert_eq!(setup["credentials"], serde_json::json!([]));
    assert!(setup["model_directory"].is_null());
    assert!(setup["reason"].is_string());
    // This fixture has an SSH definition, but no configured provision credential.
    app.json("POST", "/api/admin/actions", Some(request.clone()), 409);
    let mut bad = request;
    bad["model_directory"] = serde_json::json!("/models/../outside");
    app.json("POST", "/api/admin/actions", Some(bad), 400);
}

fn assert_closed_fields(schema: &Value, value: &Value) {
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], false);
    let required: BTreeSet<_> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|key| key.as_str().unwrap())
        .collect();
    let keys = |object: &Value| {
        object
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(keys(value), keys(&schema["properties"]));
    assert_eq!(
        required,
        value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect()
    );
}

#[test]
fn managed_gguf_layer_recipe_is_discoverable_and_round_trips() {
    use serde_json::json;

    let mut app = Fixture::new();
    std::fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{"alpha":{"local":true,"capabilities":{
            "gguf_output":true,"gguf_hf_output":true,"gguf_layer_recipe":true
        }}}})
        .to_string(),
    )
    .unwrap();
    app.launch();
    let contract: Value = serde_json::from_str(CONTRACT).unwrap();
    let discover = app.json("GET", "/api/state", None, 200);
    assert_eq!(
        discover["hosts"][0]["capabilities"]["gguf_layer_recipe"],
        true
    );
    assert!(discover["capabilities"].get("gguf_layer_recipe").is_none());
    let per_host = &contract["components"]["schemas"]["HostCapabilityMap"];
    assert!(
        per_host["propertyNames"]["anyOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|branch| branch["enum"]
                .as_array()
                .is_some_and(|names| names.contains(&json!("gguf_layer_recipe"))))
    );

    let create = &contract["paths"]["/api/conversions"]["post"]["requestBody"]["content"]["application/json"];
    let mut request = create["examples"]["ggufLayerRecipe"]["value"].clone();
    assert!(
        request.is_object(),
        "managed recipe sample is not published"
    );
    let request_schema = resolve(&contract, &create["schema"]["$ref"]);
    let recipe_options = resolve(
        &contract,
        &request_schema["then"]["then"]["properties"]["options"]["$ref"],
    );
    assert_closed_fields(recipe_options, &request["options"]);
    let recipe = resolve(&contract, &recipe_options["properties"]["layer_recipe"]["$ref"]);
    assert_closed_fields(recipe, &request["options"]["layer_recipe"]);
    let source = app.register("alpha", "hf", "/source/base");
    request["input_ids"] = json!([source]);
    let result = app.json("POST", "/api/conversions", Some(request.clone()), 201);
    assert_eq!(result["format"], "gguf");
    assert_eq!(result["host"], "alpha");
    assert_eq!(result["input_ids"], request["input_ids"]);
    assert_eq!(result["options"], request["options"]);
    let inspect = result["job_ids"]["inspect"].as_str().unwrap();
    let job = app.json("GET", &format!("/api/jobs/{inspect}"), None, 200);
    assert_eq!(job["kind"], "inspect");
    assert_eq!(job["options"], request["options"]);

    let mut mixed = request.clone();
    mixed["options"]["preset"] = json!("Q8_0");
    app.json("POST", "/api/conversions", Some(mixed), 400);
    request["host"] = json!("auto");
    app.json("POST", "/api/conversions", Some(request), 400);
}
