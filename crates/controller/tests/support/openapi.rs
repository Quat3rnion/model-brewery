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
        ("post", "/api/models", "createModel"),
        ("delete", "/api/models/{id}", "deleteModel"),
        ("post", "/api/prompts", "createPrompt"),
        ("post", "/api/jobs", "createJob"),
        ("get", "/api/jobs/{id}", "getJob"),
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
        ("post", "/api/admin/actions", "applyAdminAction"),
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
                                .all(|variant| variant["additionalProperties"] == false))
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
