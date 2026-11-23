use crate::idempotency;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Method, Request, header},
};
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, time::Duration};
use tokio::time::{Instant, sleep_until};
use tower::ServiceExt;

const API: &str = include_str!("../../../../api/openapi.json");
const DATA_MAX: usize = 4 * 1024 * 1024;
const WAIT_MS_MAX: u64 = 5000;

// Never select a route from a client-supplied URL, method or operation ID.
const ROUTES: &[(&str, &str, &str)] = &[
    ("list_models", "get", "/api/models"),
    ("register_model", "post", "/api/models"),
    ("list_scratch", "get", "/api/scratch"),
    ("get_scratch", "get", "/api/scratch/{id}"),
    ("compile_program", "post", "/api/scratch/compile_program"),
    ("run_program", "post", "/api/scratch/{id}/run_program"),
    ("start_trace", "post", "/api/jobs"),
    ("steer_feature", "post", "/api/jobs"),
    ("read_feature_trace", "get", "/api/jobs/{id}/trace/features"),
    ("get_job", "get", "/api/jobs/{id}"),
    ("list_interventions", "get", "/api/interventions"),
    ("create_intervention", "post", "/api/interventions"),
    (
        "new_intervention_version",
        "post",
        "/api/interventions/{id}/versions",
    ),
    (
        "get_intervention",
        "get",
        "/api/interventions/{id}/versions/{version}",
    ),
    (
        "export_control_vector",
        "get",
        "/api/interventions/{id}/versions/{version}/gguf",
    ),
    ("create_scratch", "post", "/api/scratch"),
    ("apply_edit", "post", "/api/scratch/{id}/edits"),
    ("merge_models", "post", "/api/scratch/merge"),
    ("rome_edit", "post", "/api/scratch/{id}/rome"),
    ("memit_edit", "post", "/api/scratch/{id}/memit"),
    ("alphaedit_edit", "post", "/api/scratch/{id}/alphaedit"),
    ("prune_model", "post", "/api/scratch/{id}/prune"),
    ("plan_experts", "post", "/api/scratch/experts/{id}/plan"),
    ("prune_experts", "post", "/api/scratch/experts/{id}/prune"),
    ("convert_scratch_gguf", "post", "/api/scratch/{id}/gguf"),
    ("discard_scratch", "post", "/api/scratch/{id}/discard"),
    ("run_checks", "post", "/api/scratch/{id}/checks"),
    ("get_checks", "get", "/api/scratch/{id}/checks"),
    ("diff_tensors", "get", "/api/diff/tensors"),
    ("diff_traces", "get", "/api/diff/traces"),
    ("diff_behavior", "get", "/api/diff/behavior"),
    (
        "request_publish",
        "post",
        "/api/scratch/{id}/publish-request",
    ),
];

const TRACE_VIEWS: &[(&str, &str)] = &[
    ("manifest", "/api/jobs/{id}/trace"),
    ("tensor", "/api/jobs/{id}/trace/tensor"),
    ("logit_lens", "/api/jobs/{id}/trace/logit-lens"),
    ("model_logits", "/api/jobs/{id}/trace/model-logits"),
];

struct Route {
    path: &'static str,
    write: bool,
    path_params: Vec<String>,
    query_params: Vec<String>,
    schema: Value,
    description: String,
}

pub(super) struct Server {
    api: Router,
    routes: BTreeMap<&'static str, Route>,
    trace_views: BTreeMap<&'static str, Route>,
    catalog: Value,
}

impl Server {
    pub(super) fn new(api: Router) -> Self {
        let contract: Value = serde_json::from_str(API).expect("valid OpenAPI");
        let routes: BTreeMap<_, _> = ROUTES
            .iter()
            .map(|&(name, method, path)| {
                (
                    name,
                    Route::new(
                        &contract,
                        method,
                        path,
                        matches!(name, "start_trace" | "steer_feature"),
                    ),
                )
            })
            .collect();
        let trace_views: BTreeMap<_, _> = TRACE_VIEWS
            .iter()
            .map(|&(view, path)| (view, Route::new(&contract, "get", path, false)))
            .collect();
        let mut tools = Vec::new();
        for (&name, route) in &routes {
            let mut schema = route.schema.clone();
            let description = match name {
                "start_trace" => {
                    "Queue a CPU trace. Enter one Library GGUF ID or one scratch revision ID with private GGUF in input_ids. Poll its ID with get_job."
                }
                "steer_feature" => {
                    schema["properties"]["options"]["required"] = json!(["feature_steering"]);
                    "Queue a CPU trace with one SAE decoder direction. Set options.feature_steering. Poll its ID with get_job."
                }
                "get_job" => {
                    schema["properties"]["wait_ms"] = json!({
                        "type":"integer", "minimum":0, "maximum":WAIT_MS_MAX, "default":0,
                        "description":"Wait for a terminal state up to this many milliseconds. Then return the latest progress."
                    });
                    "Read job progress, events and result. Optionally wait for completion; does not cancel."
                }
                "export_control_vector" => {
                    "Check that an intervention version can export as a llama.cpp control vector, and return its relative download URL."
                }
                _ => &route.description,
            };
            tools.push(tool(
                name,
                description,
                schema,
                !route.write || name == "plan_experts",
            ));
        }
        let variants: Vec<_> = trace_views
            .iter()
            .map(|(&view, route)| {
                let mut schema = route.schema.clone();
                schema["properties"]["view"] = json!({"const":view});
                schema["required"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!("view"));
                schema
            })
            .collect();
        let mut trace_props = Map::new();
        for schema in &variants {
            trace_props.extend(object(&schema["properties"]).clone());
        }
        trace_props.insert("view".into(), json!({"type":"string","enum":TRACE_VIEWS.iter().map(|(view, _)| view).collect::<Vec<_>>()}));
        tools.push(tool("read_trace", "Read a trace manifest, tensor slice, logit lens or model logits. Limit large reads with the layer and token range filters.",
            json!({"type":"object","properties":trace_props,"required":["id","view"],"oneOf":variants}), true));
        Self {
            api,
            routes,
            trace_views,
            catalog: json!({"tools":tools}),
        }
    }

    pub(super) fn list(&self) -> Value {
        self.catalog.clone()
    }

    pub(super) async fn call(&self, params: &Value) -> Result<Value, &'static str> {
        let name = params["name"].as_str().ok_or("tool name is required")?;
        let mut args = match params.get("arguments") {
            Some(value) => value
                .as_object()
                .ok_or("arguments must be an object")?
                .clone(),
            None => Map::new(),
        };
        let route = if name == "read_trace" {
            args.remove("view")
                .and_then(|v| self.trace_views.get(v.as_str()?))
                .ok_or("read_trace requires a supported view")?
        } else {
            self.routes.get(name).ok_or("unknown tool")?
        };
        let wait_ms = if name == "get_job" {
            args.remove("wait_ms").map_or(Ok(0), |ms| {
                ms.as_u64()
                    .filter(|ms| *ms <= WAIT_MS_MAX)
                    .ok_or("wait_ms must be an integer from 0 to 5000")
            })?
        } else {
            0
        };
        if name == "steer_feature"
            && !args
                .get("options")
                .and_then(|o| o.get("feature_steering"))
                .is_some_and(Value::is_object)
        {
            return Err("steer_feature requires options.feature_steering");
        }
        let (path, key, body) =
            route.request(args, matches!(name, "start_trace" | "steer_feature"))?;
        let deadline = Instant::now() + Duration::from_millis(wait_ms);
        let result = loop {
            let result = match self.send(route, &path, key.as_deref(), body.clone()).await {
                Ok(result) => result,
                Err(result) => return Ok(result),
            };
            if wait_ms == 0
                || Instant::now() >= deadline
                || !matches!(result["state"].as_str(), Some("queued" | "running"))
            {
                break result;
            }
            sleep_until(deadline.min(Instant::now() + Duration::from_millis(250))).await;
        };
        let data = match name {
            "list_models" => json!({"models":result}),
            "list_scratch" => json!({"revisions":result}),
            "list_interventions" => json!({"interventions":result}),
            _ => result,
        };
        Ok(content(data, false))
    }

    async fn send(
        &self,
        route: &Route,
        path: &str,
        key: Option<&str>,
        body: Vec<u8>,
    ) -> Result<Value, Value> {
        let mut request = Request::builder()
            .method(if route.write {
                Method::POST
            } else {
                Method::GET
            })
            .uri(path);
        if route.write {
            request = request.header(header::CONTENT_TYPE, "application/json");
        }
        if let Some(key) = key {
            request = request.header("Idempotency-Key", key);
        }
        let request = request
            .body(Body::from(body))
            .map_err(|_| failure(400, "invalid tool path or idempotency key"))?;
        // The outer HTTP request has passed the workbench guard. The fixed
        // in-process call still uses the same extractors, limits and handlers.
        let response = self
            .api
            .clone()
            .oneshot(request)
            .await
            .expect("Axum Router has an infallible service error type");
        let status = response.status();
        if status.is_success() && !route.write && route.path.ends_with("/gguf") {
            // Export has validated the rules. The client downloads the binary
            // separately through the workbench, not as a JSON tool result.
            return Ok(json!({"url":format!("/api{path}")}));
        }
        let bytes = to_bytes(response.into_body(), DATA_MAX)
            .await
            .map_err(|_| too_large())?;
        if !status.is_success() {
            let data = serde_json::from_slice::<Value>(&bytes)
                .unwrap_or_else(|_| json!({"error":String::from_utf8_lossy(&bytes)}));
            return Err(content(
                json!({"status":status.as_u16(), "details":data}),
                true,
            ));
        }
        serde_json::from_slice(&bytes)
            .map_err(|_| failure(500, "expected a JSON response from the controller"))
    }
}

impl Route {
    fn new(contract: &Value, method: &str, path: &'static str, trace: bool) -> Self {
        let operation = &contract["paths"][path][method];
        let description = operation["summary"]
            .as_str()
            .expect("tool source operation")
            .into();
        let write = method == "post";
        let mut schema = if trace {
            trace_schema(contract)
        } else if write {
            let body = resolve(contract, &operation["requestBody"]);
            resolve(contract, &body["content"]["application/json"]["schema"])
        } else {
            json!({"type":"object"})
        };
        let mut properties = schema["properties"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        let mut required = schema["required"].as_array().cloned().unwrap_or_default();
        let mut path_params = Vec::new();
        let mut query_params = Vec::new();
        if let Some(parameters) = operation["parameters"].as_array() {
            for parameter in parameters {
                let parameter = resolve(contract, parameter);
                let mut definition = parameter["schema"].clone();
                let original_name = parameter["name"].as_str().expect("parameter name");
                let name = match parameter["in"].as_str() {
                    Some("path") => {
                        path_params.push(original_name.into());
                        original_name
                    }
                    Some("query") => {
                        query_params.push(original_name.into());
                        original_name
                    }
                    Some("header") if write && original_name == "Idempotency-Key" => {
                        "idempotency_key"
                    }
                    Some("header") if write && original_name == "Brewery-Allow-Duplicate" => {
                        continue;
                    }
                    _ => panic!("unsupported tool parameter"),
                };
                if name == "idempotency_key" {
                    definition["description"] = json!(
                        "Choose one unique key per intended write. Reuse it only for a retry of that write."
                    );
                } else if let Some(description) = parameter["description"].as_str() {
                    definition["description"] = json!(description);
                }
                properties.insert(name.into(), definition);
                if name == "idempotency_key" || parameter["required"] == true {
                    required.push(json!(name));
                }
            }
        }
        assert!(
            !write
                || properties.contains_key("idempotency_key")
                || path == "/api/scratch/experts/{id}/plan"
        );
        schema["properties"] = json!(properties);
        schema["required"] = json!(required);
        schema["additionalProperties"] = json!(false);
        Self {
            write,
            path: path.strip_prefix("/api").unwrap(),
            path_params,
            query_params,
            schema,
            description,
        }
    }

    fn request(
        &self,
        mut args: Map<String, Value>,
        trace: bool,
    ) -> Result<(String, Option<String>, Vec<u8>), &'static str> {
        if !args
            .keys()
            .all(|key| object(&self.schema["properties"]).contains_key(key))
        {
            return Err("unknown tool argument");
        }
        if !self.schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .all(|key| args.contains_key(key.as_str().unwrap()))
        {
            return Err("missing required tool argument");
        }
        let key = if object(&self.schema["properties"]).contains_key("idempotency_key") {
            let key = args
                .remove("idempotency_key")
                .and_then(|v| v.as_str().map(str::to_owned))
                .filter(|key| idempotency::valid_key(key))
                .ok_or("invalid idempotency_key")?;
            Some(key)
        } else {
            None
        };
        let mut path = self.path.to_owned();
        for name in &self.path_params {
            let value = args.remove(name).ok_or("missing path argument")?;
            let value = match name.as_str() {
                "id" => {
                    let id = value.as_str().ok_or("id must be a controller UUID")?;
                    uuid::Uuid::try_parse(id)
                        .ok()
                        .filter(|uuid| uuid.hyphenated().to_string() == id)
                        .ok_or("id must be a controller UUID")?;
                    id.to_owned()
                }
                "version" => value
                    .as_u64()
                    .filter(|v| (1..=u32::MAX as u64).contains(v))
                    .ok_or("version must be a positive 32-bit integer")?
                    .to_string(),
                _ => return Err("unsupported path argument"),
            };
            path = path.replace(&format!("{{{name}}}"), &value);
        }
        let mut separator = '?';
        for name in &self.query_params {
            if let Some(value) = args.remove(name) {
                // These controller query parameters are integers, UUIDs or
                // role slugs. Never let a value add a delimiter or an escape.
                let value = match self.schema["properties"][name]["type"].as_str() {
                    Some("integer") => value.as_u64().map(|n| n.to_string()),
                    Some("string") => value
                        .as_str()
                        .filter(|s| {
                            !s.is_empty()
                                && s.len() <= 128
                                && s.bytes()
                                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                        })
                        .map(str::to_owned),
                    _ => None,
                }
                .ok_or("invalid query argument")?;
                path.push(separator);
                path.push_str(name);
                path.push('=');
                path.push_str(&value);
                separator = '&';
            }
        }
        let body = if self.write {
            if trace {
                args.insert("kind".into(), json!("trace"));
            }
            serde_json::to_vec(&args).expect("serialize tool arguments")
        } else {
            Vec::new()
        };
        Ok((path, key, body))
    }
}

fn trace_schema(contract: &Value) -> Value {
    let base = &contract["components"]["schemas"];
    let trace = &base["TraceJob"];
    let mut schema = base["JobRequestFields"].clone();
    schema["required"] = trace["required"].clone();
    schema["description"] = trace["description"].clone();
    schema["oneOf"] = trace["oneOf"].clone();
    let props = schema["properties"].as_object_mut().unwrap();
    props.remove("kind");
    props.insert(
        "host".into(),
        base["DefaultInputHost"]["properties"]["host"].clone(),
    );
    for (name, rules) in object(&trace["properties"]) {
        if name != "kind" {
            match props.get_mut(name) {
                Some(generic) => object_mut(generic).extend(object(rules).clone()),
                None => {
                    props.insert(name.clone(), rules.clone());
                }
            }
        }
    }
    // Replace the generic options schema, not a composite $ref with siblings.
    props.insert("options".into(), trace["properties"]["options"].clone());
    resolve(contract, &schema)
}

fn resolve(contract: &Value, value: &Value) -> Value {
    match value {
        Value::Object(fields) => {
            if let Some(reference) = fields.get("$ref").and_then(Value::as_str) {
                let path = reference
                    .strip_prefix("#/")
                    .expect("local OpenAPI reference");
                let target = contract
                    .pointer(&format!("/{path}"))
                    .expect("valid OpenAPI reference");
                return resolve(contract, target);
            }
            Value::Object(
                fields
                    .iter()
                    .map(|(k, v)| (k.clone(), resolve(contract, v)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.iter().map(|v| resolve(contract, v)).collect()),
        value => value.clone(),
    }
}

fn object(value: &Value) -> &Map<String, Value> {
    value.as_object().expect("OpenAPI object")
}

fn object_mut(value: &mut Value) -> &mut Map<String, Value> {
    value.as_object_mut().expect("OpenAPI object")
}

fn tool(name: &str, description: &str, schema: Value, read_only: bool) -> Value {
    json!({"name":name, "description":description, "inputSchema":schema,
        "annotations":{"readOnlyHint":read_only}})
}

fn content(data: Value, is_error: bool) -> Value {
    json!({"content":[{"type":"text","text":data.to_string()}], "structuredContent":data, "isError":is_error})
}

fn failure(status: u16, message: &str) -> Value {
    content(json!({"status":status, "details":{"error":message}}), true)
}

pub(super) fn too_large() -> Value {
    failure(
        413,
        "MCP result is too large. Use a smaller trace slice or read the controller API directly.",
    )
}
