//! Stateless MCP Streamable HTTP. Only fixed Brewery operations are exposed.
mod tools;
pub(crate) use tools::Server;

use crate::web::{has_json, single};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, FromRequest, Request, State},
    http::{HeaderMap, Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::any,
};
use serde_json::{Value, json};
use std::sync::Arc;

const VERSION: &str = "2025-06-18";
const RESPONSE_MAX: usize = 8 * 1024 * 1024;

pub fn router(server: Arc<Server>) -> Router {
    Router::new()
        .route("/mcp", any(message))
        .layer(DefaultBodyLimit::max(256 * 1024))
        .with_state(server)
}

async fn message(State(server): State<Arc<tools::Server>>, request: Request) -> Response {
    match single(request.headers(), "mcp-protocol-version") {
        Ok(None | Some(VERSION)) => {}
        _ => {
            return error(
                StatusCode::BAD_REQUEST,
                Value::Null,
                -32600,
                "unsupported MCP protocol version",
            );
        }
    }
    if request.method() != Method::POST {
        return (StatusCode::METHOD_NOT_ALLOWED, [(header::ALLOW, "POST")]).into_response();
    }
    if !has_json(request.headers()) {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Value::Null,
            -32600,
            "use application/json for MCP",
        );
    }
    if !accepts(request.headers(), "application/json")
        || !accepts(request.headers(), "text/event-stream")
    {
        return error(
            StatusCode::NOT_ACCEPTABLE,
            Value::Null,
            -32600,
            "accept application/json and text/event-stream",
        );
    }
    let Json(value) = match Json::<Value>::from_request(request, &()).await {
        Ok(body) => body,
        Err(rejection) => {
            return error(
                rejection.status(),
                Value::Null,
                -32700,
                "invalid or oversized JSON body",
            );
        }
    };
    let id = value.get("id").cloned();
    let method = value.get("method").and_then(Value::as_str);
    if value["jsonrpc"] != "2.0"
        || method.is_none()
        || id
            .as_ref()
            .is_some_and(|id| !id.is_string() && !id.is_i64() && !id.is_u64())
    {
        return error(
            StatusCode::BAD_REQUEST,
            Value::Null,
            -32600,
            "invalid JSON-RPC request",
        );
    }
    let params = value.get("params");
    let Some(id) = id else {
        // Never run a tool without a request ID. No server-initiated requests
        // or session state need a notification handler.
        return if params.is_none_or(Value::is_object) {
            StatusCode::ACCEPTED.into_response()
        } else {
            StatusCode::BAD_REQUEST.into_response()
        };
    };
    if params.is_some_and(|value| !value.is_object()) {
        return error(StatusCode::OK, id, -32602, "params must be an object");
    }
    let result = match method {
        Some("initialize") => match params {
            Some(p)
                if p["protocolVersion"].is_string()
                    && p["capabilities"].is_object()
                    && p["clientInfo"]["name"].is_string()
                    && p["clientInfo"]["version"].is_string() =>
            {
                json!({
                    "protocolVersion": VERSION,
                    "serverInfo": {"name":"model-brewery", "version":env!("CARGO_PKG_VERSION")},
                    "capabilities": {"tools":{}},
                    "instructions": "Use scratch for experiments. Results are not release-qualified. Request publish in MCP; the operator reviews it in the workbench."
                })
            }
            _ => {
                return error(
                    StatusCode::OK,
                    id,
                    -32602,
                    "initialize needs protocolVersion, capabilities, and clientInfo",
                );
            }
        },
        Some("ping") => json!({}),
        Some("tools/list") if params.and_then(|p| p.get("cursor")).is_none() => server.list(),
        Some("tools/list") => {
            return error(
                StatusCode::OK,
                id,
                -32602,
                "tools/list does not take a cursor",
            );
        }
        Some("tools/call") => match server.call(params.unwrap_or(&Value::Null)).await {
            Ok(result) => result,
            Err(message) => return error(StatusCode::OK, id, -32602, message),
        },
        _ => return error(StatusCode::OK, id, -32601, "unknown method"),
    };
    let bytes = serde_json::to_vec(&json!({"jsonrpc":"2.0","id":id,"result":result}))
        .expect("serialize JSON-RPC response");
    if bytes.len() > RESPONSE_MAX {
        return Json(json!({"jsonrpc":"2.0","id":id,"result":tools::too_large()})).into_response();
    }
    ([(header::CONTENT_TYPE, "application/json")], bytes).into_response()
}

fn accepts(headers: &HeaderMap, media: &str) -> bool {
    headers
        .get_all(header::ACCEPT)
        .iter()
        .filter_map(|header| header.to_str().ok())
        .flat_map(|header| header.split(','))
        .any(|range| {
            let mut parts = range.split(';').map(str::trim);
            parts
                .next()
                .is_some_and(|value| value.eq_ignore_ascii_case(media) || value == "*/*")
                && parts.all(|param| {
                    param.split_once('=').is_none_or(|(key, value)| {
                        !key.trim().eq_ignore_ascii_case("q")
                            || value
                                .trim()
                                .parse::<f32>()
                                .is_ok_and(|q| q > 0.0 && q <= 1.0)
                    })
                })
        })
}

fn error(status: StatusCode, id: Value, code: i32, message: &str) -> Response {
    (
        status,
        Json(json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})),
    )
        .into_response()
}
