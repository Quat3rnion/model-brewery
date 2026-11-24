use super::{Kind, MAX_CONTEXT, Provider, Summary, has_secret};
use reqwest::{Client, header::HeaderValue};
use serde_json::{Value, json};
use std::collections::BTreeSet;

const SYSTEM: &str = include_str!("system_prompt.md");
const MALFORMED: Failure =
    Failure::Provider("The provider returned an unsupported tool-use response.");
const SECRET: Failure =
    Failure::Provider("The provider returned private data. The session stopped.");
const RESPONSE_BYTES: usize = 128 * 1024;
const MAX_CALLS_PER_TURN: usize = 16;

#[cfg(test)]
#[test]
fn system_prompt_uses_catalog_tools_and_reserves_approval_for_operator() {
    const OPERATOR_ONLY: &str = "Approval and deployment are operator-only.";
    let catalog = crate::mcp::Server::new(axum::Router::new()).list();
    let tools: BTreeSet<_> = catalog["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();

    // Backticks mark tools and the exact field names and values below.
    let schema_terms = BTreeSet::from([
        "all",
        "alphas",
        "behavior_direction",
        "dense",
        "experts",
        "final_gate",
        "flat",
        "l_out",
        "last_k",
        "last_k_direction_id",
        "options.gguf_projection",
        "options.hgn_projection",
        "plain",
        "registered_marker",
        "reference_id",
        "reference_layer",
        "residual",
        "second_source_layer",
        "tune",
    ]);
    let parts: Vec<_> = SYSTEM.split('`').collect();
    assert_eq!(parts.len() % 2, 1, "unpaired backtick in system prompt");
    let referenced: BTreeSet<_> = parts.iter().skip(1).step_by(2).copied().collect();
    assert!(tools.is_disjoint(&schema_terms));
    assert_eq!(
        referenced,
        tools.union(&schema_terms).copied().collect(),
        "prompt tool or schema terms differ from the expected catalog"
    );
    for part in parts.iter().step_by(2) {
        assert!(!part.contains('_'), "unmarked identifier in system prompt");
    }

    assert_eq!(
        SYSTEM.lines().filter(|line| *line == OPERATOR_ONLY).count(),
        1
    );
    for word in SYSTEM
        .lines()
        .filter(|line| *line != OPERATOR_ONLY)
        .flat_map(|line| line.split(|c: char| !c.is_ascii_alphanumeric()))
    {
        let word = word.to_ascii_lowercase();
        assert!(
            !word.starts_with("approv") && !word.starts_with("deploy"),
            "operator action outside the operator-only line: {word}"
        );
    }
}

pub(super) struct Conversation {
    pub provider: Provider,
    client: Client,
    key: String,
    messages: Vec<Value>,
    tools: Vec<Value>,
}

pub(super) enum Failure {
    Context,
    Provider(&'static str),
}

pub(super) struct Turn {
    pub texts: Vec<String>,
    pub calls: Vec<Call>,
    pub tokens: u64,
    pub limited: bool,
}

pub(super) struct Call {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

impl Conversation {
    pub fn new(
        client: Client,
        provider: Provider,
        key: String,
        catalog: &Value,
        input: &Summary,
    ) -> Self {
        let mut messages = Vec::new();
        if provider.kind == Kind::Openai {
            messages.push(json!({"role":"system","content":SYSTEM}));
        }
        let task = json!({"goal":input.goal,"target_model_id":input.model_id}).to_string();
        messages.push(json!({"role":"user","content":task}));
        let tools = catalog["tools"].as_array().expect("MCP tool catalog").iter().map(|tool| {
            match provider.kind {
                Kind::Anthropic => json!({
                    "name":tool["name"],"description":tool["description"],"input_schema":tool["inputSchema"]
                }),
                Kind::Openai => json!({
                    "type":"function", "function":{
                        "name":tool["name"],"description":tool["description"],"parameters":tool["inputSchema"]
                    }
                }),
            }
        }).collect();
        Self {
            provider,
            client,
            key,
            messages,
            tools,
        }
    }

    pub fn contains_secret(&self, value: &Value) -> bool {
        has_secret(value, &self.key)
    }

    pub async fn turn(&mut self, used: u64) -> Result<Turn, Failure> {
        let provider = &self.provider;
        let max = provider
            .max_tokens
            .min(provider.max_session_tokens.saturating_sub(used));
        let mut body = json!({"model":provider.model,"max_tokens":max,"messages":self.messages,"tools":self.tools});
        let (suffix, auth, header) = match provider.kind {
            Kind::Anthropic => {
                body["system"] = json!(SYSTEM);
                ("/messages", self.key.clone(), "x-api-key")
            }
            Kind::Openai => (
                "/chat/completions",
                format!("Bearer {}", self.key),
                "authorization",
            ),
        };
        let bytes = serde_json::to_vec(&body).map_err(|_| MALFORMED)?;
        if bytes.len() > MAX_CONTEXT {
            return Err(Failure::Context);
        }
        let mut auth = HeaderValue::from_str(&auth)
            .map_err(|_| Failure::Provider("Set a valid provider API key."))?;
        auth.set_sensitive(true);
        let mut request = self
            .client
            .post(format!("{}{suffix}", provider.base_url))
            .header(header, auth)
            .header("Content-Type", "application/json")
            .body(bytes);
        if provider.kind == Kind::Anthropic {
            request = request.header("anthropic-version", "2023-06-01");
        }
        let transport_error = |error: reqwest::Error| {
            Failure::Provider(if error.is_timeout() {
                "The provider request reached its time limit."
            } else {
                "The controller could not contact the provider."
            })
        };
        let mut reply = request.send().await.map_err(transport_error)?;
        if !reply.status().is_success() {
            // Never read or surface a remote error body or redirect Location.
            return Err(Failure::Provider(match reply.status().as_u16() {
                401 | 403 => "The provider rejected the API key or access to the model.",
                429 => "The provider reached a request limit.",
                _ => "The provider rejected the request.",
            }));
        }
        if reply
            .content_length()
            .is_some_and(|len| len > RESPONSE_BYTES as u64)
        {
            return Err(Failure::Context);
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = reply.chunk().await.map_err(transport_error)? {
            if bytes.len() + chunk.len() > RESPONSE_BYTES {
                return Err(Failure::Context);
            }
            bytes.extend_from_slice(&chunk);
        }
        let response: Value = serde_json::from_slice(&bytes).map_err(|_| MALFORMED)?;
        if self.contains_secret(&response) {
            return Err(SECRET);
        }
        let (turn, assistant) = match provider.kind {
            Kind::Anthropic => anthropic(&response)?,
            Kind::Openai => openai(&response)?,
        };
        if turn
            .calls
            .iter()
            .any(|call| self.contains_secret(&call.arguments))
        {
            return Err(SECRET);
        }
        let mut ids = BTreeSet::new();
        if turn.calls.len() > MAX_CALLS_PER_TURN
            || turn.calls.iter().any(|call| {
                call.id.is_empty()
                    || call.id.len() > 200
                    || !ids.insert(&call.id)
                    || call.name.is_empty()
                    || call.name.len() > 128
                    || !call.arguments.is_object()
            })
        {
            return Err(MALFORMED);
        }
        self.messages.push(assistant);
        Ok(turn)
    }

    pub fn add_results(&mut self, results: Vec<(String, String, bool)>) {
        match self.provider.kind {
            Kind::Anthropic => {
                let content: Vec<_> = results
                    .into_iter()
                    .map(|(id, text, error)| {
                        json!({
                            "type":"tool_result","tool_use_id":id,"content":text,"is_error":error,
                        })
                    })
                    .collect();
                self.messages.push(json!({"role":"user","content":content}));
            }
            Kind::Openai => {
                self.messages
                    .extend(results.into_iter().map(|(id, text, _)| {
                        json!({
                            "role":"tool","tool_call_id":id,"content":text,
                        })
                    }));
            }
        }
    }
}

fn text(value: &Value) -> Result<&str, Failure> {
    value.as_str().ok_or(MALFORMED)
}

fn count(usage: &Value, input: &str, output: &str) -> Result<u64, Failure> {
    let a = usage[input].as_u64().ok_or(MALFORMED)?;
    let b = usage[output].as_u64().ok_or(MALFORMED)?;
    a.checked_add(b).ok_or(MALFORMED)
}

fn anthropic(reply: &Value) -> Result<(Turn, Value), Failure> {
    let usage = &reply["usage"];
    let tokens = ["cache_creation_input_tokens", "cache_read_input_tokens"]
        .into_iter()
        .try_fold(
            count(usage, "input_tokens", "output_tokens")?,
            |sum, field| {
                sum.checked_add(usage[field].as_u64().unwrap_or(0))
                    .ok_or(MALFORMED)
            },
        )?;
    let reason = text(&reply["stop_reason"])?;
    let mut turn = Turn {
        texts: vec![],
        calls: vec![],
        tokens,
        limited: reason == "max_tokens",
    };
    let mut content = Vec::new();
    for block in reply["content"].as_array().ok_or(MALFORMED)? {
        match text(&block["type"])? {
            "text" => {
                let text = text(&block["text"])?;
                turn.texts.push(text.to_owned());
                content.push(json!({"type":"text","text":text}));
            }
            "tool_use" => {
                let call = Call {
                    id: text(&block["id"])?.to_owned(),
                    name: text(&block["name"])?.to_owned(),
                    arguments: block["input"].clone(),
                };
                content.push(
                    json!({"type":"tool_use","id":call.id,"name":call.name,"input":call.arguments}),
                );
                turn.calls.push(call);
            }
            _ => return Err(MALFORMED),
        }
    }
    validate_end(
        &turn,
        reason,
        "tool_use",
        &["end_turn", "stop_sequence", "max_tokens"],
    )?;
    Ok((turn, json!({"role":"assistant","content":content})))
}

fn openai(reply: &Value) -> Result<(Turn, Value), Failure> {
    let usage = &reply["usage"];
    let tokens = count(usage, "prompt_tokens", "completion_tokens")?
        .max(usage["total_tokens"].as_u64().unwrap_or_default());
    let choice = &reply["choices"]
        .as_array()
        .filter(|choices| choices.len() == 1)
        .ok_or(MALFORMED)?[0];
    let reason = text(&choice["finish_reason"])?;
    let message = &choice["message"];
    if message["role"] != "assistant" {
        return Err(MALFORMED);
    }
    let mut turn = Turn {
        texts: vec![],
        calls: vec![],
        tokens,
        limited: reason == "length",
    };
    if !message["content"].is_null() {
        turn.texts.push(text(&message["content"])?.to_owned());
    }
    let mut assistant = json!({"role":"assistant","content":message["content"]});
    if !message["tool_calls"].is_null() {
        let mut calls = Vec::new();
        for block in message["tool_calls"].as_array().ok_or(MALFORMED)? {
            if block["type"] != "function" {
                return Err(MALFORMED);
            }
            let arguments = text(&block["function"]["arguments"])?;
            let call = Call {
                id: text(&block["id"])?.to_owned(),
                name: text(&block["function"]["name"])?.to_owned(),
                arguments: serde_json::from_str(arguments).map_err(|_| MALFORMED)?,
            };
            calls.push(json!({
                "type":"function","id":call.id,"function":{"name":call.name,"arguments":arguments},
            }));
            turn.calls.push(call);
        }
        assistant["tool_calls"] = json!(calls);
    }
    validate_end(&turn, reason, "tool_calls", &["stop", "length"])?;
    Ok((turn, assistant))
}

fn validate_end(turn: &Turn, reason: &str, tool: &str, endings: &[&str]) -> Result<(), Failure> {
    if (turn.calls.is_empty() && endings.contains(&reason))
        || (!turn.calls.is_empty() && reason == tool)
    {
        Ok(())
    } else {
        Err(MALFORMED)
    }
}
