use crate::{
    gsm8k::{Predictions, Question, MAX_TOKENS},
    hash_ok, read, Result,
};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, fs::File, time::Duration};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    url: String,
    model: String,
    pub source_sha256: String,
    runtime_id: String,
    bearer_env: Option<String>,
    #[serde(default)]
    synthetic: bool,
}

fn id_ok(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
}

pub fn load(id: &str) -> Result<Source> {
    if !id_ok(id) {
        return Err("invalid baseline source ID".into());
    }
    let path =
        std::env::var_os("BREWERY_BASELINE_CONFIG").ok_or("no baseline endpoint configuration")?;
    let mut config: BTreeMap<String, Source> = read(File::open(path)?, 64 * 1024)
        .map_err(|_| "invalid operator baseline configuration")?;
    config
        .remove(id)
        .ok_or_else(|| "baseline source is not configured".into())
}

impl Source {
    pub fn run(
        &self,
        id: &str,
        questions: &[Question],
        synthetic: bool,
        progress: &crate::progress::Progress,
    ) -> Result<Predictions> {
        if self.synthetic != synthetic {
            return Err("baseline endpoint synthetic designation does not match job".into());
        }
        let url = self
            .url
            .parse::<ureq::http::Uri>()
            .map_err(|_| "invalid baseline endpoint configuration")?;
        let local = matches!(url.host(), Some("127.0.0.1" | "[::1]"));
        if !(url.scheme_str() == Some("https") || (local && url.scheme_str() == Some("http")))
            || url.authority().is_none_or(|a| a.as_str().contains('@'))
            || url.query().is_some()
            || self.url.contains('#')
            || url.path() != "/v1/completions"
            || !hash_ok(&self.source_sha256)
            || !id_ok(&self.runtime_id)
            || self.model.is_empty()
            || self.model.len() > 256
            || self.model.chars().any(char::is_control)
        {
            return Err("invalid baseline endpoint or identity configuration".into());
        }
        let secret = self
            .bearer_env
            .as_ref()
            .map(|name| {
                if !name.starts_with("BREWERY_BASELINE_TOKEN_") {
                    return Err("invalid baseline token environment variable");
                }
                std::env::var(name).map_err(|_| "baseline endpoint token is unavailable")
            })
            .transpose()?;
        let client: ureq::Agent = ureq::Agent::config_builder()
            .proxy(None)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_secs(120)))
            .build()
            .into();
        let mut texts = Vec::with_capacity(questions.len());
        let mut phase = progress.start(
            "baseline_questions",
            "questions",
            Some(questions.len() as u64),
        );
        for (index, q) in questions.iter().enumerate() {
            let request = client.post(url.clone());
            let request = if let Some(secret) = &secret {
                request.header("Authorization", format!("Bearer {secret}"))
            } else {
                request
            };
            let mut response = request
                .send_json(json!({"model":self.model,
                "prompt":q.prompt,"temperature":0,"max_tokens":MAX_TOKENS,"stream":false,"n":1}))
                .map_err(|_| format!("baseline endpoint request failed at row {}", index + 1))?;
            let exact = |key, expected| {
                let mut values = response.headers().get_all(key).iter();
                values.next().and_then(|v| v.to_str().ok()) == Some(expected)
                    && values.next().is_none()
            };
            if response.status() != 200
                || !exact("X-Brewery-Source-Sha256", self.source_sha256.as_str())
                || !exact("X-Brewery-Runtime-Id", self.runtime_id.as_str())
            {
                return Err(format!(
                    "baseline endpoint identity was not verified at row {}",
                    index + 1
                )
                .into());
            }
            #[derive(Deserialize)]
            struct Reply {
                model: String,
                choices: Vec<Choice>,
            }
            #[derive(Deserialize)]
            struct Choice {
                text: String,
                finish_reason: String,
            }
            let response: Reply = read(response.body_mut().as_reader(), 256 * 1024)
                .map_err(|_| format!("invalid baseline completion at row {}", index + 1))?;
            let [choice] = response.choices.as_slice() else {
                return Err("expected one baseline completion".into());
            };
            if response.model != self.model
                || !matches!(choice.finish_reason.as_str(), "length" | "stop")
            {
                return Err(
                    "baseline completion model or finish reason differs from protocol".into(),
                );
            }
            texts.push(choice.text.clone());
            phase.advance(1);
        }
        phase.finish();
        Predictions::new(
            self.source_sha256.clone(),
            json!({
            "method":if synthetic {"synthetic-endpoint-fixture"} else {"operator-bf16-endpoint"},
            "baseline_source":id,"runtime_id":self.runtime_id,
                "model":self.model,"source_evidence":"matched-response-headers"
            }),
            questions,
            texts,
            synthetic,
        )
    }
}
