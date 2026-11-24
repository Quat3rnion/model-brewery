use crate::{App, domain::now, web::ApiError};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    routing::{get, post},
};
use brewery_worker::{
    is_digest,
    scratch_eval::registered::{Classifier, Config, Partition, Request, name_valid},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs::OpenOptions,
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    sync::Arc,
};
use subtle::ConstantTimeEq;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Operator {
    operator: String,
    token: String,
}

impl Operator {
    pub fn from_env(var: &str) -> Result<Option<Self>, crate::Error> {
        let Some(path) = std::env::var_os(var) else {
            return Ok(None);
        };
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)?;
        let meta = file.metadata()?;
        // SAFETY: geteuid has no arguments and no memory or process side effects.
        let uid = unsafe { libc::geteuid() };
        if !meta.is_file() || meta.mode() & 0o077 != 0 || meta.uid() != uid || meta.len() > 4096 {
            return Err("operator credentials must be a private, owned regular file".into());
        }
        let mut raw = Vec::new();
        file.take(4097).read_to_end(&mut raw)?;
        let record: Self =
            serde_json::from_slice(&raw).map_err(|_| "invalid operator credentials")?;
        if raw.len() > 4096 || !name_valid(&record.operator) || !is_digest(&record.token) {
            return Err("invalid operator credentials".into());
        }
        Ok(Some(record))
    }

    pub(crate) fn authenticate(&self, headers: &HeaderMap) -> Option<&str> {
        let token = crate::web::single(headers, AUTHORIZATION.as_str())
            .ok()??
            .strip_prefix("Bearer ")?;
        bool::from(token.as_bytes().ct_eq(self.token.as_bytes())).then_some(&self.operator)
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Pair<T> {
    tune: T,
    final_gate: T,
}

impl<T> Pair<T> {
    fn get(&self, partition: Partition) -> &T {
        match partition {
            Partition::Tune => &self.tune,
            Partition::FinalGate => &self.final_gate,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Source {
    url: String,
    sha256: String,
    size: u64,
    license: String,
    format: Format,
    field: String,
    system: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    filter: Option<Filter>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Format {
    Csv,
    Json,
    Jsonl,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Filter {
    field: String,
    equals: Value,
}

impl Source {
    fn valid(&self) -> bool {
        let url = reqwest::Url::parse(&self.url).ok();
        let pinned = url.as_ref().is_some_and(|url| {
            url.scheme() == "https"
                && url.has_host()
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
                && url.path().rsplit_once('/').is_some_and(|(path, tail)| {
                    !tail.is_empty()
                        && path.split('/').any(|part| {
                            part.len() == 40
                                && part
                                    .bytes()
                                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                        })
                })
        });
        let label =
            |s: &str| !s.trim().is_empty() && s.len() <= 160 && !s.chars().any(char::is_control);
        self.url.len() <= 2048
            && pinned
            && is_digest(&self.sha256)
            && (1..=48 * 1024 * 1024).contains(&self.size)
            && label(&self.license)
            && label(&self.field)
            && self.system.len() <= 4096
            && !self.system.contains(['\0'])
            && !self.system.contains("<|")
            && self.filter.as_ref().is_none_or(|filter| {
                label(&filter.field)
                    && match &filter.equals {
                        Value::String(s) => s.len() <= 512,
                        Value::Null | Value::Bool(_) | Value::Number(_) => true,
                        _ => false,
                    }
                    && (!matches!(self.format, Format::Csv) || filter.equals.is_string())
            })
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Spec {
    pub name: String,
    positive: Source,
    negative: Source,
    sizes: Pair<usize>,
    pins: Pair<String>,
    classifier: Classifier,
}

impl Spec {
    fn validate(&self) -> Result<(), ApiError> {
        if !name_valid(&self.name)
            || !self.positive.valid()
            || !self.negative.valid()
            || ![self.sizes.tune, self.sizes.final_gate]
                .iter()
                .all(|n| (1..=32).contains(n))
            || !is_digest(&self.pins.tune)
            || !is_digest(&self.pins.final_gate)
            || self.pins.tune == self.pins.final_gate
        {
            return Err(ApiError::bad(
                "invalid evaluation set pins, sources or partition sizes",
            ));
        }
        self.classifier
            .validate()
            .map_err(|e| ApiError::bad(e.to_string()))
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Record {
    pub spec: Spec,
    operator: String,
    registered_at_ms: u64,
}

impl Record {
    pub fn resolve(&self, request: &Request) -> Config {
        Config {
            name: self.spec.name.clone(),
            partition: request.partition,
            sha256: self.spec.pins.get(request.partition).clone(),
            size_per_side: *self.spec.sizes.get(request.partition),
            classifier: self.spec.classifier.clone(),
        }
    }

    pub(crate) fn summary(self) -> Summary {
        Summary {
            name: self.spec.name,
            sizes: self.spec.sizes,
            operator: self.operator,
            registered_at_ms: self.registered_at_ms,
        }
    }
}

#[derive(Serialize)]
pub(crate) struct Summary {
    name: String,
    sizes: Pair<usize>,
    operator: String,
    registered_at_ms: u64,
}

#[derive(Serialize)]
struct List {
    sets: Vec<Summary>,
}

pub fn router() -> Router<Arc<App>> {
    Router::new().route("/evaluation-sets", get(list)).route(
        "/admin/evaluation-sets",
        post(register).layer(DefaultBodyLimit::max(96 * 1024)),
    )
}

async fn list(State(app): State<Arc<App>>) -> Result<Json<List>, ApiError> {
    let store = app.store.lock().map_err(ApiError::storage)?;
    Ok(Json(List {
        sets: store.evaluation_sets().map_err(ApiError::storage)?,
    }))
}

async fn register(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Result<Json<Spec>, JsonRejection>,
) -> Result<(StatusCode, Json<Summary>), ApiError> {
    let Some(operator) = app
        .evaluation_operator
        .as_ref()
        .and_then(|operator| operator.authenticate(&headers))
    else {
        return Err(ApiError(
            StatusCode::FORBIDDEN,
            "evaluation set registration requires operator credentials".into(),
        ));
    };
    let Json(spec) = body.map_err(|_| ApiError::bad("invalid evaluation set registration"))?;
    spec.validate()?;
    let row = Record {
        spec,
        operator: operator.into(),
        registered_at_ms: now(),
    };
    let store = app.store.lock().map_err(ApiError::storage)?;
    if !store
        .insert_evaluation_set(&row)
        .map_err(ApiError::storage)?
    {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "evaluation set name is already registered".into(),
        ));
    }
    Ok((StatusCode::CREATED, Json(row.summary())))
}
