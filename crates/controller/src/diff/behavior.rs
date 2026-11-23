use super::*;
use brewery_worker::weight_source::open_regular;
use std::{collections::BTreeSet, io::Read};

const ARTIFACT: &str = "behavior.v1.json";
const MAX_BYTES: u64 = 256 << 10;
const MAX_METRICS: usize = 256;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u8,
    parent_id: String,
    child_id: String,
    metrics: Vec<Metric>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Scalar {
    Number(f64),
    Bool(bool),
    Text(String),
    Null,
}

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Better {
    Higher,
    Lower,
    #[default]
    Neither,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metric {
    name: String,
    parent: Scalar,
    child: Scalar,
    #[serde(default)]
    better: Better,
    #[serde(default)]
    threshold: f64,
}

#[derive(Serialize)]
struct MetricDiff {
    #[serde(flatten)]
    metric: Metric,
    delta: Option<f64>,
    flag: &'static str,
}

#[derive(Serialize)]
struct Behavior {
    metrics: Vec<MetricDiff>,
}

pub(super) fn run(app: &App, q: Pair) -> Result<Bytes> {
    let path = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let child = tensors::scratch(&store, &q.b)?
            .ok_or_else(|| ApiError::not_found("scratch revision"))?;
        if child.provenance.parent_id != q.a {
            return Err(invalid("a is not the parent of scratch revision b"));
        }
        tensors::location(&store, &q.b, ARTIFACT)
    };
    let mut file = open_regular(&path).map_err(|_| ApiError::not_found("behavior input"))?;
    let key = Key::Behavior(q.clone());
    let before = stamp(&file)?;
    if let Some(bytes) = app.diffs.get(&key, std::slice::from_ref(&before))? {
        return Ok(bytes);
    }
    let mut bytes = Vec::new();
    (&mut file)
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(ApiError::storage)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(too_large());
    }
    if stamp(&file)? != before {
        return Err(invalid("behavior input changed during the diff"));
    }
    let input: Input =
        serde_json::from_slice(&bytes).map_err(|_| invalid("invalid behavior input"))?;
    if input.schema_version != 1 || input.parent_id != q.a || input.child_id != q.b {
        return Err(invalid(
            "behavior input schema or revision IDs do not match",
        ));
    }
    if input.metrics.len() > MAX_METRICS {
        return Err(too_large());
    }
    let mut names = BTreeSet::new();
    for m in &input.metrics {
        if m.name.is_empty()
            || m.name.len() > 128
            || m.name.chars().any(char::is_control)
            || !names.insert(&m.name)
            || m.threshold < 0.
            || !m.threshold.is_finite()
        {
            return Err(invalid("invalid or repeated behavior metric"));
        }
    }
    let metrics = input
        .metrics
        .into_iter()
        .map(compare)
        .collect::<Result<Vec<_>>>()?;
    app.diffs
        .save(key, vec![before], &q.report(Behavior { metrics }))
}

fn compare(m: Metric) -> Result<MetricDiff> {
    let (delta, flag) = match (&m.parent, &m.child) {
        (Scalar::Null, _) | (_, Scalar::Null) => (None, "missing"),
        (Scalar::Number(a), Scalar::Number(b)) => {
            let d = b - a;
            if !d.is_finite() {
                return Err(invalid("behavior metric delta is not finite"));
            }
            let flag = if d == 0. {
                "unchanged"
            } else if d.abs() <= m.threshold {
                "within_threshold"
            } else {
                match m.better {
                    Better::Higher if d > 0. => "improved",
                    Better::Lower if d < 0. => "improved",
                    Better::Higher | Better::Lower => "regressed",
                    Better::Neither => "changed",
                }
            };
            (Some(d), flag)
        }
        (Scalar::Bool(a), Scalar::Bool(b)) => (None, if a == b { "unchanged" } else { "changed" }),
        (Scalar::Text(a), Scalar::Text(b)) => (None, if a == b { "unchanged" } else { "changed" }),
        _ => return Err(invalid("behavior metric result types differ")),
    };
    Ok(MetricDiff {
        metric: m,
        delta,
        flag,
    })
}
