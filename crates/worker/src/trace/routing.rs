//! Routing atlas and paired expert ablation.
use super::*;
use std::collections::BTreeMap;

pub const MAX_ROWS: usize = 128;
const MAX_EXPERTS: usize = 512;
const MAX_TOP_K: usize = 32;
const CLUSTERS_PER_EXPERT: usize = 5;

#[derive(Serialize, Deserialize)]
struct Input {
    tokens: Vec<Token>,
    rows: Vec<Route>,
    interventions: Vec<Applied>,
}

#[derive(Serialize, Deserialize)]
struct Route {
    position: usize,
    logits: Vec<f32>,
    ids: Vec<usize>,
    weights: Vec<f32>,
    residual: Option<Vec<f32>>,
    top_token: Option<i32>,
}

pub(super) fn input(
    dir: &Path,
    m: Manifest,
    s: &Slice,
    result: &Value,
    export: &Export,
) -> Result<ReadReply> {
    let (Some(layer), Some(start), Some(end)) = (s.layer, s.start, s.end) else {
        anyhow::bail!("routing slice needs a layer and range");
    };
    let mut rows = BTreeMap::new();
    for t in m.tensors.into_iter().filter(|t| {
        t.key.layer == layer
            && (start..end).contains(&t.key.position)
            && (t.key.role.routing() || t.key.role == Role::LOut)
    }) {
        let width = match t.key.role {
            Role::FfnMoeLogits => MAX_EXPERTS,
            Role::FfnMoeTopk | Role::FfnMoeRouteWeights => MAX_TOP_K,
            _ => 16_384,
        } as u64;
        ensure!(
            matches!(t.shape.as_slice(), [n] if (1..=width).contains(n) && *n * 4 == t.bytes),
            "invalid routing tensor shape"
        );
        // Verify the manifest-selected file before decoding it.
        let leaf = t
            .file
            .strip_prefix("tensor-")
            .and_then(|name| name.strip_suffix(".f32"));
        ensure!(
            leaf.is_some_and(|name| name.len() == 6 && name.bytes().all(|b| b.is_ascii_digit()))
                && t.dtype == "f32_le",
            "invalid routing tensor descriptor"
        );
        let bytes = read_file(&dir.join(&t.file), t.bytes)?;
        ensure!(
            bytes.len() as u64 == t.bytes && format!("{:x}", Sha256::digest(&bytes)) == t.sha256,
            "routing tensor size or SHA-256 mismatch"
        );
        let values: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite routing tensor"
        );
        ensure!(
            rows.insert((t.key.position, t.key.role), values).is_none(),
            "duplicate routing row"
        );
    }
    if !rows.keys().any(|(_, role)| role.routing()) {
        return Ok(ReadReply::Missing);
    }
    let mut selected = Vec::new();
    for position in start..end {
        let Some(logits) = rows.remove(&(position, Role::FfnMoeLogits)) else {
            return Ok(ReadReply::Missing);
        };
        let Some(ids) = rows.remove(&(position, Role::FfnMoeTopk)) else {
            return Ok(ReadReply::Missing);
        };
        let Some(weights) = rows.remove(&(position, Role::FfnMoeRouteWeights)) else {
            return Ok(ReadReply::Missing);
        };
        let ids = ids
            .into_iter()
            .map(|n| {
                ensure!(
                    n >= 0. && n.fract() == 0. && n < logits.len() as f32,
                    "invalid expert ID"
                );
                Ok(n as usize)
            })
            .collect::<Result<_>>()?;
        selected.push(Route {
            position,
            logits,
            ids,
            weights,
            residual: rows.remove(&(position, Role::LOut)),
            top_token: m
                .model_logits
                .iter()
                .find(|r| r.position == position)
                .and_then(|r| r.top.first())
                .map(|r| r.token_id),
        });
    }
    let data = json!({
        "tokens":m.tokens,"rows":selected,"interventions":m.interventions,
        "manifest_sha256":export.manifest_sha256,"model_id":result["model_id"],
        "model_sha256":result["model_sha256"],"lanternfish_commit":result["lanternfish_commit"],
    });
    let bytes = serde_json::to_vec(&data)?;
    if bytes.len() > RESPONSE_CAP {
        return Ok(ReadReply::TooLarge);
    }
    Ok(ReadReply::Ok(data))
}

impl Input {
    fn validate(&self) -> Result<usize> {
        let count = self.rows.first().map_or(0, |r| r.logits.len());
        let slots = self.rows.first().map_or(0, |r| r.ids.len());
        ensure!(
            (1..=MAX_ROWS).contains(&self.rows.len())
                && (1..=MAX_EXPERTS).contains(&count)
                && (1..=count.min(MAX_TOP_K)).contains(&slots)
                && (1..=MAX_POSITIONS).contains(&self.tokens.len()),
            "routing slice is empty or exceeds atlas limits"
        );
        ensure!(
            self.tokens
                .iter()
                .enumerate()
                .all(|(i, t)| i == t.position && t.token_id >= 0),
            "invalid trace tokens"
        );
        let mut previous = None;
        for r in &self.rows {
            ensure!(
                r.position < self.tokens.len()
                    && previous.is_none_or(|p| p + 1 == r.position)
                    && r.logits.len() == count
                    && r.logits.iter().all(|v| v.is_finite())
                    && r.ids.len() == slots
                    && r.weights.len() == slots
                    && r.ids.iter().all(|id| *id < count)
                    && r.ids.iter().collect::<BTreeSet<_>>().len() == slots
                    && r.weights.iter().all(|w| w.is_finite() && *w >= 0.),
                "invalid or inconsistent routing row"
            );
            previous = Some(r.position);
        }
        Ok(count)
    }
}

fn entropy(logits: &[f32]) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let shifted = || logits.iter().map(|&v| f64::from(v) - max);
    let z = shifted().map(f64::exp).sum::<f64>();
    (z.ln() - shifted().map(|v| v.exp() * v).sum::<f64>() / z).max(0.)
}

type ContextKey = (Option<i32>, i32);

#[derive(Default)]
struct Load {
    count: usize,
    weight: f64,
    tokens: BTreeMap<i32, usize>,
    contexts: BTreeMap<ContextKey, usize>,
}

fn clusters<K: Copy + Ord>(
    selected: BTreeMap<K, usize>,
    totals: &BTreeMap<K, usize>,
    label: impl Fn(K) -> Value,
) -> Vec<Value> {
    let mut entries: Vec<_> = selected.into_iter().collect();
    // Rank by support first to avoid presenting a one-off as a strong result.
    entries.sort_by_key(|(key, n)| (std::cmp::Reverse(*n), *key));
    entries
        .into_iter()
        .take(CLUSTERS_PER_EXPERT)
        .map(|(key, count)| {
            let mut row = label(key);
            row["selected"] = json!(count);
            row["observations"] = json!(totals[&key]);
            row["selection_rate"] = json!(count as f64 / totals[&key] as f64);
            row
        })
        .collect()
}

pub fn summarize(baseline: Value, edited: Option<Value>, layer: usize) -> Result<Value> {
    let base: Input = serde_json::from_value(baseline)?;
    let n_experts = base.validate()?;
    ensure!(
        base.interventions.is_empty(),
        "routing atlas needs an unedited baseline trace"
    );
    let n = base.rows.len();
    let mut experts: Vec<Load> = (0..n_experts).map(|_| Load::default()).collect();
    let mut pair_counts = BTreeMap::<(usize, usize), usize>::new();
    let (mut token_totals, mut context_totals) = (BTreeMap::new(), BTreeMap::new());
    let mut entropy_sum = 0.;
    for row in &base.rows {
        let token = base.tokens[row.position].token_id;
        let prev = row.position.checked_sub(1).map(|i| base.tokens[i].token_id);
        let context = (prev, token);
        *token_totals.entry(token).or_default() += 1;
        *context_totals.entry(context).or_default() += 1;
        entropy_sum += entropy(&row.logits);
        for (i, (&id, &weight)) in row.ids.iter().zip(&row.weights).enumerate() {
            let expert = &mut experts[id];
            expert.count += 1;
            expert.weight += f64::from(weight);
            *expert.tokens.entry(token).or_default() += 1;
            *expert.contexts.entry(context).or_default() += 1;
            for &other in &row.ids[..i] {
                *pair_counts
                    .entry((id.min(other), id.max(other)))
                    .or_default() += 1;
            }
        }
    }
    let ablation = edited
        .map(|value| ablation(&base, serde_json::from_value(value)?, layer, n_experts))
        .transpose()?;
    let experts: Vec<_> = experts.into_iter().enumerate().map(|(id, e)| {
        json!({
            "expert":id, "selected":e.count, "load":e.count as f64 / n as f64,
            "mean_selected_weight": (e.count > 0).then(|| e.weight / e.count as f64),
            "token_clusters":clusters(e.tokens, &token_totals, |token_id| json!({"token_id":token_id})),
            "context_clusters":clusters(e.contexts, &context_totals, |(previous_token_id, token_id)|
                json!({"previous_token_id":previous_token_id,"token_id":token_id}))
        })
    }).collect();
    let coactivation: Vec<_> = pair_counts
        .into_iter()
        .map(|((a, b), count)| json!({"a":a,"b":b,"count":count}))
        .collect();
    let report = json!({
        "schema_version":1,"layer":layer,"positions":n,"expert_count":n_experts,
        "experts_per_token":base.rows[0].ids.len(),"mean_entropy_nats":entropy_sum / n as f64,
        "experts":experts,"coactivation":coactivation,"ablation":ablation
    });
    ensure!(
        serde_json::to_vec(&report)?.len() <= RESPONSE_CAP,
        "select a smaller routing range"
    );
    Ok(report)
}

fn ablation(base: &Input, edit: Input, layer: usize, n_experts: usize) -> Result<Value> {
    ensure!(
        edit.validate()? == n_experts
            && edit.tokens == base.tokens
            && edit.rows.len() == base.rows.len(),
        "ablation trace tokens or shapes differ"
    );
    let expert = edit
        .interventions
        .first()
        .and_then(|r| r.expert)
        .filter(|&id| id < n_experts)
        .context("paired trace needs a valid expert ablation")?;
    let range = base.rows[0].position..=base.rows.last().unwrap().position;
    let mut positions = BTreeSet::new();
    for r in edit.interventions {
        ensure!(
            r.expert == Some(expert)
                && r.key.layer == layer
                && r.key.role == Role::FfnOut
                && range.contains(&r.key.position)
                && positions.insert(r.key.position),
            "paired trace must only ablate one expert in the requested layer and range"
        );
    }
    let mut rows = Vec::new();
    for (a, b) in base.rows.iter().zip(edit.rows) {
        // Earlier edits at this layer cannot change this layer's router decisions.
        ensure!(
            a.position == b.position && a.ids == b.ids && a.weights == b.weights,
            "paired trace changed the selected expert IDs or weights"
        );
        if !positions.contains(&a.position) {
            continue;
        }
        let (x, y) = a
            .residual
            .as_ref()
            .zip(b.residual.as_ref())
            .context("paired ablation needs l_out in both traces")?;
        ensure!(
            !x.is_empty() && x.len() == y.len(),
            "ablation residual shapes differ"
        );
        let residual_l2 = x
            .iter()
            .zip(y)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            .sqrt();
        ensure!(residual_l2.is_finite(), "invalid ablation residual");
        let selected = a.ids.contains(&expert);
        ensure!(
            selected || residual_l2 == 0.,
            "an unselected expert changed this layer's residual"
        );
        rows.push(json!({
            "position":a.position,"selected":selected,"residual_l2":residual_l2,
            "top_token_changed":a.top_token.zip(b.top_token).map(|(x,y)| x != y)
        }));
    }
    Ok(json!({"expert":expert,"rows":rows}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_load_entropy_coactivation_context_and_paired_effect() {
        let input = |edited| {
            json!({
                "tokens": [{"position":0,"token_id":9},{"position":1,"token_id":2},
                           {"position":2,"token_id":2}],
                "rows": [
                    {"position":1,"logits":[0.,0.,0.],"ids":[0,1],"weights":[0.75,0.25],
                     "residual":[if edited { 0. } else { 3. },4.],"top_token":if edited { 7 } else { 8 }},
                    {"position":2,"logits":[0.,0.,0.],"ids":[0,2],"weights":[0.5,0.5],
                     "residual":[1.,0.],"top_token":8}
                ],
                "interventions": if edited {
                    json!([{"layer":0,"role":"ffn_out","position":1,"expert":1}])
                } else { json!([]) }
            })
        };
        let atlas = summarize(input(false), Some(input(true)), 0).unwrap();
        assert_eq!(atlas["experts"][0]["selected"], 2);
        assert_eq!(atlas["experts"][0]["mean_selected_weight"], 0.625);
        assert!((atlas["mean_entropy_nats"].as_f64().unwrap() - 3f64.ln()).abs() < 1e-10);
        assert_eq!(
            atlas["coactivation"],
            json!([
                {"a":0,"b":1,"count":1}, {"a":0,"b":2,"count":1}
            ])
        );
        assert_eq!(atlas["experts"][1]["token_clusters"][0]["token_id"], 2);
        assert_eq!(
            atlas["experts"][1]["context_clusters"][0]["previous_token_id"],
            9
        );
        assert_eq!(atlas["ablation"]["rows"][0]["residual_l2"], 3.0);
        assert_eq!(atlas["ablation"]["rows"][0]["top_token_changed"], true);
        let mut invalid = input(true);
        invalid["rows"][0]["ids"] = json!([1, 0]);
        assert!(summarize(input(false), Some(invalid), 0).is_err());
        let mut malformed = input(false);
        malformed["rows"][0]["ids"] = json!([0, 0]);
        assert!(summarize(malformed, None, 0).is_err());
    }
}
