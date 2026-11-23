//! Trace comparisons use the verified W1 export, not the GGUF or a live engine.
use super::*;
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize)]
struct Input {
    spec: Capture,
    tokens: Vec<Token>,
    tensors: Vec<Row>,
    missing: Vec<Key>,
    logit_lens: Vec<Logits>,
    model_logits: Vec<Logits>,
}

#[derive(Serialize, Deserialize)]
struct Row {
    #[serde(flatten)]
    key: Key,
    shape: Vec<u64>,
    data_base64: String,
}

pub(super) fn input(dir: &Path, m: Manifest, s: &Slice) -> Result<ReadReply> {
    let in_range = |p: usize| (s.start.unwrap_or(0)..s.end.unwrap_or(MAX_POSITIONS)).contains(&p);
    let key_match = |k: &Key| in_range(k.position) && s.layer.is_none_or(|l| k.layer == l);
    let mut input = Input {
        spec: m.spec,
        tokens: m.tokens,
        tensors: Vec::new(),
        missing: m.missing.into_iter().filter(key_match).collect(),
        logit_lens: m
            .logit_lens
            .into_iter()
            .filter(|r| in_range(r.position) && s.layer.is_none_or(|l| r.layer == Some(l)))
            .collect(),
        model_logits: m
            .model_logits
            .into_iter()
            .filter(|r| in_range(r.position))
            .collect(),
    };
    let mut size = serde_json::to_vec(&input)?.len();
    for t in m.tensors.into_iter().filter(|t| key_match(&t.key)) {
        size += (t.bytes as usize).div_ceil(3) * 4 + 256;
        if size > RESPONSE_CAP {
            return Ok(ReadReply::TooLarge);
        }
        let data = read_file(&dir.join(&t.file), t.bytes)?;
        ensure!(
            data.len() as u64 == t.bytes && format!("{:x}", Sha256::digest(&data)) == t.sha256,
            "trace tensor size or SHA-256 mismatch"
        );
        input.tensors.push(Row {
            key: t.key,
            shape: t.shape,
            data_base64: STANDARD.encode(data),
        });
    }
    if size > RESPONSE_CAP {
        return Ok(ReadReply::TooLarge);
    }
    Ok(ReadReply::Ok(serde_json::to_value(input)?))
}

#[derive(Serialize)]
pub struct Report {
    activations: Vec<Activation>,
    logit_lens: Vec<LogitDiff>,
    model_logits: Vec<LogitDiff>,
}

#[derive(Serialize)]
struct Activation {
    #[serde(flatten)]
    key: Key,
    shape: Option<Vec<u64>>,
    missing_a: bool,
    missing_b: bool,
    l2: Option<f64>,
    linf: Option<f64>,
    cosine: Option<f64>,
}

#[derive(Serialize)]
struct LogitDiff {
    layer: Option<usize>,
    position: usize,
    a: Option<Vec<TopLogit>>,
    b: Option<Vec<TopLogit>>,
    entered: Vec<i32>,
    exited: Vec<i32>,
    shared_count: usize,
    l2: Option<f64>,
    linf: Option<f64>,
    kl_a_to_b_shared: Option<f64>,
}

pub fn compare(a: Value, b: Value) -> Result<Report> {
    let a: Input = serde_json::from_value(a)?;
    let b: Input = serde_json::from_value(b)?;
    ensure!(a.spec == b.spec, "trace capture specs differ");
    ensure!(a.tokens == b.tokens, "trace tokens differ");
    Ok(Report {
        activations: activations(a.tensors, a.missing, b.tensors, b.missing)?,
        logit_lens: logits(a.logit_lens, b.logit_lens)?,
        model_logits: logits(a.model_logits, b.model_logits)?,
    })
}

fn activations(
    a: Vec<Row>,
    a_missing: Vec<Key>,
    b: Vec<Row>,
    b_missing: Vec<Key>,
) -> Result<Vec<Activation>> {
    fn index(rows: Vec<Row>, missing: Vec<Key>) -> Result<BTreeMap<Key, Option<Row>>> {
        let mut index = BTreeMap::new();
        for (key, row) in rows
            .into_iter()
            .map(|r| (r.key, Some(r)))
            .chain(missing.into_iter().map(|k| (k, None)))
        {
            ensure!(
                index.insert(key, row).is_none(),
                "duplicate trace activation"
            );
        }
        Ok(index)
    }
    let a = index(a, a_missing)?;
    let b = index(b, b_missing)?;
    ensure!(a.keys().eq(b.keys()), "trace activation inventories differ");
    a.into_iter()
        .zip(b)
        .map(|((key, a), (_, b))| {
            let (l2, linf, cosine) = match (&a, &b) {
                (Some(a), Some(b)) => {
                    ensure!(a.shape == b.shape, "trace activation shapes differ");
                    let (x, y) = (
                        STANDARD.decode(&a.data_base64)?,
                        STANDARD.decode(&b.data_base64)?,
                    );
                    let bytes = a.shape.iter().try_fold(4, |n: u64, d| n.checked_mul(*d));
                    ensure!(
                        Some(x.len() as u64) == bytes && x.len() == y.len(),
                        "invalid trace activation bytes"
                    );
                    let float = |v: &[u8]| f64::from(f32::from_le_bytes(v.try_into().unwrap()));
                    let (mut a2, mut b2, mut dot, mut d2, mut max) = (0., 0., 0., 0., 0f64);
                    for (a, b) in x.chunks_exact(4).zip(y.chunks_exact(4)) {
                        let (a, b) = (float(a), float(b));
                        ensure!(
                            a.is_finite() && b.is_finite(),
                            "non-finite trace activation"
                        );
                        a2 += a * a;
                        b2 += b * b;
                        dot += a * b;
                        let d = b - a;
                        d2 += d * d;
                        max = max.max(d.abs());
                    }
                    let denom = (a2 * b2).sqrt();
                    (
                        Some(d2.sqrt()),
                        Some(max),
                        (denom > 0.).then(|| (dot / denom).clamp(-1., 1.)),
                    )
                }
                _ => (None, None, None),
            };
            Ok(Activation {
                key,
                shape: a.as_ref().or(b.as_ref()).map(|r| r.shape.clone()),
                missing_a: a.is_none(),
                missing_b: b.is_none(),
                l2,
                linf,
                cosine,
            })
        })
        .collect()
}

fn logits(a: Vec<Logits>, b: Vec<Logits>) -> Result<Vec<LogitDiff>> {
    let mut rows = BTreeMap::<_, [Option<Vec<TopLogit>>; 2]>::new();
    for (side, logits) in [a, b].into_iter().enumerate() {
        for r in logits {
            let pair = rows.entry((r.layer, r.position)).or_default();
            ensure!(
                pair[side].replace(r.top).is_none(),
                "duplicate trace logit row"
            );
        }
    }
    rows.into_iter()
        .map(|((layer, position), [a, b])| {
            fn scores(top: &Option<Vec<TopLogit>>) -> Result<BTreeMap<i32, f64>> {
                let mut map = BTreeMap::new();
                for t in top.iter().flatten() {
                    ensure!(
                        t.token_id >= 0
                            && t.logit.is_finite()
                            && map.insert(t.token_id, f64::from(t.logit)).is_none(),
                        "invalid top logits"
                    );
                }
                Ok(map)
            }
            let x = scores(&a)?;
            let y = scores(&b)?;
            let entered = y.keys().filter(|t| !x.contains_key(t)).copied().collect();
            let exited = x.keys().filter(|t| !y.contains_key(t)).copied().collect();
            let shared: Vec<_> = x
                .iter()
                .filter_map(|(t, &a)| y.get(t).map(|&b| (a, b)))
                .collect();
            let (l2, linf, kl) = if shared.is_empty() {
                (None, None, None)
            } else {
                let a = log_softmax(shared.iter().map(|&(a, _)| a));
                let b = log_softmax(shared.iter().map(|&(_, b)| b));
                let kl = a
                    .iter()
                    .zip(b)
                    .map(|(p, q)| p.exp() * (p - q))
                    .sum::<f64>()
                    .max(0.);
                let (mut d2, mut max) = (0., 0f64);
                for &(a, b) in &shared {
                    let d = b - a;
                    d2 += d * d;
                    max = max.max(d.abs());
                }
                (Some(d2.sqrt()), Some(max), Some(kl))
            };
            Ok(LogitDiff {
                layer,
                position,
                a,
                b,
                entered,
                exited,
                shared_count: shared.len(),
                l2,
                linf,
                kl_a_to_b_shared: kl,
            })
        })
        .collect()
}

fn log_softmax(logits: impl Iterator<Item = f64> + Clone) -> Vec<f64> {
    let top = logits.clone().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = logits.clone().map(|x| (x - top).exp()).sum::<f64>().ln();
    logits.map(|x| (x - top) - log_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires two local Lanternfish exports"]
    fn diff_real_bf16_exports() -> Result<()> {
        let load = |env: &str| -> Result<Value> {
            let dir = std::path::PathBuf::from(std::env::var(env)?);
            let (m, _) = manifest(&dir, HARD_BYTE_CAP)?;
            let expected = verify(&dir, &m.spec)?;
            let slice = Slice {
                view: View::DiffInput,
                prompt: 0,
                layer: None,
                role: None,
                start: None,
                end: None,
            };
            let ReadReply::Ok(value) = read(
                dir.parent().context("export has no parent")?,
                &slice,
                &json!({"prompts":[expected]}),
            )?
            else {
                anyhow::bail!("real export did not return a diff slice")
            };
            Ok(value)
        };
        let result = compare(load("BREWERY_DIFF_TRACE_A")?, load("BREWERY_DIFF_TRACE_B")?)?;
        let changed: Vec<_> = result
            .activations
            .iter()
            .filter(|r| r.l2.is_some_and(|x| x > 0.))
            .map(|r| (r.key.layer, r.key.role, r.key.position, r.l2.unwrap()))
            .collect();
        let native: Vec<_> = result
            .model_logits
            .iter()
            .filter(|r| {
                r.l2.is_some_and(|x| x > 0.) || !r.entered.is_empty() || !r.exited.is_empty()
            })
            .map(|r| r.position)
            .collect();
        let lens: Vec<_> = result
            .logit_lens
            .iter()
            .filter(|r| {
                r.l2.is_some_and(|x| x > 0.) || !r.entered.is_empty() || !r.exited.is_empty()
            })
            .map(|r| (r.layer, r.position))
            .collect();
        eprintln!(
            "{} activations; changed: {changed:?}; model logits: {native:?}; logit lens: {lens:?}",
            result.activations.len()
        );
        assert_eq!(result.activations.len(), 45);
        assert_eq!(
            changed
                .iter()
                .map(|&(layer, role, position, _)| (layer, role, position))
                .collect::<Vec<_>>(),
            [(23, Role::LOut, 4), (23, Role::FfnOut, 4)]
        );
        assert_eq!(native, [4]);
        assert_eq!(lens, [(Some(23), 4)]);
        Ok(())
    }
}
