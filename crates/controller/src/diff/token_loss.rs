use super::*;
use crate::scratch_token_loss::{self as artifact, Score};

const PAGE: usize = 128;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Slice {
    pub a: String,
    pub b: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentence: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<usize>,
}

#[derive(Serialize)]
struct Sentence {
    index: usize,
    text: &'static str,
    token_count: usize,
    logprob_gain: f64,
    share_of_net_gain: Option<f64>,
}

#[derive(Serialize)]
struct Token {
    position: usize,
    id: u32,
    piece: String,
    parent_logprob: f64,
    child_logprob: f64,
    loss_delta: f64,
}

impl Token {
    fn view((position, score): (usize, &Score)) -> Self {
        Self {
            position,
            id: score.id,
            piece: String::from_utf8_lossy(&score.piece().expect("validated token piece"))
                .into_owned(),
            parent_logprob: score.parent_logprob,
            child_logprob: score.child_logprob,
            loss_delta: score.loss_delta(),
        }
    }
}

#[derive(Serialize)]
struct Diff {
    check_bundle_id: String,
    artifact_sha256: String,
    heldout_sha256: String,
    total_tokens: usize,
    total_logprob_gain: f64,
    sentences: Vec<Sentence>,
    sentence: usize,
    start: usize,
    end: usize,
    tokens: Vec<Token>,
    top_tokens: Vec<Token>,
}

pub(super) async fn get_loss(
    State(app): State<Arc<App>>,
    query: Params<Slice>,
) -> Result<impl IntoResponse> {
    Ok(response(read_loss(app, parameters(query)?).await?))
}

pub(crate) async fn finding_loss(app: Arc<App>, query: Slice) -> Result<Value> {
    serde_json::from_slice(&read_loss(app, query).await?).map_err(ApiError::storage)
}

async fn read_loss(app: Arc<App>, q: Slice) -> Result<Bytes> {
    let pair = Pair {
        a: q.a.clone(),
        b: q.b.clone(),
    };
    pair.check()?;
    let permit = app.diffs.acquire()?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        run(&app, pair, q)
    })
    .await
    .map_err(ApiError::storage)?
}

fn run(app: &App, pair: Pair, q: Slice) -> Result<Bytes> {
    let (child, path) = {
        let store = app.store.lock().map_err(ApiError::storage)?;
        let child = tensors::scratch(&store, &pair.b)?
            .ok_or_else(|| ApiError::not_found("scratch revision"))?;
        if child.is_compiled_program() || child.provenance.parent_id != pair.a {
            return Err(invalid("a is not the parent of scratch revision b"));
        }
        let path = tensors::location(&store, &pair.b, artifact::FILE);
        (child, path)
    };
    let (bundle_id, reference) = child
        .token_loss()
        .ok_or_else(|| ApiError::not_found("token loss check evidence"))?;
    let data = artifact::read(&path, reference)?;
    let mut sentences: Vec<_> = artifact::HELDOUT
        .lines()
        .zip(&data)
        .enumerate()
        .map(|(index, (text, scores))| Sentence {
            index,
            text,
            token_count: scores.len(),
            logprob_gain: scores.iter().map(|s| -s.loss_delta()).sum(),
            share_of_net_gain: None,
        })
        .collect();
    let total: f64 = sentences.iter().map(|s| s.logprob_gain).sum();
    let total_tokens = sentences.iter().map(|s| s.token_count).sum();
    for s in &mut sentences {
        let share = s.logprob_gain / total;
        s.share_of_net_gain = share.is_finite().then_some(share);
    }
    sentences.sort_by(|a, b| b.logprob_gain.abs().total_cmp(&a.logprob_gain.abs()));
    let sentence = q.sentence.unwrap_or(sentences[0].index);
    let scores = data
        .get(sentence)
        .ok_or_else(|| ApiError::not_found("heldout sentence"))?;
    let start = q.start.unwrap_or(0);
    let end = q
        .end
        .unwrap_or(start.saturating_add(PAGE).min(scores.len()));
    if start >= end || end > scores.len() || end - start > PAGE {
        return Err(ApiError::bad(
            "invalid token range; use up to 128 predicted tokens",
        ));
    }
    let tokens = (start..end)
        .zip(&scores[start..end])
        .map(Token::view)
        .collect();
    let mut top: Vec<_> = scores.iter().enumerate().collect();
    top.sort_by(|a, b| b.1.loss_delta().abs().total_cmp(&a.1.loss_delta().abs()));
    let top_tokens = top.into_iter().take(8).map(Token::view).collect();
    let diff = Diff {
        check_bundle_id: bundle_id.into(),
        artifact_sha256: reference.sha256.clone(),
        heldout_sha256: artifact::heldout_sha256(),
        total_tokens,
        total_logprob_gain: total,
        sentences,
        sentence,
        start,
        end,
        tokens,
        top_tokens,
    };
    let json = serde_json::to_vec(&pair.report(diff)).map_err(ApiError::storage)?;
    if json.len() > trace::RESPONSE_CAP {
        return Err(too_large());
    }
    Ok(Bytes::from(json))
}
