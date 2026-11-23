//! The heldout scores are private check evidence, not a job or Library output.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

pub const FILE: &str = "token-loss.v1.json";
pub const MAX_BYTES: u64 = 256 << 10;
pub const MAX_TOKENS: usize = 511;
pub const HELDOUT: &str = include_str!("../../eval/data/edit-checks/heldout.txt");

#[derive(Deserialize, Serialize)]
pub struct Reference {
    pub sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub id: u32,
    pub piece_hex: String,
    pub parent_logprob: f64,
    pub child_logprob: f64,
}

impl Score {
    pub fn loss_delta(&self) -> f64 {
        self.parent_logprob - self.child_logprob
    }

    pub fn piece(&self) -> Option<Vec<u8>> {
        if self.piece_hex.len() > 1024 || !self.piece_hex.len().is_multiple_of(2) {
            return None;
        }
        self.piece_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|chunk| {
                let nibble = |c| match c {
                    b'0'..=b'9' => Some(c - b'0'),
                    b'a'..=b'f' => Some(c - b'a' + 10),
                    _ => None,
                };
                Some(nibble(chunk[0])? * 16 + nibble(chunk[1])?)
            })
            .collect()
    }
}

pub type Sentences = Vec<Vec<Score>>;

pub fn heldout_sha256() -> String {
    format!("{:x}", Sha256::digest(HELDOUT))
}

pub fn store(
    directory: &Path,
    sentences: &Sentences,
    parent_ppl: f64,
    child_ppl: f64,
) -> Result<Reference, crate::Error> {
    let mut count = 0;
    let (mut parent, mut child) = (0., 0.);
    if sentences.len() != HELDOUT.lines().count() {
        return Err("incorrect heldout sentence count".into());
    }
    for sentence in sentences {
        if sentence.is_empty() || sentence.len() > MAX_TOKENS {
            return Err("incorrect heldout token count".into());
        }
        for token in sentence {
            if ![token.parent_logprob, token.child_logprob]
                .into_iter()
                .all(|lp| lp.is_finite() && lp <= 0.)
                || token.piece().is_none()
            {
                return Err("invalid heldout token scores".into());
            }
            count += 1;
            parent += token.parent_logprob;
            child += token.child_logprob;
        }
    }
    if [(parent, parent_ppl), (child, child_ppl)]
        .into_iter()
        .any(|(sum, ppl)| !sum.is_finite() || (-sum / count as f64 - ppl.ln()).abs() > 1e-10)
    {
        return Err("heldout token scores do not match perplexity".into());
    }
    let mut bytes = serde_json::to_vec(sentences)?;
    bytes.push(b'\n'); // brewery_worker::atomic adds one newline.
    if bytes.len() as u64 > MAX_BYTES {
        return Err("heldout token scores exceed the size limit".into());
    }
    brewery_worker::atomic(&directory.join(FILE), sentences)?;
    Ok(Reference {
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    })
}

pub fn read(path: &Path, expected: &Reference) -> Result<Sentences, crate::web::ApiError> {
    use crate::web::ApiError;
    use axum::http::StatusCode;
    let file = brewery_worker::weight_source::open_regular(path)
        .map_err(|_| ApiError::not_found("token loss artifact"))?;
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(ApiError::storage)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "token loss artifact exceeds the size limit".into(),
        ));
    }
    if format!("{:x}", Sha256::digest(&bytes)) != expected.sha256 {
        return Err(ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            "token loss artifact SHA-256 does not match the check bundle".into(),
        ));
    }
    serde_json::from_slice(&bytes).map_err(ApiError::storage)
}
