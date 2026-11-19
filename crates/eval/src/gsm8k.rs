use crate::{digest, hash_ok, Result};
use brewery_hash_pool::ProgressEvent;
use serde::{Deserialize, Serialize};

pub const REVISION: &str = "3101c7d5072418e28b9008a6636bde82a006892c";
pub const SHA: &str = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14";
pub const PROTOCOL: &str = "gsm8k-zero-shot-raw-v1";
pub const SCHEMA: &str = "brewery-gsm8k-predictions-v1";
pub const MAX_TOKENS: u32 = 512;
const TEST: &str = include_str!("../data/gsm8k-test.jsonl");

pub struct Question {
    pub prompt: String,
    pub sha256: String,
    answer: String,
}

pub fn test() -> Result<Vec<Question>> {
    #[derive(Deserialize)]
    struct Row {
        question: String,
        answer: String,
    }
    if digest(TEST) != SHA || TEST.lines().count() != 1319 {
        return Err("pinned GSM8K test split digest or size does not match".into());
    }
    TEST.lines()
        .map(|line| {
            let row: Row = serde_json::from_str(line)?;
            Ok(Question {
                prompt: format!(
                    "Solve the following math word problem. Show your reasoning and put the final numeric answer on the last line as #### <number>.\n\nQuestion: {}\nAnswer:",
                    row.question
                ),
                sha256: digest(row.question),
                answer: extract(&row.answer).ok_or("invalid reference answer")?,
            })
        })
        .collect()
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Predictions {
    pub schema: String,
    pub source_sha256: String,
    pub dataset_sha256: String,
    pub protocol_id: String,
    pub synthetic: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<serde_json::Value>,
    pub predictions: Vec<Prediction>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub id: usize,
    pub question_sha256: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct Score {
    pub correct: usize,
    pub total: usize,
    pub unparsed: usize,
    pub accuracy_basis_points: usize,
}

#[derive(Serialize)]
pub struct Comparison {
    pub changed_answers: usize,
    pub changed_answer_ids: Vec<usize>,
    pub delta_correct: i64,
    pub within_two_percentage_points: bool,
}

impl Predictions {
    pub fn new(
        source_sha256: String,
        provenance: serde_json::Value,
        test: &[Question],
        completions: Vec<String>,
        synthetic: bool,
    ) -> Result<Self> {
        if completions.len() != test.len() {
            return Err("did not receive one response per GSM8K test question".into());
        }
        Ok(Self {
            schema: SCHEMA.into(),
            dataset_sha256: SHA.into(),
            protocol_id: PROTOCOL.into(),
            source_sha256,
            synthetic,
            provenance: Some(provenance),
            predictions: test
                .iter()
                .zip(completions)
                .enumerate()
                .map(|(i, (question, text))| Prediction {
                    id: i + 1,
                    question_sha256: question.sha256.clone(),
                    text,
                })
                .collect(),
        })
    }

    pub fn score(
        &self,
        test: &[Question],
        synthetic: bool,
        progress: &crate::progress::Progress,
        name: &'static str,
    ) -> Result<(Score, Vec<Option<String>>)> {
        if self.schema != SCHEMA
            || self.dataset_sha256 != SHA
            || self.protocol_id != PROTOCOL
            || self.predictions.len() != test.len()
            || !hash_ok(&self.source_sha256)
            || self.synthetic != synthetic
            || (!synthetic && self.provenance.as_ref().is_none_or(|p| !p.is_object()))
        {
            return Err("incompatible prediction schema, source, synthetic flag, or split".into());
        }
        // Check the immutable row identities in split order so malformed
        // input reports the same first offending row at every thread budget.
        for (i, (prediction, question)) in self.predictions.iter().zip(test).enumerate() {
            if prediction.id != i + 1
                || prediction.question_sha256 != question.sha256
                || prediction.text.len() > 64 * 1024
            {
                return Err(
                    format!("invalid or mismatched prediction for GSM8K row {}", i + 1).into(),
                );
            }
        }
        let mut phase = progress.start(name, "questions", Some(test.len() as u64));
        let parsed = brewery_hash_pool::map_cpu_observed(
            &self.predictions,
            |_, prediction, _| Ok(extract(&prediction.text)),
            |slot, index, event| {
                match event {
                    ProgressEvent::Started => {
                        phase.assign(slot, &format!("Question {}", index + 1))
                    }
                    ProgressEvent::Finished => {
                        phase.unassign(slot);
                        phase.advance(1);
                    }
                    ProgressEvent::Stopped => phase.unassign(slot),
                    ProgressEvent::Advanced(_) => {}
                }
                Ok(())
            },
        );
        phase.finish();
        let parsed = parsed.map_err(|error| format!("{error:#}"))?;
        let correct = parsed
            .iter()
            .zip(test)
            .filter(|(answer, question)| answer.as_ref() == Some(&question.answer))
            .count();
        let unparsed = parsed.iter().filter(|answer| answer.is_none()).count();
        let total = test.len();
        Ok((
            Score {
                correct,
                total,
                unparsed,
                accuracy_basis_points: (correct * 10_000 + total / 2) / total,
            },
            parsed,
        ))
    }
}

pub fn compare(
    candidate: &(Score, Vec<Option<String>>),
    baseline: &(Score, Vec<Option<String>>),
) -> Comparison {
    let delta_correct = candidate.0.correct as i64 - baseline.0.correct as i64;
    let changed_answer_ids: Vec<_> = candidate
        .1
        .iter()
        .zip(&baseline.1)
        .enumerate()
        .filter_map(|(i, (a, b))| (a != b).then_some(i + 1))
        .collect();
    Comparison {
        changed_answers: changed_answer_ids.len(),
        changed_answer_ids,
        delta_correct,
        within_two_percentage_points: delta_correct * 100 >= -2 * candidate.0.total as i64,
    }
}

// No fallback to a random number in the reasoning. Normalize decimal numbers
// exactly, including valid thousands separators and signed zero, without floats.
fn extract(text: &str) -> Option<String> {
    let value = text
        .trim_end()
        .lines()
        .next_back()?
        .trim()
        .strip_prefix("####")?
        .trim();
    let (negative, unsigned) = match value.as_bytes().first()? {
        b'-' => (true, &value[1..]),
        b'+' => (false, &value[1..]),
        _ => (false, value),
    };
    let (whole, decimal) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if !decimal.bytes().all(|b| b.is_ascii_digit()) || (whole.is_empty() && decimal.is_empty()) {
        return None;
    }
    for (i, group) in whole.split(',').enumerate() {
        if !group.bytes().all(|b| b.is_ascii_digit())
            || (whole.contains(',')
                && ((i == 0 && !(1..=3).contains(&group.len())) || (i > 0 && group.len() != 3)))
        {
            return None;
        }
    }
    let whole = whole.replace(',', "");
    let whole = whole.trim_start_matches('0');
    let decimal = decimal.trim_end_matches('0');
    let mut out = if negative && (!whole.is_empty() || !decimal.is_empty()) {
        "-".to_owned()
    } else {
        String::new()
    };
    out.push_str(if whole.is_empty() { "0" } else { whole });
    if !decimal.is_empty() {
        out.push('.');
        out.push_str(decimal);
    }
    Some(out)
}
