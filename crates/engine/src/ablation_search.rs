use crate::{Job, integrity, progress::Progress, trace::Runtime, write_json};
use anyhow::{Context, Result, ensure};
use brewery_worker::{
    ablation_search::{self as search, Positions, Proposals, ScreenReport, ScreenRequest, Trial},
    behavior_direction::{self as direction, Behavior, Prompt, Split},
    trace,
    weight_source::scratch::ablation::{Plan, TraceRule},
};
use serde_json::Value;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const A: usize = 4346;
const B: usize = 5349;
const OPENERS: [(&str, usize); 3] = [("Certainly", 95456), ("In", 641), ("It", 2132)];
const ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n";

struct Row {
    prompt: String,
    user_last: usize,
    tokens: Vec<i32>,
    logits: Vec<f32>,
    agree_a: bool,
}

impl Row {
    fn positions(&self, selection: Positions) -> std::collections::BTreeSet<usize> {
        match selection {
            Positions::UserLast => [self.user_last].into(),
            Positions::UserLast4 => (self.user_last.saturating_sub(3)..=self.user_last).collect(),
            Positions::UserAndAnswerLast => [self.user_last, self.tokens.len() - 1].into(),
        }
    }
}

struct Trace<'a> {
    runtime: Runtime,
    model: &'a integrity::Identity,
    path: &'a Path,
    out: PathBuf,
    rules: PathBuf,
}

impl Trace<'_> {
    fn run(
        &self,
        prompt: &str,
        spec: &trace::Capture,
        rules: &[TraceRule],
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        self.model.check()?;
        if !rules.is_empty() {
            write_json(&self.rules, &serde_json::to_value(rules)?)?;
        }
        let proof = self.runtime.capture(
            self.path,
            prompt,
            spec,
            &self.out,
            (!rules.is_empty()).then_some(self.rules.as_path()),
        )?;
        self.model.check()?;
        let rows = trace::ablation_rows(&self.out, spec, &proof, rules)?;
        fs::remove_dir_all(&self.out)?;
        if !rules.is_empty() {
            fs::remove_file(&self.rules)?;
        }
        Ok(rows)
    }

    fn tokens(&self, prompt: &str) -> Result<Vec<i32>> {
        Ok(self.run(prompt, &trace::direction_probe(), &[])?.0)
    }

    fn baseline(&self, prompt: &Prompt, agree_a: bool) -> Result<Row> {
        let user = prompt.qwen_chat_prefix();
        let user_tokens = self.tokens(&user)?;
        let text = format!("{user}{ASSISTANT}");
        let expected = self.tokens(&text)?;
        let (tokens, logits) = self.run(&text, &trace::ablation_logits(expected.len() - 1), &[])?;
        ensure!(
            tokens == expected && tokens.starts_with(&user_tokens),
            "screen Qwen prompt did not preserve the user-token prefix"
        );
        Ok(Row {
            prompt: text,
            user_last: user_tokens.len() - 1,
            tokens,
            logits,
            agree_a,
        })
    }

    fn edit(&self, row: &Row, plan: &Plan, where_: Positions) -> Result<Vec<f32>> {
        let rules = plan.trace_rules(row.positions(where_))?;
        let (tokens, logits) = self.run(
            &row.prompt,
            &trace::ablation_logits(row.tokens.len() - 1),
            &rules,
        )?;
        ensure!(
            tokens == row.tokens && logits.len() == row.logits.len(),
            "screen tokenization or vocabulary changed"
        );
        Ok(logits)
    }
}

fn log_normalizer(logits: &[f32]) -> f64 {
    let max = f64::from(*logits.iter().max_by(|a, b| a.total_cmp(b)).unwrap());
    max + logits
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>()
        .ln()
}

fn kl(parent: &[f32], child: &[f32]) -> f64 {
    let (p, q) = (log_normalizer(parent), log_normalizer(child));
    parent
        .iter()
        .zip(child)
        .map(|(&a, &b)| {
            let lp = f64::from(a) - p;
            lp.exp() * (lp - (f64::from(b) - q))
        })
        .sum::<f64>()
        .max(0.0)
}

fn probability(logits: &[f32], marker: usize) -> f64 {
    (f64::from(logits[marker]) - log_normalizer(logits)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_kl_uses_the_parent_as_reference() {
        let uniform = [0.0_f32, 0.0];
        let skewed = [3.0_f32.ln(), 0.0];
        assert_eq!(kl(&uniform, &uniform), 0.0);
        assert!((kl(&uniform, &skewed) - 0.1438410362).abs() < 1e-7);
        assert!((kl(&skewed, &uniform) - 0.1308120359).abs() < 1e-7);
        assert!((probability(&skewed, 0) - 0.75).abs() < 1e-7);
        let shift = [-1000.0, -1000.0];
        assert!((kl(&uniform, &shift)).abs() < 1e-12);
    }
}

pub fn run(job: &Job, cwd: &Path, progress: &Progress) -> Result<Value> {
    let req = ScreenRequest::parse(&job.options)?;
    let [input] = job.inputs.as_slice() else {
        anyhow::bail!("screen needs one pinned GGUF")
    };
    ensure!(
        input.kind == "gguf"
            && job.prompt.is_none()
            && input.sha256.as_ref() == Some(&req.model_sha256),
        "invalid screen GGUF input"
    );
    let cache = PathBuf::from(env::var_os(direction::CACHE).context("missing contrast cache")?);
    ensure!(cache.is_absolute(), "contrast cache must be absolute");
    let pairs = direction::load_pairs(&cache, req.behavior, req.behavior.contrast_sha256())?;
    let confirm = direction::load_pairs(&cache, req.behavior, &req.confirmation_sha256)?;
    let confirmation: Vec<_> = confirm
        .iter()
        .filter(|p| p.split == Split::HeldOut)
        .collect();
    let normalized = |s: &str| {
        s.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let used: std::collections::BTreeSet<_> = pairs
        .iter()
        .map(|p| normalized(p.positive.user()))
        .collect();
    let unique: std::collections::BTreeSet<_> = confirmation
        .iter()
        .map(|p| normalized(p.positive.user()))
        .collect();
    ensure!(
        confirmation.len() == 32
            && unique.len() == 32
            && used.is_disjoint(&unique)
            && !confirmation
                .iter()
                .any(|p| pairs.iter().any(|other| other.id == p.id)),
        "provide 32 unique confirmation prompts disjoint from fit and screen prompts"
    );
    let ids = integrity::check_inputs(&job.inputs, progress)?;
    let model = &ids[0];
    ensure!(
        model.files.len() == 1 && model.sha256 == req.model_sha256,
        "screen GGUF hash mismatch"
    );
    let runtime = Runtime::stage_screen(cwd)?;
    let trace = Trace {
        out: runtime.path().join("screen-trace"),
        rules: runtime.path().join("screen-rules.json"),
        runtime,
        model,
        path: &input.path,
    };
    let held: Vec<_> = pairs.iter().filter(|p| p.split == Split::HeldOut).collect();
    ensure!(
        held.len() == req.fit.held_out_pairs && !held.is_empty(),
        "screen holdout does not match fit"
    );
    let mut baseline = progress.start("prepare_screen", "items", Some(held.len() as u64));
    let rows: Vec<_> = held
        .iter()
        .map(|pair| {
            let row = trace.baseline(
                &pair.positive,
                pair.expected_agreement.as_deref() == Some("(A)"),
            )?;
            baseline.advance(1);
            Ok::<_, anyhow::Error>(row)
        })
        .collect::<Result<_>>()?;
    baseline.finish();
    let parent_split = pairs.iter().filter(|p| p.split == Split::Train);
    let (marker, marker_tokens, slop_token) = if req.behavior == Behavior::Slop {
        let mut effect = [0.0_f64; OPENERS.len()];
        for (i, pair) in parent_split.enumerate() {
            let pos = trace.baseline(&pair.positive, false)?;
            let neg = trace.baseline(&pair.negative, false)?;
            for (j, (text, id)) in OPENERS.iter().enumerate() {
                ensure!(
                    *id < pos.logits.len() && *id < neg.logits.len(),
                    "unsupported Qwen vocabulary"
                );
                if i == 0 {
                    let tokenized = trace.tokens(&format!("{}{text}", pos.prompt))?;
                    ensure!(
                        tokenized.starts_with(&pos.tokens)
                            && tokenized[pos.tokens.len()..] == [*id as i32],
                        "unsupported Qwen slop opener tokenization"
                    );
                }
                effect[j] += probability(&pos.logits, *id) - probability(&neg.logits, *id);
            }
        }
        let i = (0..effect.len())
            .max_by(|&a, &b| effect[a].total_cmp(&effect[b]))
            .unwrap();
        let (text, id) = OPENERS[i];
        (
            format!(
                "first-token probability of {text:?}; picked by largest mean stock positive-minus-negative gap on fit prompts"
            ),
            vec![id],
            Some((id, effect[i] > 0.0)),
        )
    } else {
        for (text, id) in [("(A", A), ("(B", B)] {
            let tokens = trace.tokens(&format!("{}{text}", rows[0].prompt))?;
            ensure!(
                tokens.starts_with(&rows[0].tokens)
                    && tokens[rows[0].tokens.len()..] == [id as i32],
                "unsupported Qwen A/B tokenization"
            );
        }
        (
            "agree minus disagree first-token logit for the (A / (B answer choices".into(),
            vec![A, B],
            None,
        )
    };
    let measure = |row: &Row, logits: &[f32]| {
        if let Some((id, _)) = slop_token {
            probability(logits, id)
        } else {
            let [a, b] = [f64::from(logits[A]), f64::from(logits[B])];
            if row.agree_a { a - b } else { b - a }
        }
    };
    ensure!(
        rows.iter()
            .all(|row| marker_tokens.iter().all(|&i| i < row.logits.len())),
        "unsupported Qwen vocabulary"
    );
    let stock: Vec<_> = rows.iter().map(|r| measure(r, &r.logits)).collect();
    let hits = stock
        .iter()
        .filter(|&&score| {
            slop_token.map_or(score > 0.0, |(_, calibrated)| calibrated && score >= 0.05)
        })
        .count();
    let n = rows.len();
    let mut trials = Vec::new();
    let mut proposal = Proposals::new(&req);
    let mut phase = progress.start(
        "screen_trials",
        "items",
        Some((req.trial_budget * n) as u64),
    );
    for _ in 0..req.trial_budget {
        let candidate = proposal.next(&req, &trials)?;
        let plan = req.plan(candidate, false)?;
        let control = req.plan(candidate, true)?;
        let mut trial = Trial {
            candidate,
            reduction: 0.0,
            random_reduction: 0.0,
            kl: 0.0,
            random_kl: 0.0,
            wins_over_random: 0,
        };
        for (row, &base) in rows.iter().zip(&stock) {
            let edited = trace.edit(row, &plan, candidate.positions)?;
            let random = trace.edit(row, &control, candidate.positions)?;
            let a = measure(row, &edited);
            let b = measure(row, &random);
            trial.wins_over_random += usize::from(a < b);
            trial.reduction += (base - a) / n as f64;
            trial.random_reduction += (base - b) / n as f64;
            trial.kl += kl(&row.logits, &edited) / n as f64;
            trial.random_kl += kl(&row.logits, &random) / n as f64;
            phase.advance(1);
        }
        trials.push(trial);
    }
    phase.finish();
    let (pareto, selected) = search::frontier(&trials, n, hits, req.max_children);
    let report = ScreenReport {
        schema_version: 1,
        model_sha256: req.model_sha256.clone(),
        direction_id: req.direction_id.clone(),
        behavior: req.behavior,
        seed: req.seed,
        lanternfish_sha256: search::SHA256.into(),
        screen_sha256: req.behavior.contrast_sha256().into(),
        confirmation_sha256: req.confirmation_sha256.clone(),
        marker,
        marker_tokens,
        parent_mean: stock.iter().sum::<f64>() / n as f64,
        parent_hits: hits,
        prompts: n,
        trials,
        pareto,
        selected,
    };
    report.verify(&req)?;
    Ok(serde_json::to_value(report)?)
}
