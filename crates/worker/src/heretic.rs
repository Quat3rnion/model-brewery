use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub const RUN: &str = "heretic";
pub const REPRODUCE: &str = "heretic_reproduce";
pub const RUNTIME: &str = "BREWERY_HERETIC_RUNTIME";
pub const COMMIT: &str = "4f89482c0d63f2552a582215939910f9f45a3e5f";
pub const MAX_REQUEST: usize = 128 << 10;

pub fn job(kind: &str) -> bool {
    matches!(kind, RUN | REPRODUCE)
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Invocation {
    pub profile: String,
    pub trials: u8,
    pub trial_index: u8,
    pub seed: u32,
}

impl Invocation {
    pub fn validate(&self) -> Result<()> {
        let bytes = self.profile.as_bytes();
        ensure!(
            (1..=64).contains(&bytes.len())
                && bytes[0].is_ascii_lowercase()
                && bytes.iter().all(|b| b.is_ascii_lowercase()
                    || b.is_ascii_digit()
                    || matches!(b, b'-' | b'_')),
            "invalid operator-installed Heretic profile name"
        );
        ensure!(
            (1..=64).contains(&self.trials) && self.trial_index < self.trials,
            "expected 1 to 64 trials and a trial index below the trial budget"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub kind: String,
    pub base_sha256: String,
    pub base_hf_commit: Option<String>,
    pub invocation: Value,
    pub manifest: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StudyTrial {
    index: u8,
    scores: Vec<StudyScore>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StudyScore {
    name: String,
    value: f64,
    baseline: f64,
}

impl Options {
    pub fn parse(kind: &str, options: &Value) -> Result<Self> {
        ensure!(
            serde_json::to_vec(options)?.len() <= MAX_REQUEST,
            "Heretic request exceeds 128 KiB"
        );
        let parsed: Self = serde_json::from_value(options.clone())?;
        ensure!(
            parsed.kind == kind && crate::is_digest(&parsed.base_sha256),
            "invalid Heretic base SHA-256"
        );
        match kind {
            RUN => {
                ensure!(
                    parsed.manifest.is_none() && parsed.base_hf_commit.is_none(),
                    "unexpected community manifest"
                );
                serde_json::from_value::<Invocation>(parsed.invocation.clone())?.validate()?;
            }
            REPRODUCE => {
                let manifest = parsed
                    .manifest
                    .as_ref()
                    .filter(|m| m["version"] == "4")
                    .ok_or_else(|| anyhow::anyhow!("expected a community v4 manifest"))?;
                let commit = parsed.base_hf_commit.as_deref().unwrap_or_default();
                ensure!(
                    commit.len() == 40
                        && commit
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                        && manifest["settings"]["model_commit"] == commit
                        && parsed.invocation.is_null(),
                    "community manifest base Hugging Face commit mismatch"
                );
            }
            _ => anyhow::bail!("unknown Heretic job kind"),
        }
        Ok(parsed)
    }

    pub fn verify(&self, receipt: &Value, run_root: Option<&Path>) -> Result<PathBuf> {
        let journal = &receipt["journal"];
        let export = receipt["export_sha256"].as_str().unwrap_or_default();
        let runtime = &journal["runtime"];
        ensure!(
            receipt["kind"] == self.kind
                && journal["kind"] == self.kind
                && journal["schema"] == "brewery.heretic-journal.v1"
                && journal["base_sha256"] == self.base_sha256
                && journal["base_hf_commit"] == serde_json::to_value(&self.base_hf_commit)?
                && journal["invocation"] == self.invocation
                && crate::is_digest(export)
                && journal["export_sha256"] == export
                && runtime["heretic"] == "2.1.0"
                && runtime["commit"] == COMMIT
                && runtime["licence"] == "AGPL-3.0-or-later"
                && ["torch", "transformers"].iter().all(|k| runtime[*k]
                    .as_str()
                    .is_some_and(|s| !s.is_empty() && s.len() <= 128))
                && ["lock_sha256", "environment_sha256", "launcher_sha256"]
                    .iter()
                    .all(|k| runtime[*k].as_str().is_some_and(crate::is_digest))
                && journal["effective_config"].is_string(),
            "invalid Heretic export or provenance receipt"
        );
        if self.kind == RUN {
            ensure!(
                journal["seed"] == self.invocation["seed"]
                    && journal["trial_index"] == self.invocation["trial_index"]
                    && journal["source_config"].is_string(),
                "Heretic trial or config mismatch"
            );
            if let Some(trials) = journal.get("trials") {
                let trials: Vec<StudyTrial> = serde_json::from_value(trials.clone())?;
                let budget = self.invocation["trials"].as_u64().unwrap_or_default();
                let mut indexes = BTreeSet::new();
                let valid = trials.iter().all(|trial| {
                    let index = u64::from(trial.index);
                    (1..=budget).contains(&index)
                        && indexes.insert(index)
                        && (1..=16).contains(&trial.scores.len())
                        && trial.scores.iter().all(|score| {
                            (1..=80).contains(&score.name.chars().count())
                                && !score.name.chars().any(char::is_control)
                                && score.value.is_finite()
                                && score.baseline.is_finite()
                        })
                });
                ensure!(
                    (1..=budget as usize).contains(&trials.len())
                        && valid
                        && journal["selected_study_trial"]
                            .as_u64()
                            .is_some_and(|i| indexes.contains(&i)),
                    "invalid Heretic trial summary"
                );
            }
        } else if let Some(manifest) = &self.manifest {
            ensure!(
                manifest["hashes"]["model.safetensors"] == export
                    && manifest["settings"]["seed"] == journal["seed"],
                "Heretic reproduction does not match its manifest"
            );
        }
        let path = PathBuf::from(receipt["run_directory"].as_str().unwrap_or_default());
        crate::absolute(&path)?;
        ensure!(
            path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n
                .strip_prefix("brewery-")
                .is_some_and(|s| s.len() == 6 && s.bytes().all(|b| b.is_ascii_alphanumeric())))
                && run_root.is_none_or(|root| path.parent() == Some(root)),
            "invalid private Heretic export directory"
        );
        Ok(path)
    }
}
