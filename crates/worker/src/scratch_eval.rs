use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub mod paired;
pub mod registered;

pub const GGUF: &str = "scratch_gguf";
pub const CHECKS: &str = "scratch_checks";
pub const SCRIPT: &str = "BREWERY_SCRATCH_CHECKS_SCRIPT";
pub const RUNNER: &str = "BREWERY_SCRATCH_CHECKS_RUNNER";
pub const RUNNER_SHA: &str = "BREWERY_SCRATCH_CHECKS_RUNNER_SHA256";
pub const RUNNER_LIB_DIR: &str = "BREWERY_SCRATCH_CHECKS_RUNNER_LIB_DIR";
pub const CACHE: &str = "BREWERY_SCRATCH_CHECKS_CACHE";
pub const CONTRAST_CACHE: &str = "BREWERY_CONTRAST_CACHE";
const BENIGN_REFUSAL: &str = "benign_refusal";
const BENIGN_REFUSAL_SHA256: &str =
    "0a3af7d442e3043a713ccb050c4291abf360614f4164cac16e1535d46ea5d35e";
const BENIGN_REFUSAL_COUNT: usize = 16;
pub const GRANT: &str = "scratch_grant";
pub const CHECK_INPUTS: [&str; 4] = ["gguf", "gguf", "safetensors", "safetensors"];
pub const CHECK_DIGESTS: [&str; 4] = [
    "parent_gguf_sha256",
    "revision_gguf_sha256",
    "parent_sha256",
    "revision_sha256",
];

pub fn private_output(kind: &str) -> bool {
    matches!(
        kind,
        GGUF | CHECKS | crate::locate_edit::JOB | crate::ablation_search::SCREEN
    ) || crate::heretic::job(kind)
        || matches!(kind, crate::hgn_edit::JOB | crate::hgn_checks::JOB)
}

pub fn granted(input: &crate::Input) -> bool {
    input.extra.get(GRANT) == Some(&Value::Bool(true))
}

pub fn gguf_options(options: &Value) -> bool {
    *options == json!({"output_format":"gguf","gguf_type":"BF16"})
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorConfig {
    pub prompt_sets: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    refusal_prefixes: Vec<String>,
}

impl BehaviorConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            ["slop", "sycophancy", "response_length"]
                .iter()
                .all(|key| self.prompt_sets.contains_key(*key))
                && self.prompt_sets.iter().all(|(key, digest)| {
                    match key.as_str() {
                        "slop" | "sycophancy" | "response_length" => crate::is_digest(digest),
                        BENIGN_REFUSAL => digest == BENIGN_REFUSAL_SHA256,
                        _ => false,
                    }
                }),
            "expected pinned slop, sycophancy and length sets, optionally the reviewed benign_refusal set"
        );
        ensure!(
            self.refusal_prefixes.len() <= 32
                && (self.refusal_prefixes.is_empty()
                    || self.prompt_sets.contains_key(BENIGN_REFUSAL))
                && self.refusal_prefixes.iter().all(|prefix| {
                    !prefix.trim().is_empty()
                        && prefix.len() <= 160
                        && !prefix.chars().any(char::is_control)
                }),
            "invalid refusal prefixes"
        );
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub behavior_direction: Option<BehaviorConfig>,
    pub registered_marker: Option<registered::Config>,
}

pub fn check_options(options: &Value) -> Result<Options> {
    let options: Options = serde_json::from_value(options.clone())?;
    if let Some(config) = &options.behavior_direction {
        config.validate()?;
    }
    if let Some(config) = &options.registered_marker {
        config.validate()?;
    }
    Ok(options)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BehaviorMetric {
    pub name: String,
    pub parent: f64,
    pub child: f64,
    pub better: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    parent: String,
    child: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RefusalFlag {
    parent: bool,
    child: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BehaviorReport {
    schema: String,
    config: BehaviorConfig,
    max_new_tokens: usize,
    max_continuation_tokens: usize,
    cpu_threads: usize,
    counts: BTreeMap<String, usize>,
    samples: BTreeMap<String, Sample>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    benign_refusal_flags: Option<Vec<RefusalFlag>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    paired_outcomes: Option<BTreeMap<String, Vec<paired::PairedOutcome>>>,
    kl_prompts: usize,
    kl_continuation_positions: usize,
    pub metrics: Vec<BehaviorMetric>,
}

pub fn validate_behavior(
    report: Option<&BehaviorReport>,
    config: Option<&BehaviorConfig>,
) -> Result<()> {
    let (report, config) = match (report, config) {
        (Some(report), Some(config)) => (report, config),
        (None, None) => return Ok(()),
        _ => anyhow::bail!("missing or unexpected behavior-direction report"),
    };
    let refusal = config.prompt_sets.contains_key(BENIGN_REFUSAL);
    let refusal_rates = match (&report.benign_refusal_flags, refusal) {
        (Some(flags), true)
            if flags.len() == BENIGN_REFUSAL_COUNT
                && report.counts.get(BENIGN_REFUSAL) == Some(&BENIGN_REFUSAL_COUNT) =>
        {
            Some((
                flags.iter().filter(|flag| flag.parent).count() as f64
                    / BENIGN_REFUSAL_COUNT as f64,
                flags.iter().filter(|flag| flag.child).count() as f64 / BENIGN_REFUSAL_COUNT as f64,
            ))
        }
        (None, false) => None,
        _ => anyhow::bail!("invalid benign refusal flags"),
    };
    ensure!(
        report.schema == "brewery.behavior-direction.v1"
            && &report.config == config
            && report.max_new_tokens == 64
            && report.max_continuation_tokens == 4
            && (1..=16).contains(&report.cpu_threads)
            && report.counts.keys().eq(config.prompt_sets.keys())
            && report.counts.values().all(|n| (1..=32).contains(n))
            && report.kl_prompts == report.counts["response_length"]
            && report.kl_continuation_positions <= 4 * report.kl_prompts
            && report.samples.keys().map(String::as_str).eq([
                "response_length",
                "slop",
                "sycophancy"
            ])
            && report
                .samples
                .values()
                .all(|s| s.parent.len() <= 160 && s.child.len() <= 160)
            && report.metrics.len() == 5 + usize::from(refusal),
        "invalid behavior-direction metadata or samples"
    );
    if let Some(outcomes) = &report.paired_outcomes {
        paired::check_outcomes(outcomes, &report.counts, |name| {
            let mut matches = report.metrics.iter().filter(|m| m.name == name);
            match (matches.next(), matches.next()) {
                (Some(m), None) => Some((m.parent, m.child)),
                _ => None,
            }
        })?;
    }
    for (name, better, max, is_kl) in [
        ("benign_refusal_rate", "lower", 1., false),
        ("slop_marker_rate", "lower", 1., false),
        ("sycophancy_agreement_rate", "lower", 1., false),
        ("response_length_tokens", "neither", 64., false),
        ("kl_first_token_nats", "lower", f64::MAX, true),
        ("kl_continuation_nats", "lower", f64::MAX, true),
    ] {
        if name == "benign_refusal_rate" && !refusal {
            continue;
        }
        let mut matches = report.metrics.iter().filter(|m| m.name == name);
        ensure!(
            matches.next().is_some_and(|m| {
                m.better == better
                    && [m.parent, m.child]
                        .into_iter()
                        .all(|x| (0. ..=max).contains(&x))
                    && (!is_kl || m.parent == 0.)
                    && (name != "benign_refusal_rate" || refusal_rates == Some((m.parent, m.child)))
                    && (name != "kl_continuation_nats"
                        || report.kl_continuation_positions != 0
                        || m.child == 0.)
            }) && matches.next().is_none(),
            "invalid behavior-direction metric: {name}"
        );
    }
    Ok(())
}

fn check_behavior(report: Option<&Value>, config: Option<&BehaviorConfig>) -> Result<()> {
    let report: Option<BehaviorReport> = report.cloned().map(serde_json::from_value).transpose()?;
    validate_behavior(report.as_ref(), config)
}

#[cfg(test)]
mod behavior_tests {
    use super::*;

    #[test]
    fn validates_opt_in_against_pins_and_changes_only_allowed_settings() {
        let pins = json!({"slop":"a".repeat(64), "response_length":"b".repeat(64),
                          "sycophancy":"c".repeat(64)});
        assert!(
            check_options(&json!({}))
                .unwrap()
                .behavior_direction
                .is_none()
        );
        let config = check_options(&json!({"behavior_direction":{"prompt_sets":pins}}))
            .unwrap()
            .behavior_direction
            .unwrap();
        assert!(
            check_options(&json!({"behavior_direction": {
                "prompt_sets": pins, "classifier": "/untrusted/path"
            }}))
            .is_err()
        );
        assert!(
            check_options(&json!({"behavior_direction": {
                "prompt_sets": {"slop":"../../outside"}, "refusal_prefixes":["\n"]
            }}))
            .is_err()
        );

        let mut report = json!({
            "schema":"brewery.behavior-direction.v1",
            "config":config, "max_new_tokens":64, "max_continuation_tokens":4, "cpu_threads":4,
            "counts":{"slop":1,"response_length":1,"sycophancy":1},
            "samples":{"slop":{"parent":"","child":""},"response_length":{"parent":"","child":""},
                       "sycophancy":{"parent":"","child":""}},
            "kl_prompts":1,"kl_continuation_positions":4,
            "metrics":[
                {"name":"slop_marker_rate","better":"lower","parent":0,"child":1},
                {"name":"sycophancy_agreement_rate","better":"lower","parent":0,"child":0},
                {"name":"response_length_tokens","better":"neither","parent":0,"child":33},
                {"name":"kl_first_token_nats","better":"lower","parent":0,"child":0.01},
                {"name":"kl_continuation_nats","better":"lower","parent":0,"child":0.03}
            ]
        });
        assert!(check_behavior(Some(&report), Some(&config)).is_ok());
        let mut unchecked = report.clone();
        unchecked["config"]["prompt_sets"]["refusal"] = json!("d".repeat(64));
        assert!(check_options(&json!({"behavior_direction":unchecked["config"]})).is_err());
        unchecked["config"]["prompt_sets"]
            .as_object_mut()
            .unwrap()
            .remove("refusal");
        unchecked["config"]["prompt_sets"]["benign_refusal"] = json!("d".repeat(64));
        assert!(check_options(&json!({"behavior_direction":unchecked["config"]})).is_err());
        unchecked["config"]["prompt_sets"]["benign_refusal"] = json!(BENIGN_REFUSAL_SHA256);
        let benign_config = check_options(&json!({"behavior_direction":unchecked["config"]}))
            .unwrap()
            .behavior_direction
            .unwrap();
        unchecked["counts"]["benign_refusal"] = json!(16);
        unchecked["metrics"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name":"benign_refusal_rate","better":"lower","parent":0.0625,"child":0}));
        assert!(check_behavior(Some(&unchecked), Some(&benign_config)).is_err());
        let mut flags = vec![json!({"parent":false,"child":false}); 16];
        flags[0]["parent"] = json!(true);
        unchecked["benign_refusal_flags"] = json!(flags);
        assert!(check_behavior(Some(&unchecked), Some(&benign_config)).is_ok());
        unchecked["benign_refusal_flags"][0]["parent"] = json!(false);
        assert!(check_behavior(Some(&unchecked), Some(&benign_config)).is_err());
        unchecked["benign_refusal_flags"][0]["parent"] = json!(true);
        unchecked["benign_refusal_flags"][0]["reply"] = json!("not allowed");
        assert!(check_behavior(Some(&unchecked), Some(&benign_config)).is_err());

        // Per-prompt paired outcomes are optional, and must match counts and rates.
        report["paired_outcomes"] = json!({
            "slop": [{"parent":false,"child":true}],
            "sycophancy": [{"parent":false,"child":false}],
        });
        assert!(check_behavior(Some(&report), Some(&config)).is_ok());
        report["paired_outcomes"]["slop"][0]["parent"] = json!(true);
        assert!(check_behavior(Some(&report), Some(&config)).is_err());
        report["paired_outcomes"]["slop"][0]["parent"] = json!(false);
        report["paired_outcomes"]["slop"][0]["extra"] = json!(1);
        assert!(check_behavior(Some(&report), Some(&config)).is_err());
        report["paired_outcomes"]["slop"][0] = json!({"parent":false,"child":true});
        report["paired_outcomes"]["slop"] = json!(vec![json!({"parent":false,"child":true}); 2]);
        assert!(check_behavior(Some(&report), Some(&config)).is_err());
        report["paired_outcomes"]["slop"] = json!([{"parent":false,"child":true}]);
        report["paired_outcomes"]["response_length"] = json!([{"parent":false,"child":false}]);
        assert!(check_behavior(Some(&report), Some(&config)).is_err());
        report.as_object_mut().unwrap().remove("paired_outcomes");

        assert!(check_behavior(None, Some(&config)).is_err());
        assert!(check_behavior(Some(&report), None).is_err());
        report["samples"]["response_length"]["child"] = json!("x".repeat(161));
        assert!(check_behavior(Some(&report), Some(&config)).is_err());
        report["samples"]["response_length"]["child"] = json!("");
        report["config"]["prompt_sets"]["response_length"] = json!("d".repeat(64));
        assert!(check_behavior(Some(&report), Some(&config)).is_err());
    }
}

pub fn check_report(
    report: &Value,
    inputs: [&str; 4],
    runner_sha: &str,
    options: &Options,
) -> Result<()> {
    let fixtures: Value =
        serde_json::from_str(include_str!("../../eval/data/edit-checks/manifest.json"))?;
    ensure!(
        report["schema"] == "brewery.scratch-behavior.v1"
            && report["qualification"] == "not release-qualified"
            && report["runner_sha256"] == runner_sha
            && CHECK_DIGESTS
                .into_iter()
                .zip(inputs)
                .all(|(key, sha)| crate::is_digest(sha) && report[key] == sha)
            && report["fixture_sha256"] == fixtures
            && ["parent_generation", "child_generation"].iter().all(|key| {
                report[key]
                    .as_str()
                    .is_some_and(|s| !s.trim().is_empty() && s.len() <= 4096)
            })
            && report["metrics"].is_array()
            && report.get("artifact").is_none(),
        "scratch checks report does not match verified inputs or protocol"
    );
    check_behavior(
        report.get("behavior_direction"),
        options.behavior_direction.as_ref(),
    )?;
    let marker = report
        .get("registered_marker")
        .cloned()
        .map(serde_json::from_value)
        .transpose()?;
    registered::validate_report(marker.as_ref(), options.registered_marker.as_ref())
}
