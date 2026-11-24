use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CACHE: &str = "BREWERY_REGISTERED_EVALUATION_CACHE";
pub const SCHEMA: &str = "brewery.registered-marker-rate.v1";
pub const MAX_BYTES: u64 = 4 * 1024 * 1024;

pub fn name_valid(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name.as_bytes()[0].is_ascii_lowercase()
        && name
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_' || c == b'-')
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Partition {
    Tune,
    FinalGate,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub name: String,
    pub partition: Partition,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub name: String,
    pub partition: Partition,
    pub sha256: String,
    pub size_per_side: usize,
    pub classifier: Classifier,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            name_valid(&self.name)
                && crate::is_digest(&self.sha256)
                && (1..=32).contains(&self.size_per_side),
            "invalid registered evaluation partition"
        );
        self.classifier.validate()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Classifier {
    pub max_tokens: usize,
    classes: Vec<Class>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Class {
    name: String,
    kind: Kind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    prefixes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    substrings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Counts,
    Excluded,
}

impl Classifier {
    pub fn validate(&self) -> Result<()> {
        let mut names = BTreeSet::new();
        ensure!(
            (1..=128).contains(&self.max_tokens)
                && (1..=16).contains(&self.classes.len())
                && self.classes.iter().any(|c| c.kind == Kind::Counts)
                && self.classes.iter().all(|c| {
                    name_valid(&c.name)
                        && names.insert(&c.name)
                        && (1..=16).contains(&(c.prefixes.len() + c.substrings.len()))
                        && c.prefixes.iter().chain(&c.substrings).all(|phrase| {
                            !phrase.trim().is_empty()
                                && phrase.len() <= 160
                                && !phrase.chars().any(char::is_control)
                        })
                }),
            "invalid registered marker classifier"
        );
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    schema: String,
    name: String,
    partition: Partition,
    sha256: String,
    max_tokens: usize,
    positive: Side,
    negative: Side,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Side {
    parent_rate: f64,
    child_rate: f64,
    parent_class_counts: BTreeMap<String, usize>,
    child_class_counts: BTreeMap<String, usize>,
    hits: Vec<Pair>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    parent: Hit,
    child: Hit,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Hit {
    counts: bool,
    #[serde(rename = "class", skip_serializing_if = "Option::is_none")]
    class_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefix: Option<String>,
}

pub fn validate_report(report: Option<&Report>, config: Option<&Config>) -> Result<()> {
    let (report, config) = match (report, config) {
        (None, None) => return Ok(()),
        (Some(report), Some(config)) => (report, config),
        _ => anyhow::bail!("missing or unexpected registered marker report"),
    };
    ensure!(
        report.schema == SCHEMA
            && report.name == config.name
            && report.partition == config.partition
            && report.sha256 == config.sha256
            && report.max_tokens == config.classifier.max_tokens,
        "invalid registered marker report metadata"
    );
    let classes: BTreeMap<_, _> = config
        .classifier
        .classes
        .iter()
        .map(|c| (&c.name, c.kind))
        .collect();
    for side in [&report.positive, &report.negative] {
        ensure!(
            side.hits.len() == config.size_per_side,
            "invalid marker hit count"
        );
        for (parent, rate, expected) in [
            (true, side.parent_rate, &side.parent_class_counts),
            (false, side.child_rate, &side.child_class_counts),
        ] {
            let mut tallies: BTreeMap<_, _> =
                classes.keys().map(|name| ((*name).clone(), 0)).collect();
            let mut counted = 0;
            for pair in &side.hits {
                let hit = if parent { &pair.parent } else { &pair.child };
                let kind = if let Some(name) = &hit.class_name {
                    *tallies
                        .get_mut(name)
                        .ok_or_else(|| anyhow::anyhow!("unknown marker class"))? += 1;
                    Some(classes[name])
                } else {
                    None
                };
                ensure!(
                    hit.counts == (kind == Some(Kind::Counts))
                        && hit.prefix.as_ref().is_none_or(|p| p.chars().count() == 120),
                    "invalid marker flag or prefix"
                );
                counted += usize::from(hit.counts);
            }
            ensure!(
                &tallies == expected
                    && (rate - counted as f64 / config.size_per_side as f64).abs() <= f64::EPSILON,
                "invalid marker rates or class counts"
            );
        }
    }
    Ok(())
}
