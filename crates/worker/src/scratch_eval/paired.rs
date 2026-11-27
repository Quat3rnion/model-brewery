//! Paired parent/child outcomes for the slop and sycophancy replay checks,
//! and the small statistics used to compare them.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Two-sided 95% normal quantile, as used by the A3 evidence scripts.
const Z95: f64 = 1.959963984540054;

/// Behaviours that report one boolean pair per prompt, with their rate metric.
pub const PAIRED_BEHAVIORS: [(&str, &str); 2] = [
    ("slop", "slop_marker_rate"),
    ("sycophancy", "sycophancy_agreement_rate"),
];

/// One held-out prompt: did the behaviour show on the parent, and on the child?
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairedOutcome {
    pub parent: bool,
    pub child: bool,
}

/// Wilson 95% interval for `positives` out of `n`, clamped to `[0, 1]`.
pub fn wilson95(positives: usize, n: usize) -> (f64, f64) {
    assert!(n > 0 && positives <= n, "invalid Wilson counts");
    let (n, p) = (n as f64, positives as f64 / n as f64);
    let d = 1. + Z95 * Z95 / n;
    let center = (p + Z95 * Z95 / (2. * n)) / d;
    let radius = Z95 * ((p * (1. - p) + Z95 * Z95 / (4. * n)) / n).sqrt() / d;
    ((center - radius).max(0.), (center + radius).min(1.))
}

/// Exact two-sided McNemar p-value from the two discordant counts.
pub fn mcnemar_exact_p(improved: usize, worsened: usize) -> f64 {
    let d = improved + worsened;
    if d == 0 {
        return 1.;
    }
    // Sum C(d, i) / 2^d for i <= min(b, c) in log space, so large d cannot overflow.
    let (mut ln_choose, mut tail) = (0f64, 0f64);
    for i in 0..=improved.min(worsened) {
        if i > 0 {
            ln_choose += (((d - i + 1) as f64) / i as f64).ln();
        }
        tail += (ln_choose - d as f64 * std::f64::consts::LN_2).exp();
    }
    (2. * tail).min(1.)
}

/// Holm step-down adjustment; the result keeps the order of `p_values`.
pub fn holm_adjust(p_values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..p_values.len()).collect();
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));
    let (mut adjusted, mut out) = (0f64, vec![0.; p_values.len()]);
    for (rank, &index) in order.iter().enumerate() {
        adjusted = adjusted.max(((p_values.len() - rank) as f64 * p_values[index]).min(1.));
        out[index] = adjusted;
    }
    out
}

/// Counts of (improved, worsened) pairs: the behaviour went from present to
/// absent, or from absent to present. Lower is better for these behaviours.
pub fn discordant(outcomes: &[PairedOutcome]) -> (usize, usize) {
    let count = |parent, child| {
        outcomes
            .iter()
            .filter(|o| (o.parent, o.child) == (parent, child))
            .count()
    };
    (count(true, false), count(false, true))
}

/// Check that the booleans match the reported counts and rates.
/// `rate` returns the reported (parent, child) metric for a metric name.
pub fn check_outcomes(
    outcomes: &BTreeMap<String, Vec<PairedOutcome>>,
    counts: &BTreeMap<String, usize>,
    rate: impl Fn(&str) -> Option<(f64, f64)>,
) -> Result<()> {
    ensure!(
        outcomes.len() == PAIRED_BEHAVIORS.len()
            && PAIRED_BEHAVIORS
                .iter()
                .all(|(name, _)| outcomes.contains_key(*name)),
        "expected both slop and sycophancy paired outcomes"
    );
    for (behavior, rows) in outcomes {
        let metric = PAIRED_BEHAVIORS
            .iter()
            .find(|(name, _)| name == behavior)
            .map(|(_, metric)| *metric);
        ensure!(
            metric.is_some(),
            "unexpected paired outcome behavior: {behavior}"
        );
        ensure!(
            !rows.is_empty() && counts.get(behavior) == Some(&rows.len()),
            "paired outcomes for {behavior} do not match its count"
        );
        let n = rows.len() as f64;
        let parent = rows.iter().filter(|o| o.parent).count() as f64 / n;
        let child = rows.iter().filter(|o| o.child).count() as f64 / n;
        ensure!(
            rate(metric.unwrap_or_default()) == Some((parent, child)),
            "paired outcomes for {behavior} do not match its rate"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn mcnemar_matches_known_exact_values() {
        near(mcnemar_exact_p(0, 0), 1.);
        near(mcnemar_exact_p(0, 5), 0.0625);
        near(mcnemar_exact_p(5, 0), 0.0625);
        near(mcnemar_exact_p(3, 3), 1.);
        near(mcnemar_exact_p(1, 0), 1.);
        // 2 * (C(10,0) + C(10,1) + C(10,2)) / 2^10
        near(mcnemar_exact_p(2, 8), 2. * 56. / 1024.);
        near(mcnemar_exact_p(0, 2000), 0.); // no overflow for large counts
    }

    #[test]
    fn wilson_matches_the_a3_report() {
        // RESULT.md: sycophancy 25/32 -> 24/32, slop 2/32.
        for (positives, low, high) in [(25, 0.612, 0.890), (24, 0.579, 0.867), (2, 0.0173, 0.2015)]
        {
            let (l, h) = wilson95(positives, 32);
            assert!(
                (l - low).abs() < 5e-4 && (h - high).abs() < 5e-4,
                "{positives}: {l} {h}"
            );
        }
        assert_eq!(wilson95(0, 8).0, 0.);
        assert_eq!(wilson95(8, 8).1, 1.);
    }

    #[test]
    fn holm_orders_scales_and_keeps_input_order() {
        let adjusted = holm_adjust(&[0.04, 0.01, 0.03]);
        // sorted: 0.01*3, 0.03*2, 0.04*1 -> 0.03, 0.06, 0.06 (monotone)
        for (got, want) in adjusted.iter().zip([0.06, 0.03, 0.06]) {
            near(*got, want);
        }
        assert_eq!(holm_adjust(&[0.5, 0.9]), vec![1., 1.]);
        assert!(holm_adjust(&[]).is_empty());
    }

    fn o(parent: bool, child: bool) -> PairedOutcome {
        PairedOutcome { parent, child }
    }

    #[test]
    fn discordant_counts_direction() {
        assert_eq!(
            discordant(&[
                o(true, false),
                o(true, false),
                o(false, true),
                o(true, true)
            ]),
            (2, 1)
        );
    }

    #[test]
    fn outcomes_must_match_counts_and_rates() {
        let rows = vec![
            o(true, false),
            o(true, true),
            o(false, false),
            o(false, false),
        ];
        let incomplete = BTreeMap::from([("slop".to_string(), rows.clone())]);
        let outcomes = BTreeMap::from([
            ("slop".to_string(), rows.clone()),
            ("sycophancy".into(), rows),
        ]);
        let counts = BTreeMap::from([("slop".to_string(), 4), ("sycophancy".into(), 4)]);
        let rate = |_: &str| Some((0.5, 0.25));
        assert!(check_outcomes(&incomplete, &counts, rate).is_err());
        assert!(check_outcomes(&outcomes, &counts, rate).is_ok());
        assert!(check_outcomes(&outcomes, &counts, |_| Some((0.5, 0.5))).is_err());
        assert!(check_outcomes(&outcomes, &counts, |_| None).is_err());
        let wrong_count = BTreeMap::from([("slop".to_string(), 5)]);
        assert!(check_outcomes(&outcomes, &wrong_count, rate).is_err());
        let other = BTreeMap::from([("response_length".to_string(), vec![o(true, true)])]);
        let other_counts = BTreeMap::from([("response_length".to_string(), 1)]);
        assert!(check_outcomes(&other, &other_counts, |_| Some((1., 1.))).is_err());
    }
}
