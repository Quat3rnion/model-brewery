use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub const GGUF: &str = "scratch_gguf";
pub const CHECKS: &str = "scratch_checks";
pub const SCRIPT: &str = "BREWERY_SCRATCH_CHECKS_SCRIPT";
pub const RUNNER: &str = "BREWERY_SCRATCH_CHECKS_RUNNER";
pub const RUNNER_SHA: &str = "BREWERY_SCRATCH_CHECKS_RUNNER_SHA256";
pub const CACHE: &str = "BREWERY_SCRATCH_CHECKS_CACHE";
pub const GRANT: &str = "scratch_grant";
pub const CHECK_INPUTS: [&str; 4] = ["gguf", "gguf", "safetensors", "safetensors"];
pub const CHECK_DIGESTS: [&str; 4] = [
    "parent_gguf_sha256",
    "revision_gguf_sha256",
    "parent_sha256",
    "revision_sha256",
];

pub fn private_output(kind: &str) -> bool {
    matches!(kind, GGUF | CHECKS | crate::locate_edit::JOB)
}

pub fn granted(input: &crate::Input) -> bool {
    input.extra.get(GRANT) == Some(&Value::Bool(true))
}

pub fn gguf_options(options: &Value) -> bool {
    *options == json!({"output_format":"gguf","gguf_type":"BF16"})
}

pub fn check_report(report: &Value, inputs: [&str; 4], runner_sha: &str) -> Result<()> {
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
    Ok(())
}
