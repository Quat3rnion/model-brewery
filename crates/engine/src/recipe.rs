use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Write, path::Path};

const SCHEMA: &str = "bf16-hgn-tensor-recipe-v1";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    schema: u8,
    revision: String,
    name: String,
    catalog_sha256: String,
    architecture: String,
    base: String,
    overrides: BTreeMap<String, String>,
}

pub(super) struct Recipe {
    selection: Selection,
    document: Vec<u8>,
    receipt: Value,
    digest: String,
    policy: String,
}

impl Recipe {
    pub(super) fn parse(value: &Value) -> Result<Self> {
        ensure!(
            serde_json::to_vec(value)?.len() <= 512 << 10,
            "profile recipe exceeds size limit"
        );
        let selection: Selection =
            serde_json::from_value(value.clone()).context("invalid profile recipe envelope")?;
        let digest_ok =
            |s: &str| s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
        ensure!(
            selection.schema == 1
                && selection.architecture == "qwen4_exp"
                && !selection.name.trim().is_empty()
                && selection.name.len() <= 256
                && !selection.name.chars().any(char::is_control)
                && digest_ok(&selection.catalog_sha256),
            "invalid profile recipe identity"
        );
        let alternative = match selection.base.as_str() {
            "v2" => "ht43",
            "ht43" => "v2",
            _ => anyhow::bail!("invalid profile recipe base"),
        };
        ensure!(
            (1..=386).contains(&selection.overrides.len())
                && selection.overrides.iter().all(|(name, profile)| {
                    profile == alternative
                        && (1..=95).contains(&name.len())
                        && name
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"._".contains(&b))
                }),
            "expected nonempty, alternate-only recipe tensor overrides"
        );
        // The controller hashes this versioned sorted compact object before
        // storing the immutable version. The native recipe has its own digest.
        let mut identity = value.clone();
        identity.as_object_mut().unwrap().remove("revision");
        identity.sort_all_objects();
        ensure!(
            hash(&serde_json::to_vec(&identity)?) == selection.revision,
            "profile recipe revision does not match its contents"
        );
        let mut receipt = json!({
            "schema":SCHEMA,
            "base_profile":selection.base,
            "alternate_tensors":selection.overrides.keys().collect::<Vec<_>>()
        });
        let document = serde_json::to_vec(&receipt)?;
        let digest = hash(&document);
        receipt["sha256"] = json!(digest);
        let policy = format!("{SCHEMA}:{digest}");
        Ok(Self {
            selection,
            document,
            receipt,
            digest,
            policy,
        })
    }

    pub(super) fn base(&self) -> &str {
        &self.selection.base
    }

    pub(super) fn snapshot(&self, work: &Path) -> Result<tempfile::NamedTempFile> {
        let mut file = tempfile::Builder::new()
            .prefix(".profile-recipe-")
            .suffix(".json")
            .tempfile_in(work)?;
        file.write_all(&self.document)?;
        Ok(file)
    }

    pub(super) fn check(&self, report: &Value) -> Result<()> {
        ensure!(
            report["recipe"] == self.receipt && report["policy"] == self.policy,
            "native converter did not attest the selected recipe"
        );
        Ok(())
    }

    pub(super) fn check_plan(&self, plan: &Value) -> Result<()> {
        let main = plan["files"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|file| file["filename"] == "model.hgn")
            .collect::<Vec<_>>();
        ensure!(
            plan["architecture"] == self.selection.architecture
                && main.len() == 1
                && main[0]["identity"] == self.main_identity(),
            "native plan did not bind the selected custom container identity"
        );
        Ok(())
    }

    pub(super) fn main_identity(&self) -> String {
        format!("qwen3.8-flash-next-recipe-{}", &self.digest[..32])
    }

    pub(super) fn result(&self) -> Value {
        json!({
            "name":self.selection.name, "revision":self.selection.revision,
            "catalog_sha256":self.selection.catalog_sha256, "architecture":self.selection.architecture,
            "base":self.selection.base, "policy":self.policy, "recipe_sha256":self.digest,
            "release_qualified":false
        })
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
