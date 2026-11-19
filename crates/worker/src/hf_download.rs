//! Shared, closed request vocabulary for model downloads.
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub repo_id: String,
    #[serde(default = "main_revision")]
    pub revision: String,
    #[serde(default, deserialize_with = "present_filename")]
    pub filename: Option<String>,
}

fn present_filename<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    String::deserialize(d).map(Some)
}

fn main_revision() -> String {
    "main".into()
}

fn alphanumeric(c: &u8) -> bool {
    c.is_ascii_alphanumeric()
}

fn component(part: &str, limit: usize) -> bool {
    part.len() <= limit
        && part.as_bytes().first().is_some_and(alphanumeric)
        && part
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
        && !part.contains("..")
        && !part.ends_with('.')
}

impl Spec {
    pub fn parse(options: &Value) -> Result<Self> {
        let spec = Self::deserialize(options)?;
        ensure!(
            spec.repo_id.split('/').count() <= 2
                && spec.repo_id.split('/').all(|p| {
                    component(p, 96)
                        && !p.contains("--")
                        && !p.ends_with(['-', '.'])
                        && !p.ends_with(".git")
                }),
            "invalid Hugging Face model repository ID"
        );
        ensure!(
            spec.revision.len() <= 128
                && spec
                    .revision
                    .split('/')
                    .all(|p| component(p, 128) && !p.ends_with(".lock")),
            "invalid Hugging Face revision"
        );
        if let Some(filename) = &spec.filename {
            ensure!(
                filename.len() <= 512
                    && filename.split('/').count() <= 16
                    && filename.split('/').all(|p| component(p, 255))
                    && (filename.ends_with(".gguf") || filename.ends_with(".safetensors")),
                "filename must be an exact relative .gguf or .safetensors path"
            );
        }
        Ok(spec)
    }

    pub fn artifact_kind(&self) -> &'static str {
        match self.filename.as_deref() {
            Some(name) if name.ends_with(".gguf") => "gguf",
            Some(_) => "safetensors",
            None => "hf",
        }
    }
}

pub fn is_commit(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

use std::path::PathBuf;

#[derive(serde::Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub hfr: PathBuf,
    pub max_bytes: u64,
    // Provisioning only. An isolated test namespace can serve a fake Hub on
    // its own loopback; a job cannot select any origin or pass credentials.
    pub fixture_port: Option<u16>,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        crate::absolute(&self.hfr)?;
        ensure!(
            (1..=i64::MAX as u64).contains(&self.max_bytes) && self.fixture_port != Some(0),
            "invalid Hugging Face download worker limits"
        );
        Ok(())
    }

    pub fn endpoint(&self) -> String {
        match self.fixture_port {
            Some(port) => format!("http://127.0.0.1:{port}"),
            None => "https://huggingface.co".into(),
        }
    }
}
