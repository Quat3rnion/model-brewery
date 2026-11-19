use crate::domain::{Job, Model, NewModel};
use brewery_worker::hf_download::{Spec, is_commit};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt<'a> {
    kind: &'a str,
    repo_id: &'a str,
    revision: &'a str,
    commit: &'a str,
    filename: Option<&'a str>,
    verified: bool,
    #[serde(borrow)]
    artifact: Artifact<'a>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact<'a> {
    kind: &'a str,
    path: &'a str,
    sha256: &'a str,
    bytes: u64,
}

pub fn verified_artifact(job: &Job, result: &Value) -> Result<Model, &'static str> {
    let invalid = "worker did not return a verified Hugging Face artifact for this request";
    let spec = Spec::parse(&Value::Object(job.options.clone())).map_err(|_| invalid)?;
    let receipt = Receipt::deserialize(result).map_err(|_| invalid)?;
    let artifact = receipt.artifact;
    let mut suffix = format!("/jobs/{}/work/download", job.id);
    if let Some(filename) = &spec.filename {
        suffix.push('/');
        suffix.push_str(filename);
    }
    if !receipt.verified
        || receipt.kind != "hf_download"
        || receipt.repo_id != spec.repo_id
        || receipt.revision != spec.revision
        || receipt.filename != spec.filename.as_deref()
        || (receipt.filename.is_none() && result.get("filename").is_some())
        || !is_commit(receipt.commit)
        || (is_commit(&spec.revision) && receipt.commit != spec.revision)
        || artifact.kind != spec.artifact_kind()
        || !artifact.path.ends_with(&suffix)
        || !artifact.path.starts_with('/')
        || artifact
            .path
            .split('/')
            .skip(1)
            .any(|component| matches!(component, "" | "." | ".."))
        || artifact.bytes == 0
        || artifact.sha256.len() != 64
        || !artifact
            .sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid);
    }
    NewModel {
        host: job.host.clone(),
        name: Some(
            spec.filename
                .as_deref()
                .map_or(spec.repo_id.as_str(), |name| {
                    name.rsplit('/').next().unwrap_or(name)
                })
                .into(),
        ),
        kind: artifact.kind.into(),
        path: artifact.path.into(),
    }
    .validate()
    .map_err(|_| invalid)
}
