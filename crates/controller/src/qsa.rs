use crate::domain::{Envelope, Job, Model, NewModel, QsaProvenance};
use brewery_worker::qsa;
use serde_json::Value;
use std::path::Path;

pub fn verified_artifact(job: &Job, envelope: &Envelope, result: &Value) -> Result<Model, String> {
    const INVALID: &str = "Invalid QSA training artifact receipt";
    let spec = qsa::Spec::parse(&Value::Object(envelope.options.clone())).map_err(|_| INVALID)?;
    qsa::check_receipt(&spec, &job.host, result).map_err(|_| INVALID)?;
    let inputs = result["inputs"].as_array().ok_or(INVALID)?;
    if inputs.len() != envelope.inputs.len()
        || inputs
            .iter()
            .zip(&envelope.inputs)
            .any(|(row, registered)| {
                row["id"] != registered.id
                    || row["kind"] != registered.kind
                    || row["path"] != registered.path
                    || row["host"] != registered.host
            })
    {
        return Err(INVALID.into());
    }
    let output = |key: &str, leaf: &str| -> Result<&str, String> {
        let path = result[key]["path"].as_str().ok_or(INVALID)?;
        managed_path(path, &job.id, leaf)
            .then_some(path)
            .ok_or_else(|| INVALID.into())
    };
    let path = output("artifact", qsa::FILE)?;
    let metrics = output("metrics_file", qsa::METRICS)?;
    let teacher_sha256 = result["teacher"]["sha256"].as_str().ok_or(INVALID)?;
    if Path::new(path).parent() != Path::new(metrics).parent() {
        return Err(INVALID.into());
    }
    if envelope
        .qsa_teacher_sha256
        .as_deref()
        .is_some_and(|expected| expected != teacher_sha256)
    {
        return Err(INVALID.into());
    }
    let mut model = NewModel {
        name: Some(format!("QSA indexer ({})", job.id)),
        host: job.host.clone(),
        path: path.into(),
        kind: qsa::INDEXER.into(),
    }
    .validate()?;
    model.qsa_provenance = Some(QsaProvenance {
        training_job_id: job.id.clone(),
        teacher_model_id: envelope.inputs[0].id.clone(),
        teacher_sha256: teacher_sha256.into(),
    });
    Ok(model)
}

fn managed_path(path: &str, id: &str, leaf: &str) -> bool {
    path.strip_prefix('/').is_some_and(|parts| {
        !path.contains('\\')
            && !path.chars().any(char::is_control)
            && parts
                .split('/')
                .all(|part| !matches!(part, "" | "." | ".."))
            && path.ends_with(&format!("/jobs/{id}/work/{}/{leaf}", qsa::OUTPUT))
    })
}
