mod endpoint;
mod gsm8k;
#[path = "../../engine/src/progress.rs"]
mod progress;

use gsm8k::{Predictions, PROTOCOL, REVISION, SHA};
use progress::Progress;
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    kind: String,
    #[serde(default)]
    inputs: Vec<Input>,
    #[serde(default)]
    options: Options,
    prompt: Option<String>,
}

#[derive(Deserialize)]
struct Input {
    kind: String,
    path: PathBuf,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    baseline_source: Option<String>,
    #[serde(default)]
    synthetic_fixture: bool,
    max_tokens: Option<u32>,
}

fn bundle_kind(kind: &str) -> bool {
    matches!(kind, "hgn" | "brewery-bundle" | "hgn-v2" | "ht43")
}

fn reference_file_kind(kind: &str) -> bool {
    matches!(
        kind,
        "gsm8k-baseline" | "gsm8k-predictions" | "gsm8k-evaluation"
    )
}

fn precision_ok(precision: &str) -> bool {
    !precision.is_empty()
        && precision.len() <= 32
        && precision
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}

fn run(job: Job) -> Result<Value> {
    let progress = Progress::new(&fs::canonicalize(".")?, job.inputs.iter().map(|i| &i.path));
    let Options {
        baseline_source,
        synthetic_fixture,
        max_tokens,
    } = job.options;
    if job.kind == "generate" {
        let [input] = job.inputs.as_slice() else {
            return Err("generate requires one HGN bundle directory".into());
        };
        if synthetic_fixture || baseline_source.is_some() || !bundle_kind(&input.kind) {
            return Err("invalid generate inputs or options".into());
        }
        let prompt = job
            .prompt
            .filter(|s| !s.trim().is_empty() && s.len() <= 16_384)
            .ok_or("generate requires a nonempty prompt of at most 16384 bytes")?;
        let max = max_tokens.unwrap_or(128);
        let prompt_sha256 = digest(&prompt);
        gpu_settings(max)?;
        let bundle = manifest(input, &progress)?;
        let (identity, mut texts) =
            generate_bundle(&bundle, &[prompt], max, &progress, "generate")?;
        let text = texts.pop().ok_or("missing generation")?;
        return Ok(json!({"kind":"generate","text":text,"identity":identity,
            "protocol":{"id":"raw-greedy-v1","temperature":0,"max_tokens":max,"prompt_sha256":prompt_sha256}}));
    }
    if max_tokens.is_some() || job.prompt.is_some() {
        return Err("GSM8K uses fixed prompts and token limits; overrides are not accepted".into());
    }
    let mut load = progress.start("load_dataset", "items", Some(1));
    let test = gsm8k::test()?;
    load.advance(1);
    load.finish();
    let dataset = json!({"name":"openai/grade-school-math","revision":REVISION,"split":"test","sha256":SHA,"rows":test.len(),"full_split":true});
    match job.kind.as_str() {
        "baseline" => {
            let run = match (job.inputs.as_slice(), baseline_source.as_deref()) {
                ([], Some(source)) => {
                    endpoint::load(source)?.run(source, &test, synthetic_fixture, &progress)?
                }
                ([input], None) if reference_file_kind(&input.kind) => {
                    import(&input.path, synthetic_fixture, &progress, 1, 1)?
                }
                _ => return Err(
                    "baseline requires an imported prediction file or a configured baseline_source"
                        .into(),
                ),
            };
            let scored = run.score(&test, synthetic_fixture, &progress, "baseline_scoring")?;
            let artifact = artifact("gsm8k-baseline", &run, &progress)?;
            Ok(
                json!({"kind":"baseline","dataset":dataset,"protocol_id":PROTOCOL,
                "score":scored.0,"identity":identity(&run),"artifact":artifact}),
            )
        }
        "evaluate" => {
            let (candidate, baseline) = match job.inputs.as_slice() {
                [candidate] => (candidate, None),
                [candidate, baseline] if reference_file_kind(&baseline.kind) => {
                    (candidate, Some(baseline))
                }
                _ => {
                    return Err(
                        "evaluate requires one candidate and at most one GSM8K reference file"
                            .into(),
                    )
                }
            };
            if baseline.is_some() && baseline_source.is_some() {
                return Err("choose a baseline file or baseline_source, not both".into());
            }
            let bundle = if !synthetic_fixture && bundle_kind(&candidate.kind) {
                gpu_settings(gsm8k::MAX_TOKENS)?;
                Some(manifest(candidate, &progress)?)
            } else if synthetic_fixture && candidate.kind == "gsm8k-predictions" {
                None
            } else {
                return Err(
                    "evaluate requires an HGN bundle, or explicit synthetic fixture predictions"
                        .into(),
                );
            };
            let reference = if let Some(file) = baseline {
                Some(import(&file.path, synthetic_fixture, &progress, 2, 2)?)
            } else if let Some(id) = baseline_source.as_deref() {
                Some(endpoint::load(id)?.run(id, &test, synthetic_fixture, &progress)?)
            } else {
                None
            };
            let reference_score = reference
                .as_ref()
                .map(|r| r.score(&test, synthetic_fixture, &progress, "baseline_scoring"))
                .transpose()?;
            let run = if let Some(bundle) = bundle.as_ref() {
                let prompts: Vec<_> = test.iter().map(|q| q.prompt.clone()).collect();
                let (identity, texts) = generate_bundle(
                    bundle,
                    &prompts,
                    gsm8k::MAX_TOKENS,
                    &progress,
                    "evaluate_questions",
                )?;
                Predictions::new(
                    bundle.source_sha256().to_owned(),
                    identity,
                    &test,
                    texts,
                    false,
                )?
            } else {
                import(&candidate.path, true, &progress, 1, job.inputs.len() as u64)?
            };
            let scored = run.score(&test, synthetic_fixture, &progress, "evaluate_scoring")?;
            let reference_scope = reference.as_ref().map(|reference| {
                if !run.synthetic
                    && run.source_sha256 == reference.source_sha256
                    && reference.provenance.as_ref().is_some_and(|p| {
                        p["method"] == "operator-bf16-endpoint"
                            && p["bf16_attested"] == true
                            && p["precision"] == "bf16"
                    })
                {
                    "attested_bf16_same_source"
                } else {
                    "diagnostic"
                }
            });
            let comparison = reference_score
                .as_ref()
                .map(|reference| gsm8k::compare(&scored, reference));
            let baseline_artifact = if baseline_source.is_some() {
                reference
                    .as_ref()
                    .map(|r| artifact("gsm8k-baseline", r, &progress))
                    .transpose()?
            } else {
                None
            };
            let artifact = artifact("gsm8k-evaluation", &run, &progress)?;
            Ok(
                json!({"kind":"evaluate","dataset":dataset,"protocol_id":PROTOCOL,"score":scored.0,
                "baseline_score": reference_score.as_ref().map(|r| &r.0),
                "baseline_identity": reference.as_ref().map(identity),
                "reference_scope":reference_scope,
               "comparison":comparison,"identity":identity(&run),"artifact":artifact,
               "baseline_artifact":baseline_artifact}),
            )
        }
        _ => Err("unsupported job kind".into()),
    }
}

fn identity(run: &Predictions) -> Value {
    json!({"source_sha256":run.source_sha256,"synthetic":run.synthetic,"provenance":run.provenance})
}

fn import(
    path: &Path,
    synthetic: bool,
    progress: &Progress,
    index: u64,
    count: u64,
) -> Result<Predictions> {
    let file = File::open(path)?;
    let meta = file.metadata()?;
    let mut phase = progress.start_for(
        "verify_input",
        "bytes",
        meta.is_file().then_some(meta.len()),
        Some(1),
        Some((index, count)),
    );
    phase.file(path);
    let mut reader = file.take(90 * 1024 * 1024 + 1);
    let mut bytes = Vec::new();
    let mut hash = Sha256::new();
    let mut chunk = vec![0; 1 << 20];
    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hash.update(&chunk[..n]);
        bytes.extend_from_slice(&chunk[..n]);
        phase.set_completed((bytes.len() as u64).min(if meta.is_file() {
            meta.len()
        } else {
            u64::MAX
        }));
    }
    if bytes.len() > 90 * 1024 * 1024 {
        return Err("input exceeded size limit".into());
    }
    let mut run: Predictions = serde_json::from_slice(&bytes)?;
    if !synthetic && run.provenance.as_ref().is_none_or(|p| !p.is_object()) {
        return Err("a real imported baseline must provide object provenance".into());
    }
    // Precision from a file is descriptive only; never relay an attestation
    // or endpoint verification claim from an imported prediction envelope.
    let mut provenance =
        json!({"method":"imported-predictions","import_sha256":format!("{:x}", hash.finalize())});
    if let Some(precision) = run
        .provenance
        .as_ref()
        .and_then(|p| p["precision"].as_str())
        .filter(|precision| precision_ok(precision))
    {
        provenance["precision"] = json!(precision);
    }
    run.provenance = Some(provenance);
    phase.file_done();
    phase.finish();
    Ok(run)
}

fn manifest(input: &Input, progress: &Progress) -> Result<halogen_proof::GenerationBundle> {
    if !bundle_kind(&input.kind) || !input.path.is_dir() {
        return Err("HGN input must be a bundle directory containing manifest.json".into());
    }
    let mut phase = progress.start("verify_input", "bytes", None);
    let bundle = halogen_proof::qualify_generation_observed(&input.path, &mut |path, bytes| {
        if bytes == 0 {
            phase.file(path);
        } else {
            phase.advance(bytes);
        }
    })?;
    phase.finish();
    if matches!(input.kind.as_str(), "hgn-v2" | "ht43") && input.kind != bundle.format_name() {
        return Err("input kind does not match the bundle format".into());
    }
    Ok(bundle)
}

fn gpu_settings(limit: u32) -> Result<String> {
    if !(1..=512).contains(&limit) {
        return Err("max_tokens must be in 1..=512".into());
    }
    if std::env::var("BREWERY_ALLOW_GPU_LOAD").as_deref() != Ok("1") {
        return Err("isolated HGN GPU load not enabled by worker operator".into());
    }
    let image = std::env::var("BREWERY_HALOGEN_IMAGE_ID")
        .map_err(|_| "worker has no pinned Halogen image")?;
    if !image.strip_prefix("sha256:").is_some_and(hash_ok) {
        return Err("worker Halogen image ID must be a pinned local sha256 ID".into());
    }
    Ok(image)
}

fn generate_bundle(
    bundle: &halogen_proof::GenerationBundle,
    prompts: &[String],
    limit: u32,
    progress: &Progress,
    name: &'static str,
) -> Result<(Value, Vec<String>)> {
    let image = gpu_settings(limit)?;
    let total = prompts.len() as u64;
    let mut loading = progress.start("admission", "items", None);
    let mut phase = None;
    let generated =
        halogen_proof::generate_observed(bundle, &image, prompts, limit, &mut |completed| {
            let phase = phase.get_or_insert_with(|| {
                loading.advance(1);
                loading.checkpoint();
                let unit = if name == "generate" {
                    "items"
                } else {
                    "questions"
                };
                progress.start_at(name, unit, Some(total), completed)
            });
            phase.set_completed(completed);
            if completed == total {
                phase.checkpoint();
            }
        })?;
    if let Some(phase) = phase {
        phase.finish();
    }
    Ok((serde_json::to_value(generated.proof)?, generated.texts))
}

fn artifact(kind: &str, value: &impl serde::Serialize, progress: &Progress) -> Result<Value> {
    let cwd = fs::canonicalize(".")?;
    let mut publish = progress.start("publish_output", "items", Some(1));
    let bytes = serde_json::to_vec(value)?;
    let mut file = tempfile::Builder::new()
        .prefix(kind)
        .suffix(".json")
        .tempfile_in(&cwd)?;
    file.write_all(&bytes)?;
    file.as_file().sync_all()?;
    let (_, path) = file.keep()?;
    publish.advance(1);
    publish.finish();
    Ok(json!({"kind":kind,"path":path,"sha256":digest(&bytes),"bytes":bytes.len()}))
}

fn bounded(reader: impl Read, limit: u64) -> Result<Vec<u8>> {
    let mut raw = Vec::new();
    reader.take(limit + 1).read_to_end(&mut raw)?;
    if raw.len() as u64 > limit {
        return Err("input exceeded size limit".into());
    }
    Ok(raw)
}

fn read<T: DeserializeOwned>(reader: impl Read, limit: u64) -> Result<T> {
    Ok(serde_json::from_slice(&bounded(reader, limit)?)?)
}

fn digest(data: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn hash_ok(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn main() -> std::process::ExitCode {
    match read(io::stdin().lock(), 256 * 1024).and_then(run) {
        Ok(value) => {
            println!("{value}");
            std::process::ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("brewery-eval: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}
