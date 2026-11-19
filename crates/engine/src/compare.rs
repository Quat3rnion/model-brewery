use crate::{check_inputs, integrity, native_json, read_json, tool, write_json, Job};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path, process::Command};

const MAX_INLINE_CELLS: usize = 4096;
const MAX_INLINE_ROUTES: usize = 32;
// Leave room for the worker RPC envelope. This also checks pre-existing
// inline fields (tokens/candidates/staging identity), not just signals.
const MAX_COMPARISON_BYTES: usize = 12 << 20;

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Options {
    n: u64,
    raw: bool,
    think: bool,
    threads: u64,
    labels: u64,
    gpu_device: Option<String>,
    gpu_layers: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            n: 16,
            raw: false,
            think: false,
            threads: 4,
            labels: 0,
            gpu_device: None,
            gpu_layers: None,
        }
    }
}

struct Trace {
    inline: Value,
    receipt: Value,
    tokens: Vec<Token>,
}

#[derive(Deserialize, PartialEq, Eq)]
struct Token {
    i: usize,
    id: u64,
    piece: String,
    phase: String,
}

pub fn run(job: &Job, cwd: &Path, progress: &crate::progress::Progress) -> Result<Value> {
    ensure!(
        job.inputs.len() == 2 && job.inputs.iter().all(|i| i.kind == "gguf"),
        "compare requires two GGUF inputs"
    );
    let mut options: Options =
        serde_json::from_value(job.options.clone()).context("invalid compare options")?;
    ensure!(
        options.n <= 128 && options.labels <= 600 && (1..=64).contains(&options.threads),
        "compare option out of range"
    );
    options.threads =
        crate::child_threads(job.options.get("threads").and_then(Value::as_u64))? as u64;
    ensure!(
        options
            .gpu_device
            .as_ref()
            .is_none_or(|device| !device.is_empty()
                && device.len() <= 64
                && device
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')),
        "invalid GPU device name"
    );
    ensure!(
        options
            .gpu_layers
            .is_none_or(|layers| (1..=4096).contains(&layers) && options.gpu_device.is_some()),
        "gpu_layers requires gpu_device and must be in 1..=4096"
    );
    let prompt = job.prompt.as_deref().context("compare requires a prompt")?;
    ensure!(
        !prompt.trim().is_empty() && prompt.len() <= 8192 && !prompt.contains('\0'),
        "compare prompt must be nonempty, NUL-free and at most 8192 bytes"
    );
    let ids = check_inputs(&job.inputs, progress)?;
    let headers = job
        .inputs
        .iter()
        .zip(&ids)
        .map(|(input, id)| crate::gguf::trace_metadata(&input.path, id.files.len()))
        .collect::<Result<Vec<_>>>()?;
    let exe = tool("BREWERY_LANTERNFISH", "lanternfish")?;
    let home = cwd.join("comparison");
    fs::create_dir(&home).context("comparison output already exists or cannot be created")?;
    // Both inputs were fully checked before the first process; no shared cache
    // (Lanternfish label cache keys use only the model basename).
    let mut phase = progress.start("compare_left", "items", None);
    let left = trace(
        job, &options, prompt, &exe, &home, "left", &ids, &headers, 0,
    )?;
    phase.advance(1);
    phase.finish();
    let mut phase = progress.start("compare_right", "items", None);
    let right = trace(
        job, &options, prompt, &exe, &home, "right", &ids, &headers, 1,
    )?;
    phase.advance(1);
    phase.finish();
    ids.iter().try_for_each(integrity::Identity::check)?;

    let same_model = if ids[0].files.len() == 1 && ids[1].files.len() == 1 {
        ids[0].sha256 == ids[1].sha256
    } else {
        ids[0].bundle_sha256 == ids[1].bundle_sha256
    };
    let same_prompt =
        left.inline["template"]["rendered_prompt"] == right.inline["template"]["rendered_prompt"];
    let matched = if same_model && same_prompt {
        left.tokens
            .iter()
            .zip(&right.tokens)
            .take_while(|(a, b)| a == b)
            .count()
    } else {
        0
    };
    let (status, reason) = if !same_model {
        ("not_aligned", "Different GGUF bytes: tokenizer, history, and layer/expert identities were not established as comparable.")
    } else if !same_prompt {
        ("not_aligned", "Rendered prompts differ.")
    } else if matched == 0 {
        ("not_aligned", "No identical token-history prefix.")
    } else {
        ("prefix_only", "Only the identical token-history prefix is paired. No numerical parity or layer/expert correspondence is asserted.")
    };
    let mut result = json!({
        "kind":"compare","host":job.inputs[0].host,
        "traces":[left.inline,right.inline],
        "alignment":{"status":status,"matched_prefix_tokens":matched,"reason":reason},
        "backend_receipt":{"tool":"lanternfish","processes":[left.receipt,right.receipt]},
        "file_verification":{"status":"passed","scope":"physical GGUF input hashes, not model numerical verification"},
        "hardware_proof": if options.gpu_device.is_some() {
            json!({"status":"observed","scope":"Lanternfish selected-device GPU matmuls at every evaluated step; CPU nodes can coexist"})
        } else {
            json!({"status":"not_proved","reason":"CPU requested; Lanternfish run emits no CPU placement receipt"})
        }
    });
    let mut publish = progress.start("publish_output", "items", Some(1));
    write_json(&home.join("comparison.json"), &result)?;
    publish.advance(1);
    publish.finish();
    result["artifact"] =
        integrity::identify_output(&home, progress)?.artifact(&home, "trace-comparison");
    ensure!(
        serde_json::to_vec(&result)?.len() <= MAX_COMPARISON_BYTES,
        "comparison result exceeds inline byte budget"
    );
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn trace(
    job: &Job,
    options: &Options,
    prompt: &str,
    exe: &Path,
    home: &Path,
    side: &str,
    ids: &[integrity::Identity],
    headers: &[Value],
    index: usize,
) -> Result<Trace> {
    let input = &job.inputs[index];
    let id = &ids[index];
    id.check()?;
    let store = home.join(side);
    fs::create_dir(&store)?;
    let mut command = Command::new(exe);
    crate::child_budget(&mut command, options.threads as usize);
    command
        .arg("--home")
        .arg(&store)
        .arg("run")
        .arg(&input.path)
        .args(["--name", "trace", &format!("--prompt={prompt}"), "--n"])
        .arg(options.n.to_string())
        .arg("--labels")
        .arg(options.labels.to_string())
        .arg("--threads")
        .arg(options.threads.to_string());
    if options.raw {
        command.arg("--raw");
    }
    if options.think {
        command.arg("--think");
    }
    if let Some(device) = &options.gpu_device {
        command.arg("--gpu-device").arg(device);
    }
    if let Some(layers) = options.gpu_layers {
        command.arg("--gpu-layers").arg(layers.to_string());
    }
    eprintln!("brewery-engine: tracing {side}");
    let (native, receipt) = native_json(&mut command, &store.join("stderr.log"))?;
    id.check()?;
    let root = store.join("traces/trace");
    let viz_file = root.join("viz.json");
    ensure!(
        native["path"].as_str().map(Path::new) == Some(&viz_file),
        "unexpected native trace path"
    );
    let viz = read_json(&viz_file)?;
    let meta = read_json(&root.join("meta.json"))?;
    ensure!(
        viz.get("native_model").is_none()
            && viz["tool"]
                .as_str()
                .is_some_and(|s| s.starts_with("lanternfish "))
            && viz["layers"].as_array().is_some_and(|l| !l.is_empty()),
        "not a supported Lanternfish llama.cpp trace"
    );
    let rendered = viz["prompt"]
        .as_str()
        .context("trace is missing the rendered prompt")?;
    let template_status = if options.raw {
        ensure!(rendered == prompt, "raw prompt was modified");
        "raw_requested"
    } else if rendered == prompt {
        "no_change_or_fallback"
    } else {
        "rendered_differs"
    };
    let tokens: Vec<Token> =
        serde_json::from_value(viz["tokens"].clone()).context("invalid trace tokens")?;
    ensure!(
        !tokens.is_empty()
            && tokens[0].phase == "prefill"
            && tokens
                .iter()
                .enumerate()
                .all(|(i, t)| t.i == i && matches!(t.phase.as_str(), "prefill" | "decode"))
            && !tokens
                .windows(2)
                .any(|w| w[0].phase == "decode" && w[1].phase == "prefill")
            && tokens.iter().filter(|t| t.phase == "decode").count() as u64 <= options.n
            && native["tokens"].as_u64() == Some(tokens.len() as u64)
            && meta["tokens"].as_u64() == Some(tokens.len() as u64),
        "inconsistent token counts, positions, or phases"
    );
    let steps = viz["steps"]
        .as_array()
        .context("trace is missing evaluated steps")?;
    let candidates = viz["next_candidates"]
        .as_array()
        .context("trace is missing next candidates")?;
    ensure!(
        !steps.is_empty()
            && steps.len() == candidates.len()
            && candidates.iter().all(Value::is_array),
        "next candidates must map one-to-one to evaluated steps"
    );
    let mut visited = 0usize;
    for step in steps {
        ensure!(
            step["first"].as_u64() == Some(visited as u64),
            "trace steps not contiguous"
        );
        let n = step["n"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
            .context("invalid step size")?;
        let end = visited.checked_add(n).context("step overflow")?;
        ensure!(
            end <= tokens.len()
                && tokens[visited..end]
                    .iter()
                    .all(|t| step["phase"].as_str() == Some(&t.phase)),
            "evaluated step does not match token positions and phases"
        );
        visited = end;
    }
    ensure!(
        visited == tokens.len() || visited + 1 == tokens.len() && tokens[visited].phase == "decode",
        "unexpected unevaluated tokens"
    );
    let observed = &viz["execution"];
    if let Some(device) = &options.gpu_device {
        ensure!(
            observed["backend"] == "gpu" && observed["device"]["name"].as_str() == Some(device),
            "Lanternfish did not attest the requested GPU"
        );
        let evidence = observed["steps"]
            .as_array()
            .context("missing per-step GPU evidence")?;
        ensure!(
            evidence.len() == steps.len()
                && evidence
                    .iter()
                    .zip(steps)
                    .all(|(e, step)| e["first"] == step["first"]
                        && e["n"] == step["n"]
                        && e["phase"] == step["phase"]
                        && e["gpu_matmul_nodes"].as_u64().is_some_and(|n| n > 0)),
            "missing GPU compute in an evaluated step"
        );
    } else {
        ensure!(
            observed.is_null() || observed["backend"] == "cpu",
            "unexpected trace execution target"
        );
    }
    let output_text: String = tokens
        .iter()
        .filter(|t| t.phase == "decode")
        .map(|t| t.piece.as_str())
        .collect();
    let signals = inline_signals(&viz, &tokens)?;
    let inline = json!({
        "input":id.input(input),
        "meta":meta,
        "template":{"mode":if options.raw {"raw"} else {"model_chat_template"},"status":template_status,
            "think_requested":options.think,"input_prompt":prompt,"rendered_prompt":rendered,
            "embedded_default_template_sha256":headers[index]["embedded_default_template_sha256"]},
        "gguf_metadata":headers[index],
        "tokens":viz["tokens"],"steps":steps,"next_candidates":candidates,
        "output_text":output_text,"signals":signals,
        "execution":{"requested":{"backend":if options.gpu_device.is_some() {"gpu"} else {"cpu"},
            "gpu_device":options.gpu_device,"gpu_layers":options.gpu_layers,
            "threads":options.threads,"labels":options.labels,"max_decode_tokens":options.n},"observed":observed},
        "viz_file":viz_file,
    });
    Ok(Trace {
        inline,
        receipt,
        tokens,
    })
}

fn inline_signals(viz: &Value, tokens: &[Token]) -> Result<Value> {
    let layers = viz["layers"].as_array().context("trace has no layers")?;
    let mut kinds = BTreeMap::new();
    let mut by_token = vec![Vec::new(); tokens.len()];
    let mut captured = 0usize;
    for row in layers {
        let layer = row["layer"]
            .as_u64()
            .and_then(|l| usize::try_from(l).ok())
            .filter(|l| *l < layers.len())
            .context("invalid native layer position")?;
        let kind = row["kind"]
            .as_str()
            .filter(|kind| matches!(*kind, "attention" | "linear"))
            .unwrap_or("unknown");
        ensure!(
            kinds.insert(layer, kind).is_none(),
            "duplicate layer position"
        );
        for (key, cell) in row["tokens"]
            .as_object()
            .context("invalid native layer cells")?
        {
            let position = key
                .parse::<usize>()
                .context("invalid native layer token position")?;
            ensure!(
                position < tokens.len() && position.to_string() == *key && cell.is_object(),
                "invalid native layer token cell"
            );
            by_token[position].push((layer, cell));
            captured += 1;
        }
    }
    let mut selected: BTreeMap<usize, Vec<(usize, Value)>> = BTreeMap::new();
    let mut unavailable = [0; 4];
    let mut token_span = None::<[usize; 2]>;
    let mut included = 0;
    // If bounded, retain as much of the decode trajectory as possible across
    // layers before spending the remainder on prefill. Display order is native.
    for (position, native_cells) in by_token
        .iter()
        .enumerate()
        .filter(|(i, _)| tokens[*i].phase == "decode")
        .chain(
            by_token
                .iter()
                .enumerate()
                .filter(|(i, _)| tokens[*i].phase == "prefill"),
        )
    {
        for &(layer, raw) in native_cells.iter().take(MAX_INLINE_CELLS - included) {
            let mut cell = json!({"token":position});
            let ids = raw["experts"].as_array().filter(|ids| {
                !ids.is_empty()
                    && ids.len() <= MAX_INLINE_ROUTES
                    && ids
                        .iter()
                        .all(|id| id.as_u64().is_some_and(|id| u32::try_from(id).is_ok()))
            });
            let weights = raw["weights"].as_array().filter(|w| {
                ids.is_some_and(|ids| ids.len() == w.len()) && w.iter().all(Value::is_number)
            });
            for (i, (key, value)) in [
                (
                    "attn",
                    raw["write"]["attn"]
                        .is_number()
                        .then_some(&raw["write"]["attn"]),
                ),
                (
                    "ffn",
                    raw["write"]["ffn"]
                        .is_number()
                        .then_some(&raw["write"]["ffn"]),
                ),
                ("experts", ids.map(|_| &raw["experts"])),
                ("weights", weights.map(|_| &raw["weights"])),
            ]
            .into_iter()
            .enumerate()
            {
                if let Some(value) = value {
                    cell[key] = value.clone();
                } else {
                    unavailable[i] += 1;
                }
            }
            selected.entry(layer).or_default().push((position, cell));
            let [first, last] = token_span.unwrap_or([position, position]);
            token_span = Some([position.min(first), position.max(last)]);
            included += 1;
        }
        if included == MAX_INLINE_CELLS {
            break;
        }
    }
    let layer_span = selected
        .keys()
        .next()
        .zip(selected.keys().next_back())
        .map(|(&first, &last)| [first, last]);
    let compact: Vec<_> = selected
        .into_iter()
        .map(|(layer, mut cells)| {
            cells.sort_unstable_by_key(|(position, _)| *position);
            json!({"layer":layer,"kind":kinds[&layer],
               "cells":cells.into_iter().map(|(_, cell)| cell).collect::<Vec<_>>()})
        })
        .collect();
    let total = layers
        .len()
        .checked_mul(tokens.len())
        .context("native trace dimension overflow")?;
    Ok(json!({
        "schema":1,"layers":compact,
        "coverage":{"total_layers":layers.len(),"total_tokens":tokens.len(),
            "captured_cells":captured,"included_cells":included,
            "layer_span":layer_span,"token_span":token_span,"truncated":included < captured},
        "omissions":{"missing_cells":total - captured,"skipped_cells":captured - included,
            "unavailable":{"attn":unavailable[0],"ffn":unavailable[1],
                           "experts":unavailable[2],"weights":unavailable[3]},
            "inline_excluded":["logit lens","next-token rank","residual and stream RMS",
                               "shared gate","expert labels"]}
    }))
}
