//! Synthetic parser and comparator fixtures only; these are not runtime receipts.

use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture(gpu: bool) -> Value {
    let positions: Vec<_> = (0..4)
        .map(|position| {
            json!({
                "position": position, "phase": if position < 2 { "prefill" } else { "decode" },
                "logits": [0.25, -0.125, 1.5, 2.0],
                "layers": [{
                    "layer": 0, "residual": [0.75, -0.5], "attn_rms": 1.125, "ffn_rms": 0.5,
                    "expert_ids": [2, 0], "expert_weights": [0.75, 0.25], "shared_gate": null
                }]
            })
        })
        .collect();
    json!({
        "schema_version": 1,
        "model": {
            "name": "test fixture", "architecture": "qwen2moe", "sha256": "a".repeat(64),
            "bytes": 256, "n_layers": 1, "hidden": 2, "vocab": 4, "n_experts": 3, "top_k": 2
        },
        "build": {
            "lanternfish_version": "0.1.0", "lanternfish_sha256": "b".repeat(64),
            "llama_system_info": "fixture",
            "runtime_libraries": { "/fixture/libllama.so": "c".repeat(64) }
        },
        "execution": {
            "backend": if gpu { "gpu" } else { "cpu" },
            "requested_gpu_layers": if gpu { -1 } else { 0 },
            "device": if gpu {
                json!({"name": "CUDA0", "description": "fixture", "backend": "CUDA", "id": null})
            } else { Value::Null },
            "steps": [
                {"first": 0, "n": 2, "phase": "prefill",
                 "gpu_compute_nodes": if gpu { 6 } else { 0 }, "gpu_matmul_nodes": if gpu { 2 } else { 0 },
                 "cpu_matmul_nodes": if gpu { 0 } else { 2 }},
                {"first": 2, "n": 1, "phase": "decode",
                 "gpu_compute_nodes": if gpu { 3 } else { 0 }, "gpu_matmul_nodes": if gpu { 1 } else { 0 },
                 "cpu_matmul_nodes": if gpu { 0 } else { 2 }},
                {"first": 3, "n": 1, "phase": "decode",
                 "gpu_compute_nodes": if gpu { 3 } else { 0 }, "gpu_matmul_nodes": if gpu { 1 } else { 0 },
                 "cpu_matmul_nodes": if gpu { 0 } else { 2 }}
            ]
        },
        "prefill_tokens": [0, 1], "decode_tokens": [2, 3], "positions": positions
    })
}

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        fs::create_dir_all(&root).unwrap();
        loop {
            let dir = root.join(format!(
                "trace-parity-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&dir) {
                Ok(()) => return Self(dir),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => panic!("{err}"),
            }
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn compare(cpu: Option<&Value>, gpu: Option<&Value>, args: &[&str]) -> Output {
    let d = Scratch::new();
    let cpu_path = d.0.join("cpu.json");
    let gpu_path = d.0.join("gpu.json");
    for (path, value) in [(&cpu_path, cpu), (&gpu_path, gpu)] {
        if let Some(value) = value {
            fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
        }
    }
    Command::new(env!("CARGO_BIN_EXE_trace-parity"))
        .args(args)
        .arg(cpu_path)
        .arg(gpu_path)
        .output()
        .unwrap()
}

#[test]
fn complete_pair_with_small_drift_passes() {
    let mut gpu = fixture(true);
    gpu["positions"][2]["logits"][0] = json!(0.2501);
    let out = compare(Some(&fixture(false)), Some(&gpu), &["--require-moe"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("PASS (fixed-v1)"));
}

fn rejected_stderr(out: Output) -> String {
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn assert_rejected(out: Output, expected: &str) {
    let stderr = rejected_stderr(out);
    assert!(
        stderr.contains(expected),
        "expected {expected:?}; got {stderr}"
    );
}

#[test]
fn a_file_for_each_backend_and_observed_matmul_are_required() {
    let cpu = fixture(false);
    let mut gpu = fixture(true);
    assert_rejected(compare(Some(&cpu), None, &[]), "gpu.json");
    assert_rejected(compare(None, Some(&gpu), &[]), "cpu.json");
    assert_rejected(compare(Some(&cpu), Some(&cpu), &[]), "expected backend Gpu");
    let mut no_cpu_matmul = cpu.clone();
    no_cpu_matmul["execution"]["steps"][0]["cpu_matmul_nodes"] = json!(0);
    assert_rejected(
        compare(Some(&no_cpu_matmul), Some(&gpu), &[]),
        "execution.steps[0]",
    );
    gpu["execution"]["steps"][1]["gpu_matmul_nodes"] = json!(0);
    assert_rejected(compare(Some(&cpu), Some(&gpu), &[]), "execution.steps[1]");
}

fn reject_gpu_change(pointer: &str, value: Value, expected: &str) {
    let mut gpu = fixture(true);
    *gpu.pointer_mut(pointer).unwrap() = value;
    assert_rejected(compare(Some(&fixture(false)), Some(&gpu), &[]), expected);
}

#[test]
fn trace_must_match_model_token_schedule_and_tensor_shapes() {
    reject_gpu_change("/model/sha256", json!("d".repeat(64)), "model identity");
    reject_gpu_change("/decode_tokens/0", json!(1), "fixed token sequences");
    reject_gpu_change("/decode_tokens/0", json!(4), "outside the vocabulary");
    reject_gpu_change("/decode_tokens", json!([]), "nonempty prefill and decode");
    reject_gpu_change("/positions/2/position", json!(1), "position 2");
    reject_gpu_change("/positions/2/phase", json!("prefill"), "phase");
    reject_gpu_change("/execution/steps/2/n", json!(2), "position coverage");
    reject_gpu_change("/positions/2/logits", json!([1]), "logits or layer count");
    reject_gpu_change("/positions/2/layers", json!([]), "logits or layer count");
    reject_gpu_change(
        "/positions/2/layers/0/residual",
        json!([]),
        "residual width",
    );
}

#[test]
fn null_omissions_are_reported_but_missing_checkpoints_fail() {
    let mut cpu = fixture(false);
    let mut gpu = fixture(true);
    let out = compare(Some(&cpu), Some(&gpu), &[]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("shared_gate=0/4"));
    assert_rejected(
        compare(Some(&cpu), Some(&gpu), &["--require-shared-gate"]),
        "missing shared_gate coverage",
    );
    gpu["positions"][0]["layers"][0]["shared_gate"] = json!(0.5);
    assert_rejected(compare(Some(&cpu), Some(&gpu), &[]), "coverage differs");
    cpu["positions"][0]["layers"][0]
        .as_object_mut()
        .unwrap()
        .remove("shared_gate");
    assert_rejected(
        compare(Some(&cpu), Some(&gpu), &[]),
        "missing field `shared_gate`",
    );
    reject_gpu_change(
        "/positions/1/layers/0/attn_rms",
        Value::Null,
        "missing attention",
    );
    reject_gpu_change(
        "/positions/1/layers/0/ffn_rms",
        Value::Null,
        "missing attention",
    );
}

#[test]
fn routing_ids_are_exact_and_missing_routing_cannot_pass() {
    reject_gpu_change(
        "/positions/1/layers/0/expert_ids",
        json!([0, 2]),
        "expert_ids[0]",
    );
    reject_gpu_change(
        "/positions/1/layers/0/expert_ids",
        json!([0, 0]),
        "duplicate",
    );
    reject_gpu_change(
        "/positions/1/layers/0/expert_ids",
        json!([3, 0]),
        "out-of-range",
    );
    reject_gpu_change("/positions/1/layers/0/expert_weights", json!([]), "top_k");
    reject_gpu_change(
        "/positions/1/layers/0/expert_weights",
        Value::Null,
        "both be present",
    );

    let mut cpu = fixture(false);
    let mut gpu = fixture(true);
    for trace in [&mut cpu, &mut gpu] {
        for p in trace["positions"].as_array_mut().unwrap() {
            p["layers"][0]["expert_ids"] = Value::Null;
            p["layers"][0]["expert_weights"] = Value::Null;
        }
    }
    assert_rejected(compare(Some(&cpu), Some(&gpu), &[]), "missing MoE routing");
    for trace in [&mut cpu, &mut gpu] {
        trace["model"]["n_experts"] = json!(0);
        trace["model"]["top_k"] = json!(0);
    }
    assert_rejected(
        compare(Some(&cpu), Some(&gpu), &["--require-moe"]),
        "--require-moe needs a routed model",
    );
    assert!(compare(Some(&cpu), Some(&gpu), &[]).status.success());
}

#[test]
fn numeric_drift_fails_fixed_profile_and_overrides_are_diagnostic() {
    reject_gpu_change("/positions/2/logits/0", json!(0.5), "position 2 logits[0]");
    reject_gpu_change(
        "/positions/2/layers/0/residual/0",
        json!(0.9),
        "residual[0]",
    );
    reject_gpu_change("/positions/2/layers/0/attn_rms", json!(2.5), "attn_rms[0]");
    reject_gpu_change("/positions/2/layers/0/ffn_rms", json!(2.5), "ffn_rms[0]");
    reject_gpu_change(
        "/positions/2/layers/0/expert_weights/1",
        json!(0.27),
        "expert_weights[1]",
    );

    let cpu = fixture(false);
    let mut gpu = fixture(true);
    gpu["positions"][2]["logits"][0] = json!(0.5);
    let out = compare(Some(&cpu), Some(&gpu), &["--atol", "0.3"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("DIAGNOSTIC PASS (not release evidence)"));
    gpu["positions"][2]["layers"][0]["expert_ids"] = json!([0, 2]);
    assert_rejected(
        compare(Some(&cpu), Some(&gpu), &["--atol", "1", "--rtol", "1"]),
        "exact ordered IDs",
    );
    for invalid in ["-1", "NaN", "inf"] {
        assert_eq!(
            compare(Some(&cpu), Some(&gpu), &["--atol", invalid])
                .status
                .code(),
            Some(2)
        );
    }
}

#[test]
fn byte_token_and_scalar_limits_are_enforced() {
    let d = Scratch::new();
    let big = d.0.join("oversize.json");
    fs::File::create(&big)
        .unwrap()
        .set_len(256 * 1024 * 1024 + 1)
        .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_trace-parity"))
        .arg(&big)
        .arg(&big)
        .output()
        .unwrap();
    assert_rejected(out, "exceeds 256 MiB");
    reject_gpu_change("/prefill_tokens", json!(vec![0; 511]), "at most 512");
    reject_gpu_change("/model/hidden", json!(usize::MAX), "exceed 16777216");
}

#[test]
fn malformed_and_nonfinite_values_are_rejected() {
    let d = Scratch::new();
    let cpu = d.0.join("cpu.json");
    let gpu = d.0.join("gpu.json");
    fs::write(&cpu, serde_json::to_vec(&fixture(false)).unwrap()).unwrap();
    let run = |data: &str| {
        fs::write(&gpu, data).unwrap();
        Command::new(env!("CARGO_BIN_EXE_trace-parity"))
            .arg(&cpu)
            .arg(&gpu)
            .output()
            .unwrap()
    };
    assert_rejected(run("{"), "EOF");
    let json = fixture(true).to_string();
    assert_rejected(run(&format!("{json}{{}}")), "trailing characters");
    let overflow = json.replacen("\"logits\":[0.25", "\"logits\":[3.5e38", 1);
    let stderr = rejected_stderr(run(&overflow));
    // With float_roundtrip, serde_json rejects f32 overflow during parsing.
    // Without it, the comparator rejects the infinite logit.
    assert!(
        stderr.contains("gpu.json: number out of range")
            || (stderr.contains("position 0 logits[0]") && stderr.contains("GPU=inf")),
        "expected GPU logit overflow; got {stderr}"
    );
}
