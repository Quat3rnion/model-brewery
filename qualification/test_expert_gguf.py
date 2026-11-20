"""Bounded offline fixtures for the expert GGUF CLI; no model downloads.

Set BREWERY_LLAMA_CPU_BUILD to a configured, built CPU-only static llama.cpp
build directory to also check runtime decode and reorder logit parity.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "expert_gguf.py"
PLANNER = Path(__file__).resolve().with_name("expert_plan.py")
ARCH = "qwen2moe"
EXPERT_COUNT = f"{ARCH}.expert_count"
ALIGNMENT = 64

LLAMA_PROBE = r"""
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static std::vector<float> decode(const char * path) {
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(path, mp);
    if (!model) throw std::runtime_error("model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 32;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.offload_kqv = cp.op_offload = false;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context init failed");
    const int n = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n != 16) throw std::runtime_error("vocabulary mismatch");
    llama_token prompt[] = {1, 4, 7, 6};
    llama_batch batch = llama_batch_get_one(prompt, 4);
    std::vector<float> values;
    for (int i = 0; i < 3; ++i) {
        if (llama_decode(ctx, batch)) throw std::runtime_error("decode failed");
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) throw std::runtime_error("missing logits");
        for (int j = 0; j < n; ++j) {
            if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite logits");
            values.push_back(logits[j]);
        }
        prompt[0] = std::max_element(logits, logits + n) - logits;
        batch = llama_batch_get_one(prompt, 1);
    }
    llama_free(ctx);
    llama_model_free(model);
    return values;
}

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    try {
        llama_backend_init();
        const auto source = decode(argv[1]);
        const auto reorder = decode(argv[2]);
        const auto prune = decode(argv[3]);
        if (source.size() != reorder.size() || source.size() != prune.size()) return 2;
        float delta = 0;
        for (size_t i = 0; i < source.size(); ++i) {
            delta = std::max(delta, std::abs(source[i] - reorder[i]));
        }
        std::printf("%zu logits per artifact; reorder max abs delta %.9g\n", source.size(), delta);
        llama_backend_free();
        return delta > 1e-6;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
"""


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def scalar(kind, value):
    if kind == 8:
        return string(value)
    return struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?"}[kind], value)


def metadata_blob(fields):
    return b"".join(string(key) + struct.pack("<I", kind) + value for key, kind, value in fields)


def padding(n):
    return b"\0" * (-n % ALIGNMENT)


def fixture(arch=ARCH):
    """Two complete tiny llama.cpp models' supported tensor layouts."""
    fields = []
    tensors = {}
    rng = random.Random(320)
    expert_ff = 32 if arch == "qwen2moe" else 64

    def key(name, kind, value):
        fields.append((name, kind, scalar(kind, value)))

    def array(name, kind, values):
        fields.append((name, 9, struct.pack("<IQ", kind, len(values))
                       + b"".join(scalar(kind, value) for value in values)))

    def floats(name, dims, *, norm=False):
        n = 1
        for d in dims:
            n *= d
        values = [1.0] * n if norm else [rng.uniform(-0.16, 0.16) for _ in range(n)]
        tensors[name] = (dims, 0, struct.pack(f"<{n}f", *values))

    key("general.architecture", 8, arch)
    key("general.alignment", 4, ALIGNMENT)
    key("general.name", 8, "Offline tiny MoE expert fixture")
    for name, value in {
            "block_count": 2, "context_length": 32, "embedding_length": 32,
            "feed_forward_length": 64, "expert_count": 4, "expert_used_count": 2,
            "expert_feed_forward_length": expert_ff, "attention.head_count": 4,
            "attention.head_count_kv": 2, "rope.dimension_count": 8,
    }.items():
        key(f"{arch}.{name}", 4, value)
    if arch == "qwen2moe":
        key(f"{arch}.expert_shared_feed_forward_length", 4, 32)
    key(f"{arch}.attention.layer_norm_rms_epsilon", 6, 1e-5)
    key("tokenizer.ggml.model", 8, "llama")
    key("tokenizer.ggml.bos_token_id", 4, 1)
    key("tokenizer.ggml.eos_token_id", 4, 2)
    key("tokenizer.ggml.unknown_token_id", 4, 0)
    array("tokenizer.ggml.tokens", 8, ["<unk>", "<s>", "</s>"] + list("abcdefghijklm"))
    array("tokenizer.ggml.scores", 6, [0.0] * 16)
    array("tokenizer.ggml.token_type", 5, [2, 3, 3] + [1] * 13)

    floats("token_embd.weight", (32, 16))
    floats("output_norm.weight", (32,), norm=True)
    floats("output.weight", (32, 16))
    for layer in range(2):
        prefix = f"blk.{layer}."
        floats(prefix + "attn_norm.weight", (32,), norm=True)
        floats(prefix + "attn_q.weight", (32, 32))
        floats(prefix + "attn_k.weight", (32, 16))
        floats(prefix + "attn_v.weight", (32, 16))
        floats(prefix + "attn_output.weight", (32, 32))
        if arch == "qwen3moe":
            floats(prefix + "attn_k_norm.weight", (8,), norm=True)
            floats(prefix + "attn_q_norm.weight", (8,), norm=True)
        floats(prefix + "ffn_norm.weight", (32,), norm=True)
        floats(prefix + "ffn_gate_inp.weight", (32, 4))
        for role in ("gate", "up", "down"):
            # Each 18-byte Q4_0 block is a binary16 d and 16 low/high-nibble pairs.
            packed = b"".join(struct.pack("<e", 0.0125) + rng.randbytes(16) for _ in range(4 * expert_ff))
            dims = (expert_ff, 32, 4) if role == "down" else (32, expert_ff, 4)
            tensors[prefix + f"ffn_{role}_exps.weight"] = (dims, 2, packed)
        if arch == "qwen2moe":
            floats(prefix + "ffn_gate_inp_shexp.weight", (32,))
            for role in ("gate", "up", "down"):
                floats(prefix + f"ffn_{role}_shexp.weight", (32, 32))
    return fields, tensors


def fixture_bytes(fields, tensors):
    info = bytearray()
    payload = bytearray()
    for name, (dims, kind, data) in tensors.items():
        info += string(name) + struct.pack("<I", len(dims))
        info += struct.pack(f"<{len(dims)}QIQ", *dims, kind, len(payload))
        payload += data + padding(len(data))
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(fields)) + metadata_blob(fields) + info
    return header + padding(len(header)) + payload


class ExpertGGUFTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/arboretum"))
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source, self.plan, self.output = [root / name for name in ("source.gguf", "plan.json", "result.gguf")]
        self.fields, self.tensors = fixture()

    def write_plan(self, layers, *, fields=None, tensors=None, **plan_fields):
        self.source.write_bytes(fixture_bytes(fields if fields is not None else self.fields,
                                              tensors if tensors is not None else self.tensors))
        self.plan.write_text(json.dumps({
            "schema": "brewery.expert-map.v1",
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "layers": layers,
            **plan_fields,
        }), encoding="utf-8")

    def run_cli(self, digest=None, *, command="apply"):
        output = [] if command == "preview" else ["--output", str(self.output)]
        return subprocess.run(
            [sys.executable, str(TOOL), command, "--source", str(self.source),
             "--plan", str(self.plan), "--plan-sha256",
             digest if digest is not None else hashlib.sha256(self.plan.read_bytes()).hexdigest(),
             *output],
            text=True, capture_output=True, timeout=10,
        )

    def test_reorders_packed_slices_and_router_together(self):
        self.write_plan({"0": [2, 0, 3, 1], "1": [1, 3, 0, 2]})
        source = self.source.read_bytes()
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = self.tensors.copy()
        for layer, permutation in enumerate(((2, 0, 3, 1), (1, 3, 0, 2))):
            for name, width in (("ffn_gate_inp", 128), ("ffn_gate_exps", 576),
                                ("ffn_up_exps", 576), ("ffn_down_exps", 576)):
                key = f"blk.{layer}.{name}.weight"
                shape, kind, packed = self.tensors[key]
                old = [packed[i:i + width] for i in range(0, len(packed), width)]
                expected[key] = (shape, kind, b"".join(old[i] for i in permutation))
        self.assertEqual(self.output.read_bytes(), fixture_bytes(self.fields, expected))
        self.assertEqual(self.source.read_bytes(), source)

    def test_previews_pruning_and_verifies_only_the_exact_output(self):
        self.write_plan({"0": [3, 1], "1": [0, 2]})
        preview = self.run_cli(command="preview")
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertFalse(self.output.exists())
        apply = self.run_cli()
        self.assertEqual(apply.returncode, 0, apply.stderr)
        verify = self.run_cli(command="verify")
        self.assertEqual(verify.returncode, 0, verify.stderr)
        receipt = json.loads(verify.stdout)
        self.assertEqual(json.loads(apply.stdout), receipt)
        self.assertEqual(receipt.pop("output_sha256"), hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(json.loads(preview.stdout), receipt)
        self.assertEqual(receipt["operation"], "prune")
        self.assertEqual(receipt["output_expert_count"], 2)
        self.assertEqual(receipt["expert_used_count"], 2)
        self.assertLess(receipt["output_bytes"], receipt["source_bytes"])
        expected = self.tensors.copy()
        for layer, selection in enumerate(((3, 1), (0, 2))):
            for name, width in (("ffn_gate_inp", 128), ("ffn_gate_exps", 576),
                                ("ffn_up_exps", 576), ("ffn_down_exps", 576)):
                key = f"blk.{layer}.{name}.weight"
                shape, kind, packed = self.tensors[key]
                expected[key] = ((*shape[:-1], 2), kind,
                                 b"".join(packed[i * width:(i + 1) * width] for i in selection))
        expected_fields = [(key, kind, scalar(kind, 2) if key == EXPERT_COUNT else value)
                           for key, kind, value in self.fields]
        self.assertEqual(self.output.read_bytes(), fixture_bytes(expected_fields, expected))
        tampered = bytearray(self.output.read_bytes())
        tampered[-ALIGNMENT] ^= 1
        self.output.write_bytes(tampered)
        verify = self.run_cli(command="verify")
        self.assertNotEqual(verify.returncode, 0)
        self.assertIn("staged tensor bytes mismatch", verify.stderr)
        self.assertEqual(self.output.read_bytes(), tampered)

    def test_requires_exact_reviewed_bytes_and_physical_source(self):
        layers = {"0": [2, 0, 3, 1], "1": [1, 3, 0, 2]}
        self.write_plan(layers, top_k=2, review={"top_k": 2, "method": "fixture"})
        digest = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        self.plan.write_bytes(self.plan.read_bytes() + b"\n")
        result = self.run_cli(digest)
        self.assertIn("reviewed plan SHA-256 mismatch", result.stderr)
        self.assertFalse(self.output.exists())
        # Same model structure, different physical source bytes.
        self.source.write_bytes(self.source.read_bytes().replace(b"Offline", b"Altered"))
        result = self.run_cli()
        self.assertIn("physical source SHA-256 mismatch", result.stderr)
        self.assertFalse(self.output.exists())
        self.write_plan(layers, top_k=2, review={"top_k": 2, "method": "fixture"})
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_non_native_top_k_in_reviewed_plan(self):
        layers = {"0": [2, 0, 3, 1], "1": [1, 3, 0, 2]}
        for fields, reason in (
            ({"top_k": 3}, "plan.top_k disagrees"),
            ({"top_k": True}, "plan.top_k must be a positive integer"),
            ({"top_k": 2, "review": {"top_k": 1}}, "review.top_k disagrees"),
        ):
            with self.subTest(fields=fields):
                self.write_plan(layers, **fields)
                self.assertIn(reason, self.run_cli().stderr)
                self.assertFalse(self.output.exists())

    def test_applies_unmodified_planner_stdout(self):
        self.source.write_bytes(fixture_bytes(self.fields, self.tensors))
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        calibration, holdout = (self.source.with_name(name) for name in ("calibration.json", "holdout.json"))
        common = {
            "schema": "brewery.expert-activations.v1", "source_sha256": digest, "observer_sha256": "a" * 64,
            "n_layers": 2, "routed_layers": [0, 1], "n_experts": 4, "top_k": 2,
        }
        positions = [{
            str(layer): [{"expert": expert, "applied_gate": 0.5, "output_l2": float(expert + layer + 1)}
                         for expert in pair]
            for layer in range(2)
        } for pair in ((0, 1), (2, 3))]
        for path, label, tokens in (
            (calibration, "calibration", [0, 1]), (holdout, "held-out", [2, 3]),
        ):
            path.write_text(json.dumps(common | {
                "split": label, "samples": [{"tokens": tokens, "positions": positions}],
            }))
        result = subprocess.run(
            [sys.executable, str(PLANNER), "reap", "--source-sha256", digest, "--calibration", str(calibration),
             "--held-out", str(holdout), "--keep", "3"], capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.plan.write_bytes(result.stdout)
        applied = self.run_cli()
        self.assertEqual(applied.returncode, 0, applied.stderr)
        self.assertEqual(json.loads(applied.stdout)["output_expert_count"], 3)
        self.assertEqual(json.loads(applied.stdout)["plan_sha256"], hashlib.sha256(result.stdout).hexdigest())

    def test_never_overwrites_files_or_follows_output_symlinks(self):
        self.write_plan({"0": [1, 0, 3, 2], "1": [0, 1, 2, 3]})
        self.output.write_bytes(b"pre-existing")
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_bytes(), b"pre-existing")
        self.output.unlink()
        target = self.output.with_name("missing")
        self.output.symlink_to(target)
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(target.exists())

    def test_rejects_unsafe_maps_before_writing(self):
        for layers, reason in (
            ({"0": [1, 0]}, "every routed layer"),
            ({"0": [1, 0], "1": [3, 0], "2": [3, 1]}, "every routed layer"),
            ({"0": [1, 0], "01": [3, 0]}, "every routed layer"),
            ({"0": [1, 0], "1": [2, 1, 0]}, "uniform"),
            ({"0": [1], "1": [2]}, "retention count"),
            ({"0": [1, 1], "1": [3, 0]}, "unique in-range integer"),
            ({"0": [1, 4], "1": [3, 0]}, "unique in-range integer"),
            ({"0": [1, False], "1": [3, 0]}, "unique in-range integer"),
        ):
            with self.subTest(layers=layers):
                self.write_plan(layers)
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(reason, result.stderr)
                self.assertFalse(self.output.exists())
        raw = self.plan.read_text()
        self.plan.write_text(raw.replace('"0":', '"0": [3, 2], "0":'))
        self.assertIn("duplicate plan key", self.run_cli().stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_ambiguous_layouts_and_truncated_storage(self):
        perm = {"0": [2, 0, 3, 1], "1": [0, 1, 3, 2]}
        tensors = self.tensors.copy()
        good = tensors["blk.0.ffn_gate_exps.weight"]
        for case, fields, edits, reason in (
            ("grouped", self.fields + [(f"{ARCH}.expert_group_count", 4, scalar(4, 2))], {},
             "expert_group_count"),
            ("sharded", self.fields + [("split.count", 4, scalar(4, 2))], {}, "sharded GGUF"),
            ("unknown", [(key, kind, scalar(8, "gpt-oss") if key == "general.architecture" else value)
                         for key, kind, value in self.fields], {}, "unsupported GGUF MoE architecture"),
            ("extra", self.fields, {"blk.1.ffn_gate_exps.scale": ((4,), 0, bytes(16))}, "unsupported GGUF tensor"),
            ("wrong-rank", self.fields,
             {"blk.0.ffn_gate_exps.weight": ((32, 32, 1, 4), good[1], good[2])}, "unexpected GGUF shape"),
        ):
            with self.subTest(case=case):
                self.write_plan(perm, fields=fields, tensors=tensors | edits)
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(reason, result.stderr)
                self.assertFalse(self.output.exists())
        self.write_plan(perm)
        self.source.write_bytes(self.source.read_bytes()[:-ALIGNMENT])
        self.assertIn("truncated", self.run_cli().stderr)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.environ.get("BREWERY_LLAMA_CPU_BUILD"), "no local CPU llama.cpp build configured")
    def test_cpu_llama_cpp_reorder_logit_parity_and_pruned_decode(self):
        build = Path(os.environ["BREWERY_LLAMA_CPU_BUILD"])
        cache = (build / "CMakeCache.txt").read_text().splitlines()
        self.assertIn("BUILD_SHARED_LIBS:BOOL=OFF", cache)
        self.assertIn("GGML_CUDA:BOOL=OFF", cache)
        for backend in ("HIP", "VULKAN", "SYCL", "METAL", "OPENCL"):
            self.assertNotIn(f"GGML_{backend}:BOOL=ON", cache)
        llama = Path(next(line.split("=", 1)[1] for line in cache if line.startswith("CMAKE_HOME_DIRECTORY:")))
        cpp = self.source.with_name("decode.cpp")
        probe = cpp.with_suffix("")
        cpp.write_text(LLAMA_PROBE)
        compiled = subprocess.run(
            [os.environ.get("CXX", "c++"), "-std=c++17", "-I", str(llama / "include"),
             "-I", str(llama / "ggml/include"), str(cpp), "-Wl,--start-group",
             str(build / "src/libllama.a"), *(str(build / f"ggml/src/lib{name}.a")
                                            for name in ("ggml", "ggml-cpu", "ggml-base")),
             "-Wl,--end-group", "-fopenmp", "-ldl", "-pthread", "-o", str(probe)],
            env={**os.environ, "TMPDIR": str(cpp.parent)}, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        for arch in ("qwen2moe", "qwen3moe"):
            with self.subTest(arch=arch):
                fields, tensors = fixture(arch)
                self.output = self.source.with_name(f"{arch}-reorder.gguf")
                self.write_plan({"0": [2, 0, 3, 1], "1": [1, 3, 0, 2]}, fields=fields, tensors=tensors)
                result = self.run_cli()
                self.assertEqual(result.returncode, 0, result.stderr)
                reordered = self.output
                self.output = self.source.with_name(f"{arch}-prune.gguf")
                self.write_plan({"0": [3, 1], "1": [0, 2]}, fields=fields, tensors=tensors)
                result = self.run_cli()
                self.assertEqual(result.returncode, 0, result.stderr)
                decoded = subprocess.run([str(probe), str(self.source), str(reordered), str(self.output)],
                                         capture_output=True, text=True, timeout=30)
                self.assertEqual(decoded.returncode, 0, decoded.stdout + decoded.stderr[-3000:])
                print(f"{arch}: {decoded.stdout.strip()}")


if __name__ == "__main__":
    unittest.main()
