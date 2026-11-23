"""Regression tests through the compiled-model reference CLI."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


CLI = Path(__file__).with_name("compiled_program.py")


def call(*args):
    return json.loads(subprocess.check_output([sys.executable, "-B", CLI, *args], text=True))


def circuit(program, directory):
    call("compile", program, directory)
    rows = json.loads(subprocess.check_output(
        [sys.executable, "-B", CLI.with_name("interp_benchmark.py"), "circuit", directory], text=True,
    ))
    return {row["method"]: row for row in rows}


class ReferenceTraceTest(unittest.TestCase):
    def test_histogram_exports_truth_and_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, trace = Path(tmp) / "model", Path(tmp) / "trace"
            model.mkdir()
            receipt = call("compile", "histogram", model)
            truth_bytes = (model / "ground-truth.v1.json").read_bytes()
            truth = json.loads(truth_bytes)
            self.assertEqual(truth["schema_version"], 1)
            self.assertEqual(truth["model_sha256"], receipt["sha256"])
            self.assertEqual(hashlib.sha256(truth_bytes).hexdigest(), receipt["ground_truth_sha256"])
            width = next(op for op in truth["operations"] if op["type"] == "SelectorWidth")
            self.assertTrue(width["selector"].startswith("same_tok_"))
            self.assertEqual([b["role"] for b in width["sites"]], ["head_out", "ffn_out"])
            self.assertGreater(len(width["sites"][0]["residual_indices"]), 0)
            self.assertEqual(len(width["residual_indices"]), 5)  # Compiler includes count zero.
            self.assertEqual(call("trace", model, "[2,0,2,1]", trace)["output"], [2, 1, 2, 1])
            manifest = json.loads((trace / "manifest.json").read_text())
            self.assertEqual((manifest["kind"], manifest["schema_version"]), ("trace", 1))
            self.assertEqual(len(manifest["tokens"]), 5)  # Includes compiler BOS.
            self.assertEqual({t["role"] for t in manifest["tensors"]}, {"l_out", "attn_out", "ffn_out"})
            rows = {}
            for t in manifest["tensors"]:
                data = (trace / t["file"]).read_bytes()
                self.assertEqual(t["sha256"], hashlib.sha256(data).hexdigest())
                rows[t["layer"], t["position"], t["role"]] = np.frombuffer(data, dtype="<f4")
            for block in manifest["compiled_heads"]:
                for position, out in enumerate(np.sum(block["heads"], axis=0)):
                    np.testing.assert_allclose(rows[block["layer"], position, "attn_out"], out, atol=1e-6)

    def test_histogram_patch_scores_compiler_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = circuit("histogram", Path(tmp))
        self.assertEqual(set(rows), {"attribution_patching", "activation_patching"})
        for row in rows.values():
            self.assertEqual(row["clean_output"], [2, 1, 2, 1])
            self.assertEqual(row["corrupt_output"], [3, 1, 3, 3])
            self.assertEqual(row["sites"], 2)
            self.assertEqual(row["compiler_sites"], 2)
            self.assertEqual(row["evaluation"], "recall_only")
            self.assertIsNone(row["precision"])
            self.assertEqual(row["recall"], 1)
            self.assertIsNone(row["roc_auc"])

    def test_sort_exact_patches_recover_three_zero_attribution_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = circuit("sort", Path(tmp))
        attribution, activation = (rows[name] for name in ("attribution_patching", "activation_patching"))
        for field in ("precision", "recall", "roc_auc"):
            self.assertEqual(attribution[field], 0.7)  # Three of six tied, three spots left.
            self.assertEqual(activation[field], 1)
        saturated = {(s["layer"], s["role"]): s["patched_delta"]
                     for s in attribution["zero_attribution_sites_with_patch_effect"]}
        self.assertEqual(set(saturated), {(0, "ffn_out"), (1, "ffn_out"), (2, "head_out")})
        for delta in saturated.values():
            self.assertAlmostEqual(delta, 2, places=4)

    def test_padded_histogram_adds_one_known_negative_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            rows = circuit("histogram_padded", model)
            self.assertEqual(call("check", model), {"cases": 340})
        attribution, activation = (rows[name] for name in ("attribution_patching", "activation_patching"))
        self.assertEqual(activation["sites"], 3)
        self.assertEqual(activation["compiler_sites"], 2)
        negative, = [s for s in activation["ranking"] if not s["compiler_site"]]
        self.assertEqual((negative["layer"], negative["role"], negative["head"], negative["patched_delta"]),
                         (0, "head_out", 1, 0))
        for field in ("precision", "recall", "roc_auc"):
            self.assertEqual(attribution[field], 0.75)  # One true and one false site tied for last spot.
            self.assertEqual(activation[field], 1)

    def test_alta_parity_tracks_shared_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            call("compile", "alta_parity_relative", model)
            truth = json.loads((model / "ground-truth.v1.json").read_text())
            self.assertEqual(truth["output_variable"], "parity")
            parity = next(op for op in truth["operations"] if op["label"] == "parity")
            self.assertEqual([(s["layer"], s["role"]) for s in parity["sites"]],
                             [(layer, "ffn_out") for layer in range(7)])
            self.assertEqual(call("trace", model, "[1,0,1,0]", model / "trace")["output"], [1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
