import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).with_name("expert_usage.py")


def replay():
    # Lanternfish replay v1: two prefill tokens and one singleton decode.
    trace = {
        "schema_version": 1,
        "model": {
            "name": "tiny", "architecture": "test-moe", "sha256": "a" * 64,
            "bytes": 1234, "n_layers": 1, "hidden": 2, "vocab": 6,
            "n_experts": 3, "top_k": 2,
        },
        "build": {
            "lanternfish_version": "0.2.0", "lanternfish_sha256": "b" * 64,
            "llama_system_info": "CPU", "runtime_libraries": {"/private/lib.so": "c" * 64},
        },
        "execution": {
            "backend": "cpu", "requested_gpu_layers": 0, "device": None,
            "steps": [
                {"first": i, "n": n, "phase": phase, "gpu_compute_nodes": 0,
                 "gpu_matmul_nodes": 0, "cpu_matmul_nodes": 2}
                for i, n, phase in ((0, 2, "prefill"), (2, 1, "decode"))
            ],
        },
        "prefill_tokens": [1, 2],
        "decode_tokens": [3],
        "positions": [],
    }
    for i, phase, ids, weights in (
        (0, "prefill", [1, 2], [0.0, 0.25]),
        (1, "prefill", None, None),
        (2, "decode", [0, 2], [-0.125, 0.75]),
    ):
        trace["positions"].append({
            "position": i, "phase": phase, "logits": [0.0] * 6,
            "layers": [{
                "layer": 0, "residual": [0.0, 0.0], "attn_rms": None, "ffn_rms": None,
                "expert_ids": ids, "expert_weights": weights, "shared_gate": None,
            }],
        })
    return trace


class ExpertUsageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, data, filename="trace.json"):
        path = Path(self.tmp.name) / filename
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return path

    def run_cli(self, *paths):
        return subprocess.run(
            [sys.executable, str(CLI), *map(str, paths)],
            capture_output=True, text=True, timeout=60,
        )

    def report(self, *paths):
        run = self.run_cli(*paths)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(run.stdout.splitlines()), 1)
        return json.loads(run.stdout)

    def test_reports_observed_denominator_and_keeps_zero_weight(self):
        trace = replay()
        trace["prompt"] = "secret prompt"
        path = self.write(trace)
        report = self.report(path)
        self.assertEqual(report["schema"], "brewery-expert-usage-v1")
        self.assertEqual(report["traces"], [{
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prefill_tokens": 2, "decode_tokens": 1,
        }])
        self.assertEqual(report["totals"], {
            "trace_count": 1, "prefill_tokens": 2, "decode_tokens": 1,
            "observed_positions": 2, "missing_positions": 1, "selections": 4,
        })
        layer, = report["layers"]
        self.assertEqual(layer["layer"], 0)
        prefill = layer["prefill"]
        self.assertEqual(prefill["observed_positions"], 1)
        self.assertEqual(prefill["missing_positions"], 1)
        self.assertEqual(prefill["selections"], 2)
        self.assertEqual(prefill["experts"][0], {
            "expert": 1, "selections": 1, "observed_positions": 1, "selection_fraction": 1.0,
            "weight_samples": 1, "weight_sum": 0.0, "weight_mean": 0.0,
        })
        self.assertEqual(layer["decode"]["experts"][0]["weight_mean"], -0.125)
        output = json.dumps(report)
        for private in ("secret prompt", '"/private/lib.so"', '"logits"', '"residual"', '"expert_ids"'):
            self.assertNotIn(private, output)

    def test_pools_compatible_traces_and_counts_each_phase_separately(self):
        first = self.write(replay())
        other = replay()
        other["prefill_tokens"] = [4, 5]
        other["positions"][0]["layers"][0]["expert_ids"] = [0, 2]
        layer = other["positions"][1]["layers"][0]
        layer["expert_ids"], layer["expert_weights"] = [0, 1], [0.125, 0.5]
        report = self.report(first, self.write(other, "other.json"))
        self.assertEqual(report["totals"], {
            "trace_count": 2, "prefill_tokens": 4, "decode_tokens": 2,
            "observed_positions": 5, "missing_positions": 1, "selections": 10,
        })
        prefill = report["layers"][0]["prefill"]
        self.assertEqual(prefill["observed_positions"], 3)
        self.assertEqual(prefill["experts"][1], {
            "expert": 1, "selections": 2, "observed_positions": 3, "selection_fraction": 2 / 3,
            "weight_samples": 2, "weight_sum": 0.5, "weight_mean": 0.25,
        })
        self.assertEqual(report["layers"][0]["decode"]["experts"][0]["selections"], 2)

    def test_dense_and_prefill_only_leave_empty_phase_rows(self):
        trace = replay()
        trace["model"].update(n_experts=0, top_k=0)
        trace["decode_tokens"].clear()
        trace["positions"].pop()
        trace["execution"]["steps"].pop()
        for pos in trace["positions"]:
            pos["layers"][0].update(expert_ids=None, expert_weights=None)
        layer = self.report(self.write(trace))["layers"][0]
        for phase, missing in (("prefill", 2), ("decode", 0)):
            self.assertEqual(layer[phase], {
                "observed_positions": 0, "missing_positions": missing,
                "selections": 0, "experts": [],
            })

    def assert_invalid(self, *paths, message):
        result = self.run_cli(*paths)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_rejects_duplicate_traces_even_at_different_paths(self):
        first = self.write(replay())
        duplicate = self.write(replay(), "duplicate.json")
        self.assert_invalid(first, duplicate, message="duplicate trace SHA-256")

    def test_rejects_incompatible_pool_identities(self):
        first = self.write(replay())
        for section, field, value, message in (
            ("model", "sha256", "e" * 64, "different model or tokenizer identity"),
            ("model", "vocab", 7, "different model or tokenizer identity"),
            ("build", "lanternfish_sha256", "f" * 64, "different runtime identity"),
            ("build", "runtime_libraries", {"/private/lib.so": "d" * 64}, "different runtime identity"),
        ):
            with self.subTest(section=section, field=field):
                other = replay()
                other[section][field] = value
                self.assert_invalid(first, self.write(other, "other.json"), message=message)

    def test_rejects_malformed_routing(self):
        for field, value, message in (
            ("expert_ids", None, "must both be present"),
            ("expert_ids", [1, 1], "duplicate expert ID"),
            ("expert_ids", [1, 3], "invalid integer"),
            ("expert_ids", [True, 2], "invalid integer"),
            ("expert_weights", [0.0], "incorrect length"),
            ("expert_weights", [float("nan"), 0.0], "non-finite JSON number"),
            ("expert_weights", [7e38, 0.0], "invalid expert weight"),
        ):
            with self.subTest(field=field, value=value):
                trace = replay()
                trace["positions"][0]["layers"][0][field] = value
                self.assert_invalid(self.write(trace), message=message)

    def test_rejects_malformed_replay(self):
        trace = replay()
        trace["positions"][2]["phase"] = "prefill"
        self.assert_invalid(self.write(trace), message="incorrect index or phase")
        raw = json.dumps(replay()).replace('"schema_version": 1', '"schema_version": 2, "schema_version": 1')
        self.assert_invalid(self.write(raw), message="duplicate JSON object key")
        self.assert_invalid(self.write("{"), message="invalid replay JSON")

    def test_rejects_oversize_and_nonregular_inputs(self):
        path = self.write(replay())
        with path.open("r+b") as file:
            file.truncate(64 * 1024 * 1024 + 1)
        self.assert_invalid(path, message="byte limit")
        self.assert_invalid(path.parent, message="regular file")
        trace = replay()
        trace["model"]["hidden"] = 16_777_217
        self.assert_invalid(self.write(trace), message="scalar slot limit")

    @unittest.skipUnless(os.environ.get("BREWERY_EXPERT_USAGE_FIXTURES"), "no real replay fixture directory")
    def test_real_lanternfish_cpu_fixtures(self):
        root = Path(os.environ["BREWERY_EXPERT_USAGE_FIXTURES"])
        moe = self.report(root / "moe.cpu.json")
        self.assertEqual(moe["totals"], {
            "trace_count": 1, "prefill_tokens": 3, "decode_tokens": 2,
            "observed_positions": 120, "missing_positions": 0, "selections": 960,
        })
        for phase, observed in (("prefill", 3), ("decode", 2)):
            layer = moe["layers"][0][phase]
            by_expert = {row["expert"]: row["selections"] for row in layer["experts"]}
            self.assertEqual(layer["observed_positions"], observed)
            self.assertEqual(by_expert[14], observed)
            self.assertEqual(by_expert[29], observed)
        dense = self.report(root / "dense.cpu.json")
        self.assertEqual(dense["totals"]["observed_positions"], 0)
        self.assertEqual(dense["totals"]["missing_positions"], 160)
        self.assertEqual(dense["totals"]["selections"], 0)


if __name__ == "__main__":
    unittest.main()
