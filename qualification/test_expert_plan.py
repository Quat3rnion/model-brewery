import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from test_expert_usage import replay


CLI = Path(__file__).with_name("expert_plan.py")
USAGE = CLI.with_name("expert_usage.py")
SHA = "a" * 64


def observed_moe():
    # Run a tiny top-2 MoE, not canned saliency samples. Token embeddings are
    # one-hot, experts run down(ReLU(up(x))) and the router normalizes selected
    # strengths. Capture pre-gate expert vectors at the weighted sum boundary.
    model = {
        "router": [[3, 1, 0], [3, 1, 0], [3, 1, 0], [1, 0, 1], [1, 3, 0], [0, 3, 1]],
        "down": [
            [[0, -2]] * 6,
            [[6, -8], [0, 2], [0, -2], [0, 2], [12, -16], [12, -16]],
            [[-6, 8]] * 6,
        ],
        "up": [[int(i == j) for j in range(6)] for i in range(6)],
    }
    source = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()

    def positions(tokens):
        for token in tokens:
            x = [int(i == token) for i in range(6)]
            strengths = [sum(x[i] * model["router"][i][j] for i in range(6)) for j in range(3)]
            ids = sorted(range(3), key=lambda j: (-strengths[j], j))[:2]
            total = sum(strengths[j] for j in ids)
            gates = [strengths[j] / total for j in ids]
            hidden = [max(0, sum(w * a for w, a in zip(row, x))) for row in model["up"]]
            outputs = [[sum(a * w for a, w in zip(hidden, col))
                        for col in zip(*model["down"][j])] for j in ids]
            yield {"0": [{"expert": j, "applied_gate": g, "output_l2": math.hypot(*output)}
                         for j, g, output in zip(ids, gates, outputs)]}

    common = {
        "schema": "brewery.expert-activations.v1", "source_sha256": source,
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_layers": 1, "routed_layers": [0], "n_experts": 3, "top_k": 2,
    }
    return (
        common | {"split": "calibration",
                  "samples": [{"tokens": [0, 1, 2, 3], "positions": list(positions([0, 1, 2, 3]))}]},
        common | {"split": "held-out",
                  "samples": [{"tokens": [4, 5], "positions": list(positions([4, 5]))}]},
    )


class ExpertPlanTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def cli(self, mode, *options, source=SHA):
        return subprocess.run(
            [sys.executable, CLI, mode, "--source-sha256", source, *map(str, options)],
            capture_output=True, text=True, timeout=20,
        )

    def result(self, *args, **kwargs):
        run = self.cli(*args, **kwargs)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(run.stdout.splitlines()), 1)
        self.assertEqual(run.stderr, "")
        plan = json.loads(run.stdout)
        self.assertEqual(set(plan), {"schema", "source_sha256", "layers", "top_k", "review"})
        return plan, plan["review"]

    def assert_invalid(self, *args, message, **kwargs):
        run = self.cli(*args, **kwargs)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(run.stdout, "")
        self.assertIn(message, run.stderr)
        self.assertNotIn("Traceback", run.stderr)

    def usage(self, trace):
        path = self.write("trace.json", trace)
        run = subprocess.run(
            [sys.executable, USAGE, path], capture_output=True, text=True, check=True, timeout=20,
        )
        return self.write("usage.json", json.loads(run.stdout))

    def test_complete_frequency_report_reorders_per_layer_and_prunes(self):
        trace = replay()
        trace["model"].update(n_experts=4, n_layers=2)
        trace["positions"][1]["layers"][0].update(expert_ids=[1, 2], expert_weights=[0, 0.25])
        for pos in trace["positions"]:
            pos["layers"].append(pos["layers"][0] | {
                "layer": 1, "expert_ids": [3, 0], "expert_weights": [0, 1],
            })
        report = self.usage(trace)
        plan, review = self.result("frequency", "--usage", report)
        self.assertEqual(plan["schema"], "brewery.expert-map.v1")
        self.assertEqual(plan["source_sha256"], SHA)
        self.assertEqual(plan["top_k"], 2)
        self.assertEqual(plan["layers"], {"0": [2, 1, 0, 3], "1": [0, 3, 1, 2]})
        coverage = review["coverage"]["0"]
        self.assertEqual(coverage["prefill"], {"observed_positions": 2, "missing_positions": 0})
        self.assertEqual(coverage["decode"], {"observed_positions": 1, "missing_positions": 0})
        self.assertEqual(coverage["selections"], [1, 2, 3, 0])
        pruned, prune_review = self.result("frequency", "--usage", report, "--keep", 2)
        self.assertEqual(pruned["layers"], {"0": [2, 1], "1": [0, 3]})
        self.assertTrue(prune_review["required"])
        self.assertFalse(prune_review["release_qualified"])
        self.assert_invalid("frequency", "--usage", report, "--keep", 1, message="top_k=2")

    def test_frequency_refuses_missing_routing_and_wrong_source(self):
        report = self.usage(replay())
        self.assert_invalid("frequency", "--usage", report, message="missing routing")
        self.assert_invalid("frequency", "--usage", report, source="c" * 64, message="does not match")

    def test_reap_reorders_and_prunes_by_conditional_weighted_output_norm(self):
        cal, held = observed_moe()
        cal_path = self.write("cal.json", cal)
        held_path = self.write("held.json", held)
        args = ("reap", "--calibration", cal_path, "--held-out", held_path)
        plan, review = self.result(*args, source=cal["source_sha256"])
        self.assertEqual(plan["layers"], {"0": [2, 0, 1]})
        calibration = review["calibration"]["coverage"]["0"]
        self.assertEqual(calibration["selections"], [4, 3, 1])
        for actual, expected in zip(calibration["scores"], [1.375, 7 / 6, 5]):
            self.assertAlmostEqual(actual, expected)
        held_out = review["held_out"]["coverage"]["0"]
        self.assertGreater(held_out["scores"][1], held_out["scores"][2])
        pruned, prune_review = self.result(*args, "--keep", 2, source=cal["source_sha256"])
        self.assertEqual(pruned["layers"], {"0": [2, 0]})
        self.assertEqual(prune_review["held_out"]["coverage"]["0"]["removed_selections"], 2)
        self.assert_invalid(*args, "--keep", 1, source=cal["source_sha256"], message="top_k=2")

    def test_reap_refuses_incomplete_or_wrong_observations_and_conflicting_splits(self):
        def fail(cal, held, message):
            self.assert_invalid("reap", "--calibration", self.write("cal.json", cal),
                                "--held-out", self.write("held.json", held),
                                source=cal["source_sha256"], message=message)

        cal, held = observed_moe()
        fail(cal, cal | {"split": "held-out"}, "overlapping token sequence")
        fail(cal, held | {"source_sha256": "f" * 64}, "held-out source SHA-256 does not match")
        fail(cal, held | {"observer_sha256": "f" * 64}, "observer differ")
        fail(cal, held | {"split": "calibration"}, "expected held-out split")
        for field in ("applied_gate", "output_l2"):
            with self.subTest(field=field):
                cal, held = observed_moe()
                del cal["samples"][0]["positions"][0]["0"][0][field]
                fail(cal, held, f"missing field '{field}'")
        cal, held = observed_moe()
        cal["samples"][0]["positions"][3]["0"][1]["expert"] = 1
        fail(cal, held, "undefined REAP score")
        cal["samples"][0]["positions"][3].clear()
        fail(cal, held, "missing or unexpected routed layer observations")

    def test_reap_keeps_measured_zeros_distinct_from_never_selected_and_ties_by_id(self):
        cal, held = observed_moe()
        for pos in cal["samples"][0]["positions"]:
            for entry in pos["0"]:
                if entry["expert"] == 0:
                    entry["applied_gate"] = 0.0
                if entry["expert"] == 2:
                    entry["output_l2"] = 0.0
        held["samples"][0]["positions"][1]["0"][1]["expert"] = 0
        args = ("reap", "--calibration", self.write("cal.json", cal),
                "--held-out", self.write("held.json", held))
        plan, review = self.result(*args, source=cal["source_sha256"])
        coverage = review["calibration"]["coverage"]["0"]
        self.assertEqual(coverage["scores"][0], 0)
        self.assertEqual(coverage["scores"][2], 0)
        self.assertIsNone(review["held_out"]["coverage"]["0"]["scores"][2])
        self.assertEqual(plan["layers"], {"0": [1, 0, 2]})
        run = self.cli(*args, source=cal["source_sha256"])
        self.assertEqual(json.loads(run.stdout), plan)

    def test_writers_apply_unmodified_planner_stdout(self):
        from test_expert_gguf import TOOL as gguf_cli, fixture, fixture_bytes
        from test_expert_hgn import CLI as hgn_cli, DOWN, ROUTER, UP, native_fixture, read_native
        from test_expert_safetensors import (
            CLI as st_cli, MTP, TRUNK, config, entries, read_weights, write_weights,
        )

        gguf = self.root / "model.gguf"
        gguf.write_bytes(fixture_bytes(*fixture()))
        hgn = self.root / "model.hgn"
        native_fixture(hgn, {
            ROUTER: (2, 0, (3, 1), struct.pack("<3e", 1, 2, 3)),
            UP: (2, 0, (3, 2, 1), struct.pack("<6e", 1, 2, 3, 4, 5, 6)),
            DOWN: (2, 0, (3, 1, 1), struct.pack("<3e", 1, 2, 3)),
        })
        st = self.root / "model.safetensors"
        write_weights(st, entries(TRUNK) | entries(MTP))
        cfg = self.write("config.json", config())

        for source, n_layers, n_experts in ((gguf, 2, 4), (hgn, 1, 3), (st, 2, 3)):
            with self.subTest(format=source.suffix):
                # Synthetic report for testing the byte-level plan contract only.
                # Don't retag a real Lanternfish report to target other model bytes.
                before = source.read_bytes()
                sha = hashlib.sha256(before).hexdigest()
                trace = replay()
                trace["model"].update(n_layers=n_layers, n_experts=n_experts, sha256=sha,
                                      bytes=len(before))
                for pos in trace["positions"]:
                    row = pos["layers"][0] | {"expert_ids": [2, 0], "expert_weights": [0.75, 0.25]}
                    pos["layers"] = [row | {"layer": layer} for layer in range(n_layers)]
                report = self.usage(trace)
                planned = self.cli("frequency", "--usage", report, "--keep", 2, source=sha)
                self.assertEqual(planned.returncode, 0, planned.stderr)
                plan = self.root / "plan.json"
                raw = planned.stdout.encode("ascii")
                plan.write_bytes(raw)
                plan_hash = hashlib.sha256(raw).hexdigest()
                dest = self.root / f"pruned{source.suffix}"
                plan_args = ["--plan", plan, "--plan-sha256", plan_hash]
                if source == gguf:
                    command = [gguf_cli, "apply", "--source", source, *plan_args, "--output", dest]
                elif source == hgn:
                    command = [hgn_cli, source, plan, dest, "--reviewed-sha256", plan_hash]
                else:
                    command = [st_cli, "apply", "--source", source, *plan_args, "--output", dest,
                               "--config", cfg, "--config-sha256", hashlib.sha256(cfg.read_bytes()).hexdigest()]
                applied = subprocess.run([sys.executable, *command], capture_output=True, text=True, timeout=20)
                self.assertEqual(applied.returncode, 0, applied.stderr)
                self.assertEqual(plan.read_bytes(), raw)
                self.assertEqual(source.read_bytes(), before)
                if source == gguf:
                    self.assertEqual(json.loads(applied.stdout)["output_expert_count"], 2)
                elif source == hgn:
                    self.assertEqual(read_native(dest)[ROUTER][3], struct.pack("<2e", 1, 3))
                else:
                    self.assertEqual(read_weights(dest / st.name)[TRUNK + "gate.bias"][1], (-5, 1))
