"""Run the HGN CLI on a two-layer, locally generated HGN fixture."""
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).parent / "hgn_trace"


class TraceTest(unittest.TestCase):
    def test_token_probe_does_not_open_model(self):
        from hgn_trace_fixture import fixture

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            spec = {"layers": [], "roles": [], "positions": [0], "top_k": 0, "byte_cap": 2048}
            (path / "spec.json").write_text(json.dumps(spec))
            subprocess.run([
                sys.executable, str(ROOT / "trace.py"), "--model", str(path / "not-opened.hgn"),
                "--sidecar", str(path / "sidecar.hgn"), "--config", str(path / "config.json"),
                "--tokenizer", str(path / "tokenizer.json"), "--spec", str(path / "spec.json"),
                "--out", str(path / "probe"), "--prompt", "a b", "--threads", "1",
            ], check=True)
            m = json.loads((path / "probe/manifest.json").read_text())
            self.assertEqual([t["token_id"] for t in m["tokens"]], [1, 2])
            self.assertEqual(m["tensors"], [])
            self.assertEqual(m["model_logits"], [])

    def test_spec_selects_full_streams_and_logit_views(self):
        from hgn_trace_fixture import fixture

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            spec = {"layers": [1, 0, 1], "roles": ["l_out", "final_norm"],
                    "positions": [1], "top_k": 2, "byte_cap": 8192}
            args = [sys.executable, str(ROOT / "trace.py"), "--model", str(path / "model.hgn"),
                    "--sidecar", str(path / "sidecar.hgn"), "--config", str(path / "config.json"),
                    "--tokenizer", str(path / "tokenizer.json"), "--spec", str(path / "spec.json"),
                    "--out", str(path / "trace"), "--prompt", "a b", "--threads", "1"]
            (path / "spec.json").write_text(json.dumps(spec))
            subprocess.run(args, check=True)
            out = path / "trace"
            m = json.loads((out / "manifest.json").read_text())
            self.assertEqual(m["tokens"], [{"position": 0, "token_id": 1},
                                           {"position": 1, "token_id": 2}])
            self.assertEqual(m["spec"]["layers"], [0, 1])
            self.assertEqual(m["missing"], [{"layer": 0, "role": "final_norm", "position": 1}])
            self.assertCountEqual([(t["layer"], t["role"], t["shape"]) for t in m["tensors"]],
                                  [(0, "l_out", [4, 128]), (1, "l_out", [4, 128]),
                                   (1, "final_norm", [128])])
            for tensor in m["tensors"]:
                raw = (out / tensor["file"]).read_bytes()
                self.assertEqual(len(raw), tensor["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), tensor["sha256"])
            self.assertEqual(m["model_logits"][0]["top"], m["logit_lens"][-1]["top"])
            self.assertEqual([x["token_id"] for x in m["model_logits"][0]["top"]], [0, 1])
            residual = (out / m["tensors"][0]["file"]).read_bytes()
            self.assertEqual(struct.unpack("<512f", residual), (1.0,) * 512)
            spec["byte_cap"] = 2048
            (path / "spec.json").write_text(json.dumps(spec))
            args[args.index("--out") + 1] = str(path / "too-small")
            result = subprocess.run(args, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"byte cap", result.stderr)
            self.assertFalse((path / "too-small" / "manifest.json").exists())
            args[args.index("--prompt") + 1] = " ".join(["a"] * 2049)
            result = subprocess.run(args, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"2048", result.stderr)

    def test_projects_each_stream_or_only_the_selected_writer_groups(self):
        from hgn_trace_fixture import fixture

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path, embedding_rows=[(0, 0), (1, 1), (1, 1), (1, 1)], writers=True)
            spec = {"layers": [0, 1], "roles": ["l_out"], "positions": [0], "top_k": 0, "byte_cap": 8192}
            (path / "spec.json").write_text(json.dumps(spec))
            vector = [1] + [0] * 127
            rng = random.Random(7)
            # The fixture has energy only in its first two coordinates.
            orthogonal_control = [0, 0] + [rng.gauss(0, 1) for _ in range(126)]

            def capture(name, target=None, direction=vector, layer_range=(1, 1)):
                out = path / name
                args = [sys.executable, str(ROOT / "trace.py"), "--model", str(path / "model.hgn"),
                        "--sidecar", str(path / "sidecar.hgn"), "--config", str(path / "config.json"),
                        "--tokenizer", str(path / "tokenizer.json"), "--spec", str(path / "spec.json"),
                        "--out", str(out), "--prompt", "a", "--threads", "1"]
                if target:
                    rule = {"target": target, "vector": direction}
                    if layer_range is not None:
                        rule.update(zip(("first_layer", "last_layer"), layer_range))
                    (path / "projection.json").write_text(json.dumps(rule))
                    args.extend(["--projection", str(path / "projection.json")])
                subprocess.run(args, check=True)
                manifest = json.loads((out / "manifest.json").read_text())
                return [struct.unpack("<512f", (out / row["file"]).read_bytes())
                        for row in manifest["tensors"]]

            before = capture("stock")
            residual = capture("residual", "residual")
            dense = capture("dense", "dense")
            experts = capture("experts", "experts")
            both = capture("all", "all")
            control = capture("control", "all", orthogonal_control)
            every_layer = capture("residual-all", "residual", layer_range=None)
            for stream in range(4):
                i = stream * 128
                self.assertEqual([residual[0][i], dense[0][i], experts[0][i], both[0][i]], [1] * 4)
                self.assertEqual(residual[1][i], 0)
                self.assertEqual(every_layer[0][i], 0)
                self.assertEqual(every_layer[1][i], 0)
                self.assertGreater(before[1][i], dense[1][i])
                self.assertGreater(dense[1][i], experts[1][i])
                self.assertGreater(experts[1][i], both[1][i])
                self.assertAlmostEqual(both[1][i], 1, places=6)
                self.assertAlmostEqual(before[1][i] + both[1][i], dense[1][i] + experts[1][i], places=6)
                self.assertEqual([x[1][i + 1] for x in (before, dense, experts, both, residual)], [1] * 5)
            self.assertLess(max(abs(x - y) for stock, random in zip(before, control)
                                for x, y in zip(stock, random)), 1e-6)

    def test_prefill_runs_past_the_dense_qsa_window(self):
        from hgn_trace_fixture import fixture

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path, qsa=True)
            out = path / "long"
            spec = {"layers": [1], "roles": ["l_out"], "positions": [519], "top_k": 0, "byte_cap": 32768}
            (path / "spec.json").write_text(json.dumps(spec))
            subprocess.run([
                sys.executable, str(ROOT / "trace.py"), "--model", str(path / "model.hgn"),
                "--sidecar", str(path / "sidecar.hgn"), "--config", str(path / "config.json"),
                "--tokenizer", str(path / "tokenizer.json"), "--spec", str(path / "spec.json"),
                "--out", str(out), "--prompt", " ".join(["a"] * 520), "--threads", "1",
            ], check=True)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(len(manifest["tokens"]), 520)
            self.assertEqual(manifest["tensors"][0]["position"], 519)

    def test_screen_scores_markers_and_paired_random_controls_without_text(self):
        from hgn_trace_fixture import fixture

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path, embedding_rows=[(0, 0), (1, 1), (0, 1), (0, 1)], marker=True)
            out = path / "screen"
            direction, control = [1] + [0] * 127, [0, 0, 1] + [0] * 125
            request = {"behavior": "response_length", "train": [],
                       "held_out": [{"positive": "a", "negative": "b", "agree_a": False}],
                       "trials": [{"target": "residual", "vector": direction, "random": control}]}
            (path / "spec.json").write_text(json.dumps(request))
            subprocess.run([
                sys.executable, str(ROOT / "trace.py"), "--model", str(path / "model.hgn"),
                "--sidecar", str(path / "sidecar.hgn"), "--config", str(path / "config.json"),
                "--tokenizer", str(path / "tokenizer.json"), "--spec", str(path / "spec.json"),
                "--out", str(out), "--screen", "--threads", "1",
            ], check=True)
            report = json.loads((out / "screen.json").read_text())
            self.assertEqual(report["marker_tokens"], [3])
            self.assertEqual(report["prompts"], 1)
            self.assertEqual(report["parent_hits"], 1)
            self.assertAlmostEqual(report["control_mean"], 0.75)
            self.assertGreater(report["parent_mean"], 0.99)
            trial = report["trials"][0]
            self.assertGreater(trial["reduction"], 0.24)
            self.assertAlmostEqual(trial["kl"], 0.1438410361, places=7)
            self.assertEqual(trial["wins_over_random"], 1)
            for key in ("control_shift", "random_reduction", "random_control_shift", "random_kl"):
                self.assertEqual(trial[key], 0.0)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["screen.json"])
