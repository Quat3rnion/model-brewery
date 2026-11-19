"""Public CLI tests on actual GGUF files, without downloads or inference."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from test_weight_edit import native_fixture, reference_xor

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def matrix_gguf(dtype, values):
    name = b"layer.weight"
    header = struct.pack("<4sIQQQ", b"GGUF", 3, 1, 0, len(name)) + name
    header += struct.pack("<IQQIQ", 2, 2, 2, dtype, 0)
    return header + bytes(-len(header) % 32) + values


class Tune(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="weight-tune-", dir="/arboretum")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.source = self.root / "source.gguf"
        self.original = matrix_gguf(1, struct.pack("<4e", 1, 2, 3, 4))
        self.source.write_bytes(self.original)
        self.cal = self.data("cal", [[1, 0], [0, 1]], [[2, 3], [2, 3]])
        self.held = self.data("held", [[1, 1], [2, -1]], [[3, 7], [0, 2]])

    def data(self, name, rows, targets):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps({"activations": rows, "targets": targets}))
        return path

    def tune(self, held=None, cap="1", indices="[[0,0],[1,1]]"):
        return subprocess.run(
            [sys.executable, "-B", TOOLS / "weight_tune.py", self.source,
             "--source-sha256", hashlib.sha256(self.original).hexdigest(), "--tensor", "layer.weight",
             "--indices", indices, "--calibration", self.cal, "--held-out", held or self.held,
             "--max-update", cap],
            text=True, capture_output=True, timeout=10,
        )

    def test_calibration_improves_without_hiding_held_out_regression(self):
        run = self.tune()
        self.assertEqual(run.returncode, 0, run.stderr)
        proposal, report = json.loads(run.stdout), json.loads(run.stderr)
        self.assertEqual(proposal["schema"], "brewery.weight-edit-proposal.v1")
        self.assertEqual([(e["index"], e["before"], e["after"]) for e in proposal["edits"]],
                         [([0, 0], 1, 2), ([1, 1], 4, 3)])
        self.assertEqual(report["calibration"], {"before_mse": 0.5, "after_mse": 0.0})
        self.assertEqual(report["held_out"], {"before_mse": 0.0, "after_mse": 1.75, "regression": True})
        alternate = self.data("different-held", [[3, 2], [-1, 2]], [[0, 0], [0, 0]])
        self.assertEqual(self.tune(held=alternate).stdout, run.stdout)
        self.assertEqual(self.tune(indices="[[1,1],[0,0]]").stdout, run.stdout)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(set(p.name for p in self.root.iterdir()),
                         {"source.gguf", "cal.json", "held.json", "different-held.json"})
        saved, applied = self.root / "proposal.json", self.root / "applied.gguf"
        saved.write_text(run.stdout)
        result = subprocess.run([sys.executable, "-B", TOOLS / "weight_edit.py", "apply",
                                 self.source, saved, applied], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(applied.read_bytes(), matrix_gguf(1, struct.pack("<4e", 2, 2, 3, 3)))
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_refuses_shared_activation_even_when_targets_differ(self):
        duplicate = self.data("overlap", [[1.0, -0.0]], [[80, 90]])
        result = self.tune(held=duplicate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overlap", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_native_hgn_tune_and_reviewed_apply(self):
        self.source = self.root / "source.hgn"
        self.original, offsets = native_fixture(self.source)
        run = self.tune()
        self.assertEqual(run.returncode, 0, run.stderr)
        proposal, report = json.loads(run.stdout), json.loads(run.stderr)
        self.assertEqual(report["calibration"]["after_mse"], 0.0)
        self.assertEqual([(e["index"], e["after"]) for e in proposal["edits"]],
                         [([0, 0], 2.), ([1, 1], 3.)])
        saved, out = self.root / "proposal.json", self.root / "edited.hgn"
        saved.write_text(run.stdout)
        result = subprocess.run([sys.executable, "-B", TOOLS / "weight_edit.py", "apply",
                                 self.source, saved, out], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = bytearray(self.original)
        offset = offsets["layer.weight"]
        struct.pack_into("<4e", expected, offset, 2., 2., 3., 3.)
        struct.pack_into("<I", expected, 104 + 152, reference_xor(expected[offset:offset + 8]))
        self.assertEqual(out.read_bytes(), expected)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_native_q8g64_tune_uses_actual_codes_within_cap(self):
        self.source = self.root / "source.hgn"
        self.original, offsets = native_fixture(self.source)
        self.cal = self.data("cal64", [[1.] + [0.] * 63], [[2.75]])
        self.held = self.data("held64", [[2.] + [0.] * 63], [[4.]])

        def run(cap):
            return subprocess.run(
                [sys.executable, "-B", TOOLS / "weight_tune.py", self.source,
                 "--source-sha256", hashlib.sha256(self.original).hexdigest(),
                 "--tensor", "packed.weight", "--indices", "[[0,0]]",
                 "--calibration", self.cal, "--held-out", self.held, "--max-update", cap],
                text=True, capture_output=True, timeout=10,
            )

        rejected = run("0.99")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("no representable update", rejected.stderr)
        result = run("2.9")
        self.assertEqual(result.returncode, 0, result.stderr)
        proposal, report = json.loads(result.stdout), json.loads(result.stderr)
        edit, = proposal["edits"]
        self.assertEqual((edit["before"], edit["after"]), (0, 2))
        self.assertEqual(report["calibration"]["after_mse"], 0.5625)
        self.assertEqual(report["held_out"]["after_mse"], 0)
        saved, dest = self.root / "proposal.json", self.root / "out.hgn"
        saved.write_text(result.stdout)
        applied = subprocess.run([sys.executable, "-B", TOOLS / "weight_edit.py", "apply",
                                  self.source, saved, dest],
                                 text=True, capture_output=True, timeout=10)
        self.assertEqual(applied.returncode, 0, applied.stderr)
        expected = bytearray(self.original)
        offset = offsets["packed.weight"]
        expected[offset] = 2
        struct.pack_into("<I", expected, 104 + 160 + 152, reference_xor(expected[offset:offset + 68]))
        self.assertEqual(dest.read_bytes(), expected)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_caps_actual_f16_value_after_rounding_and_rejects_noop(self):
        run = self.tune(cap="0.0009", indices="[[0,0]]")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("no representable", run.stderr)
        self.assertEqual(run.stdout, "")
        run = self.tune(cap="0.0016", indices="[[0,0]]")
        self.assertEqual(run.returncode, 0, run.stderr)
        edit, = json.loads(run.stdout)["edits"]
        self.assertEqual(edit["after"], 1.0009765625)
        self.assertLessEqual(abs(edit["after"] - edit["before"]), 0.0016)


if __name__ == "__main__":
    unittest.main()
