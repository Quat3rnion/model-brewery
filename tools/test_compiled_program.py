"""Exercise the offline CLI, including a fresh process for weight reload."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).with_name("compiled_program.py")


class CompiledProgramTest(unittest.TestCase):
    def test_sort_reloads_exact_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            receipt = self.call("compile", "sort", artifact)
            for name, field in (("model.safetensors", "sha256"), ("config.json", "config_sha256")):
                digest = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
                self.assertEqual(receipt[field], digest)
            self.assertEqual(self.call("run", artifact, "[2,0,2,1]"), [0, 1, 2, 2])
            self.assertEqual(self.call("check", artifact), {"cases": 340})
            other = artifact / "repeat"
            other.mkdir()
            self.assertEqual(receipt, self.call("compile", "sort", other))

    def test_histogram_reloads_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            self.call("compile", "histogram", artifact)
            self.assertEqual(self.call("run", artifact, "[2,0,2,1]"), [2, 1, 2, 1])
            self.assertEqual(self.call("check", artifact), {"cases": 340})

    def test_correctness_rejects_changed_weights(self):
        from safetensors.numpy import load_file, save_file

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            self.call("compile", "sort", artifact)
            model = artifact / "model.safetensors"
            weights = load_file(model)
            weights["unembed"].fill(0)
            save_file(weights, model)
            self.assertEqual(self.call("check", artifact), {"cases": 1, "mismatch": [1]})

    def test_other_examples_compute_every_valid_input(self):
        for p, cases in (("reverse", 340), ("frac_prevs", 340), ("alta_parity_relative", 30),
                         ("alta_parity_absolute", 30)):
            with self.subTest(program=p), tempfile.TemporaryDirectory() as tmp:
                artifact = Path(tmp)
                receipt = self.call("compile", p, artifact)
                other = artifact / "repeat"
                other.mkdir()
                self.assertEqual(receipt, self.call("compile", p, other))
                self.assertEqual(self.call("check", artifact), {"cases": cases})

    def call(self, *args):
        return json.loads(subprocess.check_output([sys.executable, CLI, *args], text=True))


if __name__ == "__main__":
    unittest.main()
