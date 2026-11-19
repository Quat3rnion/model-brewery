"""Public adapter contract for a file and separately approved metadata."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_hf_gguf import CLI, fixture


class StandaloneGguf(unittest.TestCase):
    def test_only_the_selected_file_and_assets_are_inputs(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as tmp:
            base = Path(tmp)
            assets = base / "assets"
            assets.mkdir()
            # An unsupported architecture still has a truthful inventory.
            fixture(base, "OtherModel", [("dense.weight", [2, 2], 8)])
            (base / "config.json").rename(assets / "config.json")
            weights = base / "a-revision.safetensors"
            (base / "model.safetensors").rename(weights)
            (base / "config.json").write_text("must not read")
            (base / "modeling.py").touch()
            argv = [str(CLI), "inspect", "--source", str(weights)]
            self.assertIn("assets", subprocess.run(argv, capture_output=True, text=True).stderr)
            run = subprocess.run([*argv, "--assets", assets], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual(report["source_format"], "safetensors")
            self.assertEqual(report["source_tensors"], 1)
            self.assertFalse(report["export_supported"])
            (assets / "other.safetensors").write_bytes(weights.read_bytes())
            run = subprocess.run([*argv, "--assets", assets], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("metadata", run.stderr)
