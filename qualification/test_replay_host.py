import subprocess
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).with_name("replay_host.py")


class ReplayInputs(unittest.TestCase):
    def test_comparison_failure_still_checks_other_model_without_publishing_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / RUNNER.name
            shutil.copyfile(RUNNER, runner)
            models = []
            for name in ("dense", "moe"):
                data = name.encode()
                (root / name).write_bytes(data)
                models.append(dict(kind=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            (root / "replay-inputs.json").write_text(json.dumps(dict(models=models, prefill=[1], decode=[2])))
            lf, cmp = root / "fixture-lf", root / "fixture-cmp"
            lf.write_text("#!/usr/bin/python3\nimport json,hashlib,sys\n"
                          "print(json.dumps(dict(model={'sha256':hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()},"
                          "prefill_tokens=[1],decode_tokens=[2])))\n")
            cmp.write_text('#!/bin/sh\ntest "$1" = --require-moe\n')
            for tool in (cmp, lf):
                tool.chmod(0o700)
            args = [sys.executable, str(runner), "--dense", str(root / "dense"), "--moe", str(root / "moe"),
                    "--device", "fixture", "--lanternfish", str(lf), "--comparator", str(cmp), "--out"]
            result = subprocess.run([*args, root / "mixed"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "mixed/result.json").exists())
            report = json.loads((root / "mixed/failed.json").read_text())
            self.assertEqual(report["comparison_exit_codes"], {"dense": 1, "moe": 0})
            for kind, hashes in report["pairs"].items():
                for name in (f"{kind}.cpu.json", f"{kind}.gpu.json", f"{kind}.comparison"):
                    self.assertEqual(hashes[name], hashlib.sha256((root / "mixed" / name).read_bytes()).hexdigest())
            cmp.write_text("#!/bin/sh\nexit 0\n")
            result = subprocess.run([*args, root / "pass"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "pass/result.json").is_file())
            self.assertFalse((root / "pass/failed.json").exists())

    def test_wrong_weights_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "wrong.gguf"
            fake.write_bytes(b"not the pinned weights")
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--dense", str(fake),
                 "--moe", str(fake), "--device", "CUDA0", "--out", str(root / "run")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dense: model size mismatch", result.stderr)
            self.assertFalse((root / "run" / "result.json").exists())
