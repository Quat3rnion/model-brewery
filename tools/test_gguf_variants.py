"""Exercise the public driver with a real CPU quantizer and a tiny GGUF."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).with_name("gguf_variants.py")
ENV = ("BREWERY_TEST_ENGINE", "BREWERY_TEST_LLAMA_QUANTIZE", "BREWERY_TEST_TINY_GGUF")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(all(os.environ.get(key) for key in ENV), "requires engine and CPU fixture")
class GGUFVariantsTest(unittest.TestCase):
    def test_named_outputs_attest_verified_plans(self):
        engine, quantizer, source = (Path(os.environ[key]).resolve() for key in ENV)
        self.assertLessEqual(source.stat().st_size, 8 << 20, "use a tiny fixture")
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            variants = root / "variants.json"
            out = root / "out"
            command = [
                sys.executable, CLI, "--engine", engine, "--quantizer", quantizer,
                "--quantizer-sha256", sha(quantizer), "--source", source,
                "--variants", variants, "--out", out,
            ]
            variants.write_text(json.dumps({
                "q8": {"gguf_type": "Q8_0", "threads": 1},
                "q8-embed-f16": {
                    "gguf_type": "Q8_0", "threads": 1,
                    "tensor_types": {"token_embd.weight": "F16"},
                },
            }), encoding="utf-8")
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertFalse(manifest["release_qualified"])
            self.assertEqual(manifest["source"]["sha256"], sha(source))
            self.assertEqual(manifest["quantizer"]["sha256"], sha(quantizer))
            expected = {"q8": {}, "q8-embed-f16": {"token_embd.weight": "F16"}}
            self.assertEqual(set(manifest["variants"]), set(expected))
            for name, types in expected.items():
                entry = manifest["variants"][name]
                self.assertEqual(entry["gguf_type"], "Q8_0")
                self.assertEqual(entry["requested_tensor_types"], types)
                self.assertEqual(entry["file_verification"]["status"], "passed")
                artifact = entry["artifact"]
                self.assertEqual(artifact["path"], f"{name}.gguf")
                self.assertEqual(artifact["sha256"], sha(out / artifact["path"]))
            self.assertNotEqual(manifest["variants"]["q8"]["artifact"]["sha256"],
                                manifest["variants"]["q8-embed-f16"]["artifact"]["sha256"])

            bad = root / "bad"
            variants.write_text(json.dumps({
                "bad": {"gguf_type": "Q8_0", "tensor_types": {"no.such.weight": "F16"}},
            }), encoding="utf-8")
            run = subprocess.run([*command[:-1], bad], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("unknown exact GGUF tensor override", run.stderr)
            self.assertFalse((bad / "manifest.json").exists())
            self.assertFalse((bad / "bad.gguf").exists())

            wrong_pin = command.copy()
            wrong_pin[wrong_pin.index("--quantizer-sha256") + 1] = "0" * 64
            wrong_pin[-1] = root / "wrong-pin"
            run = subprocess.run(wrong_pin, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("quantizer SHA-256 mismatch", run.stderr)


if __name__ == "__main__":
    unittest.main()
