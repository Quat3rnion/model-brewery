import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CLI = Path(__file__).resolve().parents[1] / "tools" / "exact-export"


class ExactExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "encoded.weights"
        # One GGUF v3 tensor named x, with two BF16 values.
        header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
        entry = struct.pack("<Q", 1) + b"x" + struct.pack("<IQIQ", 1, 2, 30, 0)
        raw = header + entry
        self.source.write_bytes(raw + bytes(-len(raw) % 32) + bytes.fromhex("803f0040"))
        self.pin = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.dest = self.root / "export.gguf"

    def invoke(self, pin=None, target="gguf"):
        return subprocess.run([sys.executable, CLI, "--encoded-sha256", pin or self.pin,
                               "--to", target, self.source, self.dest],
                              capture_output=True, text=True)

    def test_export_from_pinned_encoding_without_requantizing(self):
        p = self.invoke()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.dest.read_bytes(), self.source.read_bytes())
        info = json.loads(p.stdout)
        self.assertEqual(info["output_sha256"], self.pin)
        self.assertEqual(info["verification"], "identical-file")
        self.assertFalse(info["runtime_tested"])

    def test_changed_encoding_pin_never_publishes(self):
        p = self.invoke("0" * 64)
        self.assertIn("digest mismatch", p.stderr)
        self.assertFalse(self.dest.exists())
        self.assertEqual(list(self.root.iterdir()), [self.source])

    def test_no_overwrite(self):
        self.dest.write_bytes(b"reserved")
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertEqual(self.dest.read_bytes(), b"reserved")
