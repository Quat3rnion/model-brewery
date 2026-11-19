"""Public CLI regressions for SafeTensors weight revisions."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def tensor_file(name, dtype, payload, shape):
    head = json.dumps({name: {"dtype": dtype, "shape": shape, "data_offsets": [0, len(payload)]},
                       "__metadata__": {"format": "pt"}}).encode()
    head += b" " * (-len(head) % 8)
    return struct.pack("<Q", len(head)) + head + payload


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin_dir(path):
    return sha(b"".join(
        f"{f.relative_to(path).as_posix()}\0{f.stat().st_size}\0{sha(f.read_bytes())}\n".encode()
        for f in sorted(path.rglob("*"), key=lambda f: f.relative_to(path).as_posix()) if f.is_file()
    ))


class SafeTensors(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir="/arboretum", prefix="weight-safetensors-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def cli(self, *args, tool="weight_edit.py", ok=True):
        result = subprocess.run([sys.executable, "-B", TOOLS / tool, *map(str, args)],
                                capture_output=True, text=True, timeout=15)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def doc(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def test_standalone_preserves_all_other_bytes_and_dtype(self):
        source = self.root / "source.safetensors"
        original = tensor_file("layer.weight", "F16", struct.pack("<4e", 1, 2, 3, 4), [2, 2])
        source.write_bytes(original)
        pin = ("--source-sha256", sha(original))
        tile = self.cli("tile", source, *pin, "--tensor", "layer.weight",
                        "--origin", "[1,0]", "--shape", "[1,2]")
        self.assertEqual(tile["source_format"], "safetensors")
        self.assertEqual(tile["values"], [[3., 4.]])
        request = self.doc("request", [{"tensor": "layer.weight", "index": [1, 0],
                                        "op": "set", "value": 1.0009}])
        proposal = self.cli("propose", source, *pin, request)
        self.assertEqual(proposal["edits"][0]["after"], 1.0009765625)
        approval = self.doc("approved", proposal)
        out = self.root / "out.safetensors"
        result = self.cli("apply", source, approval, out)
        self.assertEqual(out.read_bytes(), original[:-4] + b"\x01\x3c" + original[-2:])
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(result["output_sha256"], sha(out.read_bytes()))
        self.cli("apply", source, approval, out, ok=False)


if __name__ == "__main__":
    unittest.main()
