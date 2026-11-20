"""CLI integration tests against a locally built and pinned ik_llama.cpp helper."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
NAMES = ("IQ1_KT", "IQ2_KT", "IQ3_KT", "IQ4_KT")
IDS = (158, 153, 154, 155)


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def padded(data):
    return data + bytes(-len(data) % 32)


def fixture(path, columns=288):
    rng = random.Random(337)
    source = {}
    body = b""
    header = struct.pack("<4sIQQ", b"GGUF", 3, 5, 3)
    header += string("general.architecture") + struct.pack("<I", 8) + string("llama")
    header += string("general.file_type") + struct.pack("<II", 4, 0)
    header += string("llama.block_count") + struct.pack("<II", 4, 4)
    for i in range(5):
        name = f"blk.{i}.attn_q.weight" if i < 4 else "output_norm.weight"
        shape = (columns + 32 * i, 3) if i < 4 else (29,)
        floats = [rng.gauss(0, 0.1) for _ in range(math.prod(shape))]
        kind = (0, 1, 30, 0, 0)[i]
        if kind == 30:
            payload = b"".join(struct.pack("<f", v)[2:] for v in floats)
        else:
            payload = struct.pack("<" + ("e" if kind == 1 else "f") * len(floats), *floats)
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", kind, len(body))
        body += padded(payload)
        source[name] = (shape, kind, payload)
    path.write_bytes(padded(header) + body)
    return source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_fixture_output(path):
    """Standalone fixture reader: does not import the converter or its layouts."""
    with path.open("rb") as f:
        def unpack(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

        def read_string():
            return f.read(unpack("Q")[0]).decode()

        magic, version, count, fields = unpack("4sIQQ")
        assert (magic, version) == (b"GGUF", 3)
        metadata = {}
        for _ in range(fields):
            key = read_string()
            kind, = unpack("I")
            metadata[key] = read_string() if kind == 8 else unpack("I")[0]
        tensors = {}
        for _ in range(count):
            name = read_string()
            rank, = unpack("I")
            shape = unpack("Q" * rank)
            kind, offset = unpack("IQ")
            tensors[name] = (shape, kind, offset)
        f.seek(-f.tell() % 32, 1)
        return metadata, tensors, f.read()


@unittest.skipUnless(os.getenv("BREWERY_TEST_IK_KT_HELPER"), "set helper and independently recorded digest")
class ExactKt(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/arboretum"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.gguf"
        self.plan = self.root / "plan.json"
        self.output = self.root / "output.gguf"
        self.source = fixture(self.input)
        self.recipes = {}
        for i, kind in enumerate(NAMES):
            name = f"blk.{i}.attn_q.weight"
            self.recipes[name] = {"type": kind, "importance_sha256": None}

    def invoke(self, output=None, helper_sha=None, source_sha=None):
        self.plan.write_text(json.dumps({
            "schema": "brewery.ik-kt.v1",
            "source_sha256": source_sha or sha(self.input),
            "tensors": self.recipes,
        }))
        return subprocess.run([
            sys.executable, str(HERE / "convert.py"), "--input", str(self.input),
            "--output", str(output or self.output), "--plan", str(self.plan),
            "--plan-sha256", sha(self.plan), "--helper", os.environ["BREWERY_TEST_IK_KT_HELPER"],
            "--helper-sha256", helper_sha or os.environ["BREWERY_TEST_IK_KT_SHA256"],
            "--importance-dir", str(self.root),
        ], text=True, capture_output=True, timeout=90)

    def test_writes_all_exact_tensor_ids_and_reads_back_mixed_gguf(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        meta, tensors, body = read_fixture_output(self.output)
        self.assertNotIn("general.file_type", meta)
        self.assertEqual(meta["general.architecture"], "llama")
        self.assertEqual(meta["brewery.quantization.backend"], "ik_llama.cpp")
        self.assertEqual(meta["brewery.quantization.commit"], "7c00f8788e303bfba2886138e288effb0fd5fbbf")
        for i, (kind, expected_id, offset) in enumerate(zip(NAMES, IDS, (0, 256, 544, 992))):
            name = f"blk.{i}.attn_q.weight"
            self.assertEqual(tensors[name], (self.source[name][0], expected_id, offset))
            stats = report["tensors"][name]
            self.assertEqual(stats["type"], kind)
            self.assertGreater(stats["rmse"], 0)
            self.assertLess(stats["rmse"], 0.12)
        name = "output_norm.weight"
        self.assertEqual(tensors[name], ((29,), 0, 1600))
        self.assertEqual(body[1600:1600 + 116], self.source[name][2])
        self.assertEqual(report["output_sha256"], sha(self.output))

    def test_explicit_column_importance_changes_each_encoded_tensor(self):
        plain = self.root / "plain.gguf"
        unweighted = self.invoke(output=plain)
        self.assertEqual(unweighted.returncode, 0, unweighted.stderr)
        for name, recipe in self.recipes.items():
            width = self.source[name][0][0]
            weights = struct.pack("<" + "f" * width, *(10.0 if j % 7 == 0 else 0.1 for j in range(width)))
            digest = hashlib.sha256(weights).hexdigest()
            (self.root / (digest + ".f32")).write_bytes(weights)
            recipe["importance_sha256"] = digest
        weighted = self.invoke()
        self.assertEqual(weighted.returncode, 0, weighted.stderr)
        _, tensors, body = read_fixture_output(self.output)
        _, before, original_body = read_fixture_output(plain)
        stats = json.loads(weighted.stdout)["tensors"]
        self.assertEqual(tensors, before)
        for (name, (_, _, start)), end in zip(tensors.items(), (256, 544, 992, 1600)):
            self.assertNotEqual(body[start:end], original_body[start:end], name)
            self.assertEqual(stats[name]["importance_sha256"], self.recipes[name]["importance_sha256"])

    def test_supports_rows_shorter_than_full_trellis_block(self):
        fixture(self.input, columns=32)
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        _, tensors, body = read_fixture_output(self.output)
        for i, expected_offset in enumerate((0, 64, 160, 320)):
            self.assertEqual(tensors[f"blk.{i}.attn_q.weight"], ((32 * (i + 1), 3), IDS[i], expected_offset))
        self.assertEqual(tensors["output_norm.weight"], ((29,), 0, 544))
        self.assertTrue(any(body[:544]))

    def test_rejects_nonfinite_importance_without_publishing(self):
        name = next(iter(self.recipes))
        width = self.source[name][0][0]
        weights = struct.pack("<f", float("nan")) + struct.pack("<f", 1) * (width - 1)
        digest = hashlib.sha256(weights).hexdigest()
        (self.root / (digest + ".f32")).write_bytes(weights)
        self.recipes[name]["importance_sha256"] = digest
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid importance weight", result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_wrong_identity_unsupported_preset_wrong_shape_and_existing_output(self):
        for arguments, message in [
            ({"helper_sha": "0" * 64}, "SHA-256 mismatch"),
            ({"source_sha": "0" * 64}, "source SHA-256 mismatch"),
        ]:
            result = self.invoke(**arguments)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
            self.assertFalse(self.output.exists())
        name = next(iter(self.recipes))
        self.recipes[name]["type"] = "MOSTLY_IQ1_KT"
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported exact KT recipe", result.stderr)
        self.assertFalse(self.output.exists())
        self.recipes[name]["type"] = "IQ1_KT"
        fixture(self.input, columns=280)
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("KT row alignment", result.stderr)
        self.assertFalse(self.output.exists())
        self.output.write_bytes(b"keep me")
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output already exists", result.stderr)
        self.assertEqual(self.output.read_bytes(), b"keep me")


if __name__ == "__main__":
    unittest.main()
