"""Public CLI tests for the bounded rotated HGN interchange."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

CLI = Path(__file__).resolve().parents[1] / "tools/cross_container/hgn_in_gguf.py"
TENSOR = "layers.0.mlp.experts.down_proj.weight"


def fixture(path, *, ht=False, extra=False):
    """Make an independently serialized HGN with valid nontrivial rotation."""
    tensor = "layers.0.attn_q.weight" if ht else TENSOR
    base = tensor.removesuffix(".weight")
    values = 128 * 128
    payload = (
        bytes((i * 17 + 3) % 256 for i in range(values * 3 // 8))
        if ht else b"\x98\xab" * (values // 4) + struct.pack("<e", 0.25) * (values // 128)
    )
    entries = [
        (tensor, 16 if ht else 23, 0x1206 if ht else 128,
         (128, 128) if ht else (1, 128, 128), payload),
        (base + ".suh", 2, 0, (128,), struct.pack("<e", -1) * 128),
        (base + ".svh", 2, 0, (128,), struct.pack("<e", 0.5) * 128),
    ]
    if extra:
        entries.append(("embed_tokens.weight", 2, 0, (1,), struct.pack("<e", 1)))
    data = (104 + 160 * len(entries) + 63) & ~63
    table, body = bytearray(), bytearray()
    for name, store, variant, shape, raw in entries:
        body.extend(bytes((-len(body)) % 64))
        checksum = 0
        for i in range(0, len(raw), 4):
            checksum ^= int.from_bytes(raw[i:i + 4], "little")
        table.extend(struct.pack("<96sII4QQQII", name.encode(), store, len(shape),
                                 *shape, *([0] * (4 - len(shape))), data + len(body),
                                 len(raw), checksum, variant))
        body.extend(raw)
    header = struct.pack("<4sI4Q64s", b"HGN1", 2, len(entries), 104, data,
                         data + len(body), b"brewery-rotated-cpu-fixture")
    path.write_bytes(header + table + bytes(data - len(header) - len(table)) + body)


class HgnInGguf(unittest.TestCase):
    def call(self, *args, ok=True):
        result = subprocess.run([sys.executable, "-B", CLI, *args], capture_output=True,
                                text=True, timeout=15)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def test_cpu_parity_and_refusal_on_corruption(self):
        for ht in (False, True):
            with self.subTest(ht=ht), tempfile.TemporaryDirectory() as tmp:
                src, dst = Path(tmp) / "source.hgn", Path(tmp) / "artifact.gguf"
                fixture(src, ht=ht)
                self.call("pack", src, dst)
                report = self.call("verify", src, dst)
                self.assertEqual(report["schema"], "brewery.cross-container.v1")
                self.assertEqual(report["scope"], "complete_inventory")
                self.assertTrue(report["native_decode_bitwise_equal"])
                self.assertTrue(report["cpu_matvec_bitwise_equal"])
                self.assertEqual(dst.read_bytes()[:8], b"GGUF\x03\x00\x00\x00")
                original = dst.read_bytes()
                dst.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                self.call("verify", src, dst, ok=False)
                dst.write_bytes(original)
                original = src.read_bytes()
                src.write_bytes(original[:40] + b"X" + original[41:])
                self.call("verify", src, dst, ok=False)

    def test_default_never_discards_other_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "source.hgn", Path(tmp) / "artifact.gguf"
            fixture(src, extra=True)
            self.assertIn("inventory", self.call("pack", src, dst, ok=False))
            self.assertFalse(dst.exists())
            self.assertEqual(self.call("pack", src, dst, "--tensor", TENSOR)["scope"],
                             "explicit_slice")
            self.call("verify", src, dst, ok=False)
            self.call("verify", src, dst, "--tensor", TENSOR)

    @unittest.skipUnless(os.environ.get("BREWERY_TEST_LLAMA_QUANTIZE"), "requires stock llama-quantize")
    def test_stock_gguf_loader_refuses_private_tensor_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "source.hgn", Path(tmp) / "artifact.gguf"
            fixture(src)
            self.call("pack", src, dst)
            result = subprocess.run([os.environ["BREWERY_TEST_LLAMA_QUANTIZE"], "--dry-run",
                                     dst, Path(tmp) / "must-not-exist.gguf", "Q8_0", "1"],
                                    text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid ggml type 1112670467", result.stderr.lower())
