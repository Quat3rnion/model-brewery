"""One-block CLI fixtures; no model download, checkpoint Python, or GPU."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


CLI = Path(__file__).resolve().parents[1] / "tools" / "exl3_decode.py"
ORACLE_SHA = "072b5233f0e1ec39615a70416a865ca3fbf3043669aa7dc9c5597cd132e8d45d"
I, O = 128, 256
MARKERS = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
CASES = ((3, "3inst"), (8, "mcg"), (1, "mul1"), (1.5, "mul1"), (2.5, "mul1"), (3.5, "mul1"))
GOLDENS = {
    (3, "3inst"): ("ad7eb4661b17e6e25e5455294479b0acfd9a62725eca78a146d6f99a9f80607c",
                  "eebc06f49002dabfcd390603b7c31e85f61166b04594023b0a26b862870d2ed0"),
    (8, "mcg"): ("7ee5f07c2158e88f639e5a7dfb37bb51bb98397724fd2e88f8ab14be94cb9ed3",
                 "b958e5dcae8a7ab1b0b6b4192973e9637dbe33c46e42e0cb31dc9e8da0fb43c8"),
    (1, "mul1"): ("e4d88090af7690e1b776529853a1a73ec9be568db8f8e6733bce7a51709e3cd6",
                  "b32f0beb469b3099fc0a52df77503df34eccf5b74877dc328e53bf21d9c52563"),
    (1.5, "mul1"): ("20f1547e7040f662dced7ae2b02f54f78aee0404eb9ee669052ed994191a0885",
                    "f444198520fe9b766ae986cfa3f4f1a67dcd03b6fdea3f7aa0e2930efbafcd88"),
    (2.5, "mul1"): ("7655ccbade36442ce1a153f5597ca09611353f9f6a319065af0da4a8c43c3ce8",
                    "06e0ab7d5993e10992461543df278033cd9009be510978e48fbf8323bda9fb36"),
    (3.5, "mul1"): ("1ffe79c661650617ed4015941068f81af9a7320f768d9881807245063edfe796",
                    "bce41bc7862ee06b4aaef9dfc91a5162c5e6809c27c72f61719ff4cc004642ee"),
}


def values(n, phase, legacy):
    choices = (1, 1, -1, 1, -1) if legacy else (-0.5, 2, -1, 0.25, 1)
    return [choices[(i + phase) % len(choices)] for i in range(n)]


def companions(codebook):
    return values(256, 1, codebook == "mcg"), values(384, 3, codebook == "3inst")


def rotation(values, packed):
    if packed:
        words = [sum((v < 0) << bit for bit, v in enumerate(values[i:i + 16]))
                 for i in range(0, len(values), 16)]
        return "I16", [len(words)], struct.pack(f"<{len(words)}H", *words)
    return "F16", [len(values)], struct.pack(f"<{len(values)}e", *values)


def packed_tiles(bits, codebook):
    size = 2 * int(bits * 16)
    return b"".join(hashlib.shake_256(f"{bits}/{codebook}/{row}/{col}".encode()).digest(size)
                    for row in range(16) for col in range(24))


def fixture(path, bits, codebook, marker_value=None):
    a, b = companions(codebook)
    tensors = {
        "linear.trellis": ("I16", [16, 24, int(bits * 16)], packed_tiles(bits, codebook)),
        "linear.su" if codebook == "mcg" else "linear.suh": rotation(a, codebook == "mcg"),
        "linear.sv" if codebook == "3inst" else "linear.svh": rotation(b, codebook == "3inst"),
    }
    if codebook in MARKERS:
        val = MARKERS[codebook] if marker_value is None else marker_value
        tensors[f"linear.{codebook}"] = ("I32", [], struct.pack("<I", val))
    header, chunks, offset = {}, [], 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))


def measured_digest(report, basis):
    fmt = "e" if basis == "rotated" else "d"
    flat = [v for row in report["values"] for v in row]
    return hashlib.sha256(struct.pack(f"<{len(flat)}{fmt}", *flat)).hexdigest()


def oracle_digests(path):
    # Only the exact primary-source test reference at the pinned commit may
    # execute here. The checkpoint is never used to obtain Python code.
    import numpy as np

    with path.open("rb") as f:
        raw = f.read(10000)
    if hashlib.sha256(raw).hexdigest() != ORACLE_SHA:
        raise ValueError("unexpected pinned EXL3 oracle digest")
    # The upstream NumPy path tests isinstance(..., torch.Tensor). No torch
    # operations are needed for it; don't import the extension or require CUDA.
    env = {"torch": SimpleNamespace(Tensor=type("UnusedTensor", (), {}))}
    exec(compile(raw.replace(b"import torch\n", b"", 1), str(path), "exec"), env)
    idx = np.arange(128, dtype=np.uint8)
    parity = np.asarray([i.bit_count() % 2 for i in range(128)])
    h = (1 - 2 * parity[idx[:, None] & idx[None, :]]).astype(np.float64)
    for bits, codebook in CASES:
        packed = np.frombuffer(packed_tiles(bits, codebook), dtype="<i2").reshape(16, 24, -1)
        rotated = env["dequant_rotated"](packed, bits, codebook)[I:I + 128, O:O + 128]
        a, b = companions(codebook)
        # Equivalent to upstream dequant() with exact dyadic arithmetic:
        # the combined normalization of the two Hadamards is 1/128.
        original = (h @ rotated.astype(np.float64) @ h) / 128
        original = original * np.array(a[I:I + 128])[:, None] * np.array(b[O:O + 128])[None, :]
        yield (bits, codebook), tuple(hashlib.sha256(array.tobytes()).hexdigest()
                                     for array in (rotated.astype("<f2"), original.astype("<f8")))


class DecodeEXL3(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "model.safetensors"

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(CLI), str(self.path), "linear", *args],
                              text=True, capture_output=True, timeout=10)

    def test_integer_block_uses_native_tile_order_and_wraparound(self):
        fixture(self.path, 3, "3inst")
        result = self.run_cli("--basis", "rotated", "--origin", str(I), str(O))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["basis"], "rotated")
        self.assertEqual(report["origin"], [I, O])
        self.assertEqual(report["shape"], [128, 128])
        self.assertEqual(report["payload_bytes_read"], 6144)
        self.assertEqual(measured_digest(report, "rotated"), GOLDENS[(3, "3inst")][0])

    def test_all_codebooks_and_half_rates_against_reference_blocks(self):
        for bits, codebook in CASES:
            fixture(self.path, bits, codebook)
            for j, basis in enumerate(("rotated", "original")):
                with self.subTest(bits=bits, codebook=codebook, basis=basis):
                    result = self.run_cli("--basis", basis, "--origin", str(I), str(O))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["validation"], "cpu-block-reconstruction")
                    self.assertEqual(report["bits_per_weight"], bits)
                    self.assertEqual(report["codebook"], codebook)
                    self.assertEqual(measured_digest(report, basis), GOLDENS[bits, codebook][j])
                    payload = int(bits * 2048) + (4 if codebook != "3inst" else 0)
                    if basis == "original":
                        payload += 512 if codebook == "mul1" else 272
                    self.assertEqual(report["payload_bytes_read"], payload)

    def test_rejects_wrong_marker_and_out_of_range_block(self):
        fixture(self.path, 2.5, "mul1", marker_value=0xCBAC1FED)
        result = self.run_cli()
        self.assertEqual(result.returncode, 2, result)
        self.assertIn("incorrect multiplier", result.stderr)
        self.assertEqual(result.stdout, "")
        fixture(self.path, 2.5, "mul1")
        for origin, message in ((("16", "0"), "128-aligned"),
                                (("0", "384"), "beyond the padded matrix")):
            result = self.run_cli("--origin", *origin)
            self.assertEqual(result.returncode, 2, result)
            self.assertIn(message, result.stderr)
            self.assertEqual(result.stdout, "")

    @unittest.skipUnless(os.environ.get("EXL3_UPSTREAM_TRELLIS"),
                         "set EXL3_UPSTREAM_TRELLIS to pinned upstream tests/testlib/trellis.py")
    def test_goldens_against_pinned_source(self):
        self.assertEqual(dict(oracle_digests(Path(os.environ["EXL3_UPSTREAM_TRELLIS"]))), GOLDENS)
