"""Public CLI tests for bounded GGUF scalar edits."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import gguf
import numpy as np

CLI = Path(__file__).resolve().parents[1] / "tools/weight_edit.py"
FIXTURE = os.environ.get("BREWERY_TEST_DENSE_GGUF")
NAME = "blk.0.attn_q.weight"


class WeightEdit(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir="/arboretum", prefix="weight-edit-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source.gguf"
        self.tensor_fixture(self.source, np.array([[0.25, -0.5, 1], [2, 3, 4]], dtype=np.float32))

    @staticmethod
    def tensor_fixture(path, data, kind=None):
        writer = gguf.GGUFWriter(path, "llama")
        writer.add_name("coordinate-edit fixture")
        writer.add_tensor(NAME, data, raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(CLI), *map(str, args)],
                              capture_output=True, text=True, timeout=10)

    def run_ok(self, *args):
        result = self.cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def sha(self):
        return hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_inventory_file_order_and_full_rank_tiles(self):
        vector = np.array([1, -2, 3], dtype=np.float16)
        experts = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        writer = gguf.GGUFWriter(self.source, "llama")
        writer.add_tensor("z.vector", vector)
        writer.add_tensor("a.experts", experts)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        pin = ("--source-sha256", self.sha())
        page = self.run_ok("inventory", self.source, *pin, "--offset", 0, "--limit", 1)
        self.assertEqual(page["tensors"], [
            {"name": "z.vector", "shape": [3], "dtype": "F16", "elements": 3}
        ])
        self.assertEqual(page["next_offset"], 1)
        page = self.run_ok("inventory", self.source, *pin, "--offset", 1, "--limit", 1)
        self.assertEqual(page["tensors"], [
            {"name": "a.experts", "shape": [2, 3, 4], "dtype": "F32", "elements": 24}
        ])
        self.assertIsNone(page["next_offset"])
        result = self.run_ok("tile", self.source, *pin, "--tensor", "z.vector",
                             "--origin", "[1]", "--shape", "[2]")
        self.assertEqual(result["values"], [[-2.0, 3.0]])
        result = self.run_ok("tile", self.source, *pin, "--tensor", "a.experts",
                             "--origin", "[1,1,2]", "--shape", "[1,2,2]")
        self.assertEqual(result["values"], [[18.0, 19.0], [22.0, 23.0]])
        self.assertNotEqual(self.cli("tile", self.source, *pin, "--tensor", "a.experts",
                                    "--origin", "[0,0,0]", "--shape", "[2,1,1]").returncode, 0)

    def document(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_read_propose_and_apply_compare_entire_file(self):
        before = self.source.read_bytes()
        sha = self.sha()
        result = self.run_ok("inspect", self.source, "--source-sha256", sha, "--tensor", NAME,
                             "--index", "[1,0]")
        self.assertEqual(result["value"], 2.0)
        request = self.document("request.json", [
            {"tensor": NAME, "index": [1, 0], "op": "delta", "value": -0.125},
        ])
        proposal = self.run_ok("propose", self.source, "--source-sha256", sha, request)
        self.assertEqual(proposal["source_sha256"], sha)
        self.assertEqual(proposal["edits"][0]["before"], 2.0)
        self.assertEqual(proposal["edits"][0]["after"], 1.875)
        destination = self.root / "edited.gguf"
        approval = self.document("proposal.json", proposal)
        receipt = self.run_ok("apply", self.source, approval, destination)
        actual = destination.read_bytes()
        self.assertEqual(receipt["output_sha256"], hashlib.sha256(actual).hexdigest())
        offset = gguf.GGUFReader(self.source).tensors[0].data_offset + 3 * 4
        self.assertEqual(actual, before[:offset] + struct.pack("<f", 1.875) + before[offset + 4:])
        self.assertEqual(self.source.read_bytes(), before)
        self.assertNotEqual(self.cli("apply", self.source, approval, destination).returncode, 0)
        self.assertEqual(destination.read_bytes(), actual)

    def test_native_hgn_retains_packed_and_companion_bytes(self):
        self.source = self.root / "mixed.hgn"
        original, offsets = native_fixture(self.source)
        sha = self.sha()
        common = ("--source-sha256", sha)
        inventory = self.run_ok("inventory", self.source, *common, "--offset", "0", "--limit", "10")
        self.assertEqual(inventory["source_format"], "hgn")
        rows = {r["name"]: r for r in inventory["tensors"]}
        self.assertTrue(rows["layer.weight"]["editable"])
        self.assertTrue(rows["packed.weight"]["editable"])
        for name in ("rotated.weight", "rotated.suh", "rotated.svh"):
            self.assertFalse(rows[name]["editable"])
            self.assertTrue(rows[name]["reason"])
        tile = self.run_ok("tile", self.source, *common, "--tensor", "layer.weight",
                           "--origin", "[0,0]", "--shape", "[2,2]")
        self.assertEqual(tile["values"], [[1., 2.], [3., 4.]])
        request = self.document("request.json", [
            {"tensor": "layer.weight", "index": [1, 0], "op": "set", "value": 4.5},
        ])
        proposal = self.run_ok("propose", self.source, *common, request)
        out = self.root / "edited.hgn"
        result = self.run_ok("apply", self.source, self.document("approval.json", proposal), out)
        self.assertEqual(result["source_format"], "hgn")
        actual = out.read_bytes()
        expected = bytearray(original)
        struct.pack_into("<e", expected, offsets["layer.weight"] + 4, 4.5)
        struct.pack_into("<I", expected, 104 + 152, reference_xor(expected[offsets["layer.weight"]:offsets["layer.weight"] + 8]))
        self.assertEqual(actual, expected)
        self.assertEqual(self.source.read_bytes(), original)

    def test_hgn_rejects_corrupt_payload_and_wrong_role_even_when_sha_matches(self):
        self.source = self.root / "mixed.hgn"
        original, offsets = native_fixture(self.source)
        wrong = bytearray(original)
        wrong[offsets["packed.weight"]] ^= 1
        self.source.write_bytes(wrong)
        self.assertIn("checksum mismatch", self.cli("inventory", self.source, "--source-sha256",
                      self.sha(), "--offset", "0", "--limit", "10").stderr)
        self.source.write_bytes(original)
        for name, index in (("rotated.weight", [0, 0]), ("rotated.suh", [0])):
            result = self.cli("propose", self.source, "--source-sha256", self.sha(),
                              self.document("request.json", [
                                  {"tensor": name, "index": index, "op": "set", "value": 5.},
                              ]))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("read-only", result.stderr)
        wrong = bytearray(original)
        # Valid checksum and scalar byte length are not sufficient for a
        # rotation companion: the expected named F16 vector must exist.
        wrong[104 + 3 * 160:104 + 3 * 160 + 96] = b"other.weight".ljust(96, b"\0")
        self.source.write_bytes(wrong)
        self.assertIn("rotation companion", self.cli("inventory", self.source, "--source-sha256",
                      self.sha(), "--offset", "0", "--limit", "10").stderr)

    def test_q8g64_manual_rounding_and_zero_scale(self):
        self.source = self.root / "packed.hgn"
        original, offsets = native_fixture(self.source)
        offset = offsets["packed.weight"]
        pin = ("--source-sha256", self.sha())
        req = self.document("request.json", [
            {"tensor": "packed.weight", "index": [0, 2], "op": "delta", "value": 0.75},
        ])
        proposal = self.run_ok("propose", self.source, *pin, req)
        self.assertEqual(proposal["edits"][0]["after"], 3)
        dest = self.root / "out.hgn"
        self.run_ok("apply", self.source, self.document("proposal.json", proposal), dest)
        expected = bytearray(original)
        expected[offset + 2] = 3
        checksum = 104 + 160 + 152
        struct.pack_into("<I", expected, checksum, reference_xor(expected[offset:offset + 68]))
        self.assertEqual(dest.read_bytes(), expected)
        self.assertEqual(self.source.read_bytes(), original)
        expected[offset + 64:offset + 66] = b"\0\0"
        struct.pack_into("<I", expected, checksum, reference_xor(expected[offset:offset + 68]))
        self.source.write_bytes(expected)
        pin = ("--source-sha256", self.sha())
        tile = self.run_ok("tile", self.source, *pin, "--tensor", "packed.weight",
                           "--origin", "[0,0]", "--shape", "[1,1]")
        self.assertEqual(tile["packed_steps"], [[{"lower": None, "upper": None}]])
        self.assertIn("positive scale", self.cli("propose", self.source, *pin, req).stderr)


    @unittest.skipUnless(FIXTURE, "set BREWERY_TEST_DENSE_GGUF to the real dense-f32.gguf fixture")
    def test_real_dense_fixture_changes_only_two_selected_scalars(self):
        self.source = Path(FIXTURE)
        before = self.source.read_bytes()
        request = self.document("request.json", [
            {"tensor": NAME, "index": [0, 3], "op": "set", "value": -0.125},
            {"tensor": "output.weight", "index": [2, 9], "op": "set", "value": 0.125},
        ])
        proposal = self.run_ok("propose", self.source, "--source-sha256", self.sha(), request)
        dest = self.root / "edited.gguf"
        self.run_ok("apply", self.source, self.document("proposal.json", proposal), dest)
        tensor_by_name = {t.name: t for t in gguf.GGUFReader(self.source).tensors}
        expected = bytearray(before)
        for name, flat, value in [(NAME, 3, -0.125), ("output.weight", 2 * 256 + 9, 0.125)]:
            struct.pack_into("<f", expected, tensor_by_name[name].data_offset + flat * 4, value)
        self.assertEqual(dest.read_bytes(), expected)
        self.assertEqual(self.source.read_bytes(), before)

    def test_proposal_reports_actual_f16_and_bf16_rounding(self):
        for kind, initial, expected, encoded in [
            ("F16", np.array([[2.]], dtype=np.float16), 1.0009765625, b"\x01\x3c"),
            ("BF16", gguf.quantize(np.array([[2.]], dtype=np.float32), gguf.GGMLQuantizationType.BF16),
             1.0078125, b"\x81\x3f"),
        ]:
            with self.subTest(kind=kind):
                self.source = self.root / f"{kind}.gguf"
                raw_dtype = gguf.GGMLQuantizationType.BF16 if kind == "BF16" else None
                self.tensor_fixture(self.source, initial, raw_dtype)
                # One binary64 step above the relevant midpoint. BF16 must
                # not lose it in an intermediate F32 cast.
                midpoint = 1.00048828125 if kind == "F16" else 1.00390625
                request = self.document("request.json", [
                    {"tensor": NAME, "index": [0, 0], "op": "set", "value": midpoint + 2**-52},
                ])
                proposal = self.run_ok("propose", self.source, "--source-sha256", self.sha(), request)
                self.assertEqual(proposal["edits"][0]["dtype"], kind)
                self.assertEqual(proposal["edits"][0]["after"], expected)
                dest = self.root / f"{kind}-edited.gguf"
                self.run_ok("apply", self.source, self.document("proposal.json", proposal), dest)
                self.assertEqual(gguf.GGUFReader(dest).tensors[0].data.tobytes(), encoded)

    def test_invalid_edits_and_proposal_are_rejected(self):
        edit = {"tensor": NAME, "index": [0, 0], "op": "set", "value": -0.125}
        for message, edits in [
            ("index rank", [{**edit, "index": [0]}]),
            ("out of bounds", [{**edit, "index": [0, -1]}]),
            ("out of bounds", [{**edit, "index": [True, 0]}]),
            ("out of bounds", [{**edit, "index": [0, 3]}]),
            ("duplicate", [edit, edit]),
            ("no-op", [{**edit, "op": "delta", "value": 1e-30}]),
            ("finite", [{**edit, "value": float("nan")}]),
            ("finite", [{**edit, "value": float("inf")}]),
            ("finite", [{**edit, "value": 1e300}]),
        ]:
            with self.subTest(edits=edits):
                result = self.cli("propose", self.source, "--source-sha256", self.sha(),
                                  self.document("request.json", edits))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
        proposal = self.run_ok("propose", self.source, "--source-sha256", self.sha(),
                               self.document("request.json", [edit]))
        dest = self.root / "result.gguf"
        result = self.cli("apply", self.source, self.document(
            "altered.json", {**proposal, "source_sha256": "0" * 64}), dest)
        self.assertIn("source SHA-256 mismatch", result.stderr)
        proposal["edits"][0]["after"] = 0.75
        result = self.cli("apply", self.source, self.document("altered.json", proposal), dest)
        self.assertIn("proposal before/after", result.stderr)
        self.assertFalse(dest.exists())

    def test_quantized_and_nonfinite_sources_are_rejected(self):
        for kind, data, message in [
            (None, np.array([[float("nan")]], dtype=np.float32), "non-finite"),
            (gguf.GGMLQuantizationType.IQ1_S, np.zeros((1, 50), dtype=np.uint8), "quantized"),
        ]:
            self.source = self.root / f"source-{message}.gguf"
            self.tensor_fixture(self.source, data, kind)
            result = self.cli("inspect", self.source, "--source-sha256", self.sha(), "--tensor", NAME,
                              "--index", "[0,0]")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)

    def test_known_mixed_store_is_edit_and_export_ineligible(self):
        writer = gguf.GGUFWriter(self.source, "llama")
        writer.add_tensor(NAME, np.array([[1.]], dtype=np.float32))
        writer.add_tensor("other.weight", np.zeros((1, 36), dtype=np.uint8),
                          raw_dtype=gguf.GGMLQuantizationType.Q8_1)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        pin = ("--source-sha256", self.sha())
        inventory = self.run_ok("inventory", self.source, *pin, "--offset", "0", "--limit", "10")
        self.assertEqual(inventory["tensors"][1]["dtype"], "Q8_1")
        self.assertFalse(inventory["tensors"][1]["editable"])
        self.assertIn("read-only", inventory["tensors"][1]["reason"])
        req = self.document("request.json", [
            {"tensor": "other.weight", "index": [0, 0], "op": "set", "value": 2.0},
        ])
        self.assertIn("read-only", self.cli("propose", self.source, *pin, req).stderr)
        out = self.root / "ineligible.hgn"
        export = subprocess.run([sys.executable, "-B", CLI.parent / "exact-export",
                                 "--encoded-sha256", self.sha(), "--to", "hgn", self.source, out],
                                text=True, capture_output=True, timeout=10)
        self.assertNotEqual(export.returncode, 0)
        self.assertIn("unsupported GGML type 9", export.stderr)
        self.assertFalse(out.exists())
        req = self.document("request.json", [
            {"tensor": NAME, "index": [0, 0], "op": "set", "value": 2.0},
        ])
        proposal = self.run_ok("propose", self.source, *pin, req)
        out = self.root / "edited.gguf"
        self.run_ok("apply", self.source, self.document("proposal.json", proposal), out)
        after = gguf.GGUFReader(out).tensors
        self.assertEqual(after[0].data.tolist(), [[2.]])
        self.assertEqual(after[1].data.tobytes(), bytes(36))

    def test_q4_shared_byte_applies_both_selected_nibbles(self):
        kind = gguf.GGMLQuantizationType.Q4_0
        raw = struct.pack("<e", 0.5) + bytes([0x88]) * 16
        self.tensor_fixture(self.source, np.frombuffer(raw, dtype=np.uint8).reshape(1, -1), kind)
        req = self.document("request.json", [
            {"tensor": NAME, "index": [0, 0], "op": "set", "value": 0.5},
            {"tensor": NAME, "index": [0, 16], "op": "delta", "value": -0.5},
        ])
        pin = ("--source-sha256", self.sha())
        tile = self.run_ok("tile", self.source, *pin, "--tensor", NAME,
                           "--origin", "[0,0]", "--shape", "[1,1]")
        self.assertEqual(tile["packed_steps"], [[{"lower": -0.5, "upper": 0.5}]])
        proposal = self.run_ok("propose", self.source, *pin, req)
        out = self.root / "out.gguf"
        response = self.run_ok("apply", self.source, self.document("approval.json", proposal), out)
        self.assertEqual(response["edit_count"], 2)
        t, = gguf.GGUFReader(out).tensors
        self.assertEqual(t.data.tobytes(), raw[:2] + b"\x79" + raw[3:])
        req = self.document("request.json", [
            {"tensor": NAME, "index": [0, 0], "op": "set", "value": 5.0},
        ])
        self.assertIn("range", self.cli("propose", self.source, *pin, req).stderr)
        blocked = subprocess.run(
            [sys.executable, "-B", CLI, "apply", self.source, self.root / "approval.json",
             self.root / "blocked.gguf"], env={**os.environ, "BREWERY_WEIGHT_PACKED": "0"},
            text=True, capture_output=True, timeout=10,
        )
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("disabled for this worker", blocked.stderr)
        self.assertFalse((self.root / "blocked.gguf").exists())

    def test_impossible_tensor_extent_is_rejected_even_with_matching_hash(self):
        info = gguf.GGUFReader(self.source).tensors[0]
        offset = info.field.offset + 8 + len(info.name.encode("utf-8")) + 4
        broken = bytearray(self.source.read_bytes())
        struct.pack_into("<Q", broken, offset, 1024)
        self.source = self.root / "bad-shape.gguf"
        self.source.write_bytes(broken)
        result = self.cli("inspect", self.source, "--source-sha256", self.sha(),
                          "--tensor", NAME, "--index", "[0,0]")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid GGUF tensor extents", result.stderr)


def reference_xor(data):
    result = 0
    for i in range(0, len(data), 4):
        result ^= int.from_bytes(data[i:i + 4], "little")
    return result


def native_fixture(path):
    entries = [
        ("layer.weight", 2, 0, (2, 2), struct.pack("<4e", 1, 2, 3, 4)),
        ("packed.weight", 7, 0, (1, 64), bytes(range(64)) + struct.pack("<2e", 1., 0.)),
        ("rotated.weight", 16, 0x1206, (128, 128), bytes(6144)),
        ("rotated.suh", 2, 0, (128,), struct.pack("<128e", *([1.] * 128))),
        ("rotated.svh", 2, 0, (128,), struct.pack("<128e", *([1.] * 128))),
    ]
    table, payload = bytearray(), bytearray()
    base = (104 + 160 * len(entries) + 63) // 64 * 64
    offsets = {}
    for name, store, variant, shape, data in entries:
        payload.extend(bytes(-len(payload) % 64))
        offset = base + len(payload)
        offsets[name] = offset
        table.extend(struct.pack("<96sII4qQQII", name.encode(), store, len(shape),
                                 *(shape + (0,) * (4 - len(shape))), offset, len(data),
                                 reference_xor(data), variant))
        payload.extend(data)
    header = struct.pack("<4sIQQQQ64s", b"HGN1", 2, len(entries), 104,
                         base, base + len(payload), b"qwen3.8-flash-next-ht43")
    original = header + table + bytes(base - len(header) - len(table)) + payload
    path.write_bytes(original)
    return original, offsets


if __name__ == "__main__":
    unittest.main()
