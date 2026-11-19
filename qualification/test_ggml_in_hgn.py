"""Run with the pinned llama.cpp gguf-py on PYTHONPATH (see tool help)."""

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

TOOL = Path(__file__).resolve().parents[1] / "tools/cross_container/ggml_in_hgn.py"
FIXTURE = os.environ.get("BREWERY_TEST_IQ1_S_GGUF")
sys.path.insert(0, str(TOOL.parent))


@unittest.skipUnless(FIXTURE, "set BREWERY_TEST_IQ1_S_GGUF to the real mixed IQ1_S fixture")
class ExactSlice(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="ggml-in-hgn-", dir="/arboretum")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.source = Path(FIXTURE)
        self.dest = self.root / "model.hgn"

    def cli(self, operation, source=None, dest=None):
        return subprocess.run(
            [sys.executable, "-B", str(TOOL), operation,
             str(self.source if source is None else source),
             str(self.dest if dest is None else dest)],
            capture_output=True, text=True, timeout=15,
        )

    def pack(self):
        result = self.cli("pack")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixed_encoded_fixture_retains_all_tensors_and_typed_metadata(self):
        self.pack()
        from ggml_in_hgn import Reader

        expected = gguf.GGUFReader(self.source)
        self.assertIn("IQ1_S", {t.tensor_type.name for t in expected.tensors})
        self.assertGreater(len({t.tensor_type.name for t in expected.tensors}), 2)
        reader = Reader(self.dest)
        self.assertEqual(list(reader.tensors), [t.name for t in expected.tensors])
        fields = list(expected.fields.values())[3:]
        self.assertEqual(list(reader.metadata), [f.name for f in fields])
        for field in fields:
            self.assertEqual(reader.metadata[field.name].raw,
                             b"".join(x.tobytes() for x in field.parts))
        for tensor in expected.tensors:
            copied = reader.tensors[tensor.name]
            self.assertEqual(copied.shape, tuple(reversed(tensor.shape.tolist())))
            self.assertEqual(copied.ggml_type, int(tensor.tensor_type))
            self.assertEqual(reader.encoded(tensor.name), tensor.data.tobytes())
            np.testing.assert_array_equal(reader.decode(tensor.name),
                                          gguf.dequantize(tensor.data, tensor.tensor_type))
        result = self.cli("verify")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["tensor_count"], len(expected.tensors))

    def test_verify_rejects_changed_source_even_outside_a_tensor(self):
        self.pack()
        changed = self.root / "changed.gguf"
        changed.write_bytes(self.source.read_bytes() + b"\0")
        result = self.cli("verify", source=changed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source identity mismatch", result.stderr)

    def test_reader_and_verify_reject_corruption_even_if_xor_is_unchanged(self):
        self.pack()
        from ggml_in_hgn import Reader

        reader = Reader(self.dest)
        name = next(t.name for t in reader.tensors.values() if t.ggml_type == 19)
        offset = reader.tensors[name].offset
        contents = bytearray(self.dest.read_bytes())
        contents[offset] ^= 1
        contents[offset + 4] ^= 1  # Same bit in two XOR words cancels.
        self.dest.write_bytes(contents)
        with self.assertRaisesRegex(ValueError, "payload integrity mismatch"):
            Reader(self.dest).decode(name)
        result = self.cli("verify")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("payload integrity mismatch", result.stderr)

    def test_verify_checks_exact_metadata_against_source_not_just_checksums(self):
        self.pack()
        contents = bytearray(self.dest.read_bytes())
        # First HGN entry is the metadata tensor. Change two bits in the
        # retained GGUF general.name without affecting its HGN XOR.
        offset, length = struct.unpack_from("<QQ", contents, 104 + 136)
        name = b"synthetic-dense-quantize-only"
        location = contents.index(name, offset, offset + length)
        contents[location] ^= 1
        contents[location + 4] ^= 1
        self.dest.write_bytes(contents)
        result = self.cli("verify")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("metadata or original tensor table changed", result.stderr)

    def test_pack_rejects_unsupported_layout_and_does_not_clobber(self):
        source = gguf.GGUFReader(self.source)
        contents = bytearray(self.source.read_bytes())
        tensor = source.tensors[0]
        type_offset = tensor.field.offset + 8 + len(tensor.name.encode()) + 4 + 8 * len(tensor.shape)
        unassigned_type = 255
        self.assertNotIn(unassigned_type, {int(kind) for kind in gguf.GGMLQuantizationType})
        struct.pack_into("<I", contents, type_offset, unassigned_type)
        changed = self.root / "unsupported.gguf"
        changed.write_bytes(contents)
        result = self.cli("pack", source=changed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"unsupported GGML type {unassigned_type}", result.stderr)
        self.assertFalse(self.dest.exists())
        self.pack()
        original = self.dest.read_bytes()
        self.assertNotEqual(self.cli("pack").returncode, 0)
        self.assertEqual(self.dest.read_bytes(), original)

    def test_unextended_reader_rejects_extension_before_model_processing(self):
        converter = os.environ.get("BREWERY_TEST_STOCK_HGN")
        if not converter:
            self.skipTest("set BREWERY_TEST_STOCK_HGN to a stock bf16-to-hgn executable")
        self.pack()
        result = subprocess.run(
            [converter, "inspect", "--source", str(self.dest), "--source-format", "hgn"],
            capture_output=True, text=True, timeout=15,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported HGN version 3", result.stderr)


if __name__ == "__main__":
    unittest.main()
