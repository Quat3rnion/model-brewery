"""Run native checks with BREWERY_TEST_GGML_ORACLE and BREWERY_TEST_GGML_CODES_DIR.

The directory contains exact-Q4_0.gguf and exact-Q8_0.gguf from the pinned
CPU llama-quantize catalog run. The oracle is qualification/ggml_hgn_oracle.c
linked against the same pinned GGML. Native checks also require pinned gguf-py.
"""

import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from weight_ggml_codes import BLOCK_CELLS, block_size, candidates, edit_cell, project_candidates, read_cell

FIXTURES = os.getenv("BREWERY_TEST_GGML_CODES_DIR")
ORACLE = os.getenv("BREWERY_TEST_GGML_ORACLE")
TENSOR = "blk.0.attn_q.weight"


class GgmlCodes(unittest.TestCase):
    def test_paired_q4_cells_compose_edits_in_the_same_byte(self):
        block = bytes.fromhex("003c" + "f0" * 16)
        self.assertEqual(block_size("Q4_0"), len(block))
        self.assertEqual(read_cell(block, "Q4_0", 0).value, -8.0)
        self.assertEqual(read_cell(block, "Q4_0", 16).value, 7.0)

        low = edit_cell(block, "Q4_0", 0, operation="delta", value=1.75)
        self.assertEqual(low.cell_after.value, -6.0)
        self.assertEqual(low.block, bytes.fromhex("003cf2" + "f0" * 15))
        high = edit_cell(low.block, "Q4_0", 16, operation="set", value=1.75)
        self.assertEqual(high.cell_after.value, 2.0)
        self.assertEqual(high.block, bytes.fromhex("003ca2" + "f0" * 15))
        self.assertEqual(low.byte_index, high.byte_index)
        self.assertEqual(read_cell(high.block, "Q4_0", 0), low.cell_after)

    def test_q8_has_signed_byte_endpoints_and_handles_negative_scale(self):
        block = bytes.fromhex("00b8817f" + "00" * 30)  # Scale -0.5.
        self.assertEqual(block_size("Q8_0"), len(block))
        self.assertEqual(read_cell(block, "Q8_0", 0).code, -127)
        self.assertEqual(read_cell(block, "Q8_0", 1).value, -63.5)
        edge = edit_cell(block, "Q8_0", 0, operation="delta", value=0.5)
        self.assertEqual(edge.cell_after.value, 64.0)
        self.assertEqual(edge.block, bytes.fromhex("00b8807f" + "00" * 30))
        for target in (-63.500001, 64.000001, math.nan, math.inf, -math.inf):
            with self.subTest(target=target), self.assertRaises(ValueError):
                edit_cell(block, "Q8_0", 0, operation="set", value=target)

    def test_noop_preserves_code_and_rejects_invalid_inputs(self):
        block = bytes.fromhex("0080" + "70" * 16)  # Negative zero.
        unchanged = edit_cell(block, "Q4_0", 16, operation="set", value=0)
        self.assertEqual(unchanged.block, block)
        self.assertEqual(unchanged.cell_after.code, -1)
        self.assertEqual(math.copysign(1, unchanged.cell_after.scale), -1)
        with self.assertRaisesRegex(ValueError, "representable range"):
            edit_cell(block, "Q4_0", 16, operation="delta", value=0.001)
        for prefix in ("007c", "00fc", "017e"):  # +inf, -inf, NaN.
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "nonfinite"):
                read_cell(bytes.fromhex(prefix) + block[2:], "Q4_0", 16)
        for candidate in (("Q4_K", block, 0), ("Q4_0", block[:-1], 0),
                          ("Q4_0", block + b"\0", 0), ("Q4_0", block, -1),
                          ("Q4_0", block, 32), ("Q4_0", block, True)):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                ggml_type, data, index = candidate
                read_cell(data, ggml_type, index)
        with self.assertRaisesRegex(ValueError, "operation"):
            edit_cell(block, "Q4_0", 16, operation="multiply", value=0)

    def test_candidate_projection_is_distinct_changed_and_bounded(self):
        block = bytes.fromhex("003c" + "c8" * 16)
        targets = (0, 0.25, 1, 1.4999, 1.5, -1.5, -8, -8.00001, 7.1, math.nan)
        projected = project_candidates(block, "Q4_0", 0, iter(targets))
        self.assertEqual([p.cell_after.value for p in projected], [1.0, 2.0, -2.0, -8.0])
        self.assertEqual([p.block[:3] for p in projected],
                         list(map(bytes.fromhex, ("003cc9", "003cca", "003cc6", "003cc0"))))
        self.assertEqual(candidates(bytes(18), "Q4_0", 0), ())
        self.assertEqual(len(candidates(block, "Q4_0", 0)), 15)
        q8 = bytes.fromhex("003c" + "00" * 32)
        levels = candidates(q8, "Q8_0", 31)
        self.assertEqual(len(levels), 255)
        self.assertEqual(levels[0].block, q8[:-1] + b"\x80")
        self.assertEqual(levels[-1].block, q8[:-1] + b"\x7f")
        with self.assertRaisesRegex(ValueError, "256"):
            project_candidates(block, "Q4_0", 0, (0 for _ in range(257)))


@unittest.skipUnless(FIXTURES and ORACLE, "set BREWERY_TEST_GGML_CODES_DIR and BREWERY_TEST_GGML_ORACLE")
class GgmlNativeCodes(unittest.TestCase):
    def native_decode(self, path):
        run = subprocess.run([str(Path(ORACLE).resolve(strict=True)), path],
                             check=True, capture_output=True, timeout=10)
        offset = 0
        tensors = {}
        while offset < len(run.stdout):
            header_end = run.stdout.index(b"\n", offset)
            type_id, _, count, name = run.stdout[offset:header_end].decode("ascii").split()
            offset = header_end + 1 + int(count) * 4
            self.assertLessEqual(offset, len(run.stdout))
            name = bytes.fromhex(name).decode()
            self.assertNotIn(name, tensors)
            tensors[name] = (int(type_id), run.stdout[header_end + 1:offset])
        return tensors

    def fixture(self, kind, type_id):
        from gguf import GGUFEndian, GGUFReader

        path = Path(FIXTURES) / f"exact-{kind}.gguf"
        # The bounded C oracle verifies limits and the original container first.
        decoded = self.native_decode(path)
        reader = GGUFReader(path)
        tensor = next(t for t in reader.tensors if t.name == TENSOR)
        self.assertEqual(reader.endianess, GGUFEndian.LITTLE)
        self.assertEqual(tensor.tensor_type, type_id)
        self.assertEqual(decoded[TENSOR][0], type_id)
        source = path.read_bytes()
        size = block_size(kind)
        actual = bytearray(tensor.n_elements * 4)
        for n in range(0, tensor.n_bytes, size):
            at = tensor.data_offset + n
            block = source[at:at + size]
            for index in range(BLOCK_CELLS):
                i = n // size * BLOCK_CELLS + index
                struct.pack_into("<f", actual, i * 4, read_cell(block, kind, index).value)
        self.assertTrue(actual == decoded[TENSOR][1], f"{kind} read differs from native GGML")
        return source, tensor.data_offset, decoded

    def verify_patch(self, source, original, block_at, block, byte_changes, cell_changes):
        patched = source[:block_at] + block + source[block_at + len(block):]
        # Exact changed byte offsets and byte values; includes both scale bytes,
        # other blocks and tensors, padding, and metadata in the comparison.
        self.assertEqual(len(source), len(patched))
        self.assertEqual({i: b for i, (a, b) in enumerate(zip(source, patched)) if a != b},
                         byte_changes)
        with tempfile.TemporaryDirectory(prefix="brewery-ggml-codes-", dir="/arboretum") as scratch:
            path = Path(scratch) / "edited.gguf"
            path.write_bytes(patched)
            actual = self.native_decode(path)
        type_id, native_values = original[TENSOR]
        expected = bytearray(native_values)
        for i, (before, after) in cell_changes.items():
            self.assertEqual(struct.unpack_from("<f", native_values, i * 4)[0], before)
            self.assertNotEqual(before, after)
            struct.pack_into("<f", expected, i * 4, after)
        self.assertTrue(actual[TENSOR] == (type_id, expected),
                        "native GGML decoded an unexpected value or changed a neighbor")
        self.assertTrue({k: v for k, v in actual.items() if k != TENSOR}
                        == {k: v for k, v in original.items() if k != TENSOR})

    def test_real_q4_negative_scale_and_shared_nibble_composition(self):
        kind = "Q4_0"
        source, at, decoded = self.fixture(kind, 2)
        at += 18  # Second block, whose scale is negative.
        block = source[at:at + 18]
        low = edit_cell(block, kind, 0, operation="delta", value=-0.2)
        [high] = project_candidates(low.block, kind, 16, (-0.6, -0.61, -99))
        self.assertEqual(low.cell_before.scale, -0.120361328125)
        self.assertEqual(low.cell_after.value, 0.72216796875)
        self.assertEqual(high.cell_after.value, -0.601806640625)
        self.verify_patch(source, decoded, at, high.block, {at + 2: 0xD2},
                          {32: (0.962890625, 0.72216796875),
                           48: (0.962890625, -0.601806640625)})

    def test_real_q8_signed_endpoint_and_projected_target(self):
        kind = "Q8_0"
        source, at, decoded = self.fixture(kind, 8)
        block = source[at:at + 34]
        first = edit_cell(block, kind, 0, operation="set", value=-0.999)
        [next_edit] = project_candidates(first.block, kind, 8, (0.499, 0.501, 100))
        self.assertEqual(first.cell_after.value, -1.0)
        self.assertEqual(next_edit.cell_after.value, 0.5)
        self.verify_patch(source, decoded, at, next_edit.block,
                          {at + 2: 0x80, at + 10: 0x40},
                          {0: (-0.9921875, -1.0), 8: (0.9921875, 0.5)})
