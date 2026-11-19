"""Fixed-scale patches checked against an independent, small C F32 decoder."""
import ctypes
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from weight_q8g64 import candidates, plan_patch, read_cell


# Decode a whole fixture matrix without importing any Python codec logic.
ORACLE = """
#include <stdint.h>
#include <math.h>
static float half(const uint8_t *p) {
    unsigned h = p[0] | ((unsigned)p[1] << 8), e = (h >> 10) & 31, f = h & 1023;
    float magnitude = e == 0 ? ldexpf(f, -24) : ldexpf(1024 + f, (int)e - 25);
    return h >> 15 ? -magnitude : magnitude;
}
void decode(const uint8_t *data, float *values) {
    for (int r = 0; r < 2; ++r) {
        const uint8_t *row = data + r * 136;
        for (int c = 0; c < 128; ++c) {
            const uint8_t *g = row + 128 + 4 * (c >> 6);
            values[r * 128 + c] = fmaf(row[c], half(g), half(g + 2));
        }
    }
}
"""


def fixture():
    return (bytes(range(128)) + struct.pack("<4e", 1, -32, 0.5, -2)
            + bytes(reversed(range(128))) + struct.pack("<4e", 0.25, 4, 0.125, -6))


class FixedScale(unittest.TestCase):
    def test_read_and_candidate_levels(self):
        data = bytearray(fixture())
        original = bytes(data)
        cell = read_cell(data, 2, 128, 0, 67)
        self.assertEqual((cell.offset, cell.code, cell.scale, cell.bias, cell.value),
                         (67, 67, 0.5, -2, 31.5))
        choices = candidates(data, 2, 128, 0, 67)
        self.assertEqual(len(choices), 255)
        self.assertEqual([choice.new_code for choice in choices], [i for i in range(256) if i != 67])
        self.assertEqual((choices[0].after, choices[-1].after), (-2, 125.5))
        self.assertTrue(all(choice.before == cell.value and choice.offset == cell.offset for choice in choices))
        self.assertEqual(bytes(data), original)
        struct.pack_into("<ee", data, 132, 0, -2)
        self.assertEqual(read_cell(data, 2, 128, 0, 67).value, -2)
        self.assertEqual(candidates(data, 2, 128, 0, 67), ())
        struct.pack_into("<ee", data, 132, 2**-24, 32768)
        self.assertEqual(candidates(data, 2, 128, 0, 67), ())

    def test_rejects_unrepresentable_noop_and_invalid_geometry(self):
        data = fixture()
        for args, error in [
            ((data, 2, 96, 0, 0, "set", 1), "columns"),
            ((data[:-1], 2, 128, 0, 0, "set", 1), "length"),
            ((data, 2, 128, 2, 0, "set", 1), "coordinate"),
            ((data, 2, 128, 0, 0, "set", float("nan")), "finite"),
            ((data, 2, 128, 0, 0, "set", -33), "range"),
            ((data, 2, 128, 0, 0, "delta", 0.25), "unchanged"),
        ]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                plan_patch(*args)
        for invalid in (0, -1, float("inf")):
            bad = bytearray(data)
            struct.pack_into("<e", bad, 128, invalid)
            with self.assertRaisesRegex(ValueError, "scale"):
                plan_patch(bad, 2, 128, 0, 0, "set", 1)

    def test_ties_even_and_actual_after(self):
        data = fixture()
        result = plan_patch(data, 2, 128, 0, 67, "set", 34.75)
        self.assertEqual((result.offset, result.old_code, result.new_code), (67, 67, 74))
        self.assertEqual((result.before, result.requested, result.after), (31.5, 34.75, 35.0))

    @unittest.skipUnless(os.getenv("BREWERY_Q8G64_C_ORACLE") == "1", "set BREWERY_Q8G64_C_ORACLE=1")
    def test_against_c_decoder_and_preserves_every_other_byte_and_value(self):
        with tempfile.TemporaryDirectory(prefix="q8g64-oracle-", dir="/arboretum") as directory:
            library = Path(directory) / "decoder.so"
            subprocess.run([
                "resource-queue", "run", "--resource", "build", "--flock", "build",
                "--holder", "brewery-q8g64-oracle", "--wait", "20", "--max-hold", "30",
                "--", "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared",
                "-fPIC", "-x", "c", "-", "-lm", "-o", str(library),
            ], input=ORACLE, text=True, check=True, timeout=40)
            decode = ctypes.CDLL(str(library)).decode
            decode.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
            decode.restype = None
            data = fixture()
            edited = bytearray(data)
            result = plan_patch(edited, 2, 128, 1, 67, "delta", 0.31)
            self.assertEqual(bytes(edited), data)  # proposal never mutates
            edited[result.offset] = result.new_code
            old, new = (ctypes.c_float * 256)(), (ctypes.c_float * 256)()
            decode(data, old)
            decode(bytes(edited), new)
            index = 128 + 67
            self.assertEqual(result.offset, 136 + 67)
            self.assertEqual((old[index], new[index]), (1.5, 1.75))
            self.assertEqual((result.before, result.after), (old[index], new[index]))
            self.assertEqual(bytes(old)[:4*index], bytes(new)[:4*index])
            self.assertEqual(bytes(old)[4*(index+1):], bytes(new)[4*(index+1):])
            self.assertEqual([i for i, (a, b) in enumerate(zip(data, edited)) if a != b],
                             [result.offset])
