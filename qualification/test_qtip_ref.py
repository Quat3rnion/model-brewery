"""Public CLI checks; set QTIP_SOURCE to enable the pinned CPU integration."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROBE = Path(__file__).resolve().parents[1] / "tools/qtip_ref.py"
PIN = "e90c6688c8dfae326a3a81b5eb032db7c6680ec0"


def run_probe(source):
    return subprocess.run(
        [sys.executable, "-I", str(PROBE), "--source", str(source)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class QtipReferenceTests(unittest.TestCase):
    def test_modified_source_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            candidate = root / "lib/utils/kernel_check.py"
            candidate.parent.mkdir(parents=True)
            candidate.write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
            )
            result = run_probe(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("digest mismatch", result.stderr)
            self.assertFalse(marker.exists())

    @unittest.skipUnless(os.environ.get("QTIP_SOURCE"), "set QTIP_SOURCE for the pinned CPU run")
    def test_native_rows_round_trip_with_upstream_cpu_reader(self):
        result = run_probe(os.environ["QTIP_SOURCE"])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["upstream_commit"], PIN)
        self.assertFalse(report["gpu_kernel_executed"])
        self.assertEqual(report["calibration"]["samples"], 64)
        self.assertEqual(report["calibration"]["kind"], "synthetic_input_second_moment")
        registry = report["upstream_fused_matvec_registry"]
        self.assertEqual(registry["mode"], "quantlut_sym")
        self.assertEqual(registry["shape_order"], ["out_features", "batch", "in_features", "K"])
        for K in (2, 3, 4):
            self.assertIn([4096, 1, 4096, K], registry["registered_shapes"])
            self.assertNotIn([32, 1, 32, K], registry["registered_shapes"])
        seen = set()
        for row in report["rows"]:
            mode, L, K, V, Q = (row[key] for key in ("decode_mode", "L", "K", "V", "tlut_bits"))
            with self.subTest(mode=mode, L=L, K=K, V=V, Q=Q):
                seen.add((mode, L, K, V, Q))
                self.assertEqual(row["td_x"], 16)
                self.assertEqual(row["td_y"], 16)
                self.assertEqual(row["weight_shape"], [32, 32])
                self.assertEqual(row["trellis_bytes"], 128 * K)
                self.assertEqual(row["native_tensors"]["trellis"], {"dtype": "int16", "shape": [4, 16 * K]})
                expected_fields = {"trellis", "SU", "SV", "Wscale", "rcp", "tp_rank", "proxy_err"}
                if Q:
                    expected_fields.add("tlut")
                    self.assertEqual(row["native_tensors"]["tlut"]["shape"], [1 << Q, V])
                self.assertEqual(set(row["native_tensors"]), expected_fields)
                self.assertEqual(row["reconstruction_compare_dtype"], "float16")
                self.assertEqual(row["reconstruction_max_abs"], 0.0)
                self.assertLess(row["linear_max_abs"], 0.003)
                self.assertGreater(row["proxy_err"], 0.0)
                self.assertLess(row["proxy_err"], 1.0)
                self.assertEqual(
                    row["packing"], "cuda_swizzled" if mode == "quantlut_sym" and Q == 9 else "reference"
                )
                self.assertFalse(row["fixture_has_fused_matvec"])
        expected = {
            (mode, L, K, V, Q)
            for mode, L, V, Q in (
                ("lut", 10, 2, 10), ("1mad", 16, 1, 0), ("2mad", 16, 1, 0),
                ("3inst", 16, 1, 0), ("quantlut", 16, 1, 8), ("quantlut", 16, 2, 8),
                ("quantlut_sym", 16, 2, 8),
            )
            for K in range(1, 5)
        }
        expected.update(("quantlut_sym", 16, K, 2, 9) for K in (2, 3, 4))
        self.assertEqual(len(report["rows"]), len(expected))
        self.assertEqual(seen, expected)


if __name__ == "__main__":
    unittest.main()
