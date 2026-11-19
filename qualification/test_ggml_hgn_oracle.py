import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.getenv("BREWERY_TEST_IQ1_S_GGUF")
ORACLE = os.getenv("BREWERY_TEST_GGML_ORACLE")


@unittest.skipUnless(SOURCE and ORACLE, "set BREWERY_TEST_IQ1_S_GGUF and BREWERY_TEST_GGML_ORACLE")
class NativeOracle(unittest.TestCase):
    def test_all_fixture_values_and_refusal(self):
        with tempfile.TemporaryDirectory(prefix="ggml-hgn-oracle-", dir="/arboretum") as name:
            dest = Path(name) / "slice.hgn"
            subprocess.run([sys.executable, "-B", ROOT / "tools/cross_container/ggml_in_hgn.py",
                            "pack", SOURCE, dest], check=True, capture_output=True, timeout=15)
            command = [sys.executable, "-B", ROOT / "qualification/ggml_hgn_oracle.py",
                       "--native", ORACLE, SOURCE, dest]
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
            report = json.loads(result.stdout)
            self.assertTrue(report["equal"])
            self.assertGreater(len(report["tensors"]), 2)
            self.assertIn("iq1_s", {t["type"] for t in report["tensors"]})
            self.assertTrue(all(t["bit_mismatches"] == 0 for t in report["tensors"]))
            # Locate the known F32 tensor only to change one source value.
            # The comparison itself still uses C for all expected values.
            from gguf import GGUFReader
            norm = next(t for t in GGUFReader(SOURCE).tensors if t.name == "blk.0.attn_norm.weight")
            changed = Path(name) / "changed.gguf"
            payload = bytearray(Path(SOURCE).read_bytes())
            old, = struct.unpack_from("<I", payload, norm.data_offset)
            struct.pack_into("<I", payload, norm.data_offset, old ^ 1)
            changed.write_bytes(payload)
            command[-2] = changed
            mismatch = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(mismatch.returncode, 1, mismatch.stderr)
            rows = json.loads(mismatch.stdout)["tensors"]
            self.assertEqual(sum(row["bit_mismatches"] for row in rows), 1)
            self.assertEqual(next(row["first_mismatch"] for row in rows if row["bit_mismatches"]),
                             {"index": 0, "native_bits": f"{old ^ 1:08x}", "reader_bits": f"{old:08x}"})
            bad = Path(name) / "oversize.gguf"
            with bad.open("wb") as file:
                file.truncate((8 << 20) + 1)
            command[-2] = bad
            denied = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("8 MiB", denied.stderr)
