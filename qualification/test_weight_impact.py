import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import gguf
import numpy as np


class WeightImpact(unittest.TestCase):
    def test_shared_block_neighbors_reported_separately_from_unrelated_changes(self):
        cli = Path(__file__).resolve().parents[1] / "tools" / "weight_impact.py"
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/arboretum")) as tmp:
            root = Path(tmp)
            for name, scales in (("old", (1, 1)), ("new", (2, 3))):
                writer = gguf.GGUFWriter(root / name, "llama")
                # Two Q8_0 blocks: one FP16 scale and 32 signed codes per block.
                raw = np.frombuffer(b"".join(np.float16(s).tobytes() + bytes([1])*32
                                             for s in scales), dtype=np.uint8).reshape(1, 68)
                writer.add_tensor("w", raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
            p = subprocess.run([sys.executable, cli, root/"old", root/"new", "--tensor", "w",
                                "--index", "[0,0]"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            r = json.loads(p.stdout)
            self.assertEqual([r["selected_before"], r["selected_after"]], [1, 2])
            self.assertEqual(r["changed_neighbors"], 31)
            self.assertEqual(r["changed_values_outside_block"], 32)
