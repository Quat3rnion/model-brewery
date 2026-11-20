"""Tiny indexed BF16 snapshot; no production weights or worker required."""
import hashlib
import importlib.util
import json
from math import prod
import os
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HALO = Path(os.environ.get("HALO_HY4_TOOLS", ROOT.parent / "halo-infer/gguf-tools"))
TARGET = "blk.1.ffn_down_exps.weight"
SOURCE = "model.layers.1.mlp.experts.down_proj"
ENGRAM = "model.layers.1.engram_layer.multi_head_embedding.embedding.weight"


def module():
    spec = importlib.util.spec_from_file_location("brewery_hy4_plan", ROOT / "tools/hy4_plan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Hy4Plan(unittest.TestCase):
    def test_metadata_only_override_and_engram_extent(self):
        plan = module()
        q, pin = plan.halo_modules(HALO)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shard = "model-00001-of-00001.safetensors"
            schema = q.fixture_schema("small")
            header, cursor = {}, 0
            for name, (dtype, shape, _) in sorted(schema.items()):
                end = cursor + prod(shape) * (2 if dtype == "BF16" else 4)
                header[name] = dict(dtype=dtype, shape=shape, data_offsets=[cursor, end])
                cursor = end
            head = json.dumps(header).encode()
            with (root / shard).open("wb") as f:
                f.write(struct.pack("<Q", len(head)) + head)
                # No actual weights: only a sparse apparent payload extent.
                f.truncate(8 + len(head) + cursor)
            source_metadata = {
                "model.safetensors.index.json": json.dumps(
                    {"weight_map": {name: shard for name in header}}
                ).encode(),
                "config.json": b'{"model_type":"hy_v4_flash","text_config":{"model_type":"hy_v4_flash"}}',
                "chat_template.jinja": b"fixture",
            }
            for name, data in source_metadata.items():
                (root / name).write_bytes(data)
            # Only the test substitutes fixture pins and geometry. The runner
            # itself cannot select a fixture or accept pins from the job.
            pins = {name: hashlib.sha256(data).hexdigest() for name, data in source_metadata.items()}
            report = plan.plan(root, q, pin, {TARGET: "Q8_0"}, pins=pins, schema=lambda _: schema)
            self.assertEqual(report["tensor_count"], 6)
            self.assertEqual(report["override"]["source"], SOURCE)
            self.assertEqual(report["override"]["shape"], [256, 256, 2])
            self.assertEqual(report["override"]["qtype_before"], 12)
            self.assertEqual(report["override"]["qtype"], 8)
            self.assertEqual(report["override"]["bytes"], 139264)
            self.assertEqual(report["resident_delta_bytes"], 65536)
            self.assertEqual(report["engram"]["file_offset"] % 65536, 0)
            self.assertEqual(report["engram"]["disk_bytes"], 576)
            self.assertEqual(report["source_metadata_read_bytes"], sum(map(len, source_metadata.values())) + 16 + len(head))
            table, = report["engram"]["tables"]
            self.assertEqual((table["source"], table["rows"], table["qtype"]), (ENGRAM, 3, 30))
            self.assertEqual(report["file_bytes"], table["file_offset"] + 576)
            self.assertFalse(report["payload_verified"])
            self.assertFalse(report["runtime_ready"])
            self.assertEqual({p.name for p in root.iterdir()}, {shard, *source_metadata})
            with self.assertRaisesRegex(ValueError, "exact"):
                plan.plan(root, q, pin, {TARGET.replace("down", "up"): "Q8_0"}, pins=pins, schema=lambda _: schema)
            with (root / shard).open("r+b") as f:
                f.write(struct.pack("<Q", (1 << 20) + 1))
            with self.assertRaisesRegex(ValueError, "read budget"):
                plan.plan(root, q, pin, {TARGET: "Q8_0"}, pins=pins, schema=lambda _: schema)
            (root / "config.json").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "pinned"):
                plan.plan(root, q, pin, {TARGET: "Q8_0"}, pins=pins, schema=lambda _: schema)


if __name__ == "__main__":
    unittest.main()
