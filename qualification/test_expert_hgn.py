"""Offline CLI checks with native HGN v2 containers."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

CLI = Path(__file__).resolve().parents[1] / "tools/expert_hgn.py"
HEADER = struct.Struct("<4sI4Q64s")
ENTRY = struct.Struct("<96sII4qQQII")
PREFIX = "layers.0.mlp."
ROUTER = PREFIX + "gate.weight"
UP = PREFIX + "experts.gate_up_proj.weight"
DOWN = PREFIX + "experts.down_proj.weight"
OTHER = "layers.1.input_layernorm.weight"
ROTATION = {
    PREFIX + "experts.su": 256,
    PREFIX + "experts.gate_up_proj.svh": 256,
    PREFIX + "experts.down_proj.suh": 128,
    PREFIX + "experts.down_proj.svh": 256,
}


def checksum(data):
    result = bytearray(4)
    for i, value in enumerate(data):
        result[i % 4] ^= value
    return int.from_bytes(result, "little")


def native_fixture(path, entries):
    """Independently serialize the stock bf16-to-hgn directory and payloads."""
    table_end = 104 + 160 * len(entries)
    data = bytearray(table_end)
    base = (table_end + 63) // 64 * 64
    for index, (name, (store, variant, dims, raw)) in enumerate(entries.items()):
        data.extend(b"\0" * (-len(data) % 64))
        ENTRY.pack_into(data, 104 + 160 * index, name.encode(), store, len(dims),
                        *dims, *([0] * (4 - len(dims))), len(data), len(raw),
                        checksum(raw), variant)
        data.extend(raw)
    HEADER.pack_into(data, 0, b"HGN1", 2, len(entries), 104, base, len(data), b"native-moe-fixture")
    path.write_bytes(data)


def read_native(path):
    data = path.read_bytes()
    magic, version, count, table, base, length, identity = HEADER.unpack_from(data)
    assert (magic, version, length, table) == (b"HGN1", 2, len(data), 104)
    assert base == ((104 + 160 * count + 63) // 64) * 64
    end = 104 + 160 * count
    tensors = {}
    for index in range(count):
        name, store, rank, *fields = ENTRY.unpack_from(data, table + 160 * index)
        dims = tuple(fields[:rank])
        offset, size, check, variant = fields[4:]
        assert offset == ((end + 63) // 64) * 64 and not any(data[end:offset])
        end = offset + size
        raw = data[offset:end]
        assert len(raw) == size and checksum(raw) == check and not any(fields[rank:4])
        tensors[name.rstrip(b"\0").decode()] = (store, variant, dims, raw)
    assert end == len(data)
    return tensors


def planes(values, rows, columns):
    return ([bytes([v]) * (rows * columns // 2) for v in values],
            [struct.pack("<e", v / 32) * (rows * columns // 128) for v in values])


class ExpertHgnTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.source = Path(temp.name, "input.hgn")
        self.mapping = Path(temp.name, "review.json")
        self.output = Path(temp.name, "new.hgn")
        self.router = [struct.pack("<e", v) * 256 for v in (1, 2, 3)]
        self.up_codes, self.up_scales = planes((0x89, 0x9A, 0xAB), 256, 256)
        self.down_codes, self.down_scales = planes((0x98, 0xA9, 0xBA), 256, 128)
        self.entries = {
            ROUTER: (2, 0, (3, 256), b"".join(self.router)),
            UP: (23, 128, (3, 256, 256), b"".join(self.up_codes + self.up_scales)),
            DOWN: (23, 128, (3, 256, 128), b"".join(self.down_codes + self.down_scales)),
            OTHER: (1, 0, (2,), struct.pack("<2f", 1.5, 0.5)),
        }
        for name, width in ROTATION.items():
            self.entries[name] = (2, 0, (width,), struct.pack("<2e", -1, 1) * (width // 2))

    def invoke(self, order, *, top_k=None, reviewed=None, source_sha=None, destination=None):
        before = self.source.read_bytes()
        payload = {"schema": "brewery.expert-map.v1",
                   "source_sha256": source_sha or hashlib.sha256(before).hexdigest(),
                   "layers": {"0": order}, "provenance": {"reviewed": "fixture"}}
        if top_k is not None:
            payload["top_k"] = top_k
        raw_map = (json.dumps(payload, indent=2) + "\n").encode()
        self.mapping.write_bytes(raw_map)
        out = subprocess.run(
            [sys.executable, CLI, self.source, self.mapping, destination or self.output,
             "--reviewed-sha256", reviewed or hashlib.sha256(raw_map).hexdigest()],
            capture_output=True, text=True,
        )
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(self.mapping.read_bytes(), raw_map)
        return out

    def test_reorder_moves_i4r_codes_scales_and_router_rows_together(self):
        native_fixture(self.source, self.entries)
        result = self.invoke([2, 0, 1])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_native(self.output), {
            **self.entries,
            ROUTER: (2, 0, (3, 256), self.router[2] + self.router[0] + self.router[1]),
            UP: (23, 128, (3, 256, 256),
                 b"".join([self.up_codes[2], self.up_codes[0], self.up_codes[1],
                           self.up_scales[2], self.up_scales[0], self.up_scales[1]])),
            DOWN: (23, 128, (3, 256, 128),
                   b"".join([self.down_codes[2], self.down_codes[0], self.down_codes[1],
                             self.down_scales[2], self.down_scales[0], self.down_scales[1]])),
        })

    def test_prune_repacks_q4cp_planes_without_changing_retained_bytes(self):
        codebook = struct.pack("<16f", *range(16))
        expert_codes = [bytes([v]) * 32 for v in (0x12, 0x34, 0x56)]
        expert_scales = [(struct.pack("<e", v) + bytes(14)) * 2 for v in (1, 2, 3)]
        router = [struct.pack("<e", v) * 32 for v in (4, 5, 6)]
        down = [struct.pack("<e", v) * 32 for v in (7, 8, 9)]
        affine = bytes(range(64)) + struct.pack("<2e", 1, 0)
        fixture = {
            ROUTER: (2, 0, (3, 32), b"".join(router)),
            UP: (5, 2, (3, 2, 32),
                 codebook + b"".join(expert_codes) + bytes(32) + b"".join(expert_scales)),
            DOWN: (2, 0, (3, 32, 1), b"".join(down)),
            "layers.1.self_attn.q_proj.weight": (7, 0, (1, 64), affine),
        }
        native_fixture(self.source, fixture)
        result = self.invoke([2, 0], top_k=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(self.output.stat().st_size, self.source.stat().st_size)
        self.assertEqual(read_native(self.output), {
            **fixture,
            ROUTER: (2, 0, (2, 32), router[2] + router[0]),
            UP: (5, 2, (2, 2, 32), codebook + expert_codes[2] + expert_codes[0]
                 + expert_scales[2] + expert_scales[0]),
            DOWN: (2, 0, (2, 32, 1), down[2] + down[0]),
        })

    def test_prune_cannot_drop_below_reviewed_top_k(self):
        native_fixture(self.source, self.entries)
        for top_k in (None, 2):
            with self.subTest(top_k=top_k):
                result = self.invoke([2], top_k=top_k)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("top_k", result.stderr)
                self.assertFalse(self.output.exists())

    def test_refuses_unknown_moe_sidecar_and_malformed_rotation_signs(self):
        for name, result, entry in (
            (PREFIX + "experts.selection_bias", "unproven MoE sidecar", (2, 0, (3,), b"\0" * 6)),
            ("router.expert_score_bias", "unproven routing sidecar", (2, 0, (3,), b"\0" * 6)),
            (PREFIX + "experts.su", "expected shared +/-1 signs", (2, 0, (256,), b"\0" * 512)),
        ):
            with self.subTest(name=name):
                native_fixture(self.source, {**self.entries, name: entry})
                out = self.invoke([2, 0, 1])
                self.assertNotEqual(out.returncode, 0)
                self.assertIn(result, out.stderr)
                self.assertFalse(self.output.exists())

    def test_affine_rows_keep_codes_scales_and_bias_together(self):
        def affine_rows(code_bytes, padding, rows_per_expert):
            return [b"".join(bytes([v + i]) * code_bytes + struct.pack("<2e", v, i)
                             + bytes(padding) for i in range(rows_per_expert))
                    for v in (2, 3, 4)]

        router = [struct.pack("<e", v) * 64 for v in (3, 4, 5)]
        up_rows = affine_rows(64, 0, 128)
        down_rows = affine_rows(48, 12, 64)
        fixture = {
            ROUTER: (2, 0, (3, 64), b"".join(router)),
            UP: (7, 0, (3, 128, 64), b"".join(up_rows)),
            DOWN: (24, 64, (3, 64, 64), b"".join(down_rows)),
        }
        native_fixture(self.source, fixture)
        result = self.invoke([2, 0], top_k=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_native(self.output), {
            ROUTER: (2, 0, (2, 64), router[2] + router[0]),
            UP: (7, 0, (2, 128, 64), up_rows[2] + up_rows[0]),
            DOWN: (24, 64, (2, 64, 64), down_rows[2] + down_rows[0]),
        })

    def test_refuses_wrong_hashes_invalid_ids_and_existing_output(self):
        native_fixture(self.source, self.entries)
        for order, kwargs, message in (
            ([2, 0, 1], {"reviewed": "0" * 64}, "reviewed map SHA-256 mismatch"),
            ([2, 0, 1], {"source_sha": "0" * 64}, "source HGN SHA-256 mismatch"),
            ([2, 0, 0], {}, "duplicate expert IDs"),
            ([3, 0, 1], {}, "out of range"),
            ([2, False, 1], {}, "invalid or duplicate expert IDs"),
            ([2, 0, 1], {"destination": self.source}, "output already exists"),
        ):
            with self.subTest(order=order, message=message):
                result = self.invoke(order, **kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.output.exists())
        self.output.write_bytes(b"preserve preexisting file")
        result = self.invoke([2, 0, 1])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_bytes(), b"preserve preexisting file")


if __name__ == "__main__":
    unittest.main()
