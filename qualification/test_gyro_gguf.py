"""Exercise the offline Gyro inspector through its CLI."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/gyro_gguf.py"
CODEBOOK = os.environ.get("GYRO_TEST_CODEBOOK")
WEIGHT = "blk.0.ffn_down_exps.weight"
# Block 3 from one tensor per type in Agention's Qwen3.8-Flash-Next-Gyro
# noPLE files (HF revision abef06b8e0d6d3dc49021460fe09bfbe12e513da).
# Digests are little-endian float32 from upstream python/tq_ref.py at
# 587a5269848eb07996c9c1433f9ad638604eb4d8, not from the tool under test.
VECTORS = {
    144: ("M", "1c220ad49b07ac2d9fb123198e86bdf801d31abf93783c0a146d9087c73006ce5653",
          "8be8364c76b72bf181fdfee677e149a2041ae8f3ad7ce258e78ec6f741260bec"),
    145: ("S", "6b23d91018efffcb05659dd4a691fd6147d6a9d852881a8bd11c",
          "5b5a59cf10a3bff97e1b776902b6163e5a1da99ec95de29a46b534f7b837fc88"),
    146: ("S", "b92387b6d863c82295efb6bf72860a8c1ce859826f54aef88814799ec084",
          "5c7b7fed852d52fee3a9098dd4f56931e9915f6ddde6c32b39518646a3f748d9"),
    147: ("L", "3d23b0711d1dd588851dac29d343b69e112805a55ff197f81ff8666dcfdae54be6ed53d4a2eb4a5df3d3",
          "d41aeedd020cf602ca16f3cdf63d1b5eb7edf1bc990a6c68f7beb219690b02aa"),
    148: ("L", "d3223bdc51154e2298e528367321b6eb99d7b2cde41bcf58e43b6a2dc28f34abe5736df74ef85e886f86de414375c608cda3",
          "25bf708f76bde86376f553b9c70487126a8f1a8c64433d893e42ef19313ccfa7"),
    149: ("L", "b922dcbd2be5e67ea403c0415335f46cab75a29362d600cdc1c473a1b9480d07de5b92473a3acb7b24edd9c4ff85bee2928163169d459858d7b7",
          "57a575a11a7d6aaa116fa8cdb2b37c12b5a32cebe82365fc7dd59a41533c4b90"),
}


def text(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def string(value):
    return struct.pack("<I", 8) + text(value)


def u32(value):
    return struct.pack("<II", 4, value)


def array(subtype, values):
    return struct.pack("<IIQ", 9, subtype, len(values)) + b"".join(
        text(v) if subtype == 8 else struct.pack("<i", v) for v in values
    )


def fixture(replacements=None, kind=145, profile="S", width=128, rows=1, payload=bytes(26)):
    fields = {
        "general.architecture": string("qwen4exp"),
        "general.type": string("model"),
        "general.quantized_by": string("Agention"),
        "agention.encoder": string(f"Agention rotor encoder (APR), Gyro-{profile} release 2026-10-01"),
        "agention.trellis.state_bits": u32(15),
        "prism.hadamard.version": u32(1),
        "prism.hadamard.block_size": u32(128),
        "prism.hadamard.transform": string("normalized-sylvester-walsh-hadamard"),
        "prism.hadamard.axis": string("input-last-dimension"),
        "prism.hadamard.sign_mode": string("explicit"),
        "prism.hadamard.sign_widths": array(5, [width]),
        "prism.hadamard.sign_values": array(5, [-1, 1] * (width // 2)),
        "prism.hadamard.weight_names": array(8, [WEIGHT]),
    }
    fields.update(replacements or {})
    fields = {key: value for key, value in fields.items() if value is not None}
    header = struct.pack("<4sIQQ", b"GGUF", 3, 1, len(fields))
    header += b"".join(text(key) + value for key, value in fields.items())
    header += text(WEIGHT) + struct.pack("<IQQIQ", 2, width, rows, kind, 0)
    return header + bytes(-len(header) % 32) + payload


class GyroQualification(unittest.TestCase):
    def inspect(self, raw, *args):
        with tempfile.TemporaryDirectory(dir="/arboretum") as directory:
            path = Path(directory) / "arbitrary-name.gguf"
            path.write_bytes(raw)
            run = subprocess.run(
                [sys.executable, str(TOOL), str(path), *args],
                capture_output=True, text=True, check=False, timeout=10,
            )
            self.assertEqual(path.read_bytes(), raw)
            return run

    def test_typed_marker_and_profile_remap_colliding_disk_type(self):
        run = self.inspect(fixture())
        self.assertEqual(run.returncode, 0, run.stderr)
        out = json.loads(run.stdout)
        self.assertEqual(out["family"], "agention-gyro-gguf")
        self.assertEqual(out["trellis_types"], [{"disk_id": 145, "ik_id": 162, "type": "TQK6", "count": 1}])
        self.assertEqual(out["evidence"], "format-identified")

    def test_collider_is_not_identified_by_disk_id_or_marker_alone(self):
        for replacements, missing in (
            ({"agention.trellis.state_bits": None}, "state_bits"),
            ({"agention.trellis.state_bits": u32(16)}, "state_bits"),
            ({"agention.trellis.state_bits": struct.pack("<Ii", 5, 15)}, "state_bits"),
            ({"general.quantized_by": string("ik_llama.cpp")}, "quantized_by"),
            ({"agention.encoder": None}, "agention.encoder"),
        ):
            with self.subTest(missing=missing, replacements=replacements):
                run = self.inspect(fixture(replacements))
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(missing, run.stderr)
                self.assertEqual(run.stdout, "")

    def test_rejects_malformed_ancillary_tables(self):
        for changes, expected in (
            ({"prism.hadamard.sign_widths": array(4, [128])}, "sign_widths"),
            ({"prism.hadamard.sign_widths": array(5, [256])}, "sign_values"),
            ({"prism.hadamard.sign_values": array(5, [1] * 127 + [0])}, "sign_values"),
            ({"prism.hadamard.sign_values": array(5, [1] * 129)}, "sign_values"),
            ({"prism.hadamard.weight_names": array(8, ["absent.weight"])}, "weight_names"),
            ({"prism.hadamard.weight_names": array(8, [])}, "weight_names"),
            ({"prism.hadamard.inverse_weight_names": array(8, [])}, "inverse_weight_names"),
        ):
            with self.subTest(expected=expected, changes=changes.keys()):
                run = self.inspect(fixture(changes))
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(expected, run.stderr)
                self.assertEqual(run.stdout, "")

    def test_rejects_truncated_tensor_payload(self):
        run = self.inspect(fixture()[:-1])
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("truncated", run.stderr)

    def test_pinned_native_collision_layouts_are_not_gyro(self):
        # (native ik disk ID, width, row bytes including row metadata), per
        # ggml.c type_traits and ggml-common.h block sizes at the reader pin.
        # IQ4_KS, IQ2_KS, IQ4_KSS, Q8_K16, Q8_K32, Q8_KR8 respectively.
        for kind, width, size in (
            (144, 256, 140), (145, 256, 72), (146, 256, 132),
            (147, 128, 148), (148, 256, 296), (149, 256, 296),
        ):
            ordinary = dict(kind=kind, profile=VECTORS[kind][0], width=width, payload=bytes(size))
            for fields, message in (
                ({"agention.trellis.state_bits": None}, "state_bits"),
                ({}, "extra bytes"),
            ):
                with self.subTest(kind=kind, fields=fields):
                    run = self.inspect(fixture(fields, **ordinary))
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn(message, run.stderr)

    def test_decode_rejects_unverified_codebook_or_out_of_bounds_block(self):
        with tempfile.NamedTemporaryFile(dir="/arboretum") as book:
            book.write(bytes(8192))
            book.flush()
            args = ("--tensor", WEIGHT, "--codebook", book.name)
            for block, error in (("0", "invalid Gyro codebook"), ("1", "outside Gyro tensor")):
                run = self.inspect(fixture(), *args, "--block", block)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(error, run.stderr)

    @unittest.skipUnless(CODEBOOK, "set GYRO_TEST_CODEBOOK to the pinned raw Agention codebook")
    def test_decode_matches_independent_reference_for_all_six_disk_ids(self):
        for kind, (profile, encoded, expected) in VECTORS.items():
            with self.subTest(kind=kind):
                block = bytes.fromhex(encoded)
                payload = bytes(3 * len(block)) + block
                run = self.inspect(
                    fixture(kind=kind, profile=profile, rows=4, payload=payload),
                    "--tensor", WEIGHT, "--block", "3", "--codebook", CODEBOOK,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                sample = json.loads(run.stdout)["sample"]
                self.assertEqual(sample["basis"], "stored-rotated")
                self.assertEqual(sample["packed_sha256"], hashlib.sha256(block).hexdigest())
                self.assertEqual(hashlib.sha256(struct.pack("<128f", *sample["values"])).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
