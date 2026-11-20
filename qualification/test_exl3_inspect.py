"""Offline EXL3 inspector contract; fixtures are schemas, not working models."""

import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).resolve().parents[1] / "tools" / "exl3_inspect.py"
WIDTHS = {"I16": 2, "I32": 4, "F16": 2, "F32": 4, "BF16": 2}


def write_tensors(path, specs):
    header, scalars, offset = {}, [], 0
    for name, (dtype, shape, *value) in specs.items():
        size = WIDTHS[dtype] * math.prod(shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        if value:
            scalars.append((offset, struct.pack("<I", value[0])))
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(raw)) + raw)
        file.truncate(8 + len(raw) + offset)
        for start, contents in scalars:
            file.seek(8 + len(raw) + start)
            file.write(contents)


class InspectEXL3(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def inspect(self, path):
        return subprocess.run([sys.executable, "-B", str(CLI), str(path)],
                              text=True, capture_output=True, timeout=10)

    def success(self, path):
        result = self.inspect(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def failure(self, path, diagnostic):
        result = self.inspect(path)
        self.assertEqual(result.returncode, 2, result)
        self.assertEqual(result.stdout, "")
        self.assertIn(diagnostic, result.stderr)

    def test_default_codebook_with_asymmetric_padded_linear(self):
        # Header sizes from the published Llama-3.2-1B-Instruct lm_head at
        # 88d3885d2c12afe44827a61a8298aa5ab7d76eec. No weight data is copied.
        path = self.root / "weights.safetensors"
        write_tensors(path, {
            "lm_head.trellis": ("I16", [128, 8016, 96]),
            "lm_head.suh": ("F16", [2048]),
            "lm_head.svh": ("F16", [128256]),
        })
        report = self.success(path)
        self.assertEqual(report["representation"], "exl3")
        self.assertEqual(report["validation"], "metadata-only")
        self.assertEqual(report["marker_payload_bytes_read"], 0)
        layer, = report["modules"]
        self.assertEqual(layer["module"], "lm_head")
        self.assertEqual(layer["layout"], "linear")
        self.assertEqual(layer["packed_shape"], [128, 8016, 96])
        self.assertEqual(layer["padded_in_features"], 2048)
        self.assertEqual(layer["padded_out_features"], 128256)
        self.assertEqual(layer["bits_per_weight"], 6)
        self.assertEqual(layer["codebook"], "3inst")
        self.assertIsNone(layer["marker"])

    def test_indexed_shards_cross_check_mapping_and_sidecar(self):
        # The loader discovers a linear group by tensor names; each header
        # must agree with the official index before joining split companions.
        first = "model-00001-of-00002.safetensors"
        second = "model-00002-of-00002.safetensors"
        left = {"q.trellis": ("I16", [8, 16, 40]), "q.suh": ("F16", [128]),
                "vision.trellis": ("I16", [8, 8, 32]), "vision.suh": ("F16", [128]),
                "vision.svh": ("F16", [128])}
        right = {"q.sv": ("I16", [16]), "q.mul1": ("I32", [], 0x83DCD12D)}
        write_tensors(self.root / first, left)
        write_tensors(self.root / second, right)
        (self.root / "cal_trace.safetensors").write_text("not inspected")
        index = self.root / "model.safetensors.index.json"
        mapping = {**dict.fromkeys(left, first), **dict.fromkeys(right, second)}
        index.write_text(json.dumps({"metadata": {"total_size": 15140}, "weight_map": mapping}))
        sidecar = self.root / "quantization_config.json"
        record = {
            "quant_format": "exl3",
            "bits_per_weight": 2.5,
            "mul1_multiplier": 2212286765,
            "stored_tensors": {
                "q.trellis": {"dtype": "torch.int16", "shape": [8, 16, 40], "n_bytes": 10240},
                "q.suh": {"dtype": "torch.float16", "shape": [128], "n_bytes": 256},
                "q.sv": {"dtype": "torch.int16", "shape": [16], "n_bytes": 32},
                "q.mul1": {"dtype": "torch.int32", "shape": [], "n_bytes": 4},
            },
        }
        sidecar.write_text(json.dumps({"quant_method": "exl3", "bits": 2.2,
                                       "tensor_storage": {"q": record}}))
        report = self.success(self.root)
        self.assertEqual(report["members"], [first, second])
        self.assertEqual(report["uninspected_safetensors"], ["cal_trace.safetensors"])
        self.assertEqual(report["marker_payload_bytes_read"], 4)
        layer, vision = report["modules"]
        self.assertEqual(vision["module"], "vision")
        self.assertEqual(report["sidecar_linear_rows_checked"], 1)
        self.assertEqual(layer["bits_per_weight"], 2.5)
        self.assertEqual(layer["codebook"], "mul1")
        self.assertEqual(layer["marker"], {"tensor": "q.mul1", "multiplier": 2212286765})
        self.assertEqual(layer["rotations"]["output"]["packing"], "i16-sign-bits")
        record["mul1_multiplier"] = 3417055213
        sidecar.write_text(json.dumps({"quant_method": "exl3", "tensor_storage": {"q": record}}))
        self.failure(self.root, "sidecar codebook")
        record["mul1_multiplier"] = 2212286765
        record["stored_tensors"]["q.trellis"]["n_bytes"] = 10238
        sidecar.write_text(json.dumps({"quant_method": "exl3", "tensor_storage": {"q": record}}))
        self.failure(self.root, "sidecar descriptor disagrees")
        sidecar.unlink()
        mapping["q.svh"] = second
        index.write_text(json.dumps({"weight_map": mapping}))
        self.failure(self.root, "index does not match")
        mapping["q.svh"] = "../" + second
        index.write_text(json.dumps({"weight_map": mapping}))
        self.failure(self.root, "canonical flat SafeTensors shards")

    def test_token_embedding_uses_scaled_row_rings_not_linear_tiles(self):
        path = self.root / "model.safetensors"
        specs = {
            "model.embed_tokens.trellis": ("I16", [17, 99]),
            "model.embed_tokens.signs": ("F16", [3, 256]),
            "model.norm.weight": ("BF16", [640]),
        }
        write_tensors(path, specs)
        layer, = self.success(self.root)["modules"]
        self.assertEqual(layer["layout"], "token-embedding")
        self.assertEqual(layer["padded_embedding_features"], 768)
        self.assertEqual(layer["rows"], 17)
        self.assertEqual(layer["words_per_group"], 33)
        self.assertEqual(layer["bits_per_weight"], 2)
        self.assertEqual(layer["codebook"], "mul1")
        self.assertIsNone(layer["marker"])
        specs["model.embed_tokens.trellis"] = ("I16", [17, 123])
        write_tensors(path, specs)
        self.failure(self.root, "unsupported words per embedding group")

    def test_mcg_integer_with_packed_signs_and_bias(self):
        path = self.root / "model.safetensors"
        write_tensors(path, {
            "g.trellis": ("I16", [16, 8, 128]),
            "g.su": ("I16", [16]),
            "g.sv": ("I16", [8]),
            "g.bias": ("F32", [128]),
            "g.mcg": ("I32", [], 0xCBAC1FED),
        })
        layer, = self.success(path)["modules"]
        self.assertEqual(layer["bits_per_weight"], 8)
        self.assertEqual(layer["marker"], {"tensor": "g.mcg", "multiplier": 3417055213})
        self.assertEqual(layer["codebook"], "mcg")

    def test_rejects_unsupported_widths_and_conflicting_companions(self):
        specs = {"q.trellis": ("I16", [8, 16, 40]), "q.suh": ("F16", [128]),
                 "q.svh": ("F16", [256]), "q.mul1": ("I32", [], 0x83DCD12D)}
        path = self.root / "model.safetensors"
        for replace, remove, diagnostic in (
            ({"q.trellis": ("F16", [8, 16, 40])}, (), "rank-3 I16"),
            ({"q.trellis": ("I16", [9, 16, 40])}, (), "multiples of 128"),
            ({"q.trellis": ("I16", [8, 16, 72])}, (), "unsupported words per tile"),
            ({}, ("q.mul1",), "unsupported words per tile"),
            ({"q.mcg": ("I32", [], 0xCBAC1FED)}, ("q.mul1",), "unsupported words per tile"),
            ({"q.mcg": ("I32", [], 0xCBAC1FED)}, (), "ambiguous codebook"),
            ({"q.mul1": ("I32", [1], 0x83DCD12D)}, (), "scalar codebook marker"),
            ({"q.mul1": ("I32", [], 0xCBAC1FED)}, (), "incorrect mul1 multiplier"),
            ({"orphan.mcg": ("I32", [], 0xCBAC1FED)}, (), "orphan codebook marker"),
            ({"q.su": ("I16", [8])}, (), "ambiguous input rotation"),
            ({"q.svh": ("F16", [128])}, (), "invalid output rotation"),
            ({}, ("q.svh",), "missing output rotation"),
            ({"q.weight": ("F16", [256, 128])}, (), "conflicting .weight"),
            ({"q.bias": ("F16", [128])}, (), "padded output vector"),
        ):
            with self.subTest(replace=replace, remove=remove):
                case = {k: v for k, v in (specs | replace).items() if k not in remove}
                write_tensors(path, case)
                self.failure(path, diagnostic)

    def test_marker_is_checked_per_module_not_once_per_shard(self):
        path = self.root / "model.safetensors"
        write_tensors(path, {
            "a.trellis": ("I16", [8, 8, 32]), "a.suh": ("F16", [128]),
            "a.svh": ("F16", [128]), "a.mul1": ("I32", [], 0x83DCD12D),
            "b.trellis": ("I16", [8, 8, 32]), "b.suh": ("F16", [128]),
            "b.svh": ("F16", [128]), "b.mul1": ("I32", [], 0),
        })
        self.failure(path, "b.mul1: incorrect mul1 multiplier")

    def test_sidecar_label_does_not_identify_a_dense_checkpoint(self):
        path = self.root / "model.safetensors"
        write_tensors(path, {"lm_head.weight": ("F16", [128, 128])})
        (self.root / "quantization_config.json").write_text('{"quant_method":"exl3"}')
        self.failure(self.root, "no supported EXL3 trellis")
        path.write_bytes(struct.pack("<Q", 0xffff_ffff_ffff_ffff))
        self.failure(path, "header exceeds bounds")

    def test_does_not_follow_checkpoint_links_or_accept_pickle(self):
        pickle = self.root / "pytorch_model.bin"
        pickle.write_bytes(b"\x80\x04N.")
        self.failure(pickle, "regular .safetensors")
        link = self.root / "model.safetensors"
        link.symlink_to(pickle)
        self.failure(link, "regular .safetensors")
        self.failure(self.root, "model.safetensors")
