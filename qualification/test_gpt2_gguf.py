"""CLI contracts for bounded GPT-2 SafeTensors admission."""
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


CLI = Path(__file__).resolve().parents[1] / "tools/gguf-export-hf"


class Gpt2Gguf(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir="/arboretum")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "weights.safetensors"
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.config = dict(architectures=["GPT2LMHeadModel"], model_type="gpt2",
                           n_embd=8, n_head=2, n_layer=1, n_ctx=16, n_positions=16, vocab_size=2,
                           activation_function="gelu_new", layer_norm_epsilon=1e-5, tie_word_embeddings=False)
        self.shapes = {
            "transformer.wte.weight": [2, 8], "transformer.wpe.weight": [16, 8],
            "transformer.ln_f.weight": [8], "transformer.ln_f.bias": [8], "lm_head.weight": [2, 8],
        }
        self.dtypes = {}
        for part, shape in (
            ("ln_1", [8]), ("ln_2", [8]), ("attn.c_attn", [8, 24]), ("attn.c_proj", [8, 8]),
            ("mlp.c_fc", [8, 32]), ("mlp.c_proj", [32, 8]),
        ):
            prefix = f"transformer.h.0.{part}."
            self.shapes[prefix + "weight"] = shape
            self.shapes[prefix + "bias"] = shape[-1:]
        tokenizer = {"model": {"type": "BPE", "vocab": {"x": 0, "y": 1}}, "pre_tokenizer": {"type": "ByteLevel"}}
        (self.assets / "tokenizer.json").write_text(json.dumps(tokenizer))
        (self.assets / "tokenizer_config.json").write_text('{"tokenizer_class":"GPT2Tokenizer"}')

    def cli(self, operation="inspect"):
        (self.assets / "config.json").write_text(json.dumps(self.config))
        header, offset = {}, 0
        for name, shape in self.shapes.items():
            dtype = self.dtypes.get(name, "F32")
            size = {"F32": 4, "F16": 2, "BF16": 2}[dtype] * math.prod(shape)
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            offset += size
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        self.source.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
        args = [CLI, operation, "--source", self.source, "--assets", self.assets]
        if operation == "export":
            args += ["--output", self.root / "output"]
        return subprocess.run(args, capture_output=True)

    def inspect(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        return json.loads(result.stdout)

    def test_explicit_untied_head_and_fused_conv1d_mapping(self):
        report = self.inspect()
        self.assertTrue(report["export_supported"])
        self.assertFalse(report["export_runtime_checked"])
        self.assertFalse(report["tensor_payloads_read"])
        dims = report["expected_gguf_tensors"]
        self.assertEqual(len(dims), 17)
        self.assertEqual(dims["output.weight"], [8, 2])
        self.assertEqual(dims["position_embd.weight"], [8, 16])
        self.assertEqual(dims["blk.0.attn_qkv.weight"], [8, 24])
        self.assertEqual(dims["blk.0.ffn_up.weight"], [8, 32])
        self.assertEqual(dims["blk.0.ffn_down.weight"], [32, 8])

    def test_standalone_receipt_reports_dtypes_before_conversion(self):
        self.dtypes.update({
            "transformer.wpe.weight": "BF16",
            "transformer.h.0.attn.c_attn.weight": "F16",
            "transformer.h.0.attn.c_attn.bias": "BF16",
        })
        report = self.inspect()
        self.assertFalse(report["tensor_payloads_read"])
        dtypes = report["source_dtypes"]
        self.assertEqual(dtypes.keys(), report["expected_gguf_tensors"].keys())
        self.assertEqual(dtypes["position_embd.weight"], "BF16")
        self.assertEqual(dtypes["blk.0.attn_qkv.weight"], "F16")
        self.assertEqual(dtypes["blk.0.attn_qkv.bias"], "BF16")
        self.assertEqual(dtypes["output.weight"], "F32")

    def test_no_bare_or_undeclared_head_inference(self):
        del self.shapes["lm_head.weight"]
        del self.config["architectures"]
        report = self.inspect()
        self.assertEqual(report["source_tensors"], 16)
        self.assertIsNone(report["architecture"])
        self.assertFalse(report["export_supported"])
        result = self.cli("export")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"explicit GPT2LMHeadModel", result.stderr)
        self.assertFalse((self.root / "output").exists())
        self.config["architectures"] = ["GPT2Model"]
        self.assertFalse(self.inspect()["export_supported"])

    def test_head_tying_is_explicit_and_additional_components_are_rejected(self):
        self.config["tie_word_embeddings"] = True
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"ambiguous explicit lm_head", result.stderr)
        del self.shapes["lm_head.weight"]
        report = self.inspect()
        self.assertTrue(report["export_supported"])
        self.assertNotIn("output.weight", report["expected_gguf_tensors"])
        self.config["tie_word_embeddings"] = False
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"missing GPT-2 tensor: lm_head.weight", result.stderr)
        self.config["tie_word_embeddings"] = True
        self.shapes["score.weight"] = [2, 8]
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"unknown GPT-2 tensor: score.weight", result.stderr)

    def test_rejects_nonstandard_runtime_semantics_and_wrong_projection_shape(self):
        self.config["n_inner"] = 12
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"n_inner", result.stderr)
        del self.config["n_inner"]
        self.shapes["transformer.h.0.attn.c_attn.weight"] = [24, 8]
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"transformer.h.0.attn.c_attn.weight", result.stderr)


if __name__ == "__main__":
    unittest.main()
