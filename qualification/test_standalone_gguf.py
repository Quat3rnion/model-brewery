"""Public adapter contract for a file and separately approved metadata."""
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_hf_gguf import CLI, fixture


class StandaloneGguf(unittest.TestCase):
    def test_private_bf16_accepts_dense_llama_down_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            shapes = {"model.embed_tokens.weight": [16, 8], "model.norm.weight": [8]}
            shapes.update({f"model.layers.0.{name}.weight": shape for name, shape in (
                ("input_layernorm", [8]), ("post_attention_layernorm", [8]),
                ("self_attn.q_proj", [8, 8]), ("self_attn.k_proj", [4, 8]),
                ("self_attn.v_proj", [4, 8]), ("self_attn.o_proj", [8, 8]),
                ("mlp.up_proj", [16, 8]), ("mlp.gate_proj", [16, 8]), ("mlp.down_proj", [8, 16]),
            )})
            fixture(assets, "LlamaForCausalLM",
                    [(name, shape, 2 * math.prod(shape)) for name, shape in shapes.items()],
                    model_type="llama", num_hidden_layers=1, hidden_size=8, num_attention_heads=2,
                    num_key_value_heads=1, intermediate_size=16, vocab_size=16, tie_word_embeddings=True)
            (assets / "tokenizer.json").write_text('{"model":{"type":"BPE","vocab":{"a":0}}}')
            (assets / "tokenizer_config.json").write_text('{"tokenizer_class":"LlamaTokenizerFast"}')
            weights = root / "private.safetensors"
            (assets / "model.safetensors").rename(weights)
            argv = [str(CLI), "inspect", "--source", str(weights), "--assets", str(assets)]
            ordinary = subprocess.run(argv, capture_output=True, text=True, check=True)
            self.assertFalse(json.loads(ordinary.stdout)["export_supported"])
            private = subprocess.run([*argv, "--scratch-private"], capture_output=True, text=True, check=True)
            report = json.loads(private.stdout)
            self.assertTrue(report["export_supported"])
            self.assertEqual(report["expected_gguf_tensors"]["blk.0.ffn_down.weight"], [16, 8])

    def test_private_bf16_accepts_tied_qwen2_without_a_second_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            prefix = "model.layers.0."
            shapes = {"model.embed_tokens.weight":[16, 8], "model.norm.weight":[8],
                      prefix + "input_layernorm.weight":[8], prefix + "post_attention_layernorm.weight":[8],
                      prefix + "self_attn.o_proj.weight":[8, 8], prefix + "mlp.gate_proj.weight":[16, 8],
                      prefix + "mlp.up_proj.weight":[16, 8], prefix + "mlp.down_proj.weight":[8, 16]}
            for part, width in [("q", 8), ("k", 4), ("v", 4)]:
                shapes[prefix + f"self_attn.{part}_proj.weight"] = [width, 8]
                shapes[prefix + f"self_attn.{part}_proj.bias"] = [width]
            fixture(assets, "Qwen2ForCausalLM",
                    [(name, shape, 2 * math.prod(shape)) for name, shape in shapes.items()],
                    model_type="qwen2", num_hidden_layers=1, hidden_size=8, num_attention_heads=2,
                    num_key_value_heads=1, intermediate_size=16, vocab_size=16, tie_word_embeddings=True)
            (assets / "tokenizer.json").write_text('{"model":{"type":"BPE","vocab":{"a":0}}}')
            (assets / "tokenizer_config.json").write_text('{"tokenizer_class":"Qwen2Tokenizer"}')
            weights = root / "private.safetensors"
            (assets / "model.safetensors").rename(weights)
            argv = [str(CLI), "inspect", "--source", str(weights), "--assets", str(assets)]
            ordinary = subprocess.run(argv, capture_output=True, text=True)
            self.assertNotEqual(ordinary.returncode, 0, ordinary.stdout)
            private = subprocess.run([*argv, "--scratch-private"], capture_output=True, text=True)
            self.assertEqual(private.returncode, 0, private.stderr)
            report = json.loads(private.stdout)
            self.assertTrue(report["export_supported"])
            self.assertEqual(report["expected_gguf_tensors"]["token_embd.weight"], [8, 16])
            self.assertNotIn("output.weight", report["expected_gguf_tensors"])
            self.assertNotEqual(subprocess.run([*argv, "--scratch-private", "--outtype", "f16"],
                                              capture_output=True).returncode, 0)

    def test_only_the_selected_file_and_assets_are_inputs(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as tmp:
            base = Path(tmp)
            assets = base / "assets"
            assets.mkdir()
            # An unsupported architecture still has a truthful inventory.
            fixture(base, "OtherModel", [("dense.weight", [2, 2], 8)])
            (base / "config.json").rename(assets / "config.json")
            weights = base / "a-revision.safetensors"
            (base / "model.safetensors").rename(weights)
            (base / "config.json").write_text("must not read")
            (base / "modeling.py").touch()
            argv = [str(CLI), "inspect", "--source", str(weights)]
            self.assertIn("assets", subprocess.run(argv, capture_output=True, text=True).stderr)
            run = subprocess.run([*argv, "--assets", assets], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual(report["source_format"], "safetensors")
            self.assertEqual(report["source_tensors"], 1)
            self.assertFalse(report["export_supported"])
            (assets / "other.safetensors").write_bytes(weights.read_bytes())
            run = subprocess.run([*argv, "--assets", assets], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("metadata", run.stderr)
