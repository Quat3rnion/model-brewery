"""Offline adapter contract tests; the tiny fixtures use real safetensors bytes."""
import json
from pathlib import Path
import runpy
import struct
import subprocess
import tempfile
import unittest


CLI = Path(__file__).resolve().parents[1] / "tools/gguf-export-hf"


def fixture(root, arch, tensors, **config):
    (root / "config.json").write_text(json.dumps({"architectures": [arch], **config}))
    header = {}
    offset = 0
    for name, shape, size in tensors:
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


class HfGguf(unittest.TestCase):
    def test_qwen4exp_inventory_never_silently_drops_roles(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as tmp:
            root = Path(tmp)
            source = root / "model"
            source.mkdir()
            fixture(source, "Qwen4ExpForConditionalGeneration", [
                ("model.visual.a", [1], 2),
                ("mtp.a", [1], 2),
                ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", [1, 1], 2),
            ])
            run = subprocess.run([CLI, "inspect", "--source", source], capture_output=True, check=True)
            report = json.loads(run.stdout)
            self.assertEqual(report["required_roles"], ["engram", "mtp", "projector", "text"])
            self.assertEqual(report["source_tensors"], 3)
            self.assertIs(report["export_supported"], False)
            out = root / "export"
            run = subprocess.run([CLI, "export", "--source", source, "--output", out], capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn(b"sidecar coverage", run.stderr)
            self.assertFalse(out.exists())

    def test_qwen2_metadata_checks_every_tensor_and_rejects_remote_code(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as tmp:
            source = Path(tmp)
            model = [
                ("model.embed_tokens.weight", [16, 8]), ("model.norm.weight", [8]),
                ("lm_head.weight", [16, 8]),
            ]
            for suffix, dims in [
                ("input_layernorm.weight", [8]), ("post_attention_layernorm.weight", [8]),
                ("self_attn.o_proj.weight", [8, 8]), ("mlp.gate_proj.weight", [12, 8]),
                ("mlp.up_proj.weight", [12, 8]), ("mlp.down_proj.weight", [8, 12]),
            ]:
                model.append((f"model.layers.0.{suffix}", dims))
            for proj, width in (("q", 8), ("k", 4), ("v", 4)):
                model += [(f"model.layers.0.self_attn.{proj}_proj.weight", [width, 8]),
                          (f"model.layers.0.self_attn.{proj}_proj.bias", [width])]
            import math
            tensors = [(name, dims, 2 * math.prod(dims)) for name, dims in model]
            cfg = dict(model_type="qwen2", num_hidden_layers=1, hidden_size=8,
                       num_attention_heads=2, num_key_value_heads=1, intermediate_size=12, vocab_size=16)
            fixture(source, "Qwen2ForCausalLM", tensors, **cfg)
            (source / "tokenizer.json").write_text('{"model":{"type":"BPE","vocab":{"x":0}}}')
            tokenizer = source / "tokenizer_config.json"
            tokenizer.write_text('{"tokenizer_class":"Qwen2Tokenizer"}')
            argv = [CLI, "inspect", "--source", source]
            report = json.loads(subprocess.run(argv, capture_output=True, check=True).stdout)
            self.assertEqual(report["source_tensors"], 15)
            self.assertFalse(report["export_runtime_checked"])
            self.assertEqual(report["expected_gguf_tensors"]["blk.0.attn_k.weight"], [8, 4])
            self.assertEqual(report["expected_gguf_tensors"]["blk.0.ffn_down.weight"], [12, 8])
            tokenizer.write_text('{"tokenizer_class":"Qwen2Tokenizer","auto_map":{"AutoTokenizer":"evil.Tokenizer"}}')
            rejected = subprocess.run(argv, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(b"remote code metadata", rejected.stderr)
            tokenizer.write_text('{"tokenizer_class":"Qwen2Tokenizer"}')
            fixture(source, "Qwen2ForCausalLM", tensors[:-1], **cfg)
            rejected = subprocess.run(argv, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(b"missing or additional tensors", rejected.stderr)

    def test_output_validator_rejects_inventory_loss_and_payload_truncation(self):
        read = runpy.run_path(str(CLI))["gguf_inventory"]
        u32 = lambda n: struct.pack("<I", n)
        u64 = lambda n: struct.pack("<Q", n)

        def string(s):
            return u64(len(s)) + s.encode()

        with tempfile.TemporaryDirectory(dir="/arboretum") as tmp:
            part = b"GGUF" + u32(3) + u64(1) + u64(4)
            part += string("general.architecture") + u32(8) + string("qwen2")
            part += string("general.file_type") + u32(4) + u32(32)
            part += string("tokenizer.ggml.model") + u32(8) + string("gpt2")
            part += string("tokenizer.ggml.tokens") + u32(9) + u32(8) + u64(1) + string("x")
            part += string("token_embd.weight") + u32(2) + u64(4) + u64(2) + u32(30) + u64(0)
            full = part + bytes(-len(part) % 32 + 16)
            gguf = Path(tmp) / "model.gguf"
            gguf.write_bytes(full)
            expected = {"token_embd.weight": [4, 2]}
            self.assertEqual(read(gguf, "bf16", expected)["token_embd.weight"]["ggml_type"], 30)
            with self.assertRaisesRegex(ValueError, "tensor or metadata count"):
                read(gguf, "bf16", {**expected, "output_norm.weight": [4]})
            gguf.write_bytes(full[:-1])
            with self.assertRaisesRegex(ValueError, "payload size"):
                read(gguf, "bf16", expected)
