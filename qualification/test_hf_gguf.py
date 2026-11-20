"""Offline adapter contract tests; the tiny fixtures use real safetensors bytes."""
import json
import math
from pathlib import Path
import runpy
import struct
import subprocess
import tempfile
import unittest


CLI = Path(__file__).resolve().parents[1] / "tools/gguf-export-hf"


def shard(path, tensors):
    header = {}
    offset = 0
    for name, shape, size, *dtype in tensors:
        header[name] = {"dtype": dtype[0] if dtype else "BF16", "shape": shape,
                        "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


def fixture(root, arch, tensors, *, sharded=False, **config):
    (root / "config.json").write_text(json.dumps({"architectures": [arch], **config}))
    if not sharded:
        shard(root / "model.safetensors", tensors)
        return
    parts = {"model-00001-of-00002.safetensors": tensors[::2],
             "model-00002-of-00002.safetensors": tensors[1::2]}
    for name, entries in parts.items():
        shard(root / name, entries)
    index = {entry[0]: name for name, entries in parts.items() for entry in entries}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))


def flash_next():
    text = dict(model_type="qwen4_exp_text", hidden_size=8, vocab_size=16,
                hc_count=2, hc_lowrank=3, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                layer_types=["linear_attention", "full_attention"],
                num_experts=3, moe_intermediate_size=4, shared_expert_intermediate_size=5,
                indexer_n_heads=2, indexer_head_dim=2, indexer_kv_heads=1, indexer_budget=4,
                indexer_compress_ratio=2, num_experts_per_tok=2,
                linear_num_key_heads=2, linear_key_head_dim=2, linear_num_value_heads=4,
                linear_value_head_dim=2, linear_conv_kernel_dim=3,
                mtp_num_hidden_layers=1, mtp={"num_hidden_layers": 1, "layer_types": ["full_attention"]},
                eos_token_id=1, ple_layer_ids=[1], ngram_size=3, heads_per_ngram=2,
                ple_embed_dim=8, ple_conv_kernel_size=3, split_ngram_parts=2)
    vision = dict(depth=1, hidden_size=4, num_heads=2, out_hidden_size=8, intermediate_size=6,
                  patch_size=2, temporal_patch_size=2, in_channels=3, spatial_merge_size=2,
                  num_position_embeddings=4, deepstack_visual_indexes=[])
    lm = "model.language_model."
    mixer = "hyper_connection_mixer."
    mix = {"hc_norm.weight": [16], "input_mix_weight_down.weight": [3, 16],
           "input_mix_weight_up.weight": [16, 3]}
    shapes = {lm + "embed_tokens.weight": [16, 8], "lm_head.weight": [16, 8],
              "mtp.fc_embedding.weight": [8, 8], "mtp.fc_hidden.weight": [8, 8],
              "mtp.pre_fc_norm_embedding.weight": [8], "mtp.pre_fc_norm_hidden.weight": [16]}
    for prefix in (lm, "mtp."):
        shapes.update({prefix + mixer + k: v for k, v in mix.items()})
    for layer in (lm + "layers.0.", lm + "layers.1.", "mtp.layers.0."):
        for hc in ("attn_hyper_connection.", "mlp_hyper_connection."):
            shapes.update({layer + hc + k: v for k, v in mix.items()})
            shapes[layer + hc + "block_inject_weight.weight"] = [2, 16]
        for key, dims in {
            "experts.gate_up_proj": [3, 8, 8], "experts.down_proj": [3, 8, 4],
            "gate.weight": [3, 8], "shared_expert_gate.weight": [1, 8],
            "shared_expert.gate_proj.weight": [5, 8], "shared_expert.up_proj.weight": [5, 8],
            "shared_expert.down_proj.weight": [8, 5],
        }.items():
            shapes[layer + "mlp." + key] = dims
        if layer == lm + "layers.0.":
            prefix = layer + "linear_attn."
            attn = {"A_log": [4], "dt_bias": [4], "conv1d.weight": [16, 1, 3],
                    "in_proj_a.weight": [4, 8], "in_proj_b.weight": [4, 8], "in_proj_z.weight": [8, 8],
                    "in_proj_qkv.weight": [16, 8], "norm.weight": [2], "out_proj.weight": [8, 8]}
        else:
            prefix = layer + "self_attn."
            attn = {"q_proj.weight": [16, 8], "k_proj.weight": [4, 8], "v_proj.weight": [4, 8],
                    "o_proj.weight": [8, 8], "q_norm.weight": [4], "k_norm.weight": [4],
                    "indexer.index_qk_proj.weight": [6, 8], "indexer.q_layernorm.weight": [2],
                    "indexer.k_layernorm.weight": [2]}
        shapes.update({prefix + k: v for k, v in attn.items()})
    ple = lm + "layers.0.ple."
    for key, dims in {
        "key_proj.weight": [16, 8], "value_proj.weight": [8, 8], "norm_key.weight": [16],
        "norm_query.weight": [16], "norm_conv.weight": [16], "conv1d.weight": [16, 1, 3],
        "ple_embedding.ngram_embedding.shard_1.weight": [3, 2],
        "ple_embedding.ngram_embedding.shard_0.weight": [5, 2],
    }.items():
        shapes[ple + key] = dims
    vis = "model.visual."
    shapes[vis + "pos_embed.weight"] = [4, 4]
    shapes[vis + "patch_embed.proj.weight"] = [4, 3, 2, 2, 2]
    shapes[vis + "patch_embed.proj.bias"] = [4]
    for key, dims in {
        "blocks.0.norm1": [4], "blocks.0.norm2": [4], "blocks.0.attn.proj": [4, 4],
        "blocks.0.attn.qkv": [12, 4], "blocks.0.mlp.linear_fc1": [6, 4],
        "blocks.0.mlp.linear_fc2": [4, 6], "merger.norm": [4],
        "merger.linear_fc1": [16, 16], "merger.linear_fc2": [8, 16],
    }.items():
        shapes[vis + key + ".weight"] = dims
        shapes[vis + key + ".bias"] = dims[:1]
    tensors = [(k, v, 2 * math.prod(v)) for k, v in shapes.items()]
    tensors += [(ple + "ple_embedding." + k, [n], 8 * n, "I64") for k, n in
                (("layer_multipliers", 3), ("ngram_heads_offsets", 4), ("ngram_heads_vocab_sizes", 4))]
    return tensors, dict(model_type="qwen4_exp", text_config=text, vision_config=vision, image_token_id=2)


class HfGguf(unittest.TestCase):
    def test_qwen4exp_inventory_never_silently_drops_roles(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as tmp:
            root = Path(tmp)
            source = root / "model"
            source.mkdir()
            tensors, config = flash_next()
            fixture(source, "Qwen4ExpForConditionalGeneration", tensors, sharded=True, **config)
            run = subprocess.run([CLI, "inspect", "--source", source], capture_output=True, check=True)
            report = json.loads(run.stdout)
            self.assertEqual(report["required_roles"], ["engram", "mtp", "projector", "text"])
            self.assertEqual(report["source_tensors"], len(tensors))
            self.assertIs(report["tensor_payloads_read"], False)
            plan = report["source_coverage_plan"]
            self.assertEqual(set(plan["sources"]), {item[0] for item in tensors})
            text = plan["artifacts"]["model"]["tensors"]
            mmproj = plan["artifacts"]["mmproj"]
            self.assertEqual(mmproj["vision_projector_type"], "qwen3vl_merger")
            vision = mmproj["tensors"]
            self.assertEqual(text["blk.2.nextn.eh_proj.weight"], {
                "role": "mtp", "gguf_dimensions": [16, 8],
                "sources": ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"],
                "operations": [{"op": "concat", "axis": 1}],
            })
            self.assertEqual(text["blk.1.indexer.k_proj.weight"]["gguf_dimensions"], [8, 2])
            self.assertEqual(text["blk.0.ssm_conv1d.weight"]["gguf_dimensions"], [3, 16])
            self.assertEqual(vision["v.patch_embd.weight.1"]["operations"], [{"op": "select", "axis": 2, "index": 1}])
            self.assertEqual(vision["mm.2.weight"]["gguf_dimensions"], [16, 8])
            table = text["per_layer_token_embd.weight"]
            self.assertEqual(table["gguf_dimensions"], [2, 8])
            self.assertEqual([name.rsplit(".", 2)[-2] for name in table["sources"]], ["shard_0", "shard_1"])
            self.assertEqual([plan["sources"][name]["file"] for name in table["sources"]],
                             ["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"])
            metadata = plan["artifacts"]["model"]["metadata"]["qwen4exp.ple.layer_multipliers"]
            self.assertEqual(metadata["gguf_type"], "ARRAY<UINT64>")
            self.assertEqual(metadata["element_count"], 3)
            self.assertIs(metadata["values_checked"], False)
            self.assertEqual(plan["sources"][metadata["sources"][0]]["dtype"], "I64")
            self.assertIs(report["export_supported"], False)
            out = root / "export"
            run = subprocess.run([CLI, "export", "--source", source, "--output", out], capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn(b"sidecar coverage", run.stderr)
            self.assertFalse(out.exists())

    def test_qwen4exp_rejects_incomplete_or_ambiguous_coverage(self):
        tensors, cfg = flash_next()
        arch = "Qwen4ExpForConditionalGeneration"
        source = "model.language_model.layers.0.ple.ple_embedding."
        first = source + "ngram_embedding.shard_0.weight"
        last = source + "ngram_embedding.shard_1.weight"
        multiplier = source + "layer_multipliers"
        index = "model.safetensors.index.json"
        with tempfile.TemporaryDirectory(dir="/arboretum") as tmp:
            root = Path(tmp)
            argv = [CLI, "inspect", "--source", root]
            for name, replace, diagnostic in [
                ("mtp.fc_hidden.weight", [], "missing Qwen4Exp mtp tensor: mtp.fc_hidden.weight"),
                ("model.visual.merger.linear_fc2.weight", [], "missing Qwen4Exp projector tensor"),
                (first, [], "missing shard_0.weight"),
                (last, [(last.replace("shard_1", "shard_2"), [3, 2], 12)], "unexpected shard_2.weight"),
                (last, [(last, [3, 4], 24)], "expected rank 2 and row width 2"),
                (multiplier, [(multiplier, [3], 12, "F32")], "expected I64, got F32"),
                (multiplier, [(multiplier, [1, 3], 24, "I64")], "expected shape [3], got [1, 3]"),
                ("", [("model.language_model.layers.0.unmapped.weight", [1], 2)], "unknown Qwen4Exp source"),
            ]:
                with self.subTest(name=name, diagnostic=diagnostic):
                    fixture(root, arch, [e for e in tensors if e[0] != name] + replace, **cfg)
                    result = subprocess.run(argv, capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(diagnostic.encode(), result.stderr)
                    self.assertFalse(result.stdout)

            (root / "model.safetensors").unlink()
            fixture(root, arch, tensors, sharded=True, **cfg)
            declared = json.loads((root / index).read_text())
            declared["weight_map"][first], declared["weight_map"][last] = (
                declared["weight_map"][last], declared["weight_map"][first])
            (root / index).write_text(json.dumps(declared))
            result = subprocess.run(argv, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"index/header tensor inventory mismatch", result.stderr)

            fixture(root, arch, tensors, sharded=True, **cfg)
            shard(root / "model-00002-of-00002.safetensors", tensors[1::2] + [tensors[0]])
            result = subprocess.run(argv, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"duplicate tensor across shards", result.stderr)

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

            (source / "model.safetensors").unlink()
            source_types = {"model.norm.weight": "F32", "model.layers.0.self_attn.k_proj.weight": "F16"}
            mixed = []
            for name, dims in model:
                dtype = source_types.get(name, "BF16")
                size = (4 if dtype == "F32" else 2) * math.prod(dims)
                mixed.append((name, dims, size, dtype))
            fixture(source, "Qwen2ForCausalLM", mixed, sharded=True, **cfg)
            indexed = json.loads(subprocess.run(argv, capture_output=True, check=True).stdout)
            self.assertFalse(indexed["tensor_payloads_read"])
            self.assertEqual(indexed["expected_gguf_tensors"], report["expected_gguf_tensors"])
            dtypes = indexed["source_dtypes"]
            self.assertEqual(dtypes.keys(), indexed["expected_gguf_tensors"].keys())
            self.assertEqual(dtypes["output_norm.weight"], "F32")
            self.assertEqual(dtypes["blk.0.attn_k.weight"], "F16")
            self.assertEqual(dtypes["blk.0.attn_k.bias"], "BF16")

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
