"""Tiny HGN fixture with zero layer writes and a known embedding row."""
import json
from pathlib import Path
import struct
import sys

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual,
)

sys.path.insert(0, str(Path(__file__).parent / "hgn_trace"))
from trace import read_config
from wsp.hgn import H


def write_hgn(path, weights):
    offset = data_start = 128 + 160 * len(weights)
    table = []
    payload = []
    for name, (store, variant, shape, data) in weights.items():
        e = H.Entry()
        e.name, e.store, e.variant, e.shape = name, store, variant, shape
        e.size, e.offset, e.checksum = len(data), offset, H.xor_fold(data)
        table.append(e.pack())
        padded = data + bytes(-len(data) % 64)
        payload.append(padded)
        offset += len(padded)
    header = struct.pack("<IIQQQQ", 0x314E4748, 2, len(table), 128, data_start, offset)
    path.write_bytes(header.ljust(0x28, b"\0") + b"fixture".ljust(88, b"\0")
                     + b"".join(table + payload))


def fixture(path):
    raw = {"model_type": "qwen4_exp_text", "num_hidden_layers": 2, "hidden_size": 128,
           "vocab_size": 4, "hc_count": 4, "hc_lowrank": 16,
           "linear_key_head_dim": 16, "linear_value_head_dim": 16,
           "linear_num_value_heads": 2, "linear_num_key_heads": 1,
           "moe_intermediate_size": 128, "shared_expert_intermediate_size": 128,
           "num_experts": 2, "num_experts_per_tok": 1,
           "layer_types": ["linear_attention", "linear_attention"], "ple_layer_ids": [],
           "head_dim": 16, "num_attention_heads": 2, "num_key_value_heads": 1}
    cfg = path / "config.json"
    cfg.write_text(json.dumps({"text_config": raw}))
    config = read_config(cfg)
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "a": 1, "b": 2, "c": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    weights = {
        "embed_tokens.weight": (5, 2, (4, 128), np.ones(16, "<f4").tobytes() + bytes(256)
                                + np.ones((4, 8), "<f2").tobytes()),
        "lm_head.weight": (0, 0, (4, 128), bytes(4 * 128 * 2)),
    }

    def zeros(prefix, state):
        for name, value in state.items():
            weights[prefix + name] = (99, 0, tuple(value.shape), bytes(value.numel() * 4))

    with torch.device("meta"):
        mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        zeros("hyper_connection_mixer.", mixer.state_dict())
        for i in range(2):
            layer = Qwen4ExpTextDecoderLayer(config, i)
            root = f"layers.{i}."
            experts = root + "mlp.experts."
            zeros(root, {n: v for n, v in layer.state_dict().items() if not n.startswith("mlp.experts.")})
            for name, outputs in (("gate_up_proj", 256), ("down_proj", 128)):
                shape = (2, outputs, 128)
                weights[experts + name + ".weight"] = (23, 128, shape, bytes(2 * outputs * 66))
                for suffix, width in ((".suh", 128), (".svh", outputs)):
                    weights[experts + name + suffix] = (99, 0, (width,), np.ones(width, "<f2").tobytes())
    write_hgn(path / "model.hgn", weights)
    write_hgn(path / "sidecar.hgn", {})
