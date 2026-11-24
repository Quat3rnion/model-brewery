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
from wsp.hgn import C, H


def write_hgn(path, weights, *, canonical=False):
    header_size = 104 if canonical else 128
    offset = data_start = (header_size + 160 * len(weights) + 63) & ~63 if canonical else 128 + 160 * len(weights)
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
    header = struct.pack("<IIQQQQ", 0x314E4748, 2, len(table), header_size, data_start, offset)
    path.write_bytes((header + b"fixture".ljust(header_size - len(header), b"\0") + b"".join(table))
                     .ljust(data_start, b"\0") + b"".join(payload))


def fixture(path, *, embedding_rows=None, tokenizer=None, editable=False):
    vocab = len(embedding_rows) if embedding_rows is not None else 4
    raw = {"model_type": "qwen4_exp_text", "num_hidden_layers": 2, "hidden_size": 128,
           "vocab_size": vocab, "hc_count": 4, "hc_lowrank": 16,
           "linear_key_head_dim": 16, "linear_value_head_dim": 16,
           "linear_num_value_heads": 2, "linear_num_key_heads": 1,
           "moe_intermediate_size": 128, "shared_expert_intermediate_size": 128,
           "num_experts": 2, "num_experts_per_tok": 1,
           "layer_types": ["linear_attention", "linear_attention"], "ple_layer_ids": [],
           "head_dim": 16, "num_attention_heads": 2, "num_key_value_heads": 1}
    cfg = path / "config.json"
    cfg.write_text(json.dumps({"text_config": raw}))
    config = read_config(cfg)
    if tokenizer is None:
        tokenizer = Tokenizer(WordLevel({"<unk>": 0, "a": 1, "b": 2, "c": 3}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    if embedding_rows is None:
        codebook = np.ones(16, "<f4")
        packed = bytes(vocab * 64)
    else:
        codebook = np.array([0, 1, -1, 2, -2, 4, -4] + [0] * 9, "<f4")
        codes = {value: index for index, value in enumerate(codebook[:7])}
        # Put x and y in the first two coordinates. All other coordinates are zero.
        packed = b"".join(bytes([codes[x] | codes[y] << 4]) + bytes(63)
                          for x, y in embedding_rows)
    embedding = codebook.tobytes() + packed + np.ones((vocab, 8), "<f2").tobytes()
    weights = {
        "embed_tokens.weight": (5, 2, (vocab, 128), embedding),
        "lm_head.weight": (0, 0, (vocab, 128), bytes(vocab * 128 * 2)),
    }

    def zeros(prefix, state):
        for name, value in state.items():
            weights[prefix + name] = (1 if editable else 99, 0, tuple(value.shape), bytes(value.numel() * 4))

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
                    weights[experts + name + suffix] = (2 if editable else 99, 0, (width,), np.ones(width, "<f2").tobytes())
    if editable:
        root = "layers.0."
        shared = root + "mlp.shared_expert."

        def matrix(name, values, bf16=False):
            bits = np.asarray(values, dtype="<f4").view("<u4")
            data = (bits >> 16).astype("<u2").tobytes() if bf16 else bits.tobytes()
            weights[name] = (0 if bf16 else 1, 0, bits.shape, data)

        # All edited writers have a measurable e0 component. The shared
        # expert also drives the logit change; the other inputs stay zero.
        attention = np.eye(128, 32) / 8
        attention[0] /= 4
        matrix(root + "linear_attn.out_proj.weight", attention, bf16=True)
        matrix(shared + "down_proj.weight", np.eye(128) / 8, bf16=True)
        # In the rotated basis a column of constant values is a pure e0
        # contribution. The other columns sum to zero along that axis.
        expert = np.zeros((2, 128, 128), dtype=np.float32)
        for row in range(128):
            for shift in range(4):
                expert[:, row, (row + shift) % 128] = 1 if row % 2 else -1
        expert[:, :, 0] = 1 / 8
        codes, scales = C.i4r_encode_rot(expert)
        weights[root + "mlp.experts.down_proj.weight"] = (
            23, 128, expert.shape, codes.tobytes() + scales.astype("<f2").tobytes(),
        )
        matrix(shared + "gate_proj.weight", np.eye(128) / 4)
        matrix(shared + "up_proj.weight", np.eye(128) / 4)
        for name in (root + "mlp_hyper_connection.hc_norm.weight", "hyper_connection_mixer.hc_norm.weight"):
            matrix(name, np.ones(weights[name][2]))
        matrix("lm_head.weight", np.eye(vocab, 128), bf16=True)
    write_hgn(path / "model.hgn", weights, canonical=editable)
    write_hgn(path / "sidecar.hgn", {})
