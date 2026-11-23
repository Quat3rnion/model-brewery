"""HGN CPU codec and layers, adapted from model-install-and-ablite (t_9afc1ddb)."""
from functools import lru_cache
import mmap
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from wsp.hgn import C, H, Checkpoint


def q6g64(raw, shape):
    cols = shape[-1]
    if cols % 64:
        raise ValueError("Q6G64 width must be divisible by 64")
    stride = (cols * 13 // 16 + 15) // 16 * 16
    rows = np.frombuffer(raw, np.uint8).reshape(-1, stride)
    if np.any(rows[:, cols * 13 // 16:]):
        raise ValueError("Q6G64 row padding is not zero")
    low = np.stack((rows[:, :cols // 2] & 15, rows[:, :cols // 2] >> 4), -1)
    high = rows[:, cols // 2:3 * cols // 4, None]
    high = (high >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
    codes = low.reshape(-1, cols // 64, 64) | (high.reshape(-1, cols // 64, 64) << 4)
    params = rows[:, 3 * cols // 4:cols * 13 // 16].copy().view("<f2").astype(np.float32)
    params = params.reshape(-1, cols // 64, 2)
    return (codes * params[..., :1] + params[..., 1:]).reshape(shape)


def linear_weight(checkpoint, name):
    entry = checkpoint.entries[name]
    if entry.store == 24 and entry.variant == 64:
        return torch.from_numpy(q6g64(checkpoint.raw(name), entry.shape))
    return torch.from_numpy(checkpoint.get(name))


def token_embedding(checkpoint, ids):
    name = "embed_tokens.weight"
    entry = checkpoint.entries[name]
    if (entry.store, entry.variant) != (5, 2):
        raise ValueError("expected Q4Cp v2 token embedding")
    rows, cols = entry.shape
    scale_start = 64 + (rows * cols // 2 + 63) // 64 * 64
    scale_stride = (cols // 16 + 15) // 16 * 16
    unique, inverse = np.unique(ids.numpy(), return_inverse=True)
    if unique[0] < 0 or unique[-1] >= rows:
        raise ValueError("token embedding ID is out of bounds")
    raw = H.payload(checkpoint.path, entry)
    raw._mmap.madvise(mmap.MADV_RANDOM)
    codes = raw[64:64 + rows * cols // 2].reshape(rows, cols // 2)[unique]
    codes = np.stack((codes & 15, codes >> 4), -1).reshape(-1, cols // 32, 32)
    scale = raw[scale_start:].reshape(rows, scale_stride)[unique, :cols // 16]
    weights = raw[:64].view("<f4")[codes] * scale.copy().view("<f2")[..., None]
    return torch.from_numpy(weights.reshape(-1, cols)[inverse].reshape(*ids.shape, cols))


class NGramEmbedding(nn.Module):
    def __init__(self, checkpoint, name):
        super().__init__()
        entry = checkpoint.entries[name]
        shards, rows, width = entry.shape
        self.rows, self.width = shards * rows, width
        size = self.rows * width
        scale_offset = (size + 63) // 64 * 64
        if (entry.store, entry.variant, entry.size) != (10, 0, scale_offset + 4):
            raise ValueError("expected Fp8G128 n-gram table with a trailing global scale")
        with open(checkpoint.path, "rb") as fp:
            fp.seek(entry.offset + scale_offset)
            scale = np.frombuffer(fp.read(4), "<f4")[0]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("invalid Fp8G128 n-gram scale")
        self.table = np.memmap(checkpoint.path, np.uint8, mode="r", offset=entry.offset,
                               shape=(self.rows, self.width))
        self.table._mmap.madvise(mmap.MADV_RANDOM)
        raw = np.arange(256)
        exp, mant = (raw >> 3) & 15, raw & 7
        magnitude = np.where(exp == 0, mant / 512, np.ldexp(1 + mant / 8, exp - 7))
        self.codebook = np.copysign(magnitude, 1 - (raw >> 7) * 2).astype(np.float32) * scale
        # HF only checks .weight.device for embedding placement.
        self.weight = torch.empty((self.rows, self.width), device="meta")

    def forward(self, ids):
        unique, inverse = np.unique(ids.numpy(), return_inverse=True)
        if unique[0] < 0 or unique[-1] >= self.rows:
            raise ValueError("n-gram embedding ID is out of bounds")
        value = self.codebook[self.table[unique]]
        return torch.from_numpy(value[inverse].reshape(*ids.shape, self.width))


class RoutedExperts(nn.Module):
    """Apply I4r experts in their stored basis; rotate activations, not matrices."""
    def __init__(self, checkpoint, root, act_fn):
        super().__init__()
        self.act_fn = act_fn
        self.had = torch.from_numpy(C.H128.copy())
        self.planes = {}
        self.input_signs = {}
        self.output_signs = {}
        for key in ("gate_up_proj", "down_proj"):
            name = root + key + ".weight"
            entry = checkpoint.entries[name]
            experts, outputs, inputs = entry.shape
            if (entry.store, entry.variant) != (23, 128):
                raise ValueError(f"{name}: expected I4r variant 128")
            p = H.payload(checkpoint.path, entry)
            codes, scales = C.i4r_split(p, experts, outputs, inputs)
            self.planes[key] = codes, scales
            su, sv = checkpoint.i4r_planes(name)
            self.input_signs[key] = torch.from_numpy(su)
            self.output_signs[key] = torch.from_numpy(sv)
        # Allow within-layer reuse across microbatches without retaining any
        # expert from a previous layer.
        self.weight = lru_cache(maxsize=128)(self.decode)

    def decode(self, key, expert):
        codes, scales = self.planes[key]
        return torch.from_numpy(C.i4r_decode_rot(codes[expert:expert + 1],
                                                 scales[expert:expert + 1])[0])

    def linear(self, key, expert, x):
        # The signs and H128 blocks are exactly those used by Checkpoint.get's
        # inverse rotation. Moving them to the activations saves two GEMMs
        # over each expert matrix.
        source = ((x * self.input_signs[key]).reshape(len(x), -1, 128) @ self.had).reshape_as(x)
        dest = F.linear(source, self.weight(key, expert))
        return ((dest.reshape(len(x), -1, 128) @ self.had).reshape_as(dest)
                * self.output_signs[key])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        for expert in torch.unique(top_k_index).tolist():
            token_idx, slot = torch.where(top_k_index == expert)
            gate, up = self.linear("gate_up_proj", expert, hidden_states[token_idx]).chunk(2, -1)
            value = self.linear("down_proj", expert, self.act_fn(gate) * up)
            result.index_add_(0, token_idx, value * top_k_weights[token_idx, slot, None])
        return result


def load_layer(checkpoint, sidecar, config, index):
    root = f"layers.{index}."
    with torch.device("meta"):
        layer = Qwen4ExpTextDecoderLayer(config, index)
    if layer.ple is not None:
        name = root + "ple.ngram_embedding.weight"
        layer.ple.ple_embedding.ngram_embedding = NGramEmbedding(sidecar, name)
    layer.mlp.experts = RoutedExperts(checkpoint, root + "mlp.experts.", layer.mlp.experts.act_fn)
    state = {name: linear_weight(checkpoint, root + name) for name in layer.state_dict()}
    layer.load_state_dict(state, strict=True, assign=True)
    if hasattr(layer, "self_attn"):
        # Short sequences do not exhaust the QSA budget. Every visible key is
        # selected, so the indexer returns the original causal mask.
        def retain_all(_self, hidden_states, position_embeddings, attention_mask, past_key_values):
            return attention_mask

        layer.self_attn.indexer.forward = types.MethodType(retain_all, layer.self_attn.indexer)
    return layer.eval()


