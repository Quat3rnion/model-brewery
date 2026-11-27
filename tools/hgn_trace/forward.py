"""HGN CPU codec and layers, adapted from model-install-and-ablite (t_9afc1ddb)."""
from functools import lru_cache
import math
import mmap
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from wsp.hgn import C, H, Checkpoint


class Projection:
    """An orthonormal basis shared by the four residual streams."""
    def __init__(self, raw, config):
        if not isinstance(raw, dict) or raw.keys() - {
                "target", "vector", "second_vector", "alpha", "first_layer", "last_layer", "last_k"}:
            raise ValueError("invalid HGN projection fields")
        self.target = raw["target"]
        first, last = raw.get("first_layer"), raw.get("last_layer")
        if first is None and last is None:
            first, last = 0, config.num_hidden_layers - 1
        if (type(first) is not int or type(last) is not int
                or not 0 <= first <= last < config.num_hidden_layers):
            raise ValueError("invalid HGN projection layer range")
        self.first, self.last = first, last
        if self.target not in ("residual", "dense", "experts", "all"):
            raise ValueError("invalid HGN projection target")
        self.alpha = raw.get("alpha", 1)
        if type(self.alpha) not in (int, float) or not 0 < self.alpha <= 4:
            raise ValueError("HGN projection alpha must be in (0,4]")
        self.last_k = raw.get("last_k")
        if self.last_k is not None and (type(self.last_k) is not int or not 1 <= self.last_k <= 16):
            raise ValueError("invalid HGN direction fit position count")

        def unit(values):
            if (not isinstance(values, list) or len(values) != config.hidden_size
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
                raise ValueError("invalid HGN projection vector")
            norm = math.hypot(*values)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("invalid HGN projection norm")
            return [x / norm for x in values]

        first = unit(raw["vector"])
        self.vector = torch.tensor(first, dtype=torch.float32)
        self.second = None
        self.cosine = None
        if "second_vector" in raw:
            second = unit(raw["second_vector"])
            self.cosine = min(1, max(-1, math.fsum(a * b for a, b in zip(first, second))))
            residual = [b - self.cosine * a for a, b in zip(first, second)]
            if math.hypot(*residual) <= 1e-5:
                raise ValueError("HGN projection directions are collinear")
            self.second = torch.tensor(unit(residual), dtype=torch.float32)

    def project(self, value):
        v = self.vector
        delta = (value @ v).unsqueeze(-1) * v
        if self.second is not None:
            delta = delta + (value @ self.second).unsqueeze(-1) * self.second
        return value - self.alpha * delta

    def residual(self, hidden):
        return self.project(hidden.unflatten(-1, (-1, self.vector.numel()))).flatten(-2)

    def initial(self, hidden):
        # The token embedding is a dense writer at layer zero.
        return self.residual(hidden) if self.first == 0 and self.target != "experts" else hidden

    def hook(self, _module, _args, output):
        return self.project(output)

    def step(self, layer, index, hidden, **kwargs):
        if not self.first <= index <= self.last:
            return layer(hidden, **kwargs)
        if self.target == "residual":
            return self.residual(layer(hidden, **kwargs))
        writers = []
        if self.target in ("dense", "all"):
            writers.extend((layer.mlp.shared_expert.down_proj,
                            layer.self_attn.o_proj if hasattr(layer, "self_attn")
                            else layer.linear_attn.out_proj))
            if layer.ple is not None:
                writers.append(layer.ple.value_proj)
        if self.target in ("experts", "all"):
            # Routing is a weighted sum, so projecting the sum is equivalent
            # to projecting each selected expert's down output before weighting.
            writers.append(layer.mlp.experts)
        handles = [writer.register_forward_hook(self.hook) for writer in writers]
        try:
            return layer(hidden, **kwargs)
        finally:
            for handle in handles:
                handle.remove()


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
        # Below the QSA budget every visible key is selected. Above it, use
        # the pinned model's actual indexer instead of substituting dense attention.
        index = layer.self_attn.indexer.forward

        def select(_self, hidden_states, position_embeddings, attention_mask, cache):
            if cache is None and hidden_states.shape[1] <= config.indexer_budget:
                return attention_mask
            return index(hidden_states, position_embeddings, attention_mask, cache)

        layer.self_attn.indexer.forward = types.MethodType(select, layer.self_attn.indexer)
    return layer.eval()
