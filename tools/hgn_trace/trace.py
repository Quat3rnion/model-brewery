#!/usr/bin/env python3
"""Run a raw CPU prefill and export a Lanternfish schema v1 trace."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
import torch
from torch.nn import functional as F
from transformers import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
)

from forward import Checkpoint, linear_weight, load_layer, token_embedding

ROLES = {"l_out", "final_norm", "attn_out", "ffn_inp", "ffn_out", "kq_soft_max"}


def capture_spec(raw, n_layers, n_tokens, vocab):
    if not isinstance(raw, dict) or raw.keys() != {"layers", "roles", "positions", "top_k", "byte_cap"}:
        raise ValueError("invalid capture spec fields")
    spec = dict(raw)
    for key, limit in (("layers", n_layers), ("positions", n_tokens)):
        if not isinstance(raw[key], list) or any(type(x) is not int or not 0 <= x < limit for x in raw[key]):
            raise ValueError(f"{key} must be zero-based indices below {limit}")
        spec[key] = sorted(set(raw[key]))
    if not isinstance(raw["roles"], list) or any(not isinstance(r, str) or r not in ROLES for r in raw["roles"]):
        raise ValueError("invalid capture roles")
    spec["roles"] = sorted(set(raw["roles"]))
    k, cap = spec["top_k"], spec["byte_cap"]
    token_probe = not spec["layers"] and not spec["roles"] and spec["positions"] == [0]
    if (type(k) is not int or not 0 <= k <= min(100, vocab)
            or type(cap) is not int or not 1 <= cap <= 32 << 20
            or not spec["positions"] or (not spec["layers"] and spec["roles"])
            or (not k and not spec["roles"] and not token_probe)):
        raise ValueError("invalid capture selection, top_k, or byte cap")
    return spec


def read_config(path):
    raw = json.loads(path.read_text())
    text = raw["text_config"]
    if text["model_type"] != "qwen4_exp_text":
        raise ValueError("expected Qwen3.8 Flash Next text config")
    config = Qwen4ExpTextConfig(**text)
    config._attn_implementation = "sdpa"
    return config


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


class Forward:
    def __init__(self, model, sidecar, config):
        self.config = config
        self.checkpoint = Checkpoint(str(model))
        self.sidecar = Checkpoint(str(sidecar))
        if self.checkpoint.version != 2 or self.sidecar.version != 2 or config.hc_count != 4:
            raise ValueError("unsupported HGN container or hyper-connection layout")
        if self.checkpoint.entries["lm_head.weight"].shape != (config.vocab_size, config.hidden_size):
            raise ValueError("HGN output head does not match config")
        root = "hyper_connection_mixer."
        with torch.device("meta"):
            self.mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.mixer.load_state_dict(
            {n: linear_weight(self.checkpoint, root + n) for n in self.mixer.state_dict()},
            strict=True, assign=True,
        )
        self.mixer.eval()

    def layers(self, tokens):
        budget = self.config.indexer_budget or 512
        if not 1 <= len(tokens) <= min(512, budget):
            raise ValueError("expected 1 to 512 tokens within the dense QSA window")
        ids = torch.tensor([tokens])
        hidden = token_embedding(self.checkpoint, ids).repeat(1, 1, self.config.hc_count)
        pos = torch.arange(len(tokens)).view(1, 1, -1).expand(3, 1, -1)
        rotary = Qwen4ExpTextRotaryEmbedding(self.config).eval()(hidden, pos)
        causal = torch.ones((len(tokens), len(tokens)), dtype=torch.bool).tril()[None, None]
        for i in range(self.config.num_hidden_layers):
            layer = load_layer(self.checkpoint, self.sidecar, self.config, i)
            hidden = layer(hidden, position_embeddings=rotary, attention_mask=causal, ple_input_ids=ids)
            layer.mlp.experts.weight.cache_clear()
            del layer
            if not torch.isfinite(hidden).all():
                raise ValueError(f"layer {i}: non-finite hidden states")
            print(f"layer {i:02}", flush=True)
            yield i, hidden[0]

    def head(self, normalized):
        for lo in range(0, self.config.vocab_size, 2048):
            hi = min(lo + 2048, self.config.vocab_size)
            head = torch.from_numpy(self.checkpoint.get("lm_head.weight", lo, hi))
            yield lo, F.linear(normalized, head)

    def top(self, normalized, k):
        # Stable order is by score, then by ID. Chunks arrive in ID order;
        # discard all but the most competitive candidates between chunks.
        values = torch.empty((len(normalized), 0), dtype=torch.float32)
        ids = torch.empty((len(normalized), 0), dtype=torch.long)
        for lo, chunk in self.head(normalized):
            if not torch.isfinite(chunk).all():
                raise ValueError("non-finite HGN logits")
            values = torch.cat((values, chunk), dim=-1)
            ids = torch.cat((ids, torch.arange(lo, lo + chunk.shape[1]).expand(len(normalized), -1)), dim=-1)
            best = torch.argsort(values, descending=True, stable=True)[..., :k]
            values = values.gather(-1, best)
            ids = ids.gather(-1, best)
        return [[{"token_id": token, "logit": v} for token, v in zip(index, scores)]
                for index, scores in zip(ids.tolist(), values.tolist())]


def setup_threads(threads):
    if not 1 <= threads <= 64:
        raise ValueError("threads must be in [1,64]")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


@torch.inference_mode()
def run(args):
    setup_threads(args.threads)
    tokens = Tokenizer.from_file(str(args.tokenizer)).encode(args.prompt, add_special_tokens=False).ids
    if not 1 <= len(tokens) <= 512:
        raise ValueError("expected 1 to 512 prompt tokens")
    config = read_config(args.config)
    spec = capture_spec(json.loads(args.spec.read_text()), config.num_hidden_layers, len(tokens), config.vocab_size)
    selected, roles, positions, k, cap = (spec[n] for n in ("layers", "roles", "positions", "top_k", "byte_cap"))
    last = config.num_hidden_layers - 1
    shape = (config.hc_count, config.hidden_size)
    payload_bytes = 4 * len(positions) * config.hidden_size * (
        config.hc_count * len(selected) * ("l_out" in roles) + (last in selected and "final_norm" in roles)
    )
    if payload_bytes >= cap:
        raise ValueError("trace exceeds byte cap")
    forward = Forward(args.model, args.sidecar, config) if roles or k else None
    args.out.mkdir(mode=0o700)
    m = {
        "schema_version": 1, "kind": "trace", "spec": spec,
        "model": {"arch": "qwen4_exp_text", "name": forward.checkpoint.ident if forward else "",
                  "n_layers": config.num_hidden_layers, "hidden": config.hidden_size,
                  "n_experts": config.num_experts, "top_k": config.num_experts_per_tok,
                  "tied_head": False, "final_norm": "hyper_connection_mixer.hc_norm",
                  "l_out_axes": ["stream", "hidden"], "path": str(args.model)},
        "tokens": [{"position": i, "token_id": token} for i, token in enumerate(tokens)],
        "tensors": [], "missing": [], "interventions": [], "model_logits": [], "logit_lens": [],
    }
    consumed = payload_bytes + len(encode(m))

    def append(field, row):
        nonlocal consumed
        consumed += len(encode(row)) + bool(m[field])  # Comma after the first row.
        if consumed > cap:
            raise ValueError("trace exceeds byte cap")
        m[field].append(row)

    def save(key, value):
        data = value.numpy().astype("<f4", copy=False).tobytes()
        index = len(m["tensors"])
        file = f"tensor-{index:06}.f32"
        (args.out / file).write_bytes(data)
        append("tensors", dict(key, file=file, node=f"{key['role']}-{key['layer']}", shape=list(value.shape),
                               dtype="f32_le", bytes=len(data), sha256=sha256(data).hexdigest()))

    for layer, hidden in (forward.layers(tokens) if forward else ()):
        if layer not in selected and layer != last:
            continue
        rows = hidden[positions]
        norm = forward.mixer(rows) if k or (layer == last and "final_norm" in roles) else None
        if norm is not None and not torch.isfinite(norm).all():
            raise ValueError(f"layer {layer}: non-finite final mixer output")
        if layer in selected:
            for role in roles:
                for j, p in enumerate(positions):
                    key = {"layer": layer, "role": role, "position": p}
                    if role == "l_out":
                        save(key, rows[j].reshape(shape))
                    elif role == "final_norm" and layer == last:
                        save(key, norm[j])
                    else:
                        append("missing", key)
        if k:
            top = [{"position": p, "top": scores}
                   for p, scores in zip(positions, forward.top(norm, k))]
            for row in top:
                if layer in selected:
                    append("logit_lens", dict(row, layer=layer))
                if layer == last:
                    append("model_logits", row)
    data = encode(m)
    if consumed != len(data) + payload_bytes or consumed > cap:
        raise ValueError("trace exceeds byte cap")
    (args.out / "manifest.json").write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "sidecar", "config", "tokenizer", "spec", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--threads", type=int, default=8)
    try:
        run(parser.parse_args())
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"HGN trace: {error}\n")
