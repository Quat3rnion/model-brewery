#!/usr/bin/env python3
"""Fit a small TopK SAE to separate CPU trace activation matrices."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


TORCH = "2.7.1+cpu"
SEED = 11
BATCH = 64
LR = 0.003


def fit(train, heldout, *, features, k, steps):
    if any(x.device.type != "cpu" or x.dtype != torch.float32 or x.ndim != 2
           or not torch.isfinite(x).all() or not 2 <= len(x) <= 16384
           for x in (train, heldout)):
        raise ValueError("Expect separate, finite CPU F32 row matrices with 2 to 16384 rows each")
    width = train.shape[1]
    if (width != heldout.shape[1] or not 1 <= width <= 4096
            or not 1 <= k <= features <= 4096 or not 1 <= steps <= 10000):
        raise ValueError("Invalid SAE dimensions, TopK, or training step count")
    rng = torch.Generator().manual_seed(SEED)
    mean = train.mean(dim=0)
    rms = (train - mean).square().mean().sqrt().item()
    variance = (heldout - heldout.mean(dim=0)).square().mean().item()
    if not 0 < rms < float("inf") or not 0 < variance < float("inf"):
        raise ValueError("Training and holdout rows must have finite, nonzero centered variance")
    scale = 1.0 / rms
    decoder = torch.nn.functional.normalize(torch.randn((features, width), generator=rng), dim=1)
    encoder = torch.nn.Parameter(decoder.T.contiguous())
    decoder = torch.nn.Parameter(decoder)
    b_enc = torch.nn.Parameter(torch.zeros(features))
    b_dec = torch.nn.Parameter(mean * scale)
    params = dict(W_enc=encoder, W_dec=decoder, b_enc=b_enc, b_dec=b_dec)
    adam = torch.optim.Adam(params.values(), lr=LR)

    def active(x):
        dense = ((x * scale - b_dec) @ encoder + b_enc).relu()
        strength, index = dense.topk(k, dim=-1)
        return torch.zeros_like(dense).scatter(1, index, strength)

    for _ in range(steps):
        sample = train[torch.randint(len(train), (BATCH,), generator=rng)]
        reconstruction = active(sample) @ decoder + b_dec
        loss = (reconstruction - sample * scale).square().mean()
        adam.zero_grad()
        loss.backward()
        adam.step()
        with torch.no_grad():
            decoder.copy_(torch.nn.functional.normalize(decoder, dim=1))

    with torch.no_grad():
        mse_sum = l0_sum = 0.
        for x in heldout.split(BATCH):
            a = active(x)
            r = (a @ decoder + b_dec) / scale
            mse_sum += (x - r).double().square().sum().item()
            l0_sum += a.count_nonzero().item()
        seen = torch.zeros(features, dtype=torch.bool)
        for x in train.split(BATCH):
            seen |= active(x).gt(0).any(dim=0)
        mse = mse_sum / heldout.numel()
        metrics = {"mse": mse, "explained_variance": 1 - mse / variance,
                   "l0": l0_sum / len(heldout), "dead_features": int((~seen).count_nonzero())}
        return ({name: value.detach().contiguous() for name, value in params.items()},
                {"scale": scale, "subtract_decoder_bias": True}, metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path, help="SafeTensors with separate F32 'train' and 'heldout' tensors")
    p.add_argument("binding", type=Path, help="JSON with name, model_id, model_sha256, layer, role, provenance.corpus")
    p.add_argument("output", type=Path, help="New artifact directory")
    p.add_argument("--features", type=int, default=1024)
    p.add_argument("--k", type=int, default=48)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--target", type=float, default=0.60, help="Minimum held-out explained variance")
    args = p.parse_args()
    if torch.__version__ != TORCH or not 0 < args.target < 1:
        p.error(f"Use torch=={TORCH} and a reconstruction target strictly between zero and one")
    if args.dataset.stat().st_size > 160 << 20 or args.binding.stat().st_size > 32 << 10:
        p.error("Training data or binding exceeds the tool limit")
    torch.set_num_threads(4)
    tensors = load_file(args.dataset, device="cpu")
    weights, norm, metrics = fit(tensors["train"], tensors["heldout"], features=args.features,
                                  k=args.k, steps=args.steps)
    print(json.dumps(metrics, allow_nan=False), flush=True)
    if metrics["explained_variance"] < args.target:
        p.error("Held-out reconstruction target was not met; no artifact was written")
    metadata = json.loads(args.binding.read_text())
    with args.dataset.open("rb") as stream:
        dataset_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata.update(schema_version=1, d_in=tensors["train"].shape[1], d_sae=args.features,
                    activation={"kind":"top_k", "k":args.k}, normalization=norm)
    metadata["provenance"].update(method="cpu-top-k", tool="tools/train_sae.py", torch=TORCH,
                                  seed=SEED, steps=args.steps, optimizer="Adam", lr=LR, batch=BATCH,
                                  dataset_sha256=dataset_sha256, train_rows=len(tensors["train"]),
                                  heldout_rows=len(tensors["heldout"]), target=args.target, metrics=metrics)
    args.output.mkdir()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    save_file(weights, args.output / "sae.safetensors")


if __name__ == "__main__":
    main()
