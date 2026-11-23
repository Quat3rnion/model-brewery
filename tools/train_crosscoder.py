#!/usr/bin/env python3
"""Fit a shared TopK code with two decoders to paired W1 activations."""

import argparse
import base64
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from train_sae import BATCH, LR, SEED, TORCH


SIDES = ("parent", "child")
W1_EXECUTABLE = "0e4ca38c55a9afa0436eeadd06e89bf295eb201b986dd2f40305f95d3a61e46f"
ROLES = ("l_out", "attn_out", "ffn_inp", "ffn_out")
SPLITS = ("train", "heldout")
MAX_WEIGHTS = 128 << 20
MAX_DATASET = 256 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def codes(rows, weights, scale, k):
    x = (rows * scale - weights["b_dec"]).flatten(1)
    dense = (x @ weights["W_enc"] + weights["b_enc"]).relu()
    strength, index = dense.topk(k, dim=-1)
    return torch.zeros_like(dense).scatter(1, index, strength)


def fit(train, heldout, *, features, k, steps, difference_weight=1.0):
    if any(x.device.type != "cpu" or x.dtype != torch.float32 or x.ndim != 3
           or x.shape[1] != 2 or not 2 <= len(x) <= 16384 or not torch.isfinite(x).all()
           for x in (train, heldout)):
        raise ValueError("Expect finite paired CPU F32 matrices shaped [rows, 2, width]")
    width = train.shape[2]
    weight_bytes = 4 * (4 * features * width + features + 2 * width)
    if (width != heldout.shape[2] or not 1 <= width <= 4096
            or not 1 <= k <= features <= 4096 or not 1 <= steps <= 10000
            or not 1 <= difference_weight <= 100 or weight_bytes + 4096 > MAX_WEIGHTS):
        raise ValueError("Invalid crosscoder dimensions, TopK or training settings")
    rng = torch.Generator().manual_seed(SEED)
    mean = train.mean(dim=0)
    rms = (train - mean).square().mean().sqrt().item()
    variance = (heldout - heldout.mean(dim=0)).square().mean(dim=(0, 2))
    if not 0 < rms < float("inf") or not ((variance > 0) & torch.isfinite(variance)).all():
        raise ValueError("Training and holdout rows must have finite, nonzero centered variance")
    scale = 1.0 / rms
    # Start with shared directions. The optimizer can separate them. Normalize
    # the pair jointly so the norm of one decoder relative to the other is free.
    common = torch.randn((features, width), generator=rng)
    initial = torch.stack((common, common), dim=1).flatten(1)
    initial = torch.nn.functional.normalize(initial, dim=1)
    enc = torch.nn.Parameter(initial.T.contiguous())
    dec = torch.nn.Parameter(initial.reshape(features, 2, width).transpose(0, 1).contiguous())
    b_enc = torch.nn.Parameter(torch.zeros(features))
    b_dec = torch.nn.Parameter(mean * scale)
    weights = dict(W_enc=enc, W_dec=dec, b_enc=b_enc, b_dec=b_dec)
    adam = torch.optim.Adam(weights.values(), lr=LR)

    for _ in range(steps):
        rows = train[torch.randint(len(train), (BATCH,), generator=rng)]
        error = codes(rows, weights, scale, k) @ dec + b_dec[:, None, :] - rows.transpose(0, 1) * scale
        # In sum/difference coordinates, ordinary paired MSE weights both
        # halves equally. A larger weight emphasizes small revision deltas.
        loss = ((error[1] + error[0]).square()
                + difference_weight * (error[1] - error[0]).square()).mean() / 4
        adam.zero_grad()
        loss.backward()
        adam.step()
        with torch.no_grad():
            dec.copy_(torch.nn.functional.normalize(dec.transpose(0, 1).flatten(1), dim=1)
                      .reshape(features, 2, width).transpose(0, 1))

    with torch.no_grad():
        square_error = torch.zeros(2, dtype=torch.float64)
        support = dec.norm(dim=2) > 1e-8
        l0 = torch.zeros(2, dtype=torch.int64)
        for rows in heldout.split(BATCH):
            active = codes(rows, weights, scale, k)
            reconstruction = (active @ dec + b_dec[:, None, :]) / scale
            square_error += (rows.transpose(0, 1) - reconstruction).double().square().sum(dim=(1, 2))
            l0 += (active.gt(0) & support[:, None, :]).sum(dim=(1, 2))
        seen = torch.zeros(features, dtype=torch.bool)
        for rows in train.split(BATCH):
            seen |= codes(rows, weights, scale, k).gt(0).any(dim=0)
        mse = square_error / (len(heldout) * width)
        metrics = {
            side: {"mse": float(mse[i]), "explained_variance": float(1 - mse[i] / variance[i]),
                   "l0": float(l0[i] / len(heldout)),
                   "dead_features": int((~(support[i] & seen)).count_nonzero())}
            for i, side in enumerate(SIDES)
        }
        return {name: value.detach().contiguous() for name, value in weights.items()}, scale, metrics


def digest(data):
    return hashlib.sha256(data).hexdigest()


def capture_pair(get, pair, layer, role):
    jobs = []
    revisions = {}
    for side in SIDES:
        job_id = pair[f"{side}_job_id"]
        require(re.fullmatch(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", job_id),
                "Expected canonical trace job ID")
        job = get(f"/api/jobs/{job_id}")
        receipt = job["result"]
        require(job["kind"] == "trace" and job["state"] == "succeeded"
                and receipt["kind"] == "trace" and receipt["schema_version"] == 1
                and receipt["executable_sha256"] == W1_EXECUTABLE
                and receipt["model_id"] == job["input_ids"][0],
                "Expected a completed, pinned W1 trace")
        sha = receipt["model_sha256"]
        require(re.fullmatch(r"[a-f0-9]{64}", sha), "Missing GGUF revision SHA-256")
        revisions[side] = {"id": receipt["model_id"], "sha256": sha}
        jobs.append((job_id, receipt, job["options"].get("prompts") or [job["prompt_text"]]))

    prompts = jobs[0][2]
    require(prompts == jobs[1][2] and 1 <= len(prompts) <= 4, "Trace prompts differ")
    rows, contexts = [], []
    captured_bytes = 0
    for prompt, text in enumerate(prompts):
        manifests = [get(f"/api/jobs/{job_id}/trace?prompt={prompt}") for job_id, _, _ in jobs]
        for m, (_, receipt, _) in zip(manifests, jobs, strict=True):
            require(m["schema_version"] == 1 and m["prompt"] == prompt
                    and m["manifest_sha256"] == receipt["prompts"][prompt]["manifest_sha256"]
                    and m["model_id"] == receipt["model_id"] and m["model_sha256"] == receipt["model_sha256"]
                    and m["intervention_count"] == 0,
                    "Trace manifest does not match the unmodified revision receipt")
        a, b = manifests
        tokens = {t["position"]: t["token_id"] for t in a["tokens"]}
        positions = [p for p in a["spec"]["positions"] if p in tokens]
        require(a["tokens"] == b["tokens"] and a["spec"]["positions"] == b["spec"]["positions"]
                and positions and all(layer in m["spec"]["layers"] and role in m["spec"]["roles"]
                                      for m in manifests), "Trace tokens or capture selections differ")
        # Small chunks stay below the controller response cap for all allowed widths.
        for i in range(0, len(positions), 64):
            chunk = positions[i:i + 64]
            query = urllib.parse.urlencode(dict(prompt=prompt, layer=layer, role=role,
                                               start=chunk[0], end=chunk[-1] + 1))
            side_rows = []
            for job_id, _, _ in jobs:
                data = get(f"/api/jobs/{job_id}/trace/tensor?{query}")
                by_pos = {r["position"]: r for r in data["rows"]}
                require((data["prompt"], data["layer"], data["role"], data["dtype"])
                        == (prompt, layer, role, "f32_le") and not data["missing_positions"]
                        and sorted(by_pos) == chunk and len(by_pos) == len(data["rows"]),
                        "Missing or unaligned activation rows")
                ordered = []
                for p in chunk:
                    row = by_pos[p]
                    raw = base64.b64decode(row["data_base64"], validate=True)
                    shape = row["shape"]
                    require(len(shape) == 1 and 1 <= shape[0] <= 4096
                            and len(raw) == 4 * shape[0] and digest(raw) == row["sha256"],
                            "Expected a verified F32 activation vector")
                    ordered.append(np.frombuffer(raw, dtype="<f4"))
                side_rows.append(np.stack(ordered))
            aligned = np.stack(side_rows, axis=1)
            captured_bytes += aligned.nbytes
            require(captured_bytes <= MAX_DATASET, "Paired trace exceeds the trainer limit")
            rows.append(aligned)
            contexts.extend(dict(prompt=text, position=p, token_id=tokens[p]) for p in chunk)
    return np.concatenate(rows), revisions, contexts


def load_pairs(get, spec):
    layer, role, pairs = spec["layer"], spec["role"], spec["pairs"]
    require(type(layer) is int and 0 <= layer < 4096 and role in ROLES and 2 <= len(pairs) <= 64,
            "Expected one vector role and layer and 2 to 64 trace pairs")
    rows = {split: [] for split in SPLITS}
    contexts = {split: [] for split in SPLITS}
    revisions = None
    width = None
    jobs = set()
    for pair in pairs:
        require(set(pair) == {"parent_job_id", "child_job_id", "split"}, "Invalid trace pair fields")
        split = pair["split"]
        ids = {pair[f"{side}_job_id"] for side in SIDES}
        require(split in rows and len(ids) == 2 and ids.isdisjoint(jobs), "Invalid or repeated trace pair")
        jobs.update(ids)
        paired, bindings, labels = capture_pair(get, pair, layer, role)
        if revisions is None:
            revisions, width = bindings, paired.shape[2]
        require(bindings == revisions and paired.shape[2] == width, "Revision or activation width changed")
        require(revisions["parent"]["sha256"] != revisions["child"]["sha256"], "Expected distinct model revisions")
        rows[split].append(paired)
        contexts[split].extend(labels)
        require(len(contexts[split]) <= 16384 and sum(map(len, contexts.values())) * width * 8 <= MAX_DATASET,
                "Activation corpus exceeds the trainer limit")
    require(all(len(contexts[s]) >= 2 for s in SPLITS), "Expected separate train and heldout rows")
    train_text, heldout_text = ({row["prompt"] for row in contexts[s]} for s in SPLITS)
    require(train_text.isdisjoint(heldout_text), "Training and heldout prompts overlap")
    tensors = {s: torch.from_numpy(np.concatenate(rows[s])) for s in SPLITS}
    return tensors, revisions, contexts


def controller(url):
    base = urllib.parse.urlsplit(url)
    require(base.scheme in ("http", "https") and base.hostname and not base.username and not base.password
            and base.path in ("", "/") and not base.query and not base.fragment, "Expected a controller base URL")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(path):
        with opener.open(url.rstrip("/") + path, timeout=30) as response:
            data = response.read((4 << 20) + 1)
            require(len(data) <= 4 << 20, "Controller response exceeds the trace read limit")
            return json.loads(data)

    return get


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("controller", help="Base URL of the controller with both sets of W1 traces")
    p.add_argument("spec", type=Path, help="JSON with layer, role and ordered pairs (parent_job_id, child_job_id, split)")
    p.add_argument("output", type=Path, help="New artifact directory")
    p.add_argument("--features", type=int, default=1024)
    p.add_argument("--k", type=int, default=48)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--difference-weight", type=float, default=1.0)
    args = p.parse_args()
    if torch.__version__ != TORCH or args.spec.stat().st_size > 32 << 10:
        p.error(f"Use torch=={TORCH} and a trace spec of at most 32 KiB")
    if args.output.exists():
        p.error("Output directory already exists")
    torch.set_num_threads(2)
    source = json.loads(args.spec.read_text())
    data, revisions, _ = load_pairs(controller(args.controller), source)
    weights, scale, metrics = fit(data["train"], data["heldout"], features=args.features,
                                  k=args.k, steps=args.steps, difference_weight=args.difference_weight)
    with torch.no_grad():
        seen = torch.stack([codes(x, weights, scale, args.k).gt(0).any(dim=0)
                            for x in data["train"].split(BATCH)]).any(dim=0).tolist()
    dataset_hash = hashlib.sha256()
    for split in SPLITS:
        rows = data[split].numpy()
        dataset_hash.update(json.dumps(rows.shape).encode())
        dataset_hash.update(memoryview(rows.astype("<f4", copy=False)).cast("B"))
    metadata = dict(schema_version=1, layer=source["layer"], role=source["role"], pairs=source["pairs"],
                    **revisions, width=data["train"].shape[2], k=args.k, scale=scale, seen=seen, metrics=metrics,
                    training=dict(torch=TORCH, steps=args.steps, difference_weight=args.difference_weight,
                                  train_rows=len(data["train"]), heldout_rows=len(data["heldout"]),
                                  dataset_sha256=dataset_hash.hexdigest()))
    args.output.mkdir()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    save_file(weights, args.output / "crosscoder.safetensors")
    print(json.dumps(metrics, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
