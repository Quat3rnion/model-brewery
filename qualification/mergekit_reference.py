#!/usr/bin/env python3
"""Check or generate the small W10 fixture with the pinned mergekit CPU kernels."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct

import torch
from mergekit.merge_methods.base import TensorBatch, TensorGroup
from mergekit.merge_methods.registry import get


FIXTURE = Path(__file__).resolve().parents[1] / "crates/worker/src/weight_source/scratch/merge/reference.json"


def reference():
    assert importlib.metadata.version("mergekit") == "0.1.5"
    assert torch.__version__ == "2.9.1+cpu"
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    base = [0.5, -0.5, 1, -1, 1, 2, 0, 0, 2, -2]
    a = [2, -3.5, -1, -0.5, 5, -3, 0, -1, 3, -2]
    b = [-3.5, -3, 4.5, -0.5, 0, 1.5, 0, -3, 2, -1.5]
    tensors = [torch.tensor(x, dtype=torch.float32) for x in (base, a, b)]
    cases = {}
    for name in ("linear", "task_arithmetic", "ties", "dare_linear", "dare_ties"):
        if name == "linear":
            merged = get(name).merge_batch(
                TensorBatch(tuple(t.unsqueeze(0) for t in tensors[1:])),
                weight=torch.tensor([[0.5, 2.0]]), normalize=True,
            )[0]
        else:
            merged = get(name).merge_group(
                TensorGroup.from_tensors(tensors, base_index=0),
                weight={1: 0.5, 2: 2.0},
                density={1: 0.4, 2: 0.4} if name == "ties" else {1: 1.0, 2: 1.0},
                normalize=name == "ties", rescale=name.startswith("dare"), int8_mask=False,
                **{"lambda": 1.0},
            )
        values = merged.tolist()
        cases[name] = {
            "values": values,
            "sha256_f32le": hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest(),
        }
    return json.dumps({"inputs": [base, a, b], "outputs": cases}, indent=2) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="update the checked-in reference")
    args = parser.parse_args()
    actual = reference()
    if args.write:
        FIXTURE.write_text(actual)
    elif FIXTURE.read_text() != actual:
        raise SystemExit("mergekit reference differs")
    print("mergekit 0.1.5 / torch 2.9.1+cpu: five pinned references")
