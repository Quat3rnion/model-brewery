#!/usr/bin/env python3
"""Compare Brewery CPU prefill against guarded Halogen ppl FP32 rows."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/hgn_trace"))
from trace import Forward, read_config, setup_threads


@torch.inference_mode()
def check(args):
    config = read_config(args.config)
    ids = np.fromfile(args.ids, dtype="<i4")
    if ids.shape != (257,):
        raise ValueError("expected the 257-token reference input (256 prefill tokens)")
    tokens = ids[:-1].tolist()
    gpu = torch.from_numpy(np.fromfile(args.rows, dtype="<f4").reshape(64, config.vocab_size)).double()
    setup_threads(args.threads)
    start = time.monotonic()
    forward = Forward(args.model, args.sidecar, config)
    for _, hidden in forward.layers(tokens):
        pass
    cpu = torch.cat([x for _, x in forward.head(forward.mixer(hidden[192:256]))], dim=-1).double()
    p, q = gpu.log_softmax(-1), cpu.log_softmax(-1)
    kl = (p.exp() * (p - q)).sum(-1)
    top1 = int((cpu.argmax(-1) == gpu.argmax(-1)).sum())
    passed = top1 >= 63 and kl.mean().item() <= 0.01
    report = {"positions": [192, 256], "top1": top1, "mean_kl": kl.mean().item(),
              "max_kl": kl.max().item(), "seconds": time.monotonic() - start,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "passed": passed}
    print(json.dumps(report, indent=2, allow_nan=False))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "sidecar", "config", "ids", "rows"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    sys.exit(not check(parser.parse_args()))
