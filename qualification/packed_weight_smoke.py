"""Verify reviewed manual/tuned edits to actual packed GGUF fixture bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import gguf
import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
NAME = "blk.0.attn_q.weight"


def cli(name, *args):
    p = subprocess.run([sys.executable, "-B", TOOLS/name, *args], capture_output=True,
                       text=True, timeout=60)
    if p.returncode:
        raise ValueError(p.stderr.strip())
    return json.loads(p.stdout)


def run(source, out, kind):
    data = source.read_bytes()
    pin = hashlib.sha256(data).hexdigest()
    out.mkdir(exist_ok=False)
    tensor = next(t for t in gguf.GGUFReader(source).tensors if t.name == NAME)
    assert tensor.tensor_type.name == kind
    w = gguf.dequantize(tensor.data, tensor.tensor_type)
    # Set one code to a different representable code in the same block.
    j = next(i for i in range(1, 32) if w[0, i] != w[0, 0])
    goal = float(w[0, j])
    for split, factors in (("cal", [1., -1.]), ("held", [2., -3.])):
        x = np.zeros((2, w.shape[1]))
        x[:, 0] = factors
        y = x @ w.T
        y[:, 0] = x[:, 0] * goal
        (out/split).write_text(json.dumps({"activations": x.tolist(), "targets": y.tolist()}))
    req = out/"request"
    req.write_text(json.dumps([dict(tensor=NAME, index=[0, 0], op="set", value=goal)]))
    for mode in ("manual", "auto"):
        args = (["weight_edit.py", "propose", source, "--source-sha256", pin, req]
                if mode == "manual" else
                ["weight_tune.py", source, "--source-sha256", pin, "--tensor", NAME,
                 "--indices", "[[0,0]]", "--max-update", str(2*abs(goal-float(w[0, 0]))),
                 "--calibration", out/"cal", "--held-out", out/"held"])
        proposal = cli(*args)
        pp = out/(mode+".json")
        pp.write_text(json.dumps(proposal))
        assert proposal["edits"][0]["after"] == goal
        dest = out/(mode+".gguf")
        cli("weight_edit.py", "apply", source, pp, dest)
        other = dest.read_bytes()
        assert len(other) == len(data) and sum(x != y for x, y in zip(data, other)) == 1
        again = next(t for t in gguf.GGUFReader(dest).tensors if t.name == NAME)
        assert again.tensor_type == tensor.tensor_type
        z = gguf.dequantize(again.data, again.tensor_type)
        assert z[0, 0] == goal and np.count_nonzero(z != w) == 1
    assert source.read_bytes() == data
    return {"codec": kind, "manual_and_tuned": True, "before": float(w[0, 0]), "after": goal}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixtures", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.mkdir(exist_ok=False)
    for k in ("Q4_0", "Q8_0"):
        print(json.dumps(run(args.fixtures/f"exact-{k}.gguf", args.out/k, k)))
