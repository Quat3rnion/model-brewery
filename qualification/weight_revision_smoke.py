"""Offline manual and calibrated scalar-edit → Q8_0 → GGUF and HGN regression."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import gguf
import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
NAME = "blk.0.attn_q.weight"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=60)


def cli(tool, *args):
    return cmd(sys.executable, "-B", TOOLS / tool, *args)


def run(source, quantizer, out):
    out.mkdir(exist_ok=False)
    src_hash = sha(source)
    weights = next(t.data for t in gguf.GGUFReader(source).tensors if t.name == NAME)
    if weights.shape != (256, 256) or weights.dtype != np.float32:
        raise ValueError("expected dense-f32.gguf synthetic catalog fixture")
    # Two distinct input splits. Known target differs in exactly one weight.
    target = weights.astype(float)
    target[0, 0] = 4.
    for key, coeffs in (("calibration", [1., -1.]), ("held", [2., -3.])):
        x = np.zeros((len(coeffs), weights.shape[1]))
        x[:, 0] = coeffs
        (out/key).write_text(json.dumps(dict(activations=x.tolist(), targets=(x @ target.T).tolist())))
    request = out/"request.json"
    request.write_text(json.dumps([dict(tensor=NAME, index=[0, 0], op="set", value=4.)]))
    baseline = out/"baseline.gguf"
    cmd(quantizer, source, baseline, "Q8_0", "1")
    reports = {}
    for mode in ("manual", "automatic"):
        props = (cli("weight_edit.py", "propose", source, "--source-sha256", src_hash, request)
                 if mode == "manual" else
                 cli("weight_tune.py", source, "--source-sha256", src_hash, "--tensor", NAME,
                     "--indices", "[[0,0]]", "--calibration", out/"calibration",
                     "--held-out", out/"held", "--max-update", "8"))
        (out/(mode+".report")).write_text(props.stderr)
        prop = out/(mode+".json")
        prop.write_text(props.stdout)
        derived, encoded = (out/(mode+s) for s in (".f32.gguf", ".q8.gguf"))
        cli("weight_edit.py", "apply", source, prop, derived)
        v = next(t.data for t in gguf.GGUFReader(derived).tensors if t.name == NAME)
        if v[0, 0] != 4 or np.count_nonzero(v != weights) != 1:
            raise ValueError(f"{mode}: wrong edits applied")
        cmd(quantizer, derived, encoded, "Q8_0", "1")
        impact = json.loads(cli("weight_impact.py", baseline, encoded, "--tensor", NAME,
                                "--index", "[0,0]").stdout)
        if (not impact["changed_neighbors"] or impact["selected_before"] == impact["selected_after"]
                or impact["changed_values_outside_block"] or impact["other_changed_tensors"]):
            raise ValueError(f"unexpected Q8_0 collateral: {impact}")
        pin = sha(encoded)
        for fmt in ("gguf", "hgn"):
            cli("exact-export", "--encoded-sha256", pin, "--to", fmt, encoded, out/(mode+"."+fmt))
        reports[mode] = dict(derived_sha256=sha(derived), encoded_sha256=pin, impact=impact)
    if sha(source) != src_hash or reports["manual"] != reports["automatic"]:
        raise ValueError("source mutated or equivalent edits diverged")
    return reports


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "quantizer", "out"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    try:
        print(json.dumps(run(args.source, args.quantizer, args.out), indent=2))
    except subprocess.CalledProcessError as e:
        sys.exit(f"{e}: {e.stderr}")
