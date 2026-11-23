#!/usr/bin/env python3
"""Compare the pinned Qwen BF16, W12 prune, W10 merge, and Q4_K_M on an isolated stack."""
import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import weight_space_acceptance as w8
from weight_space_pruning_acceptance import Agent, LAYER, PERCENTAGES


Q4 = w8.BASE.with_name("qwen2.5-0.5b-instruct-q4_k_m.gguf")
EXPECTED = {
    "w12-prune": "51e027be30a7e179bb51b25bfc6a419bf750e5ba1791059e2cfd8ee93d17975b",
    "w10-merge": "ae652819a2e5b02de660e90e766dfd5223db2d41dc8f0ceaf624e68b6ba6d965",
    "q4-k-m": "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
}


def numpy_fixture(agent, directory, evidence):
    import numpy as np

    w = np.diag(np.array([15, 12, 9, 7, 5, 4, 3.5, 3.1, 2.9, 2.8, 2.7, 2.65, 2.61, 2.6, 2.59, 2.57], dtype="<f4"))
    fixture = directory / "numpy.safetensors"
    header = json.dumps({"fixture": {"dtype": "F32", "shape": list(w.shape), "data_offsets": [0, w.nbytes]}}).encode()
    fixture.write_bytes(len(header).to_bytes(8, "little") + header + w.tobytes())
    model = agent.write("register_model", "numpy-library", host="local", kind="safetensors", path=str(fixture))
    job = agent.write("weight_spectra", "numpy-queued", id=model["id"])
    row, = agent.wait(job, "numpy-job")["tensors"]
    actual = row["source"]
    w64 = w.astype(np.float64)
    singulars = np.linalg.svd(w64, compute_uv=False)
    p = singulars / singulars.sum()
    eig = np.sort(singulars ** 2)
    fits = []
    for cut in eig[:-7]:
        tail = eig[eig >= cut]
        alpha = 1 + len(tail) / np.log(tail / cut).sum()
        cdf = 1 - (tail / cut) ** (1 - alpha)
        rank = np.arange(len(tail))
        ks = np.maximum(abs(rank / len(tail) - cdf), abs((rank + 1) / len(tail) - cdf)).max()
        fits.append((ks, alpha, cut, len(tail)))
    ks, alpha, xmin, n = min(fits)
    expected = {
        "singular_values": singulars.tolist(),
        "effective_rank": float(np.exp(-np.sum(p * np.log(p)))),
        "stable_rank": float(np.linalg.norm(w64) ** 2 / singulars[0] ** 2),
        "alpha": float(alpha), "alpha_xmin": float(xmin), "alpha_ks": float(ks), "alpha_tail_count": n,
    }
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=1e-12, atol=1e-12, err_msg=key)
    w8.save(evidence, "numpy-oracle", {"version": np.__version__, "expected": expected, "actual": actual})


def quantized_case(agent, model, spectra):
    q4 = agent.write("register_model", "q4-library", host="local", kind="gguf",
                     path=str(Q4), name="Qwen2.5-0.5B-Instruct Q4_K_M")
    rows = spectra("q4-k-m", q4, layer=LAYER, compare_id=model["id"])
    # The file label is a preset, not a per-tensor type. This pinned file
    # stores the layer 8 down and value projections as Q6_K and Q8_0.
    types = {r["source"]["dtype"] for r in rows if r["status"] == "compared"
             and r["delta"]["sigma_max"] != 0}
    w8.require({"Q6_K", "Q8_0"} <= types, f"Expected changed dequantized spectra ({types})")


def exercise(base, evidence, bins, run, skip_prune, gguf_only):
    a = Agent(base, evidence)
    a.post("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                                 "revision": 0, "enabled": True})
    model = a.write("register_model", "library", host="local", kind="safetensors",
                    path=str(w8.SAFE), name="Qwen2.5-0.5B-Instruct BF16 SafeTensors")
    pin = w8.PINS[w8.SAFE]
    report = {"qualification": w8.QUALIFICATION, "bin_sha256": bins, "cases": {}}

    def spectra(label, source, **spec):
        job = a.write("weight_spectra", label + "-queued", id=source["id"], **spec)
        result = a.wait(job, label + "-job")
        w8.require(result["source"]["sha256"] == EXPECTED[label], f"{label}: source hash mismatch")
        w8.require(result["reference"]["sha256"] == pin, f"{label}: reference hash mismatch")
        w8.require(result["qualification"] == w8.QUALIFICATION, f"{label}: qualification mismatch")
        report["cases"][label] = {
            "job": job["id"], "source": result["source"], "reference": result["reference"],
            "tensors": [{k: row[k] for k in ("name", "status", "delta")} for row in result["tensors"]],
        }
        w8.save(evidence, "report", report)
        return result["tensors"]

    if gguf_only:
        quantized_case(a, model, spectra)
        report["status"] = "passed"
        w8.save(evidence, "report", report)
        return

    root = a.write("create_scratch", "base", parent_id=model["id"], parent_sha256=pin)
    if not skip_prune:
        plan = PERCENTAGES["magnitude-30"]
        child = a.write("prune_model", "w12-prune", id=root["id"], sha256=root["sha256"], pruning=plan)
        w8.require(child["sha256"] == EXPECTED["w12-prune"], "W12 child hash mismatch")
        rows = spectra("w12-prune", child, layer=LAYER)
        for name in plan["tensors"]:
            row = next(r for r in rows if r["name"] == name)
            w8.require(row["status"] == "compared" and row["delta"]["sigma_max"] != 0,
                       f"W12 {name}: expected changed spectrum")
            w8.require(row["source"]["method"] == "randomized_top_32", f"W12 {name}: method")
        w8.require(all(row["delta"]["sigma_max"] == 0 for row in rows if row["name"] not in plan["tensors"]),
                   "W12: a spectrum outside the pruned tensors changed")
        a.write("discard_scratch", "discard-w12", id=child["id"])

    # Recreate the two deterministic edits and the W10 task-arithmetic merge.
    parents = []
    for i, weight in enumerate((0.5, 0.75)):
        values = [(((j * (3 + i) + i) % 17) - 8) / 128 for j in range(640)]
        edit = a.write("apply_edit", f"w10-input-{i}", id=root["id"], operations=[
            {"op": "add", "tensor": w8.TENSOR, "origin": [i * 64], "shape": [640], "values": values},
        ])
        parents.append({"id": edit["id"], "sha256": edit["sha256"], "weight": weight})
    child = a.write("merge_models", "w10-merge", base_id=root["id"], base_sha256=root["sha256"],
                    algorithm={"method": "task_arithmetic"}, inputs=parents)
    for i, parent in enumerate(parents):
        a.write("discard_scratch", f"discard-w10-input-{i}", id=parent["id"])
    w8.require(child["sha256"] == EXPECTED["w10-merge"], "W10 child hash mismatch")
    row, = spectra("w10-merge", child, tensor=w8.TENSOR)
    w8.require(row["status"] == "compared" and row["source"]["method"] == "exact"
               and row["delta"]["sigma_max"] != 0 and row["source"]["alpha"] is None,
               "W10: expected a changed vector norm and no alpha")
    a.write("discard_scratch", "discard-w10", id=child["id"])
    a.write("discard_scratch", "discard-base", id=root["id"])

    quantized_case(a, model, spectra)
    numpy_fixture(a, run / "assets", evidence)
    report["status"] = "passed"
    w8.save(evidence, "report", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="New evidence directory under /arboretum")
    parser.add_argument("--bins", type=Path, required=True, help="Directory with copied Brewery binaries")
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument("--numpy-only", action="store_true", help="Run only the small NumPy oracle fixture")
    cases.add_argument("--skip-prune", action="store_true", help="Use retained W12 evidence; run W10, GGUF and NumPy")
    cases.add_argument("--gguf-only", action="store_true", help="Run only the GGUF comparison without scratch copies")
    args = parser.parse_args()
    args.root, args.bins = args.root.resolve(), args.bins.resolve()
    w8.require(w8.ARB in args.root.parents, "Evidence root must be under /arboretum")
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True)
    args.web_dist = w8.ARB / "w8-accept/web"
    pins = {name: w8.sha(args.bins / name) for name in (
        "brewery", "brewery-dispatch", "brewery-worker", "brewery-engine",
    )}
    if not args.numpy_only:
        w8.require(w8.sha(w8.SAFE) == w8.PINS[w8.SAFE], "SafeTensors pin mismatch")
        w8.require(w8.sha(Q4) == EXPECTED["q4-k-m"], "Q4_K_M pin mismatch")
    with tempfile.TemporaryDirectory(dir=args.root, prefix="isolated-") as name:
        run = Path(name)
        env = w8.setup(args, run)
        log = evidence / "controller.log"
        with log.open("w") as out, subprocess.Popen(
            [args.bins / "brewery"], env=env, cwd=w8.REPO,
            stdin=subprocess.DEVNULL, stdout=out, stderr=out,
        ) as controller:
            try:
                for _ in range(200):
                    w8.require(controller.poll() is None, "Controller exited; see controller.log")
                    match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$", log.read_text(), re.M)
                    if match:
                        w8.require(int(match[2]) not in {0, 3000, 4095, 4141}, "Unsafe controller port")
                        if args.numpy_only:
                            agent = Agent(match[1], evidence)
                            agent.post("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                                                              "revision": 0, "enabled": True})
                            numpy_fixture(agent, run / "assets", evidence)
                        else:
                            exercise(match[1], evidence, pins, run, args.skip_prune, args.gguf_only)
                        break
                    time.sleep(0.1)
                else:
                    raise TimeoutError("Controller did not listen")
            finally:
                controller.terminate()
                try:
                    controller.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    controller.kill()
                    controller.wait()


if __name__ == "__main__":
    main()
