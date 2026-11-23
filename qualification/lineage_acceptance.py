#!/usr/bin/env python3
"""Check X8 against pinned Qwen BF16 on an isolated loopback controller."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.request import Request

import weight_space_acceptance as w8
from weight_space_pruning_acceptance import Agent, PERCENTAGES


def require_space(copies):
    free = shutil.disk_usage(w8.ARB).free
    needed = copies * w8.SAFE.stat().st_size + (128 << 20)
    w8.require(free >= needed, f"Need {needed} bytes free, have {free}")


def unrelated_fixture(path):
    import numpy as np

    # Keep all of Qwen's names, shapes and BF16 storage. Draw independent
    # small random weights; do not copy any trained weight bytes.
    rng = np.random.default_rng(806)
    with w8.SAFE.open("rb") as source:
        start, entries = w8.safetensors_header(source)
        source.seek(0)
        header = source.read(start)
    with path.open("wb") as out:
        out.write(header)
        for tensor in sorted((t for k, t in entries.items() if k != "__metadata__"),
                             key=lambda t: t["data_offsets"][0]):
            w8.require(tensor["dtype"] == "BF16", "Unexpected fixture dtype")
            lo, hi = tensor["data_offsets"]
            for i in range(0, (hi - lo) // 2, 1 << 19):
                values = rng.standard_normal(min(1 << 19, (hi - lo) // 2 - i), dtype=np.float32)
                values *= 0.02
                # Round float32 to nearest-even BF16.
                bits = values.view(np.uint32)
                rounded = bits + 0x7fff + ((bits >> 16) & 1)
                out.write((rounded >> 16).astype("<u2").tobytes())


def exercise(agent, run, cases, bins):
    agent.post("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                                     "revision": 0, "enabled": True})
    assets = run / "assets"
    pin = w8.PINS[w8.SAFE]

    def library(label, path):
        return agent.write("register_model", label, host="local", kind="safetensors",
                           path=str(path), name=label)

    def discard(label, row):
        agent.write("discard_scratch", "discard-" + label, id=row["id"])
        w8.require(not scratch_file(row).exists(), "Scratch payload still exists")

    def scratch_file(row):
        return run / "state" / "scratch" / row["id"] / "model.safetensors"

    def root(label):
        return agent.write("create_scratch", label + "-root", parent_id=base["id"], parent_sha256=pin)

    def expose(label, row):
        # Snapshot the immutable scratch bytes by hard link into the isolated
        # test directory. Do not bypass scratch admission in the product.
        path = assets / f"{label}.safetensors"
        os.link(scratch_file(row), path)
        return library(label, path), path, row["sha256"]

    def remove(snapshot):
        model, path, _ = snapshot
        request = Request(f"{agent.base}/api/models/{model['id']}", method="DELETE")
        with agent.client.open(request, timeout=10) as response:
            w8.require(response.status == 204, "Did not remove isolated Library entry")
        path.unlink()

    base = library("qwen-base", w8.SAFE)
    report = {"qualification": w8.QUALIFICATION, "base_sha256": pin, "bin_sha256": bins, "cases": {}}

    def check(label, subject, kind, *parents):
        model, _, digest = subject
        refs = {"candidate_1": base["id"]}
        refs.update({f"candidate_{i}": m["id"] for i, (m, _, _) in enumerate(parents, 2)})
        before = agent.call("list_models")
        begin = time.monotonic()
        answer = agent.call("lineage", id=model["id"], **refs)
        duration = time.monotonic() - begin
        w8.save(agent.evidence, label + "-lineage", answer)
        w8.require(agent.call("list_models") == before, "Lineage changed Library records")
        w8.require(answer["target"]["sha256"] == digest, f"{label}: target hash")
        matches = answer["most_likely"]
        candidate = next(c for c in answer["candidates"] if c["id"] == base["id"])
        w8.require(candidate["checkpoint"]["sha256"] == pin, f"{label}: base hash")
        if kind is None:
            w8.require(matches is None and candidate["status"] == "no_match", f"{label}: false link")
        else:
            w8.require(matches and matches["base_id"] == base["id"] and matches["kind"] == kind,
                       f"{label}: wrong base or relation: {matches}")
        row = candidate["evidence"]
        merge = row["merge"]
        if kind == "merge":
            w8.require(merge["parent_ids"] == [m["id"] for m, _, _ in parents]
                       and all(abs(x - y) < 0.02 for x, y in zip(merge["weights"], (0.5, 0.75)))
                       and merge["relative_residual"] < 0.05, "Expected W10 parent fit")
        report["cases"][label] = {
            "elapsed_seconds": duration, "target_sha256": digest, "match": matches,
            "changed_fraction": row["changed_fraction"], "relative_delta": row["relative_delta"],
            "changed_to_zero_fraction": row["changed_to_zero_fraction"],
            "rank_one_energy_fraction": row["rank_one_energy_fraction"], "merge": merge,
        }
        w8.save(agent.evidence, "report", report)

    if "rome-style" in cases:
        require_space(2)
        parent = root("rome-style")
        u = [0.] * 896
        v = [0.] * 4864
        u[10:14] = [1., -0.5, 0.75, -1.25]
        v[17:25] = [0.5, 1., -1., 0.75, -0.5, 0.25, 0.75, 1.]
        child = agent.write("apply_edit", "rome-style-scratch", id=parent["id"], operations=[{
            "op": "low_rank", "tensor": "model.layers.8.mlp.down_proj.weight",
            "u": u, "v": v, "scale": 0.125,
        }])
        snapshot = expose("rome-style", child)
        discard("rome-style-child", child)
        discard("rome-style-parent", parent)
        check("rome-style", snapshot, "rank_one_edit")
        remove(snapshot)

    if "w12-pruning" in cases:
        require_space(2)
        parent = root("w12")
        child = agent.write("prune_model", "w12-scratch", id=parent["id"], sha256=parent["sha256"],
                            pruning=PERCENTAGES["magnitude-30"])
        snapshot = expose("w12-pruning", child)
        discard("w12-child", child)
        discard("w12-parent", parent)
        check("w12-pruning", snapshot, "pruning")
        remove(snapshot)

    if "w10-merge" in cases:
        require_space(4)
        parents = []
        for i in range(2):
            label = f"w10-parent-{i}"
            parent = root(label)
            child = agent.write("apply_edit", label + "-scratch", id=parent["id"], operations=[{
                "op": "add", "tensor": "model.layers.8.self_attn.q_proj.bias",
                "origin": [i * 64], "shape": [640],
                # Use a signal above the BF16 rounding error of the merge.
                "values": [(((j * (3 + i) + i) % 17) - 8) / 16 for j in range(640)],
            }])
            parents.append(expose(label, child))
            discard(label + "-edit", child)
            discard(label + "-root", parent)
        child = agent.write("merge_models", "w10-scratch", base_id=base["id"], base_sha256=pin,
                            algorithm={"method": "task_arithmetic"},
                            inputs=[{"id": m["id"], "sha256": digest, "weight": weight}
                                    for (m, _, digest), weight in zip(parents, (0.5, 0.75))])
        snapshot = expose("w10-merge", child)
        discard("w10-merge", child)
        check("w10-merge", snapshot, "merge", *parents)
        for source in [snapshot, *parents]:
            remove(source)

    if "unrelated" in cases:
        require_space(1)
        path = assets / "synthetic-unrelated.safetensors"
        unrelated_fixture(path)
        snapshot = library("synthetic-unrelated", path), path, w8.sha(path)
        check("unrelated", snapshot, None)
        remove(snapshot)

    w8.require(not list(assets.glob("*.safetensors")), "Large Library snapshots remain")
    w8.require(not list((run / "state" / "scratch").glob("*/model.safetensors")),
               "Large scratch checkpoints remain")
    report["status"] = "passed"
    w8.save(agent.evidence, "report", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="New evidence root under /arboretum")
    parser.add_argument("--bins", type=Path, required=True, help="Copied Brewery binaries")
    parser.add_argument("--web-dist", type=Path, required=True, help="Built workbench")
    cases = ("rome-style", "w12-pruning", "w10-merge", "unrelated")
    parser.add_argument("--case", choices=cases, action="append")
    args = parser.parse_args()
    args.root, args.bins, args.web_dist = (p.resolve() for p in (args.root, args.bins, args.web_dist))
    w8.require(w8.ARB in args.root.parents, "Evidence root must be under /arboretum")
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True)
    w8.require(w8.sha(w8.SAFE) == w8.PINS[w8.SAFE], "Qwen SafeTensors hash mismatch")
    bins = {name: w8.sha(args.bins / name) for name in (
        "brewery", "brewery-dispatch", "brewery-worker", "brewery-engine",
    )}
    with tempfile.TemporaryDirectory(dir=args.root, prefix="isolated-") as name:
        run = Path(name)
        log = evidence / "controller.log"
        with log.open("w") as out, subprocess.Popen(
            [args.bins / "brewery"], env=w8.setup(args, run), cwd=w8.REPO,
            stdin=subprocess.DEVNULL, stdout=out, stderr=out,
        ) as controller:
            try:
                for _ in range(200):
                    w8.require(controller.poll() is None, "Controller exited; see controller.log")
                    match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$", log.read_text(), re.M)
                    if match:
                        w8.require(int(match[2]) not in {0, 3000, 4095, 4141}, "Unsafe controller port")
                        exercise(Agent(match[1], evidence), run, args.case or cases, bins)
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
