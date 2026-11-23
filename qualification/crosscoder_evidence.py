#!/usr/bin/env python3
"""Check X6 REST/MCP parity and measure held-out feature contributions."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_crosscoder import codes, controller, load_pairs, require


DIFFERENT = {"parent_unique", "child_unique", "changed"}


def contributions(rows, contexts, anchors, weights, report):
    results = []
    scale = report["scale"]
    changed = torch.tensor([f["classification"] in DIFFERENT for f in report["features"]])
    directions = (weights["W_dec"][1] - weights["W_dec"][0]) / scale
    bias = (weights["b_dec"][1] - weights["b_dec"][0]) / scale
    active = codes(rows, weights, scale, report["k"])
    for target in anchors:
        indices = [i for i, row in enumerate(contexts)
                   if (row["prompt"], row["position"]) == (target["prompt"], target["position"])]
        require(len(indices) == 1, f"Missing or ambiguous anchor: {target}")
        index = indices[0]
        z = active[index]
        delta = rows[index, 1] - rows[index, 0]
        norm = float(delta.norm())
        if norm < 1e-8:
            results.append(target | {"activation_delta_l2": norm, "top_features": []})
            continue
        projection = z * (directions @ (delta / norm))
        ranked = projection.topk(min(8, len(projection))).indices.tolist()
        results.append(target | {
            "token_id": contexts[index]["token_id"],
            "activation_delta_l2": norm,
            "reconstruction_delta_cosine": float(torch.nn.functional.cosine_similarity(
                (z @ directions + bias)[None], delta[None])),
            "bias_projected_delta": float(bias @ (delta / norm)),
            "changed_or_unique_projected_delta": float(projection[changed].sum()),
            "top_features": [
                {"feature": i, "classification": report["features"][i]["classification"],
                 "activation": float(z[i]), "projected_delta": float(projection[i]),
                 "measured_delta_fraction": float(projection[i]) / norm}
                for i in ranked if projection[i] > 0
            ],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("spec", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--anchors", type=Path, help="Array of {label, prompt, position}; label edited fact rows 'edited-fact*'")
    args = parser.parse_args()
    get = controller(args.base)
    torch.set_num_threads(4)
    weights = args.artifact / "crosscoder.safetensors"
    metadata = args.artifact / "metadata.json"
    started = time.monotonic()
    report = json.loads(subprocess.check_output([
        "curl", "--noproxy", "*", "--fail-with-body", "--silent", "--show-error", "--max-time", "120",
        "-H", f"Origin: {args.base}", "-F", f"metadata=@{metadata};type=application/json",
        "-F", f"weights=@{weights};type=application/octet-stream", args.base + "/api/crosscoders",
    ]))
    artifact_id = report["id"]
    require(report == get(f"/api/crosscoders/{artifact_id}"), "REST import/read mismatch")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    require(digest == report["weights_sha256"], "Report weight SHA-256 mismatch")
    exported = subprocess.check_output([
        "curl", "--noproxy", "*", "--fail", "--silent", "--show-error", "--max-time", "120",
        args.base + f"/api/crosscoders/{artifact_id}/safetensors",
    ])
    require(hashlib.sha256(exported).hexdigest() == digest, "Export weight SHA-256 mismatch")
    rpc = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "crosscoder_diff", "arguments": {"id": artifact_id}}}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(args.base + "/mcp", data=json.dumps(rpc).encode(), headers={
        "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": "2025-06-18",
    })
    with opener.open(req, timeout=30) as response:
        mcp = json.load(response)["result"]
    require(not mcp["isError"] and mcp["structuredContent"] == report, "MCP and REST results differ")
    summary = {k: report[k] for k in ("id", "qualification", "version", "method", "layer", "role",
                                     "parent", "child", "scratch_revision_sha256", "weights_sha256",
                                     "training", "metrics", "manifests")}
    summary["counts"] = dict(Counter(f["classification"] for f in report["features"]))
    summary["rest_mcp_export_parity_seconds"] = time.monotonic() - started
    if args.anchors:
        data, bindings, contexts = load_pairs(get, json.loads(args.spec.read_text()))
        require(all(report[side] == binding for side, binding in bindings.items()), "Evidence binding differs")
        anchors = json.loads(args.anchors.read_text())
        summary["heldout_positions"] = contributions(
            data["heldout"], contexts["heldout"], anchors, load_file(weights), report)
    args.evidence.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    require(all(m["explained_variance"] > 0.5 for m in report["metrics"].values()),
            "Held-out explained variance did not exceed 0.5 on both sides")
    if args.anchors:
        targets = [row for row in summary["heldout_positions"] if row["label"].startswith("edited-fact")]
        require(targets and any(f["classification"] in DIFFERENT for row in targets for f in row["top_features"][:5]),
                "No changed or unique crosscoder feature among the five strongest positive target contributions")
    print(json.dumps({"id": artifact_id, "metrics": summary["metrics"], "counts": summary["counts"]}))


if __name__ == "__main__":
    main()
