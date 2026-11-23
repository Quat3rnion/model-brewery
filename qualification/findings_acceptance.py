#!/usr/bin/env python3
"""Record and replay X5 and X1 evidence through MCP on an isolated CPU stack."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

from circuit_acceptance import MANIFEST_SHA
from weight_space_acceptance import ARB, BASE, PINS, REPO, SAFE, TENSOR, W0_LIBS, require, save, setup, sha
from weight_space_pruning_acceptance import Agent

CLIENT = "X10 CPU acceptance"
PIN = ARB / "pins/lanternfish-x3-fa0e8c5"
RUNNER = ARB / "pins/scratch-check-logits-x1/scratch-check-logits"
RUNNER_SHA = "3a3d3afa7100460c59adbf661737352211b13fb302210e6663ebc080a6e29fd7"
SCRATCH = Path("/mnt/games/model-brewery/xa-scratch/x10")


def exercise(agent, assets_path, prior):
    agent.post("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                                     "enabled": True, "revision": 0})
    init = agent.post("/mcp", {"jsonrpc": "2.0", "id": "init", "method": "initialize",
                             "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                        "clientInfo": {"name": CLIENT, "version": "1"}}})
    require(init["result"]["protocolVersion"] == "2025-06-18", init)
    require(not agent.call("list_models")["models"], "isolated Library is not empty")
    write = agent.write
    model = write("register_model", "source", kind="safetensors", host="local", path=str(SAFE))
    assets = write("register_model", "assets", kind="assets", host="local", path=str(assets_path))
    base = write("create_scratch", "base", parent_id=model["id"], parent_sha256=PINS[SAFE])
    gguf = agent.convert(base, assets["id"], "base")
    initial = write("create_finding", "initial", client_id=CLIENT, model_id=base["id"], model_sha256=base["sha256"],
                    hypothesis="The Paris/Berlin contrast has a positive clean and negative corrupt margin; "
                    "a small layer 8 prune does not increase heldout mean token loss by 0.01 or more.")
    options = {**prior["options"], "model_sha256": gguf["sha256"],
               "layers": [23], "positions": [3, 4], "roles": ["l_out", "ffn_out"],
               "top_k": 3, "sample_lower": 1}
    circuit = write("find_circuit", "circuit-queued", input_ids=[base["id"]], options=options)
    c = agent.wait(circuit, "circuit-job")
    observed = write("add_evidence", "with-circuit", id=initial["id"], client_id=CLIENT,
                     evidence={"kind": "job", "job_id": circuit["id"],
                               "paths": ["/clean_metric", "/corrupt_metric", "/pearson_r"]})
    pruning = {"mode": "unstructured", "basis_points": 1000,
               "tensors": ["model.layers.8.mlp.down_proj.weight"]}
    child = write("prune_model", "prune", id=base["id"], sha256=base["sha256"], pruning=pruning)
    agent.convert(child, assets["id"], "prune")
    checks = write("run_checks", "checks-queued", id=child["id"], sha256=child["sha256"])
    bundle = agent.wait(checks, "checks-job")["check_bundle"]
    require(all(bundle[stage]["status"] == "passed" for stage in ("integrity", "smoke", "behavior")), bundle)
    diff = agent.call("read_token_loss_diff", a=base["id"], b=child["id"], sentence=5)
    save(agent.evidence, "token-loss", diff)
    delta = -diff["total_logprob_gain"] / diff["total_tokens"]
    support = c["clean_metric"] > 0 > c["corrupt_metric"] and delta < 0.01
    original = write("add_evidence", "original", id=observed["id"], client_id=CLIENT,
                     evidence={"kind": "token_loss", "a": base["id"], "b": child["id"], "sentence": 5,
                               "paths": ["/total_tokens", "/total_logprob_gain"]},
                     verdict={"status": "supported" if support else "refuted",
                              "reasoning": f"The clean/corrupt margins are {c['clean_metric']:.6g} and "
                                           f"{c['corrupt_metric']:.6g}. Mean token loss change is {delta:.6g}."})
    require(original["experiments"][1]["artifact"]["sha256"] == diff["artifact_sha256"], original)

    contributors = []
    for i, scale in enumerate((0.0625, -0.046875)):
        row = write("apply_edit", f"contributor-{i}", id=base["id"], operations=[
            {"op": "add", "tensor": TENSOR, "origin": [i * 64], "shape": [64], "values": [scale] * 64}
        ])
        contributors.append({"id": row["id"], "sha256": row["sha256"], "weight": 0.75 - i * 0.25})
    merged = write("merge_models", "merged", base_id=base["id"], base_sha256=base["sha256"],
                   algorithm={"method": "task_arithmetic"}, inputs=contributors)
    require(merged["sha256"] != base["sha256"], "merge did not change weights")
    for i, contributor in enumerate(contributors):
        write("discard_scratch", f"discard-contributor-{i}", id=contributor["id"])
    merged_gguf = agent.convert(merged, assets["id"], "merged")
    replay = write("replay_finding", "running-replay", id=original["id"], client_id=CLIENT,
                   model_id=merged["id"], model_sha256=merged["sha256"])
    deadline = time.monotonic() + 1200
    while replay["state"] == "running" and time.monotonic() < deadline:
        time.sleep(2)
        replay = agent.call("get_finding", id=replay["id"])
    save(agent.evidence, "replay", replay)
    require(replay["state"] == "ready", replay)
    require(replay["parent_id"] == replay["replay_of"] == original["id"]
            and replay["version"] == original["version"] + 1
            and replay["model"]["sha256"] == merged["sha256"], "replay link or model mismatch")
    require(agent.call("get_finding", id=original["id"]) == original, "original finding changed")
    old, new = original["experiments"], replay["experiments"]
    require(len(old) == len(new) == 2, "not all experiments replayed")
    require(all(l["artifact"]["job_id"] != r["artifact"]["job_id"] for l, r in zip(old, new, strict=True)),
            "the replay reused an old job")
    require(new[0]["spec"]["request"]["options"] == {**options, "model_sha256": merged_gguf["sha256"]},
            "circuit spec differs beyond model rebinding")
    require(new[1]["spec"]["preparation"] == old[1]["spec"]["preparation"], "prune recipe changed")
    require(new[0]["artifact"]["pins"] == old[0]["artifact"]["pins"], "circuit runtime changed")
    for key in ("heldout_sha256", "fixture_sha256", "runner_sha256", "converter_version"):
        require(new[1]["artifact"]["pins"][key] == old[1]["artifact"]["pins"][key], f"{key} changed")
    comparison = replay["comparison"]
    require(len(comparison) == 5 and all(isinstance(row["replay"], (float, int)) for row in comparison), comparison)
    counts = next(row for row in comparison if row["path"] == "/total_tokens")
    require(counts["original"] == counts["replay"] and counts["replay"] > 0, counts)
    require(len(agent.call("list_findings")["findings"]) == 4, "missing immutable versions")
    return {"status": "passed", "qualification": "not release-qualified",
            "original": original["id"], "replay": replay["id"], "comparison": comparison,
            "original_sha256": base["sha256"], "merged_sha256": merged["sha256"],
            "original_gguf_sha256": gguf["sha256"], "merged_gguf_sha256": merged_gguf["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("bins", "evidence", "web-dist", "x5-evidence"):
        parser.add_argument(f"--{option}", type=lambda path: Path(path).resolve(), required=True)
    args = parser.parse_args()
    require(sha(SAFE) == PINS[SAFE] and sha(BASE) == PINS[BASE], "model pin mismatch")
    require(sha(PIN / "SHA256SUMS") == MANIFEST_SHA and sha(RUNNER) == RUNNER_SHA, "runtime pin mismatch")
    subprocess.run(["sha256sum", "--quiet", "-c", ARB / "pins/lanternfish-0ea30b3.libs.sha256"], check=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    prior = json.loads((args.x5_evidence / "job.json").read_text())
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="stack-", dir=SCRATCH) as tmp:
        run = Path(tmp)
        env = setup(args, run)
        env["TMPDIR"] = str(run)
        config = run / "worker.json"
        worker = json.loads(config.read_text())
        worker["engine_env"].update(
            BREWERY_LANTERNFISH_SENSITIVITY_PIN=str(PIN),
            BREWERY_LANTERNFISH_SENSITIVITY_MANIFEST_SHA256=MANIFEST_SHA,
            BREWERY_SCRATCH_CHECKS_RUNNER=str(RUNNER),
            BREWERY_SCRATCH_CHECKS_RUNNER_SHA256=RUNNER_SHA,
            BREWERY_SCRATCH_CHECKS_RUNNER_LIB_DIR=str(W0_LIBS),
            BREWERY_CPU_THREADS="4",
        )
        config.write_text(json.dumps(worker))
        config = run / "dispatch.json"
        dispatch = json.loads(config.read_text())
        dispatch["hosts"]["local"]["capabilities"]["circuit"] = True
        config.write_text(json.dumps(dispatch))
        log = args.evidence / "controller.log"
        with log.open("w") as stream, subprocess.Popen(
            [args.bins / "brewery"], env=env, cwd=REPO, stdout=stream, stderr=stream,
        ) as controller:
            try:
                match = None
                for _ in range(200):
                    require(controller.poll() is None, "controller exited")
                    match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$", log.read_text(), re.M)
                    if match:
                        break
                    time.sleep(0.1)
                require(match and int(match[2]) not in {0, 3000, 4095, 4141}, "unsafe port")
                report = exercise(Agent(match[1], args.evidence), run / "assets", prior)
                report["binary_sha256"] = {path.name: sha(path) for path in args.bins.glob("brewery*")}
                report["elapsed_seconds"] = round(time.monotonic() - start, 1)
                save(args.evidence, "report", report)
            except Exception:
                for worker in [run / "worker/jobs", run / "state/local-edit/local/worker/jobs"]:
                    for path in worker.glob("*/*.log"):
                        with path.open("rb") as source:
                            source.seek(max(0, path.stat().st_size - 16384))
                            (args.evidence / f"{path.parent.name}-{path.name}").write_bytes(source.read(16384))
                raise
            finally:
                controller.terminate()
                controller.wait(timeout=20)


if __name__ == "__main__":
    main()
