"""Repeat W12 magnitude-30 on an isolated CPU stack and verify the X1 report."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.parse import urlencode

from weight_space_acceptance import ARB, PINS, REPO, SAFE, W0_LIBS, require, save, setup, sha
from weight_space_pruning_acceptance import Agent, PERCENTAGES


def compare_rest_mcp(rest, mcp, path="$"):
    if isinstance(rest, dict) and isinstance(mcp, dict) and rest.keys() == mcp.keys():
        for key in rest:
            compare_rest_mcp(rest[key], mcp[key], f"{path}.{key}")
    elif isinstance(rest, list) and isinstance(mcp, list) and len(rest) == len(mcp):
        for index, (left, right) in enumerate(zip(rest, mcp, strict=True)):
            compare_rest_mcp(left, right, f"{path}[{index}]")
    elif isinstance(rest, float) and isinstance(mcp, float):
        # MCP parses and reserializes the REST body. That can round a float.
        require(math.isclose(rest, mcp, rel_tol=1e-14), f"REST/MCP differ at {path}: {rest} != {mcp}")
    else:
        require(rest == mcp, f"REST/MCP differ at {path}: {rest} != {mcp}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("bins", "runner", "web-dist", "evidence", "shots", "w12"):
        parser.add_argument(f"--{arg}", type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    require((args.web_dist / "index.html").is_file(), "build the workbench before starting the replay")
    previous = json.loads((args.w12 / "magnitude-30-repeat.json").read_text())
    w12 = json.loads((args.w12 / "report.json").read_text())
    require(w12["dense_scope"]["layer"] == 8, "unexpected W12 scope")
    for path, digest in w12["llama_cpp_runtime_pins"].items():
        require(sha(Path(path)) == digest, f"incorrect runtime pin: {path}")
    args.evidence.mkdir(parents=True, exist_ok=True)
    args.shots.mkdir(parents=True, exist_ok=True)
    binaries = {path.name: sha(path) for path in args.bins.glob("brewery*")}
    runner_sha = sha(args.runner)
    with tempfile.TemporaryDirectory(prefix="x1-", dir=ARB / "tmp") as tmp:
        run = Path(tmp)
        env = setup(args, run)
        worker = json.loads((run / "worker.json").read_text())
        worker["engine_env"].update(
            BREWERY_SCRATCH_CHECKS_RUNNER=str(args.runner),
            BREWERY_SCRATCH_CHECKS_RUNNER_SHA256=runner_sha,
            BREWERY_SCRATCH_CHECKS_RUNNER_LIB_DIR=str(W0_LIBS),
        )
        # Keep the worker config mode from setup().
        (run / "worker.json").write_text(json.dumps(worker))
        log = args.evidence / "controller.log"
        with log.open("w") as stream, subprocess.Popen(
            [args.bins / "brewery"], env=env, cwd=REPO, stdout=stream, stderr=stream,
        ) as controller:
            try:
                for _ in range(200):
                    require(controller.poll() is None, "controller exited")
                    match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$",
                                      log.read_text(), re.M)
                    if match:
                        break
                    time.sleep(0.1)
                require(match and int(match[2]) not in {0, 3000, 4095, 4141}, "unsafe port")
                agent = Agent(match[1], args.evidence)
                agent.post("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                                                 "enabled": True, "revision": 0})
                model = agent.write("register_model", "model", kind="safetensors", host="local",
                                    path=str(SAFE), name="Qwen2.5-0.5B-Instruct")
                assets = agent.write("register_model", "assets", kind="assets", host="local",
                                     path=str(run / "assets"))
                parent = agent.write("create_scratch", "parent", parent_id=model["id"],
                                     parent_sha256=PINS[SAFE])
                pg = agent.convert(parent, assets["id"], "parent")
                child = agent.write("prune_model", "child", id=parent["id"], sha256=parent["sha256"],
                                    pruning=PERCENTAGES["magnitude-30"])
                cg = agent.convert(child, assets["id"], "child")
                require(pg["sha256"] == previous["parent_gguf_sha256"]
                        and cg["sha256"] == previous["child_gguf_sha256"]
                        and child["sha256"] == previous["child_safetensors_sha256"], "W12 pin mismatch")
                check = agent.write("run_checks", "checks-queued", id=child["id"], sha256=child["sha256"])
                bundle = agent.wait(check, "checks-job")["check_bundle"]
                report = bundle["measured_report"]
                ppl = next(m for m in report["metrics"] if m["name"] == "perplexity")
                for role in ("parent", "child"):
                    require(math.isclose(ppl[role], previous[f"{role}_perplexity"], rel_tol=1e-9),
                            f"W12 {role} perplexity mismatch")
                pair = {"a": parent["id"], "b": child["id"]}
                mcp = agent.call("read_token_loss_diff", **pair)
                with agent.client.open(agent.base + "/api/diff/token-loss?" + urlencode(pair)) as response:
                    diff = json.load(response)
                save(args.evidence, "token-loss-diff", diff)
                save(args.evidence, "token-loss-mcp", mcp)
                compare_rest_mcp(diff, mcp)
                digest = sha(run / "state" / "scratch" / child["id"] / "token-loss.v1.json")
                require(digest == bundle["token_loss"]["sha256"] == diff["artifact_sha256"]
                        and diff["check_bundle_id"] == bundle["id"]
                        and diff["heldout_sha256"] == previous["heldout_split_sha256"], "evidence pin mismatch")
                require(diff["total_tokens"] == previous["predicted_tokens"]
                        and len(diff["sentences"]) == previous["sentences"], "heldout coverage differs")
                for entry in diff["sentences"]:
                    expected = previous["per_sentence_logprob_delta"][entry["index"]]
                    require(math.isclose(entry["logprob_gain"], expected, abs_tol=1e-8),
                            f"sentence {entry['index']} differs")
                lead = diff["sentences"][0]
                require(lead["index"] == diff["sentence"] == 5
                        and round(100 * lead["share_of_net_gain"], 1) == 52.5, "dominant sentence differs")
                save(args.evidence, "report", {
                    "qualification": "not release-qualified", "model": "Qwen2.5-0.5B-Instruct",
                    "pruning": PERCENTAGES["magnitude-30"], "binaries": binaries, "runner_sha256": runner_sha,
                    "parent_gguf_sha256": pg["sha256"], "child_gguf_sha256": cg["sha256"],
                    "perplexity": ppl, "total_tokens": diff["total_tokens"],
                    "total_logprob_gain": diff["total_logprob_gain"], "largest_contributor": lead,
                    "artifact_sha256": digest, "heldout_sha256": diff["heldout_sha256"],
                    "rest_matches_mcp": True, "matches_w12_per_sentence": True,
                })
                subprocess.run(["node", REPO / "qualification/token_loss_acceptance.mjs",
                                agent.base, child["id"], args.shots], check=True)
                agent.write("discard_scratch", "child-discard", id=child["id"])
                agent.write("discard_scratch", "parent-discard", id=parent["id"])
            finally:
                controller.terminate()
                controller.wait(timeout=20)


if __name__ == "__main__":
    main()
