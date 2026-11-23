#!/usr/bin/env python3
"""Exercise W12 through MCP on a new loopback-only CPU controller."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.request import ProxyHandler, Request, build_opener

from weight_space_acceptance import (
    ARB, ASSETS, HF, LANTERN, PINS, REPO, RUNNER, SAFE, SNAPSHOT_PIN,
    require, save, setup, sha,
)
from test_expert_gguf import fixture, fixture_bytes


LAYER = 8
CALIBRATION = "A prism splits the light into a rainbow."
POSITIONS = list(range(6))
TRACE = {"layers": [LAYER], "roles": ["ffn_inp", "l_out"], "positions": POSITIONS, "top_k": 20}
PERCENTAGES = {
    "magnitude-10": {
        "mode": "unstructured", "basis_points": 1000,
        "tensors": [f"model.layers.{LAYER}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")],
    },
    "magnitude-30": {
        "mode": "unstructured", "basis_points": 3000,
        "tensors": [f"model.layers.{LAYER}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")],
    },
    "neurons": {"mode": "mlp_neurons", "basis_points": 3000, "layer": LAYER},
    "heads": {"mode": "attention_heads", "basis_points": 3000, "layer": LAYER, "heads": 14},
}


class Agent:
    def __init__(self, base, evidence):
        self.base = base
        self.evidence = evidence
        self.client = build_opener(ProxyHandler({}))
        self.serial = 0

    def post(self, path, body):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        request = Request(self.base + path, json.dumps(body).encode(), headers=headers)
        with self.client.open(request, timeout=120) as response:
            return json.load(response)

    def reply(self, tool, **args):
        self.serial += 1
        return self.post("/mcp", {"jsonrpc": "2.0", "method": "tools/call", "id": self.serial,
                                 "params": {"name": tool, "arguments": args}})

    def call(self, tool, **args):
        reply = self.reply(tool, **args)
        result = reply.get("result", {})
        require(result.get("isError") is False, f"{tool}: {reply}")
        return result["structuredContent"]

    def write(self, tool, label, **args):
        result = self.call(tool, idempotency_key=f"w12-{label}", **args)
        save(self.evidence, label, result)
        print(f"{tool}: {label}", flush=True)
        return result

    def wait(self, job, label):
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            state = self.call("get_job", id=job["id"], wait_ms=5000)
            if state["state"] not in ("queued", "running"):
                save(self.evidence, label, state)
                require(state["state"] == "succeeded", f"{label}: {state.get('error')}")
                return state["result"]
        raise TimeoutError(label)

    def convert(self, row, assets, label):
        free = shutil.disk_usage(self.evidence).free
        require(free >= 3 * (1 << 30), f"{label}: less than 3 GiB free before GGUF conversion")
        job = self.write("convert_scratch_gguf", f"{label}-gguf-queued",
                         id=row["id"], sha256=row["sha256"], assets_id=assets)
        gguf = self.wait(job, f"{label}-gguf-job")["gguf"]
        require(gguf["revision_sha256"] == row["sha256"], "private GGUF pin mismatch")
        return gguf

    def trace(self, row, gguf, label):
        job = self.write("start_trace", f"{label}-trace-queued", input_ids=[row["id"]],
                         prompt=CALIBRATION, options=TRACE)
        result = self.wait(job, f"{label}-trace-job")
        require(result["model_sha256"] == gguf["sha256"] and result["executable_sha256"] == PINS[LANTERN],
                "trace pin mismatch")
        return job["id"]


def dense(agent, run):
    before_models = agent.call("list_models")["models"]
    model = agent.write("register_model", "safe-library", host="local", kind="safetensors",
                        path=str(SAFE), name="Pinned W12 Qwen2.5-0.5B-Instruct")
    assets = agent.write("register_model", "assets-library", host="local", kind="assets",
                         path=str(run / "assets"))
    root = agent.write("create_scratch", "parent", parent_id=model["id"], parent_sha256=PINS[SAFE])
    parent_gguf = agent.convert(root, assets["id"], "parent")
    parent_trace = agent.trace(root, parent_gguf, "parent")
    capture = agent.call("read_trace", id=parent_trace, view="tensor", role="ffn_inp", layer=LAYER,
                         start=0, end=len(POSITIONS))
    require(len(capture["rows"]) == len(POSITIONS) and capture["missing_positions"] == [],
            "expected dense calibration coverage")
    plans = PERCENTAGES | {"wanda": {
        "mode": "wanda_mlp", "basis_points": 3000, "layer": LAYER, "rms_eps": 1e-6,
        "trace": {"id": parent_trace, "prompt": 0, "start": 0, "end": len(POSITIONS)},
    }}
    cases = []
    for label, plan in plans.items():
        child = agent.write("prune_model", label, id=root["id"], sha256=root["sha256"], pruning=plan)
        require(child["provenance"]["pruning"] == plan and child["state"] == "scratch"
                and child["qualification"] == "not release-qualified", "prune provenance mismatch")
        if label == "wanda":
            require(child["provenance"]["pruning_trace"]["gguf_sha256"] == parent_gguf["sha256"],
                    "Wanda used a different GGUF")
        tensor = agent.call("diff_tensors", a=root["id"], b=child["id"])
        require(len(tensor["tensors"]) == (3 if label.startswith("magnitude-") else 1), "wrong tensor count")
        save(agent.evidence, f"{label}-diff-tensors", tensor)
        child_gguf = agent.convert(child, assets["id"], label)
        child_trace = agent.trace(child, child_gguf, label)
        traces = agent.call("diff_traces", a=parent_trace, b=child_trace, layer=LAYER)
        require(traces["activations"] and traces["model_logits"], "missing trace diff")
        save(agent.evidence, f"{label}-diff-traces", traces)
        job = agent.write("run_checks", f"{label}-checks-queued", id=child["id"], sha256=child["sha256"])
        agent.wait(job, f"{label}-checks-job")
        checks = agent.call("get_checks", id=child["id"])
        save(agent.evidence, f"{label}-checks", checks)
        measured = checks["measured_report"]
        statuses = {key: checks[key]["status"] for key in ("integrity", "smoke", "behavior")}
        require(set(statuses.values()) == {"passed"} and len(measured["metrics"]) == 17
                and measured["runner_sha256"] == PINS[RUNNER]
                and measured["parent_gguf_sha256"] == parent_gguf["sha256"]
                and measured["revision_gguf_sha256"] == child_gguf["sha256"], "W4 checks did not pass")
        behavior = agent.call("diff_behavior", a=root["id"], b=child["id"])
        require(len(behavior["metrics"]) == 17, "missing behavior diff")
        save(agent.evidence, f"{label}-diff-behavior", behavior)
        perplexity, = [v for v in behavior["metrics"] if v["name"] == "perplexity"]
        require(all(math.isfinite(perplexity[k]) for k in ("parent", "child")), "invalid perplexity")
        cases.append({"label": label, "plan": plan, "id": child["id"], "sha256": child["sha256"],
                      "check_status": statuses, "perplexity": perplexity, "delta": child["delta"]})
        agent.write("discard_scratch", f"{label}-discard", id=child["id"])
    require({m["id"] for m in agent.call("list_models")["models"]}
            == {m["id"] for m in (*before_models, model, assets)} and sha(SAFE) == PINS[SAFE],
            "an unreviewed child entered the Library or the source changed")
    agent.write("discard_scratch", "parent-discard", id=root["id"])
    return cases


def experts(agent, run):
    """Plan on the existing tiny writer fixture, then check private-apply refusal."""
    require(agent.call("list_models")["models"] == [], "expected a new Library")
    source = run / "moe-fixtures" / "tiny-qwen2moe.gguf"
    source.write_bytes(fixture_bytes(*fixture()))
    digest = sha(source)
    model = agent.write("register_model", "moe-library", host="local", kind="gguf", path=str(source))
    common = {"schema": "brewery.expert-activations.v1", "source_sha256": digest, "observer_sha256": "a" * 64,
              "n_layers": 2, "routed_layers": [0, 1], "n_experts": 4, "top_k": 2}
    # Synthetic observer data as in test_expert_gguf. This does not represent quality.
    positions = [{
        str(layer): [{"expert": expert, "applied_gate": 0.5, "output_l2": float(expert + layer + 1)}
                     for expert in pair] for layer in range(2)
    } for pair in ((0, 1), (2, 3))]
    split = lambda name, tokens: json.dumps(common | {"split": name, "samples": [
        {"tokens": tokens, "positions": positions},
    ]})
    proposal = agent.call("plan_experts", id=model["id"], source_sha256=digest, keep=3,
                           calibration=split("calibration", [0, 1]), held_out=split("held-out", [2, 3]))
    save(agent.evidence, "moe-plan", proposal)
    plan = json.loads(proposal["plan"])
    require(plan["layers"] == {"0": [3, 2, 1], "1": [3, 2, 1]} and plan["source_sha256"] == digest
            and plan["review"]["operation"] == "prune" and proposal["plan"].endswith("\n")
            and sha256(proposal["plan"].encode()).hexdigest() == proposal["plan_sha256"], "wrong REAP map")
    reply = agent.reply("prune_experts", id=model["id"], plan=proposal["plan"],
                        plan_sha256=proposal["plan_sha256"], reviewed=True, idempotency_key="w12-moe-apply")
    refused = reply["result"]["structuredContent"]
    require(refused["status"] == 503 and "private scratch" in refused["details"]["error"]
            and agent.call("list_models")["models"] == [model]
            and agent.call("list_scratch")["revisions"] == [] and sha(source) == digest,
            "pending apply must leave the parent and both registries unchanged")
    save(agent.evidence, "moe-apply-refused", refused)
    return {"status": "pending verified private scratch writer", "plan": proposal, "apply": refused,
            "fixture": "test_expert_gguf.py tiny Qwen2MoE with synthetic observations"}


def preserve_failure(run, evidence):
    for receipt in evidence.glob("*-job.json"):
        job = json.loads(receipt.read_text())
        if job["state"] != "failed":
            continue
        for name in ("engine-stderr.log", "scratch-checks.log"):
            logs = list((run / "worker" / "jobs" / job["id"]).rglob(name))
            logs.extend((run / "state" / "local-edit" / "local" / "worker" / "jobs" / job["id"]).rglob(name))
            for log in logs:
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 64 * 1024))
                    (evidence / f"{receipt.stem}-{name}").write_bytes(stream.read(64 * 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="new evidence root under /arboretum")
    parser.add_argument("--bins", type=Path, default=ARB / "brewery-w3-target/debug")
    parser.add_argument("--web-dist", type=Path, default=ARB / "w8-accept/web")
    parser.add_argument("--expert-only", action="store_true", help="exercise only the small MoE fixture")
    parser.add_argument("--dense-only", action="store_true", help="exercise only the dense masks")
    parser.add_argument("--lanternfish", type=Path, default=ARB / "pins/lanternfish-0ea30b3/lanternfish",
                        help="path to the exact W1 binary")
    parser.add_argument("--runtime-manifest", type=Path, default=ARB / "pins/lanternfish-0ea30b3.libs.sha256",
                        help="SHA-256 manifest for the W1 and W4 CPU runtime libraries")
    args = parser.parse_args()
    args.root = args.root.resolve()
    require(ARB in args.root.parents and not args.root.exists(), "use a new root under /arboretum")
    require(not (args.expert_only and args.dense_only), "select at most one subset")
    if not args.expert_only:
        for path, pin in PINS.items():
            actual = args.lanternfish if path == LANTERN else path
            require(sha(actual) == pin, f"input pin mismatch: {actual}")
    runtime_pins = {}
    if not args.expert_only:
        for line in args.runtime_manifest.read_text().splitlines():
            digest, path = line.split(maxsplit=1)
            require(sha(Path(path)) == digest, f"runtime library pin mismatch: {path}")
            runtime_pins[path] = digest
        require(len(runtime_pins) == 4, "expected four pinned CPU runtime libraries")
    inputs = PINS | {HF / name: sha(HF / name) for name in ASSETS}
    assets_manifest = "".join(f"{inputs[HF / name]}  {name}\n" for name in sorted((*ASSETS, SAFE.name)))
    require(sha256(assets_manifest.encode()).hexdigest() == SNAPSHOT_PIN, "HF snapshot pin mismatch")
    args.root.mkdir()
    with tempfile.TemporaryDirectory(prefix="isolated-", dir=args.root) as tmp:
        run = Path(tmp)
        env = setup(args, run)
        config = run / "worker.json"
        worker = json.loads(config.read_text())
        fixture_dir = run / "moe-fixtures"
        fixture_dir.mkdir()
        worker["source_roots"].append(str(fixture_dir))
        if args.lanternfish != LANTERN:
            worker["engine_env"]["BREWERY_LANTERNFISH_TRACE"] = str(args.lanternfish)
        save(run, "worker", worker)
        config.chmod(0o600)
        log_path = args.root / "controller.log"
        with log_path.open("w") as log, subprocess.Popen(
            [args.bins / "brewery"], env=env, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        ) as process:
            try:
                for _ in range(200):
                    require(process.poll() is None, "controller exited during setup")
                    match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$", log_path.read_text(), re.M)
                    if match:
                        break
                    time.sleep(0.1)
                require(match and int(match[2]) not in {0, 3000, 4095, 4141}, "no safe loopback port")
                report = {"qualification": "not release-qualified"}
                agent = Agent(match[1], args.root)
                agent.post("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                                                 "enabled": True, "revision": 0})
                if not args.dense_only:
                    report["experts"] = experts(agent, run)
                if not args.expert_only:
                    report["dense"] = dense(agent, run)
                    require(all(sha(Path(p)) == v for p, v in runtime_pins.items()), "runtime library changed")
                    report["llama_cpp_runtime_pins"] = runtime_pins
                    report["input_pins"] = {str(args.lanternfish if p == LANTERN else p): v for p, v in inputs.items()}
                save(args.root, "report", report)
            except Exception:
                preserve_failure(run, args.root)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
