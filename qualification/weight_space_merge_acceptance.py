#!/usr/bin/env python3
"""Run CPU merge acceptance through MCP on a new, isolated controller."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.request import ProxyHandler, Request, build_opener

import weight_space_acceptance as w8


def exercise(base, args, run):
    http = build_opener(ProxyHandler({}))
    rpc_id = 0

    def send(path, data=None):
        request = Request(base + path, data=None if data is None else json.dumps(data).encode(),
                          headers={"Accept": "application/json, text/event-stream",
                                   "Content-Type": "application/json",
                                   "MCP-Protocol-Version": "2025-06-18"})
        with http.open(request, timeout=600) as reply:
            return json.load(reply)

    def rpc(method, params):
        nonlocal rpc_id
        rpc_id += 1
        result = send("/mcp", {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params})
        w8.require("result" in result, f"{method}: {result}")
        return result["result"]

    def call(tool, **params):
        result = rpc("tools/call", {"name": tool, "arguments": params})
        w8.require(not result["isError"], f"{tool}: {result}")
        return result["structuredContent"]

    def submit(tool, label, **params):
        return call(tool, idempotency_key="w10-" + label, **params)

    def wait(job):
        end = time.monotonic() + 1200
        while time.monotonic() < end:
            state = call("get_job", id=job["id"], wait_ms=5000)
            if state["state"] not in ("queued", "running"):
                w8.require(state["state"] == "succeeded", f"{job['id']}: {state}")
                return state["result"]
        raise TimeoutError(job["id"])

    send("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                                "revision": 0, "enabled": True})
    init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "W10 acceptance", "version": "1"}})
    catalog = [t["name"] for t in rpc("tools/list", {})["tools"]]
    w8.require("merge_models" in catalog and "approve" not in catalog, "incorrect MCP catalog")
    w8.save(args.evidence, "mcp-session", {"initialize": init, "tools": catalog})
    w8.require(not call("list_models")["models"], "isolated Library is not empty")
    model = submit("register_model", "model", host="local", kind="safetensors",
                   path=str(w8.SAFE), name="Qwen2.5-0.5B-Instruct SafeTensors")
    assets = submit("register_model", "assets", host="local", kind="assets",
                    path=str(run / "assets"), name="Qwen2.5-0.5B-Instruct assets")
    root = submit("create_scratch", "base", parent_id=model["id"], parent_sha256=w8.PINS[w8.SAFE])
    print("Created pinned scratch base", flush=True)

    def convert(revision, name):
        result = wait(submit("convert_scratch_gguf", "convert-" + name,
                             id=revision["id"], sha256=revision["sha256"], assets_id=assets["id"]))["gguf"]
        w8.require(result["revision_sha256"] == revision["sha256"], f"{name}: GGUF pin differs")
        return result

    parents = []
    for i in (0, 1):
        values = [(((j * (3 + i) + i) % 17) - 8) / 128 for j in range(640)]
        op = {"op": "add", "tensor": w8.TENSOR, "origin": [i * 64], "shape": [640], "values": values}
        parent = submit("apply_edit", f"input-{i}", id=root["id"], operations=[op])
        parents.append({"id": parent["id"], "sha256": parent["sha256"], "weight": (0.5, 0.75)[i]})
    w8.save(args.evidence, "inputs", parents)
    parent_gguf = convert(root, "base")
    print("Converted scratch base to private BF16 GGUF", flush=True)

    methods = [
        {"method": "task_arithmetic"},
        {"method": "ties", "density": 0.45},
        {"method": "dare_linear", "drop_rate": 0.5, "seed": 2026},
        {"method": "dare_ties", "drop_rate": 0.5, "seed": 2026},
    ]
    report = {"qualification": w8.QUALIFICATION, "controller": base, "base": root["sha256"],
              "base_gguf": parent_gguf["sha256"], "cases": {}}
    for algorithm in methods:
        name = algorithm["method"]
        start = time.monotonic()
        merged = submit("merge_models", name, base_id=root["id"], base_sha256=root["sha256"],
                        algorithm=algorithm, inputs=parents)
        w8.require(merged["state"] == "scratch" and merged["qualification"] == w8.QUALIFICATION
                   and merged["provenance"]["parent_id"] == root["id"]
                   and merged["provenance"]["merge"]["inputs"] == parents, f"{name}: provenance mismatch")
        w8.save(args.evidence, name + "-revision", merged)
        diff = call("diff_tensors", a=root["id"], b=merged["id"])
        tensors = diff["tensors"]
        w8.require(len(tensors) == 1 and tensors[0]["name"] == w8.TENSOR
                   and tensors[0]["l2"] > 0, f"{name}: unexpected diff")
        gguf = convert(merged, name)
        wait(submit("run_checks", "checks-" + name, id=merged["id"], sha256=merged["sha256"]))
        checks = call("get_checks", id=merged["id"])
        measured = checks["measured_report"]
        status = {k: checks[k]["status"] for k in ("integrity", "smoke", "behavior")}
        w8.require(set(status.values()) == {"passed"} and len(measured["metrics"]) == 17
                   and measured["runner_sha256"] == w8.PINS[w8.RUNNER]
                   and measured["parent_gguf_sha256"] == parent_gguf["sha256"]
                   and measured["revision_gguf_sha256"] == gguf["sha256"], f"{name}: checks differ")
        report["cases"][name] = {"tensor_diff": tensors[0], "check_status": status,
                                 "measured": measured, "elapsed_seconds": round(time.monotonic() - start, 1)}
        w8.save(args.evidence, "report", report)
        print(f"{name}: {tensors[0]['changed_elements']} changed, L2={tensors[0]['l2']:.6f}; W4 passed",
              flush=True)
        submit("discard_scratch", "discard-" + name, id=merged["id"])
    w8.require({m["id"] for m in call("list_models")["models"]} == {model["id"], assets["id"]},
               "scratch revision leaked into the Library")
    report["status"] = "passed"
    w8.save(args.evidence, "report", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="new evidence root under /arboretum")
    parser.add_argument("--bins", type=Path, required=True, help="directory with isolated Brewery binaries")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.bins = args.bins.resolve()
    w8.require(w8.ARB in args.root.parents, "root must be under /arboretum")
    args.web_dist = w8.ARB / "w8-accept/web"
    args.evidence = args.root / "evidence"
    args.evidence.mkdir(parents=True)
    expected = {path: w8.PINS[path] for path in (
        w8.SAFE, w8.QUANT, w8.RUNNER, w8.FIXTURES / "counterfact.json", w8.FIXTURES / "zsre.json",
    )}
    pins = {str(path): w8.sha(path) for path in expected}
    w8.require(all(pins[str(path)] == value for path, value in expected.items()), "input pin mismatch")
    pins.update({str(w8.HF / name): w8.sha(w8.HF / name) for name in w8.ASSETS})
    manifest = "".join(f"{pins[str(w8.HF / name)]}  {name}\n" for name in sorted((*w8.ASSETS, w8.SAFE.name)))
    w8.require(hashlib.sha256(manifest.encode()).hexdigest() == w8.SNAPSHOT_PIN, "HF snapshot pin mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=w8.REPO, text=True).strip()
    bins = {name: w8.sha(args.bins / name) for name in ("brewery", "brewery-dispatch", "brewery-engine", "brewery-worker")}
    w8.save(args.evidence, "input-pins", {"brewery_commit": commit, "files": pins, "bin_sha256": bins})
    with tempfile.TemporaryDirectory(dir=args.root, prefix="isolated-") as isolated:
        run = Path(isolated)
        env = w8.setup(args, run)
        worker = json.loads((run / "worker.json").read_text())
        del worker["engine_env"]["BREWERY_LANTERNFISH_TRACE"]
        w8.save(run, "worker", worker)
        (run / "worker.json").chmod(0o600)
        dispatch = json.loads((run / "dispatch.json").read_text())
        del dispatch["hosts"]["local"]["capabilities"]["trace"]
        w8.save(run, "dispatch", dispatch)
        log = args.evidence / "controller.log"
        with log.open("w") as out, subprocess.Popen(
            [args.bins / "brewery"], env=env, cwd=w8.REPO,
            stdin=subprocess.DEVNULL, stdout=out, stderr=out,
        ) as process:
            try:
                for _ in range(200):
                    w8.require(process.poll() is None, "controller exited; see controller.log")
                    match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$", log.read_text(), re.M)
                    if match:
                        w8.require(int(match[2]) not in {0, 3000, 4095, 4141}, "unsafe controller port")
                        exercise(match[1], args, run)
                        break
                    time.sleep(0.1)
                else:
                    raise TimeoutError("controller did not listen")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
