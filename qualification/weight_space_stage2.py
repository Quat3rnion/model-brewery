#!/usr/bin/env python3
"""Run the three CPU edit methods through MCP with the same pinned Qwen parent."""

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from weight_space_acceptance import (
    ARB, REPO, FIXTURES, PINS, RUNNER, SAFE, require, save, setup, sha,
)

RUNTIME = ARB / "w9-runtime"
LIBRARIES = ARB / "llama-w9/build/bin"
LIBRARY_NAMES = ("libllama.so.0", "libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0")
CASE_IDS = (7, 0, 1, 2, 3)
METHODS = ("rome", "memit", "alphaedit")
CHECKS = REPO / "tools/scratch_checks.py"
spec = importlib.util.spec_from_file_location("scratch_checks", CHECKS)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def facts():
    source = FIXTURES / "counterfact.json"
    require(sha(source) == PINS[source], "CounterFact source hash mismatch")
    rows = {x["case_id"]: x for x in json.loads(source.read_text())}
    subset = [rows[i]["requested_rewrite"] for i in CASE_IDS]
    return [{"subject": r["subject"], "prompt": r["prompt"],
             "target": r["target_new"]["str"]} for r in subset]


def targeted_rows(ids):
    src = (FIXTURES / "counterfact.json").read_bytes()
    rows = [json.loads(line) for line in checks.project("counterfact.jsonl", src, ids).splitlines()]
    result = []
    for row in rows:
        result.append(("counterfact_target", *row["target"]))
        result.extend(("counterfact_paraphrase", *pair) for pair in row["paraphrases"])
        result.extend(("counterfact_locality", *pair) for pair in row["locality"])
    return result


def exact_scores(parent, child, ids):
    rows = targeted_rows(ids)
    _, before = checks.score(RUNNER, parent, 4, rows)
    _, after = checks.score(RUNNER, child, 4, rows)
    require(all(a.tokens == b.tokens for k in before for a, b in zip(before[k], after[k], strict=True)),
            "paired edit prompt token mismatch")
    a, b = checks.aggregate(before), checks.aggregate(after)
    return {k: {"before": a[k][0], "after": b[k][0], "delta": b[k][0] - a[k][0]} for k in a}


async def exercise(base, run, args):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    targets = facts()
    async with httpx.AsyncClient(trust_env=False, timeout=120) as http, \
            streamable_http_client(base + "/mcp", http_client=http) as (read, write, _), \
            ClientSession(read, write) as client:
        # Only the isolated operator bootstrap is outside MCP. There is no
        # publish, approval, or direct controller access to the edited weights.
        response = await http.post(base + "/api/admin/actions", json={
            "action": "set_weight_editor", "worker": "local", "revision": 0, "enabled": True,
        })
        require(response.status_code == 200, f"isolated operator bootstrap: {response.text}")
        await client.initialize()
        catalog = {tool.name for tool in (await client.list_tools()).tools}
        require(all(f"{name}_edit" in catalog for name in METHODS) and "approve" not in catalog,
                "incorrect MCP edit catalog")

        async def call(tool, **params):
            r = await client.call_tool(tool, params)
            require(not r.isError and r.structuredContent is not None, f"{tool}: {r}")
            return r.structuredContent

        async def submit(tool, label, **params):
            return await call(tool, idempotency_key=f"w9-{label}", **params)

        async def wait(job, label):
            end = time.monotonic() + 1800
            while time.monotonic() < end:
                current = await call("get_job", id=job["id"], wait_ms=5000)
                if current["state"] not in ("queued", "running"):
                    save(args.evidence, label + "-job", current)
                    if current["state"] != "succeeded":
                        job_work = run / "state/local-edit/local/worker/jobs" / job["id"]
                        for name in ("engine-stderr.log", "supervisor.log"):
                            diagnostic = job_work / name
                            if diagnostic.is_file():
                                with diagnostic.open("rb") as src:
                                    src.seek(max(0, diagnostic.stat().st_size - 8192))
                                    (args.evidence / f"{label}-{name}").write_bytes(src.read())
                    require(current["state"] == "succeeded", f"{label}: {current.get('error')}")
                    print(f"{label}: {job['id']}", flush=True)
                    return current["result"]
            raise TimeoutError(label)

        async def register(kind, path):
            model = await submit("register_model", kind, host="local", kind=kind, path=str(path))
            return model["id"]

        safe_id = await register("safetensors", SAFE)
        assets_id = await register("assets", run / "assets")
        parent = await submit("create_scratch", "parent", parent_id=safe_id, parent_sha256=PINS[SAFE])
        parent_id, parent_sha = parent["id"], parent["sha256"]
        parent_gguf = await wait(await submit("convert_scratch_gguf", "parent-gguf", id=parent_id,
                                          sha256=parent_sha, assets_id=assets_id), "parent-gguf")
        root = run / "state/scratch"
        parent_path = root / parent_id / "model.gguf"
        require(sha(parent_path) == parent_gguf["gguf"]["sha256"], "private parent pin mismatch")
        report = {"qualification": "not release-qualified", "parent_safetensors": parent_sha,
                  "parent_gguf": parent_gguf["gguf"]["sha256"],
                  "lanternfish_sha256": sha(RUNTIME / "lanternfish"),
                  "library_sha256": {n: sha(LIBRARIES / n) for n in LIBRARY_NAMES},
                  "methods": {}}

        for method in (args.method or METHODS):
            ids = CASE_IDS[:1] if method == "rome" else CASE_IDS
            layers = [12] if method == "rome" else [10, 12]
            write = dict(id=parent_id, sha256=parent_sha, layers=layers,
                         facts=targets[:len(ids)], steps=args.steps)
            job = await submit(method + "_edit", method, **write)
            repeated = await submit(method + "_edit", method, **write)
            require(job["id"] == repeated["id"], "idempotent job replay failed")
            result = await wait(job, method)
            child_id, child_sha = result["scratch_id"], result["sha256"]
            require(child_sha != parent_sha, "unexpected unchanged child")
            diff = await call("diff_tensors", a=parent_id, b=child_id)
            expected = {f"model.layers.{layer}.mlp.down_proj.weight" for layer in layers}
            require({t["name"] for t in diff["tensors"] if t["changed_elements"]} == expected
                    and all(t["dtype"] == "BF16" for t in diff["tensors"]), "unexpected edited tensors or dtype")
            await wait(await submit("convert_scratch_gguf", method + "-gguf", id=child_id,
                                    sha256=child_sha, assets_id=assets_id), method + "-gguf")
            await wait(await submit("run_checks", method + "-checks", id=child_id, sha256=child_sha),
                       method + "-checks")
            bundle = await call("get_checks", id=child_id)
            measured = bundle["measured_report"]
            require(bundle["integrity"]["status"] == "passed"
                    and all(measured.get(k) for k in ("parent_generation", "child_generation"))
                    and len(measured["metrics"]) == 17, "W4 checks incomplete")
            child_path = root / child_id / "model.gguf"
            exact = exact_scores(parent_path, child_path, ids)
            summary = {
                "case_ids": list(ids), "layers": layers, "steps": args.steps,
                "optimization_nll": result["optimization_nll"],
                "targeted_counterfact": exact,
                "common_case_7": exact if method == "rome" else exact_scores(parent_path, child_path, CASE_IDS[:1]),
                "w4_metrics": measured["metrics"], "integrity": bundle["integrity"],
                "smoke": [measured[k] for k in ("parent_generation", "child_generation")],
                "child": child_sha, "tensors": sorted(expected),
            }
            report["methods"][method] = summary
            save(args.evidence, "report", report)
            await submit("discard_scratch", method + "-discard", id=child_id)
        require(sha(SAFE) == parent_sha and sha(parent_path) == report["parent_gguf"],
                "parent changed during editing")
        await submit("discard_scratch", "parent-discard", id=parent_id)


async def run_acceptance(run, args):
    env = setup(args, run)
    config = run / "worker.json"
    doc = json.loads(config.read_text())
    doc["engine_env"].update({
        "BREWERY_LANTERNFISH_EDIT": str(RUNTIME / "lanternfish"),
        "BREWERY_LANTERNFISH_EDIT_LIB_DIR": str(LIBRARIES),
        "BREWERY_CPU_THREADS": "6",
    })
    config.write_text(json.dumps(doc))
    path = args.evidence / "controller.log"
    with path.open("w") as log:
        controller = await asyncio.create_subprocess_exec(
            args.bins / "brewery", env=env, cwd=REPO, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                require(controller.returncode is None, f"controller exited; see {path}")
                match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$",
                                  path.read_text(), re.M)
                if match:
                    require(int(match[2]) not in {0, 3000, 4095, 4141}, "unsafe port")
                    await exercise(match[1], run, args)
                    return
                await asyncio.sleep(0.2)
            raise TimeoutError("isolated controller listen")
        finally:
            if controller.returncode is None:
                controller.terminate()
                try:
                    await asyncio.wait_for(controller.wait(), 10)
                except TimeoutError:
                    controller.kill()
                    await controller.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--bins", type=Path, default=RUNTIME / "bin")
    parser.add_argument("--web-dist", type=Path, default=ARB / "w8-accept/web")
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--steps", type=int, choices=range(1, 33), default=16)
    args = parser.parse_args()
    require(args.evidence.is_absolute() and args.evidence.is_relative_to(ARB), "use an absolute evidence directory under /arboretum")
    require(not args.evidence.exists(), "use a new evidence directory")
    for path, pin in PINS.items():
        require(sha(path) == pin, f"pin mismatch: {path}")
    args.evidence.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="stage2-", dir=ARB / "tmp") as work:
        asyncio.run(run_acceptance(Path(work), args))


if __name__ == "__main__":
    main()
