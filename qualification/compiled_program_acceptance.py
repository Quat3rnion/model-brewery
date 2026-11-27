#!/usr/bin/env python3
"""Run the compiled program MCP workflow on a new isolated loopback controller."""

import argparse
import json
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from weight_space_acceptance import ARB, HELPER, REPO, require, save


def accept(args, root):
    source = root / "sources"
    source.mkdir()
    worker = root / "worker"
    worker.mkdir(mode=0o700)
    dist = root / "dist"
    dist.mkdir()
    save(root, "worker", {
        "root": str(worker), "engine": str(args.bins / "brewery-engine"),
        "source_roots": [str(source)], "disk_reserve_bytes": 0,
    })
    config = root / "worker.json"
    config.chmod(0o600)
    save(root, "dispatch", {"hosts": {"local": {"local": True,
         "worker": str(args.bins / "brewery-worker"), "config": str(config)}}})
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(root), "TMPDIR": str(ARB / "tmp"),
        "BREWERY_LISTEN": "127.0.0.1:0", "BREWERY_DATA": str(root / "state"),
        "BREWERY_WEB_DIST": str(dist), "BREWERY_DISPATCH_CONFIG": str(root / "dispatch.json"),
        "BREWERY_DISPATCH": str(args.bins / "brewery-dispatch"),
        "BREWERY_LOCAL_EDIT_DISPATCH": str(args.bins / "brewery-dispatch"),
        "BREWERY_LOCAL_EDIT_WORKER": str(args.bins / "brewery-worker"),
        "BREWERY_LOCAL_EDIT_ENGINE": str(args.bins / "brewery-engine"),
        "BREWERY_LOCAL_EDIT_HELPER": str(args.helper),
        "BREWERY_COMPILED_PROGRAM_PYTHON": str(args.python),
        "BREWERY_COMPILED_PROGRAM_SCRIPT": str(REPO / "tools/compiled_program.py"),
    }
    log = args.evidence / "controller.log"
    with log.open("w") as stream, subprocess.Popen(
        [args.controller], env=env, cwd=REPO, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
    ) as process:
        try:
            for _ in range(200):
                require(process.poll() is None, "controller exited; see controller.log")
                match = re.search(r"^brewery listening on http://(127\.0\.0\.1:(\d+))$", log.read_text(), re.M)
                if match:
                    break
                time.sleep(0.05)
            require(match and int(match[2]) not in {0, 3000, 4095, 4141}, "no safe loopback port")
            return exercise("http://" + match[1])
        finally:
            if process.poll() is None:
                process.terminate()


def exercise(base):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(path, data):
        request = urllib.request.Request(base + path, json.dumps(data).encode(), headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
            "Origin": base,
        })
        with opener.open(request, timeout=40) as response:
            return json.load(response)

    def mcp(method, params=None):
        body = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            body["params"] = params
        result = post("/mcp", body)
        require("error" not in result, f"{method}: {result}")
        return result["result"]

    def call(name, **args):
        result = mcp("tools/call", {"name": name, "arguments": args})
        require(not result["isError"], f"{name}: {result}")
        return result["structuredContent"]

    post("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                                "revision": 0, "enabled": True})
    init = mcp("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "compiled-program-acceptance", "version": "1"}})
    tools = {t["name"] for t in mcp("tools/list")["tools"]}
    require({"compile_program", "run_program", "run_checks", "request_publish"} <= tools
            and not any("approve" in t for t in tools), "unexpected MCP tools")
    require(call("list_models")["models"] == [], "Library is not empty")
    examples = (
        ("sort", [3, 0, 3, 1], [0, 1, 3, 3], 340),
        ("histogram", [2, 0, 2, 2], [3, 1, 3, 3], 340),
        ("reverse", [1, 3, 0], [0, 3, 1], 340),
        ("frac_prevs", [1, 0, 0, 1], [1, 0.5, 1 / 3, 0.5], 340),
        ("alta_parity_relative", [1, 0, 1, 1], [1, 1, 0, 1], 30),
        ("alta_parity_absolute", [1, 1, 1, 1], [1, 0, 1, 0], 30),
    )
    programs = {}
    registered = []
    for program, inputs, expected, cases in examples:
        create = {"worker": "local", "program": program}
        first = call("compile_program", idempotency_key=f"{program}-compile", **create)
        require(first["kind"] == "compiled_program", f"{program}: not a compiled root")
        replay = call("compile_program", idempotency_key=f"{program}-compile", **create)
        require(first["id"] == replay["id"], f"{program}: compile not idempotent")
        second = call("compile_program", idempotency_key=f"{program}-repeat", **create)
        pin = first["sha256"]
        recipe = first["provenance"]["compiled_program"]
        require(pin == second["sha256"] and recipe == second["provenance"]["compiled_program"],
                f"{program}: compile not deterministic")
        revision = {"id": first["id"], "sha256": pin}
        output = call("run_program", idempotency_key=f"{program}-run", tokens=inputs, **revision)
        require(output["output"] == expected, f"{program}: incorrect output {output}")
        call("run_checks", idempotency_key=f"{program}-checks", **revision)
        bundle = call("get_checks", id=first["id"])
        require(all(bundle[k]["status"] == "passed" for k in ("integrity", "smoke", "program_correctness"))
                and bundle["program_cases"] == cases and bundle["behavior"]["status"] == "not_run",
                f"{program}: failed checks: {bundle}")
        pending = call("request_publish", idempotency_key=f"{program}-publish",
                       acknowledge_bundle=bundle["id"], **revision)
        require(pending["state"] == "publish_requested" and pending["library_model_id"] is None,
                f"{program}: wrong publish state")
        require(len(call("list_models")["models"]) == len(registered), "MCP published to the Library")
        approved = post(f"/api/admin/scratch/{first['id']}/approve", {"sha256": pin})
        require(approved["state"] == "approved", f"{program}: no Library approval")
        registered.append(approved["library_model_id"])
        programs[program] = {"revision": first["id"], "library_model": registered[-1], "sha256": pin,
                             "config_sha256": recipe["config_sha256"],
                             "compiler_commit": first["provenance"]["tool_version"],
                             "checks": bundle, "inputs": inputs, "output": output["output"]}
    models = call("list_models")["models"]
    require(len(models) == len(registered) and {m["id"] for m in models} == set(registered)
            and all(m["kind"] == "safetensors" for m in models), "wrong Library records")
    return {"qualification": "not release-qualified", "mcp": init, "programs": programs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--bins", type=Path, default=ARB / "brewery-w3-target/debug")
    parser.add_argument("--helper", type=Path, default=HELPER)
    parser.add_argument("--python", type=Path, default=ARB / "tmp/brewery-w13/.venv/bin/python")
    args = parser.parse_args()
    args.evidence = args.evidence.resolve()
    require(args.evidence.is_relative_to(ARB), "put evidence under /arboretum")
    args.evidence.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="compiled-program-", dir=ARB / "tmp") as directory:
        report = accept(args, Path(directory))
    save(args.evidence, "report", report)
    print(f"{len(report['programs'])} compiled programs passed; not release-qualified")


if __name__ == "__main__":
    main()
