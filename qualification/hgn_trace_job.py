#!/usr/bin/env python3
"""Check an HGN v2 trace over MCP on a private loopback stack."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import os
import signal
import subprocess
import time
from urllib.request import ProxyHandler, Request, build_opener


def save(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def stop_job(bins, env, host, job):
    def dispatch(action):
        return json.loads(subprocess.check_output(
            [bins / "brewery-dispatch", action, host, job], env=env,
            stdin=subprocess.DEVNULL, timeout=5,
        ))

    reply = dispatch("cancel")
    deadline = time.monotonic() + 5
    while reply["state"] not in {"cancelled", "failed", "succeeded"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"isolated worker did not stop {job}")
        time.sleep(0.2)
        reply = dispatch("status")


def exercise(args):
    os.umask(0o077)
    registry = args.assets.resolve(strict=True)
    model_sha = args.model_sha256
    assets = json.loads(registry.read_text())[model_sha]
    config = json.loads(Path(assets["config"]["path"]).read_text())["text_config"]
    root = args.out.resolve()
    root.mkdir(mode=0o700)
    (root / "worker").mkdir(mode=0o700)
    bins = args.bins.resolve()
    model = args.model.resolve()
    python = args.python.absolute()
    host = "frame-hgn-trace"
    bounds = {"cpu_threads": {"min": 1, "max": args.threads}}
    capabilities = dict.fromkeys(("inspect", "plan", "convert", "verify", "compare",
                                  "transfer", "transfer_preview", "cache_cleanup"), False)
    capabilities["trace"] = True
    save(root, "worker.json", {
        "root": str(root / "worker"), "engine": str(bins / "brewery-engine"),
        "resource_bounds": bounds,
        "source_roots": [str(model.parent)], "engine_env": {
            "BREWERY_HGN_TRACE_PYTHON": str(python),
            "BREWERY_HGN_TRACE_PYTHON_SHA256": args.python_sha256,
            "BREWERY_HGN_TRACE_ASSETS": str(registry),
            "BREWERY_CPU_THREADS": str(args.threads),
        },
    })
    save(root, "dispatch.json", {"hosts": {host: {"local": True,
         "worker": str(bins / "brewery-worker"), "config": str(root / "worker.json"),
         "capabilities": capabilities, "resource_bounds": bounds}}})
    dist = root / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("Isolated HGN trace check")
    stamp = time.monotonic()
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(root), "TMPDIR": str(root),
        "BREWERY_LISTEN": "127.0.0.1:0", "BREWERY_DATA": str(root / "state"),
        "BREWERY_WEB_DIST": str(dist), "BREWERY_DISPATCH_CONFIG": str(root / "dispatch.json"),
        "BREWERY_DISPATCH": str(bins / "brewery-dispatch"), "BREWERY_POLL_MS": "200",
    }
    log_path = root / "controller.log"
    active_job = None
    with log_path.open("w") as log, subprocess.Popen([bins / "brewery"], env=env, stdout=log, stderr=log) as proc:
        try:
            base = None
            for _ in range(100):
                if proc.poll() is not None:
                    raise RuntimeError(f"controller exited; see {log_path}")
                for line in log_path.read_text().splitlines():
                    if line.startswith("brewery listening on http://127.0.0.1:"):
                        base = line.split()[-1]
                        break
                if base:
                    break
                time.sleep(0.1)
            if not base or int(base.rsplit(":", 1)[1]) in {0, 3000, 4095, 4141}:
                raise RuntimeError("no safe isolated loopback listener")
            opener = build_opener(ProxyHandler({}))

            def mcp(method, params):
                body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                with opener.open(Request(base + "/mcp", data=json.dumps(body).encode(), headers={
                    "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-06-18", "Origin": base,
                }), timeout=30) as response:
                    reply = json.load(response)
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply["result"]

            def call(tool, **arguments):
                result = mcp("tools/call", {"name": tool, "arguments": arguments})
                if result["isError"]:
                    raise RuntimeError(f"{tool}: {result['structuredContent']}")
                return result["structuredContent"]

            def read(job, view, **filters):
                return call("read_trace", id=job, view=view, **filters)

            init = mcp("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "hgn-trace-acceptance", "version": "1"}})
            registered = call("register_model", host=host, path=str(model), kind="hgn-v2",
                              idempotency_key="h1-register")
            last = config["num_hidden_layers"] - 1
            position = 2
            top_k = min(5, config["vocab_size"])
            queued = call("start_trace", idempotency_key="h1-trace", input_ids=[registered["id"]],
                          prompt="The first prime number is", options={
                              "layers": sorted({0, last}), "roles": ["l_out", "final_norm"],
                              "positions": [position], "top_k": top_k, "byte_cap": 1 << 20})
            assert queued["host"] == host
            job = queued["id"]
            active_job = job
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                detail = call("get_job", id=job, wait_ms=5000)
                if detail["state"] == "succeeded":
                    active_job = None
                    break
                if detail["state"] in {"failed", "cancelled"}:
                    active_job = None
                    raise RuntimeError(f"trace {job}: {detail}")
            else:
                raise TimeoutError(f"trace {job} did not finish")
            summary = read(job, "manifest")
            result = detail["result"]
            assert result["model_sha256"] == model_sha and summary["runtime"] == "hgn_cpu"
            assert summary["model"]["l_out_axes"] == ["stream", "hidden"]
            assert result["assets_sha256"] == {name: pin["sha256"] for name, pin in assets.items()}
            for role, shape in [("l_out", [4, config["hidden_size"]]), ("final_norm", [config["hidden_size"]])]:
                row, = read(job, "tensor", layer=last, role=role)["rows"]
                assert row["shape"] == shape and row["position"] == position
                raw = base64.b64decode(row["data_base64"], validate=True)
                assert hashlib.sha256(raw).hexdigest() == row["sha256"]
            logits, = read(job, "model_logits")["rows"]
            lens, = read(job, "logit_lens", layer=last)["rows"]
            assert logits["top"] == lens["top"] and len(logits["top"]) == top_k
            diff = call("diff_traces", a=job, b=job)
            assert any(row["l2"] == 0 for row in diff["activations"])
            report = {"job": job, "qualification": "not release-qualified",
                      "mcp": init, "host": host, "model": registered,
                      "summary": summary, "result": result, "self_diff": diff,
                      "seconds": time.monotonic() - stamp,
                      "binaries": {name: digest(bins / name) for name in
                                   ("brewery", "brewery-worker", "brewery-engine", "brewery-dispatch")},
                      "asset_registry_sha256": digest(registry)}
            save(root, "evidence.json", report)
            print(json.dumps({"job": job, "runtime": summary["runtime"], "bytes": summary["bytes"]}))
        finally:
            try:
                if active_job is not None:
                    stop_job(bins, env, host, active_job)
            finally:
                proc.terminate()
                proc.wait(timeout=5)


def terminate(*_):
    raise SystemExit("isolated trace interrupted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bins", "model", "assets", "python", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("model-sha256", "python-sha256"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=900)
    signal.signal(signal.SIGTERM, terminate)
    exercise(parser.parse_args())
