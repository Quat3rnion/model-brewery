#!/usr/bin/env python3
"""Check W1 and W5 with one HGN job on a private loopback stack."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import os
import subprocess
import time
from urllib.request import Request, urlopen


def save(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def exercise(args):
    os.umask(0o077)
    root = args.out.resolve()
    root.mkdir()
    bins = args.bins.resolve()
    model = args.model.resolve()
    python = args.python.absolute()
    pins = {"sidecar": args.sidecar, "config": args.config, "tokenizer": args.tokenizer}
    assets = {name: {"path": str(p.resolve()), "sha256": digest(p)} for name, p in pins.items()}
    model_sha = digest(model)
    save(root, "assets.json", {model_sha: assets})
    save(root, "worker.json", {
        "root": str(root / "worker"), "engine": str(bins / "brewery-engine"),
        "source_roots": [str(model.parent)], "engine_env": {
            "BREWERY_HGN_TRACE_PYTHON": str(python),
            "BREWERY_HGN_TRACE_PYTHON_SHA256": digest(python),
            "BREWERY_HGN_TRACE_ASSETS": str(root / "assets.json"),
            "BREWERY_CPU_THREADS": str(args.threads),
        },
    })
    save(root, "dispatch.json", {"hosts": {"local": {"local": True,
         "worker": str(bins / "brewery-worker"), "config": str(root / "worker.json"),
         "capabilities": {"trace": True}}}})
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
            if not base:
                raise RuntimeError("no isolated loopback listener")

            def api(path, body=None):
                data = None if body is None else json.dumps(body).encode()
                with urlopen(Request(base + path, data=data,
                                     headers={"Content-Type": "application/json"}), timeout=600) as response:
                    return json.load(response)

            registered = api("/api/models", {"host": "local", "path": str(model), "kind": "hgn"})
            config = json.loads(args.config.read_text())["text_config"]
            last = config["num_hidden_layers"] - 1
            position = 2
            top_k = min(5, config["vocab_size"])
            job = api("/api/jobs", {"kind": "trace", "host": "local", "input_ids": [registered["id"]],
                      "prompt": "The first prime number is", "options": {
                          "layers": [0, last], "roles": ["l_out", "final_norm"], "positions": [position],
                          "top_k": top_k, "byte_cap": 1 << 20}})["id"]
            job_url = f"/api/jobs/{job}"
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                detail = api(job_url)
                if detail["state"] == "succeeded":
                    break
                if detail["state"] in {"failed", "canceled"}:
                    raise RuntimeError(f"trace {job}: {detail}")
                time.sleep(1)
            else:
                raise TimeoutError(f"trace {job} did not finish")
            summary = api(job_url + "/trace")
            result = detail["result"]
            assert result["model_sha256"] == model_sha and summary["runtime"] == "hgn_cpu"
            assert summary["model"]["l_out_axes"] == ["stream", "hidden"]
            assert result["assets_sha256"] == {name: pin["sha256"] for name, pin in assets.items()}
            for role, shape in [("l_out", [4, config["hidden_size"]]), ("final_norm", [config["hidden_size"]])]:
                row, = api(f"{job_url}/trace/tensor?layer={last}&role={role}")["rows"]
                assert row["shape"] == shape and row["position"] == position
                raw = base64.b64decode(row["data_base64"], validate=True)
                assert hashlib.sha256(raw).hexdigest() == row["sha256"]
            logits, = api(job_url + "/trace/model-logits")["rows"]
            lens, = api(f"{job_url}/trace/logit-lens?layer={last}")["rows"]
            assert logits["top"] == lens["top"] and len(logits["top"]) == top_k
            diff = api(f"/api/diff/traces?a={job}&b={job}")
            assert any(row["l2"] == 0 for row in diff["activations"])
            report = {"job": job, "summary": summary, "result": result, "self_diff": diff,
                      "seconds": time.monotonic() - stamp,
                      "binaries": {b.name: digest(b) for b in sorted(bins.iterdir())}}
            save(root, "evidence.json", report)
            print(json.dumps({"job": job, "runtime": summary["runtime"], "bytes": summary["bytes"]}))
        finally:
            proc.terminate()
            proc.wait(timeout=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bins", "model", "sidecar", "config", "tokenizer", "python", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    exercise(parser.parse_args())
