#!/usr/bin/env python3
"""Run a scaled trace and last-K fit on a synthetic Qwen2 GGUF."""
import argparse
import base64
import json
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
import time

from rocmfpx_smoke import fixture, sha

CONTRAST = "f42e2aac3d2c0faad8d0f0e5e2a2c2a5c8580bb1f51ece3e6a30b26eac3c24b6"


def exercise(args):
    assert sha(args.cache / f"{CONTRAST}.json") == CONTRAST
    with TemporaryDirectory(prefix="gguf-projection-") as temp:
        root = Path(temp)
        source = root / "source"
        source.mkdir(mode=0o700)
        worker_root = root / "worker"
        worker_root.mkdir(mode=0o700)
        model = source / "fixture.gguf"
        fixture(model, context=1024)
        digest = sha(model)

        def save(name, value):
            path = root / name
            path.write_text(json.dumps(value))
            path.chmod(0o600)
            return str(path)

        worker = save("worker.json", {
            "root": str(worker_root), "source_roots": [str(source)],
            "engine": str(args.bins / "brewery-engine"), "disk_reserve_bytes": 0,
            "engine_env": {"BREWERY_CPU_THREADS": "1", "BREWERY_CONTRAST_CACHE": str(args.cache),
                           "BREWERY_LANTERNFISH_TRACE": str(args.trace),
                           "BREWERY_LANTERNFISH_TRACE_LIB_DIR": str(args.trace_libs),
                           "BREWERY_LANTERNFISH_ABLATION": str(args.ablation),
                           "BREWERY_LANTERNFISH_PROJECTION_LIB_DIR": str(args.ablation_libs)},
        })
        env = {"PATH": "/usr/bin:/bin", "BREWERY_DISPATCH_CONFIG": save("dispatch.json", {
            "hosts": {"local": {"local": True, "worker": str(args.bins / "brewery-worker"), "config": worker}},
        })}
        inputs = [{"host": "local", "path": str(model), "id": "tiny", "kind": "gguf"}]

        def call(action, job, payload=None):
            cli = subprocess.run([args.bins / "brewery-dispatch", action, "local", job], env=env,
                                 input=json.dumps(payload) if payload is not None else "",
                                 capture_output=True, text=True, timeout=15)
            assert cli.returncode == 0, (action, cli.stdout, cli.stderr)
            return json.loads(cli.stdout)

        def run(job, kind, options, prompt=None):
            payload = {"kind": kind, "inputs": inputs, "options": options}
            if prompt is not None:
                payload["prompt"] = prompt
            status = call("start", job, payload)
            deadline = time.monotonic() + 120
            try:
                while status["state"] in ("running", "queued") and time.monotonic() < deadline:
                    time.sleep(0.25)
                    status = call("status", job)
                assert status["state"] == "succeeded", status
            finally:
                if status["state"] in ("running", "queued"):
                    call("cancel", job)
            return status["result"]

        fit = run("last-k", "behavior_direction", {
            "behavior": "response_length", "model_sha256": digest, "layers": [0], "last_k": 3,
        })
        assert fit["model_sha256"] == digest and fit["last_k"] == 3
        direction = fit["fit"]["layers"][0]["unit_vector"]
        assert abs(sum(v * v for v in direction) - 1) < 1e-6
        options = {"layers": [0], "roles": ["l_out"], "positions": [0], "top_k": 0,
                   "retain_jobs": 4, "byte_cap": 32768, "expected_model_sha256": digest}
        stock = run("stock", "trace", options, "a")
        projection = {"direction_id": "last-k", "source_layer": 0, "layer": 0,
                      "positions": [0], "last_k": 3, "alpha": 2}
        edited = run("overshoot", "trace", dict(options, gguf_projection=dict(projection, vector=direction)), "a")
        assert edited["gguf_projection"] == dict(projection, rank=1)
        assert stock["lanternfish_commit"] != edited["lanternfish_commit"]

        def values(job):
            row = call("trace-read", job, {"view": "tensor", "prompt": 0, "layer": 0, "role": "l_out",
                                          "start": 0, "end": 1})["data"]["rows"][0]
            return struct.unpack("<256f", base64.b64decode(row["data_base64"]))

        before, after = [sum(a * b for a, b in zip(values(job), direction)) for job in ("stock", "overshoot")]
        assert abs(before) > 1e-5 and abs(before + after) < 1e-7, (before, after)
        return {"model_sha256": digest, "fit_last_k": fit["last_k"], "before": before, "after": after,
                "projection": edited["gguf_projection"], "lanternfish_commit": edited["lanternfish_commit"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("bins", "cache", "trace", "trace-libs", "ablation", "ablation-libs"):
        parser.add_argument(f"--{argument}", type=lambda p: Path(p).absolute(), required=True)
    print(json.dumps(exercise(parser.parse_args()), indent=2))
