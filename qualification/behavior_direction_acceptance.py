"""Check CPU behavior directions on an isolated controller and worker."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen


ARB = Path("/arboretum")
PIN = ARB / "pins/lanternfish-d842085/lanternfish"
LIBS = ARB / "llama-a4/build/bin"
W0_PIN = ARB / "pins/lanternfish-0ea30b3/lanternfish"
W0_LIBS = ARB / "llama-w0/build/bin"
MODEL = ARB / "models/qwen2.5-0.5b-instruct-w0/Qwen2.5-0.5B-Instruct.bf16.gguf"
MODEL_SHA = "a33250ab73bbef0d876bf04e01517f5f43292dd8ede34e12d3a3a886e49f36f0"
BEHAVIORS = ("slop", "sycophancy", "response_length")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(args, *, pin=PIN, libs=LIBS, exercise_stack=None):
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=pin.parent, check=True)
    subprocess.run(["sha256sum", "-c", "bin.sha256"], cwd=args.bins, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="isolated-", dir=args.evidence) as directory:
        run = Path(directory)
        worker = run / "worker"
        worker.mkdir(mode=0o700)
        (run / "web").mkdir()
        config = run / "worker.json"
        config.touch(mode=0o600)
        save(config, {
            "root": str(worker), "engine": str(args.bins / "brewery-engine"),
            "source_roots": [str(MODEL.parent)], "disk_reserve_bytes": 0,
            "engine_env": {
                "BREWERY_LANTERNFISH_TRACE": str(W0_PIN),
                "BREWERY_LANTERNFISH_TRACE_LIB_DIR": str(W0_LIBS),
                "BREWERY_LANTERNFISH_PROJECTION": str(pin),
                "BREWERY_LANTERNFISH_PROJECTION_LIB_DIR": str(libs),
                "BREWERY_CONTRAST_CACHE": str(args.cache),
                "BREWERY_CPU_THREADS": "4",
            },
        })
        dispatch = run / "dispatch.json"
        save(dispatch, {"hosts": {"local": {"local": True,
            "worker": str(args.bins / "brewery-worker"), "config": str(config),
            "capabilities": {"behavior_direction": True, "trace": True},
        }}})
        env = {
            "PATH": "/usr/bin:/bin", "HOME": directory, "TMPDIR": str(ARB / "tmp"),
            "BREWERY_LISTEN": "127.0.0.1:0", "BREWERY_DATA": str(run / "state"),
            "BREWERY_WEB_DIST": str(run / "web"), "BREWERY_DISPATCH_CONFIG": str(dispatch),
            "BREWERY_DISPATCH": str(args.bins / "brewery-dispatch"), "BREWERY_POLL_MS": "250",
        }
        log = args.evidence / "controller.log"
        with log.open("w") as output, subprocess.Popen(
            [args.bins / "brewery"], env=env, stdout=output, stderr=output,
        ) as controller:
            try:
                for _ in range(100):
                    assert controller.poll() is None, "controller stopped"
                    listen = re.search(r"listening on (http://127\.0\.0\.1:(\d+))", log.read_text())
                    if listen:
                        break
                    time.sleep(0.1)
                assert listen and int(listen[2]) not in {0, 3000, 4095, 4141}
                if exercise_stack is None:
                    exercise(args.evidence, listen[1], commit)
                else:
                    exercise_stack(args.evidence, listen[1], commit, pin)
            finally:
                controller.terminate()


def exercise(evidence, base, commit):
    def request(path, body=None, headers=None):
        if body is not None:
            body = json.dumps(body).encode()
        req = Request(base + path, body, headers={
            "Content-Type": "application/json", **(headers or {}),
        })
        with urlopen(req, timeout=30) as response:
            return json.load(response)

    def mcp(name, **args):
        reply = request("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": args}},
            {"Accept": "application/json, text/event-stream"})
        assert not reply.get("error") and reply["result"]["isError"] is False, reply
        return reply["result"]["structuredContent"]

    model_sha = sha(MODEL)
    assert model_sha == MODEL_SHA
    model = request("/api/models", {"host": "local", "kind": "gguf", "path": str(MODEL)})
    summary = {"model_sha256": model_sha, "brewery_commit": commit,
               "lanternfish_sha256": sha(W0_PIN), "release_qualified": False, "directions": {}}
    for behavior in BEHAVIORS:
        queued = mcp("compute_direction", input_ids=[model["id"]],
                     idempotency_key=f"accept-{behavior}", options={
                         "behavior": behavior, "model_sha256": model_sha, "layers": list(range(24)),
                     })
        job_id = queued["id"]
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            job = mcp("get_job", id=job_id, wait_ms=5000)
            if job["state"] in {"succeeded", "failed", "cancelled"}:
                break
        save(evidence / f"{behavior}-job.json", job)
        assert job["state"] == "succeeded", job
        artifact = mcp("get_direction", id=job_id, version=2)
        assert artifact == request(f"/api/directions/{job_id}/versions/2")
        direction = artifact["direction"]
        assert direction == job["result"]
        assert artifact["version"] == direction["schema_version"] == 2
        assert direction["model_sha256"] == model_sha and direction["position"] is None
        assert direction["release_qualified"] is False
        fit = direction["fit"]
        layers = fit["layers"]
        assert (fit["train_pairs"], fit["held_out_pairs"]) == (12, 8)
        assert [x["layer"] for x in layers] == list(range(24))
        for layer in layers:
            assert len(layer["unit_vector"]) == 896
            assert abs(math.hypot(*layer["unit_vector"]) - 1) < 1e-5
            assert math.isfinite(layer["held_out_margin"]) and layer["norm"] > 0
            assert math.isfinite(layer["held_out_cohens_d"])
            assert math.isfinite(layer["mean_residual_norm"]) and layer["mean_residual_norm"] > 0
        best = max(layers, key=lambda x: x["held_out_cohens_d"])
        assert fit["selected_layer"] == best["layer"] and best["held_out_cohens_d"] > 0
        single = mcp("get_direction", id=job_id, version=2, layer=best["layer"])
        assert single["direction"]["fit"]["layers"] == [best]
        summary["directions"][behavior] = {
            "job_id": job_id, "contrast_sha256": direction["contrast_sha256"],
            "best": {key: val for key, val in best.items() if key != "unit_vector"},
            "layers": [{k: v for k, v in layer.items() if k != "unit_vector"} for layer in layers],
        }
        save(evidence / "report.json", summary)
        print(behavior, summary["directions"][behavior]["best"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--bins", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    for key in ("evidence", "bins", "cache"):
        path = getattr(args, key).resolve()
        if ARB not in path.parents:
            parser.error(f"--{key} must be inside /arboretum")
        setattr(args, key, path)
    args.evidence.mkdir(mode=0o700, parents=True, exist_ok=False)
    run(args)
