#!/usr/bin/env python3
"""Run the X5 factual-recall sweep on an isolated CPU workbench."""

import argparse
import json
from pathlib import Path
from statistics import correlation
import subprocess
import tempfile
from urllib.request import Request, urlopen

from sensitivity_acceptance import ARB, MODEL, MODEL_SHA, require, save, sha, wait


MANIFEST_SHA = "a6b0816755e6030d22fdce0a140911a3e0d64b407bac6043289a5f090ae85eb6"
LAYERS = [0, 12, 23]
BINARIES = ["brewery", "brewery-dispatch", "brewery-worker", "brewery-engine"]


def run(args, directory):
    worker, state, dist = (directory / name for name in ("worker", "state", "web"))
    worker.mkdir(mode=0o700)
    dist.mkdir()
    (dist / "index.html").write_text("<title>Isolated X5 acceptance</title>")
    config, dispatch = (directory / name for name in ("worker.json", "dispatch.json"))
    save(config, {
        "root": str(worker), "source_roots": [str(MODEL.parent)],
        "engine": str(args.bins / "brewery-engine"), "disk_reserve_bytes": 0,
        "engine_env": {
            "BREWERY_LANTERNFISH_SENSITIVITY_PIN": str(args.pin),
            "BREWERY_LANTERNFISH_SENSITIVITY_MANIFEST_SHA256": MANIFEST_SHA,
            "BREWERY_CPU_THREADS": "4",
        },
    })
    config.chmod(0o600)
    save(dispatch, {"hosts": {"local": {
        "local": True, "worker": str(args.bins / "brewery-worker"),
        "config": str(config), "capabilities": {"circuit": True},
    }}})
    env = {"PATH": "/usr/bin:/bin", "HOME": str(directory), "TMPDIR": str(directory),
           "BREWERY_LISTEN": "127.0.0.1:0", "BREWERY_DATA": str(state),
           "BREWERY_WEB_DIST": str(dist), "BREWERY_DISPATCH": str(args.bins / "brewery-dispatch"),
           "BREWERY_DISPATCH_CONFIG": str(dispatch), "BREWERY_POLL_MS": "100"}
    log = args.evidence / "controller.log"
    with log.open("wb") as stream:
        controller = subprocess.Popen([args.bins / "brewery"], env=env, stdout=stream, stderr=stream)
    try:
        def listen():
            require(controller.poll() is None, f"Controller stopped: {log}")
            return next((line.rsplit(" ", 1)[1] for line in log.read_text().splitlines()
                         if line.startswith("brewery listening on http://127.0.0.1:")), None)

        origin = wait(listen, timeout=10)

        def http(path, data=None):
            headers = {"Content-Type": "application/json"}
            if path == "/mcp":
                headers["Accept"] = "application/json, text/event-stream"
            req = Request(origin + path, None if data is None else json.dumps(data).encode(), headers)
            with urlopen(req, timeout=10) as response:
                return json.load(response)

        model = http("/api/models", {"host": "local", "kind": "gguf", "path": str(MODEL)})
        options = {
            "model_sha256": MODEL_SHA,
            "clean_prompt": "The capital of France is", "corrupt_prompt": "The capital of Germany is",
            "target_id": 12095, "contrast_id": 19846,
            "layers": LAYERS, "positions": [0, 1, 2, 3, 4], "roles": ["l_out", "attn_out", "ffn_out"],
            "top_k": 10, "sample_lower": 5,
        }
        call = http("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "find_circuit", "arguments": {
                "idempotency_key": "x5-france-germany-cpu", "input_ids": [model["id"]], "options": options,
            },
        }})
        require(call.get("result", {}).get("isError") is False, call)
        job_id = call["result"]["structuredContent"]["id"]

        def terminal():
            job = http(f"/api/jobs/{job_id}")
            return job if job["state"] in ("succeeded", "failed", "cancelled") else None

        job = wait(terminal, timeout=1800)
        save(args.evidence / "job.json", job)
        for filename in ("engine-stderr.log", "supervisor.log"):
            file = worker / "jobs" / job_id / filename
            if file.exists():
                (args.evidence / filename).write_bytes(file.read_bytes()[-32768:])
        require(job["state"] == "succeeded", job)
        endpoint = f"/api/jobs/{job_id}/circuit"
        receipt = http(endpoint)
        save(args.evidence / "artifact.json", receipt)
        require(job["result"] == receipt and receipt["backend"] == "cpu", "Wrong receipt or backend")
        require(receipt["clean_tokens"] == [785, 6722, 315, 9625, 374]
                and receipt["corrupt_tokens"] == [785, 6722, 315, 9856, 374], "Wrong tokenization")
        require(receipt["clean_metric"] > 0 > receipt["corrupt_metric"], "The model did not recall the two capitals")
        sites = receipt["sites"]
        measured = [(rank, s) for rank, s in enumerate(sites, 1) if s["patched_delta"] is not None]
        require(len(sites) == len(LAYERS) * 5 * 3 and len(measured) == 15
                and [s for _, s in measured[:10]] == sites[:10], "Wrong sweep coverage")
        x = [s["attribution"] for _, s in measured]
        y = [s["patched_delta"] for _, s in measured]
        def ranks(values):
            ordered = sorted(values)
            return [(ordered.index(x) + len(ordered) - ordered[::-1].index(x) - 1) / 2 for x in values]
        require(abs(correlation(x, y) - receipt["pearson_r"]) < 1e-12
                and abs(correlation(ranks(x), ranks(y)) - receipt["spearman_rho"]) < 1e-12,
                "Reported correlation does not match the measured pairs")
        require(measured[0][1]["patched_delta"] > 0, "The strongest patch did not increase the Paris margin")
        return {"artifact_route": endpoint, "pearson_r": receipt["pearson_r"],
                "spearman_rho": receipt["spearman_rho"],
                "clean_metric": receipt["clean_metric"], "corrupt_metric": receipt["corrupt_metric"],
                "site_count": len(sites),
                "measured_sites": [{"rank": rank, **site} for rank, site in measured],
                "qualification": receipt["qualification"]}
    finally:
        controller.terminate()
        controller.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bins", "evidence", "pin"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=ARB / "xa-scratch/x5")
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    require(sha(MODEL) == MODEL_SHA and sha(args.pin / "SHA256SUMS") == MANIFEST_SHA, "Input pin mismatch")
    subprocess.run(["sha256sum", "--check", "--quiet", "SHA256SUMS"], cwd=args.pin, check=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="x5-", dir=args.scratch.resolve()) as directory:
        report = run(args, Path(directory))
    report["binary_sha256"] = {name: sha(args.bins / name) for name in BINARIES}
    save(args.evidence / "report.json", report)
    print(f"CPU Pearson r = {report['pearson_r']:.6f}, Spearman rho = {report['spearman_rho']:.6f}; "
          f"{report['site_count']} sites, {len(report['measured_sites'])} measured; not release-qualified")


if __name__ == "__main__":
    main()
