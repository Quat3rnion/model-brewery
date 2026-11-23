#!/usr/bin/env python3
"""Check X3 on an isolated workbench and compare selected CPU/CUDA gradients."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen

ARB = Path("/arboretum")
MODEL = ARB / "models/qwen2.5-0.5b-instruct-w0/Qwen2.5-0.5B-Instruct.bf16.gguf"
MODEL_SHA = "a33250ab73bbef0d876bf04e01517f5f43292dd8ede34e12d3a3a886e49f36f0"
CORPUS = Path(__file__).resolve().parents[1] / "crates/eval/data/edit-checks/heldout.txt"
CORPUS_SHA = "9678f9afdc31b2a9be7fec2045997e2360693ee72b893287c0d539983bff4923"
# A value projection has a nonzero derivative even for one attended token.
TENSOR = "blk.0.attn_v.weight"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def wait(get, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := get():
            return value
        time.sleep(0.2)
    raise TimeoutError("Acceptance operation timed out")


def accept(args, run):
    dist = run / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<title>Isolated X3 acceptance</title>")
    worker, state = run / "worker", run / "state"
    worker.mkdir(mode=0o700)
    queue = [
        str(args.queue), "run", "--resource", "gpu", "--exclusive", "--flock", "gpu",
        "--dir", str(args.queue_dir),
        "--holder", "brewery-x3-acceptance", "--purpose", "X3 Qwen Fisher acceptance",
        "--wait", "120", "--max-hold", "120", "--",
    ]
    admission = run / "admission"
    # The worker reads the engine result on stdout. Send the queue's own logs
    # to stderr. The admitted engine writes a private result in the job directory.
    admission.write_text(
        "#!/bin/sh\nset -eu\n[ \"$#\" -eq 3 ] && [ \"$1\" = --wait ] && [ \"$2\" = -- ]\n"
        "shift 2\ntrap 'rm -f .admitted-output' EXIT\n"
        f"{shlex.join(queue)} /bin/sh -c 'exec \"$@\" > .admitted-output' sh \"$@\" >&2\n"
        "cat .admitted-output\n"
    )
    admission.chmod(0o700)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(run), "TMPDIR": str(ARB / "tmp"),
           "BREWERY_LISTEN": "127.0.0.1:0", "BREWERY_DATA": str(state),
           "BREWERY_WEB_DIST": str(dist), "BREWERY_DISPATCH": str(args.bins / "brewery-dispatch"),
           "BREWERY_DISPATCH_CONFIG": str(run / "dispatch.json"), "BREWERY_POLL_MS": "100"}
    save(run / "worker.json", {
        "root": str(worker), "source_roots": [str(MODEL.parent)],
        "engine": str(args.bins / "brewery-engine"), "admission": str(admission), "disk_reserve_bytes": 0,
        "engine_env": {
            "BREWERY_LANTERNFISH_SENSITIVITY_PIN": str(args.pin),
            "BREWERY_LANTERNFISH_SENSITIVITY_MANIFEST_SHA256": args.manifest_sha,
            "BREWERY_ALLOW_GPU_LOAD": "1", "BREWERY_CPU_THREADS": "4",
            "CUDA_VISIBLE_DEVICES": args.cuda_device,
        },
    })
    (run / "worker.json").chmod(0o600)
    save(run / "dispatch.json", {"hosts": {"local": {
        "local": True, "worker": str(args.bins / "brewery-worker"),
        "config": str(run / "worker.json"), "capabilities": {"sensitivity": True},
    }}})
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
            request = Request(origin + path, None if data is None else json.dumps(data).encode(), headers)
            with urlopen(request, timeout=10) as response:
                return json.load(response)

        model = http("/api/models", {"host": "local", "kind": "gguf", "path": str(MODEL)})
        options = {"model_sha256": MODEL_SHA, "corpus_sha256": CORPUS_SHA,
                   "tensors": [TENSOR], "observations": 1}
        receipts = {}
        for backend in ("cpu", "cuda"):
            request = {"input_ids": [model["id"]], "options": {**options, "backend": backend}}
            if backend == "cpu":
                job_id = http("/api/jobs", {"kind": "sensitivity", **request})["id"]
            else:
                call = http("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                    "name": "tensor_sensitivity",
                    "arguments": {"idempotency_key": "x3-cuda-acceptance", **request},
                }})
                require(call.get("result", {}).get("isError") is False, call)
                job_id = call["result"]["structuredContent"]["id"]

            def terminal():
                job = http(f"/api/jobs/{job_id}")
                return job if job["state"] in ("succeeded", "failed", "cancelled") else None

            job = wait(terminal, timeout=300)
            save(args.evidence / f"{backend}-job.json", job)
            logs = worker / "jobs" / job_id
            for filename in ("engine-stderr.log", "supervisor.log"):
                file = logs / filename
                if file.exists():
                    (args.evidence / f"{backend}-{filename}").write_bytes(file.read_bytes()[-32768:])
            require(job["state"] == "succeeded", job)
            receipt = http(f"/api/jobs/{job_id}/sensitivity")
            require(job["result"] == receipt and receipt["backend"] == backend, "Invalid job result")
            receipts[backend] = receipt
        return receipts
    finally:
        controller.terminate()
        controller.wait(timeout=10)


def compare(args, run, receipts):
    line = CORPUS.read_text().splitlines()[0]

    def call(backend, command, queued=False):
        extra = {"CUDA_VISIBLE_DEVICES": args.cuda_device if backend == "cuda" else "",
                 "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "NVIDIA_TF32_OVERRIDE": "0",
                 "GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32", "OMP_NUM_THREADS": "4", "LANTERNFISH_THREADS": "4"}
        env = {**os.environ, **extra, "LD_LIBRARY_PATH": str(args.pin / "lib" / backend)}
        launch = ([str(args.queue), "run", "--resource", "gpu", "--exclusive", "--flock", "gpu",
                   "--dir", str(args.queue_dir), "--holder", "brewery-x3-vector-parity", "--wait", "120",
                   "--max-hold", "120", "--"] if queued else [])
        return subprocess.check_output([*launch, str(args.pin / "lanternfish"), *command],
                                       env=env, cwd=run, timeout=250)

    def tokens(text):
        return json.loads(call("cpu", ["tokenize", "--model", str(MODEL), "--text", text]))["tokens"]

    prefix, full = tokens(line.split(" ", 1)[0]), tokens(line)
    require(full[:len(prefix)] == prefix, "Prefix mismatch")
    obs = {"tokens": prefix, "target_id": full[len(prefix)]}
    encoded = json.dumps([obs], separators=(",", ":")).encode()
    spec = run / "spec.json"
    save(spec, {**obs, "site": {"kind": "weight", "name": TENSOR}})
    metadata, vectors = {}, {}
    for backend, device in (("cpu", "CPU"), ("cuda", "CUDA0")):
        out = run / backend
        call(backend, ["grad-select", "--model", str(MODEL), "--spec", str(spec),
                       "--out", str(out), "--backend", backend], queued=backend == "cuda")
        row = json.loads((out / "gradient.json").read_text())
        require(row["placement"] == {"device": device, "strict": True}, "Backend fallback")
        require(row["sha256"] == sha(out / "gradient.f32"), "Gradient SHA-256 mismatch")
        vector = [x for x, in struct.iter_unpack("<f", (out / "gradient.f32").read_bytes())]
        fisher = sum(x*x for x in vector) / len(vector)
        receipt = receipts[backend]
        require(receipt["observations_sha256"] == hashlib.sha256(encoded).hexdigest(), "Observation mismatch")
        require(abs(fisher - receipt["tensors"][0]["fisher"]) <= max(fisher, 1e-40) * 1e-12, "Fisher mismatch")
        vectors[backend] = vector
        metadata[backend] = row
    cpu, cuda = vectors["cpu"], vectors["cuda"]
    denominator = max(map(abs, cpu))
    require(denominator > 0, "Parity requires a nonzero CPU gradient")
    rel_linf = max(abs(x-y) for x, y in zip(cpu, cuda, strict=True)) / denominator
    require(rel_linf <= 1e-3, f"Relative L-infinity failed: {rel_linf}")
    return {"relative_linf": rel_linf, "limit": 1e-3, "gradient_count": len(cpu),
            "gradient_metadata": metadata, "artifacts": receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("bins", "pin", "evidence", "queue", "queue-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--cuda-device", default="0")
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    require(sha(MODEL) == MODEL_SHA and sha(CORPUS) == CORPUS_SHA, "Input pin mismatch")
    require(sha(args.pin / "SHA256SUMS") == args.manifest_sha, "Runtime manifest pin mismatch")
    subprocess.run(["sha256sum", "--check", "--quiet", "SHA256SUMS"], cwd=args.pin, check=True)
    with tempfile.TemporaryDirectory(prefix="x3-run-", dir=args.evidence) as tmp:
        run = Path(tmp)
        receipts = accept(args, run)
        report = compare(args, run, receipts)
    report["binary_sha256"] = {b.name: sha(b) for b in args.bins.iterdir()}
    report["qualification"] = "not release-qualified"
    save(args.evidence / "report.json", report)
    print(f"CPU/CUDA relative L-infinity: {report['relative_linf']:.6g} (limit {report['limit']}); not release-qualified")


if __name__ == "__main__":
    main()
