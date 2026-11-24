#!/usr/bin/env python3
"""Run an operator direction import, native HGN edit, and stored checks offline."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

from hgn_trace_job import digest, save, stop_job

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hgn_trace_fixture import fixture


def exercise(args, root):
    bins, source = args.bins.resolve(), root / "source"
    source.mkdir()
    (root / "worker").mkdir(mode=0o700)
    fixture(source, embedding_rows=[(1, 1)] * 4, editable=True)
    model = source / "model.hgn"
    pin = digest(model)
    save(root, "assets.json", {pin: {
        kind: {"path": str(source / name), "sha256": digest(source / name)}
        for kind, name in (("sidecar", "sidecar.hgn"), ("tokenizer", "tokenizer.json"),
                           ("config", "config.json"))
    }})
    token = os.urandom(32).hex()
    save(root, "operator.json", {"operator": "fixture-operator", "token": token})
    (root / "key").write_bytes(os.urandom(32))
    host = "h4i-fixture"
    # The whole test runs under the exclusive desktop build gate.
    admission = root / "admit"
    admission.write_text('#!/bin/sh\nset -eu\ntest "$1" = --\nshift\nexec "$@"\n')
    admission.chmod(0o700)
    save(root, "worker.json", {
        "root": str(root / "worker"), "source_roots": [str(source)],
        "engine": str(bins / "brewery-engine"), "disk_reserve_bytes": 0,
        "hgn_edit": {"host": host, "grant_key_file": str(root / "key"),
                     "build_admission": str(admission)},
        "engine_env": {
            "BREWERY_HGN_TRACE_PYTHON": sys.executable,
            "BREWERY_HGN_TRACE_PYTHON_SHA256": digest(Path(sys.executable)),
            "BREWERY_HGN_TRACE_ASSETS": str(root / "assets.json"),
            "BREWERY_CPU_THREADS": "1",
        },
    })
    # The controller requires the source-worker lane for HGN edits. Bind the
    # real dispatch binary to a local transport; no SSH command can run.
    save(root, "controller-dispatch.json", {"hosts": {
        host: {"ssh": host, "capabilities": {"scratch_hgn_ablation": True, "trace": True}}
    }})
    save(root, "local-dispatch.json", {"hosts": {
        host: {"local": True, "worker": str(bins / "brewery-worker"),
               "config": str(root / "worker.json")}
    }})
    shim = root / "dispatch"
    shim.write_text(
        f"#!/bin/sh\nset -eu\nexport BREWERY_DISPATCH_CONFIG={shlex.quote(str(root / 'local-dispatch.json'))}\n"
        f"exec {shlex.quote(str(bins / 'brewery-dispatch'))} \"$@\"\n"
    )
    shim.chmod(0o700)
    dist = root / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("Isolated HGN direction import")
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(root), "TMPDIR": str(root),
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "BREWERY_LISTEN": "127.0.0.1:0", "BREWERY_DATA": str(root / "state"),
        "BREWERY_WEB_DIST": str(dist), "BREWERY_POLL_MS": "75",
        "BREWERY_DISPATCH": str(shim), "BREWERY_DISPATCH_CONFIG": str(root / "controller-dispatch.json"),
        "BREWERY_HGN_EDIT_KEY_FILE": str(root / "key"),
        "BREWERY_DIRECTION_OPERATOR_FILE": str(root / "operator.json"),
    }
    log_path = root / "controller.log"
    with log_path.open("w") as log, subprocess.Popen([bins / "brewery"], env=env, stdout=log, stderr=log) as proc:
        active = None
        try:
            base = None
            for _ in range(100):
                assert proc.poll() is None, log_path.read_text()
                base = next((line.split()[-1] for line in log_path.read_text().splitlines()
                             if line.startswith("brewery listening on http://127.0.0.1:")), None)
                if base:
                    break
                time.sleep(0.1)
            assert base and int(base.rsplit(":", 1)[1]) not in {0, 3000, 4095, 4141}
            opener = build_opener(ProxyHandler({}))

            def send(path, data, headers, expected=200):
                try:
                    response = opener.open(Request(base + path, data, headers), timeout=30)
                except HTTPError as err:
                    response = err
                with response:
                    reply = json.load(response)
                    assert response.code == expected, (path, expected, response.code, reply)
                    return reply

            def call(name, **arguments):
                if name in {"register_model", "ablate_direction", "run_checks"}:
                    arguments["idempotency_key"] = f"h4i-{name}"
                reply = send("/mcp", json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                }).encode(), {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
                assert "error" not in reply and not reply["result"]["isError"], (name, reply)
                return reply["result"]["structuredContent"]

            def wait(job):
                nonlocal active
                active = job
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    result = call("get_job", id=job, wait_ms=3000)
                    if result["state"] in {"failed", "cancelled", "succeeded"}:
                        active = None
                        worker_log = root / f"worker/jobs/{job}/engine-stderr.log"
                        assert result["state"] == "succeeded", (
                            result, worker_log.read_text() if worker_log.exists() else log_path.read_text())
                        return result["result"]
                raise TimeoutError(job)

            model_id = call("register_model", host=host, path=str(model), kind="hgn-v2")["id"]
            vector = np.zeros(128, dtype=np.float32)
            vector[0] = 1
            npy = BytesIO()
            np.save(npy, vector, allow_pickle=False)
            payload = npy.getvalue()
            manifest = {
                "layer": 0, "position": -1, "model_sha256": pin,
                "npy_sha256": sha256(payload).hexdigest(), "provenance": "Synthetic e0 fixture",
            }

            def upload(note, expected):
                boundary = "brewery-h4i-smoke"
                body = bytearray()
                for name, data in [("model_id", model_id.encode()),
                                   ("manifest", json.dumps(note).encode()), ("vector", payload)]:
                    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
                    body.extend(data + b"\r\n")
                body.extend(f"--{boundary}--\r\n".encode())
                return send("/api/admin/directions/import", bytes(body), {
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Authorization": f"Bearer {token}",
                }, expected)

            upload(dict(manifest, model_sha256="f" * 64), 400)
            upload(dict(manifest, layer=2), 400)
            imported = upload(manifest, 201)
            direction = imported["id"]
            receipt = imported["direction"]["import"]
            assert receipt["operator"] == "fixture-operator" and receipt["manifest"] == manifest
            listed = call("list_directions")["directions"][0]
            assert listed["id"] == direction and listed["origin"] == "imported"
            result = call("get_direction", id=direction, version=2, layer=0,
                          reference_id=direction, reference_layer=0)
            assert result["direction"]["import"] == receipt
            assert result["reference"]["cosine_by_layer"] == [{"layer": 0, "cosine": 1}]
            child = wait(call("ablate_direction", id=model_id, sha256=pin,
                              direction={"id": direction, "version": 2, "layer": 0},
                              first_layer=0, last_layer=0, variant="plain",
                              kernel={"shape": "flat", "peak": 0, "width": 1, "min_weight": 0})["id"])
            revision = call("get_scratch", id=child["scratch_id"])
            assert revision["provenance"]["ablation"]["direction_import"] == receipt
            wait(call("run_checks", id=revision["id"], sha256=revision["sha256"])["id"])
            bundle = call("get_checks", id=revision["id"])
            assert bundle["direction_import"] == receipt
            assert bundle["integrity"]["status"] == bundle["smoke"]["status"] == "passed"
            assert bundle["qualification"] == "not release-qualified"
            stored = bundle["hgn_stored"]
            assert len(stored["tensors"]) == 3 and len(stored["prompts"]) == 2
            assert all(row["kl_parent_child"] > 0 for row in stored["prompts"])
            args.evidence.write_text(json.dumps({
                "import": receipt, "direction_id": direction, "cosine": 1,
                "revision_id": revision["id"], "check_bundle": bundle,
                "python_sha256": digest(Path(sys.executable)), "bad_model_hash_rejected": True,
                "out_of_range_layer_rejected": True,
            }, indent=2) + "\n")
        except Exception:
            print(log_path.read_text(), file=sys.stderr)
            raise
        finally:
            if active:
                stop_job(bins, dict(env, BREWERY_DISPATCH_CONFIG=str(root / "local-dispatch.json")), host, active)
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("bins", "scratch", "evidence"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    with tempfile.TemporaryDirectory(dir=args.scratch.resolve(), prefix="h4i-smoke-") as directory:
        exercise(args, Path(directory))
