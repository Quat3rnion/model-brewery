"""Exercise managed weight jobs on a fresh loopback controller and real CPU tools."""
import argparse
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import time
from urllib.error import HTTPError

from gguf_workflow_smoke import checksum, controller, request, save

REPO = Path(__file__).resolve().parents[1]
HOST = "fixture"
KINDS = ("weight_inspect", "weight_propose", "weight_tune", "weight_apply")
TENSOR = "layer.weight"


def matrix(weights):
    name = TENSOR.encode()
    header = struct.pack("<4sIQQQ", b"GGUF", 3, 1, 0, len(name)) + name
    header += struct.pack("<IQQIQ", 2, 2, 2, 1, 0)  # rank two, F16
    return header + bytes(-len(header) % 32) + struct.pack("<4e", *weights)


def setup(root, bins, gguf_python):
    (root / "bin").symlink_to(bins)
    inputs = root / "inputs"
    inputs.mkdir(mode=0o700)
    (root / "worker").mkdir(mode=0o700)
    environment = {"BREWERY_CPU_THREADS": "2", "BREWERY_WEIGHT_VISUAL": "1"}
    packaged = [bins / f"brewery-weight-{kind}" for kind in ("edit", "tune")]
    if any(p.exists() for p in packaged) and not all(p.is_file() for p in packaged):
        raise ValueError("incomplete packaged weight runtime")
    for kind in ("edit", "tune"):
        helper = bins / f"brewery-weight-{kind}"
        if not helper.is_file():
            helper = root / f"python-weight-{kind}"
            command = ["/usr/bin/env", f"PYTHONPATH={gguf_python}", "/usr/bin/python3", "-B",
                       str(REPO / "tools" / f"weight_{kind}.py")]
            helper.write_text(f"#!/bin/sh\nexec {shlex.join(command)} \"$@\"\n")
            helper.chmod(0o700)
        environment[f"BREWERY_WEIGHT_{kind.upper()}"] = str(helper)
    save(root / "worker.json", {"root": str(root / "worker"), "source_roots": [str(inputs)],
                                 "engine": str(bins / "brewery-engine"), "engine_env": environment})
    save(root / "dispatch.json", {"hosts": {HOST: {
        "local": True, "worker": str(bins / "brewery-worker"), "config": str(root / "worker.json"),
         "capabilities": dict.fromkeys((*KINDS, "weight_visual"), True),
    }}})
    return inputs


def hgn_matrix(weights):
    # Single native v2 BF16 matrix; no GGUF payload or external dependencies.
    raw = b"".join(struct.pack("<f", v)[2:] for v in weights)
    start = 320  # align64(104 + 160)
    header = struct.pack("<4sI4Q64s", b"HGN1", 2, 1, 104, start, start+len(raw), b"weight-fixture")
    a, b = struct.unpack("<II", raw)
    entry = struct.pack("<96sII4QQQII", TENSOR.encode(), 0, 2, 2, 2, 0, 0,
                        start, len(raw), a ^ b, 0)
    return header + entry + bytes(start-len(header)-len(entry)) + raw


def run(root, bins, gguf_python, fmt="gguf"):
    inputs = setup(root, bins, gguf_python)
    make_matrix = hgn_matrix if fmt == "hgn" else matrix
    if fmt == "hgn":
        for filename in ("dispatch.json", "worker.json"):
            path = root / filename
            cfg = json.loads(path.read_text())
            if filename == "worker.json":
                cfg["engine_env"]["BREWERY_WEIGHT_HGN"] = "1"
            else:
                cfg["hosts"][HOST]["capabilities"]["weight_hgn"] = True
            save(path, cfg)
    source = inputs / f"source.{fmt}"
    original = make_matrix((1, 2, 3, 4))
    source.write_bytes(original)
    other = inputs / f"other.{fmt}"
    other.write_bytes(make_matrix((6, 2, 3, 4)))
    for name, activations, targets in (
        ("calibration", [[1, 0], [0, 1]], [[2, 3], [2, 3]]),
        ("held-out", [[1, 1], [2, -1]], [[3, 7], [0, 2]]),
        ("overlap", [[1, 0]], [[100, 100]]),
    ):
        save(inputs / f"{name}.json", {"activations": activations, "targets": targets})
    digest = checksum(source)

    with controller(root, 0) as url:
        assert url.startswith("http://127.0.0.1:") and url != "http://127.0.0.1:4141"

        def http(route, body=None):
            return request(url, route, body)[0]

        def register(filename, kind):
            return http("models", {"name": filename, "host": HOST, "kind": kind,
                                   "path": str(inputs / filename)})["id"]

        def payload(kind, ids, options):
            return {"kind": kind, "host": HOST, "input_ids": ids, "options": options}

        def job(label, kind, ids, options, expect="succeeded"):
            ident = http("jobs", payload(kind, ids, options))["id"]
            deadline = time.monotonic() + 20
            while True:
                detail = http(f"jobs/{ident}")
                if detail["state"] in ("succeeded", "failed", "cancelled"):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"timed out: {label} {ident}")
                time.sleep(0.1)
            save(root / f"{label}.json", detail)
            assert detail["state"] == expect, (label, detail.get("error"), detail)
            return detail

        def models():
            return {m["id"]: m for m in http("state")["models"]}

        def refuse(label, ids, options):
            before = {j["id"] for j in http("state")["jobs"]}
            try:
                http("jobs", payload("weight_apply", ids, options))
            except HTTPError as exc:
                body = exc.read().decode()
                assert exc.code == 400, (label, exc.code, body)
                save(root / f"refused-{label}.json", {"status": exc.code, "body": body})
            else:
                raise AssertionError(f"{label} was accepted")
            assert {j["id"] for j in http("state")["jobs"]} == before

        state = http("state")
        assert state["models"] == [] and state["jobs"] == []
        assert all(state["hosts"][0]["capabilities"].get(k) for k in KINDS), "release lacks managed weight jobs"
        if fmt == "hgn":
            assert state["hosts"][0]["capabilities"].get("weight_hgn"), "release lacks HGN editing"
        source_id = register(source.name, fmt)
        other_id = register(other.name, fmt)
        cal, held, overlap = [register(f"{p}.json", "weight-calibration")
                              for p in ("calibration", "held-out", "overlap")]
        initial = models()
        inspected = job("inspect", "weight_inspect", [source_id], {"tensor": TENSOR, "index": [0, 1]})
        assert inspected["result"]["report"]["value"] == 2
        inventory = job("inventory", "weight_inspect", [source_id],
                        {"mode": "inventory", "offset": 0, "limit": 16})["result"]["report"]
        pin = inventory["source_sha256"]
        assert pin == digest and inventory["total"] == 1
        tile_opts = {"mode": "tile", "tensor": TENSOR, "origin": [0, 0], "shape": [2, 2],
                     "source_sha256": pin}
        tile = job("tile", "weight_inspect", [source_id], tile_opts)["result"]["report"]
        assert tile["source_sha256"] == pin and tile["values"] == [[1, 2], [3, 4]]
        job("stale-tile", "weight_inspect", [other_id], tile_opts, expect="failed")
        manual = job("manual-review", "weight_propose", [source_id], {"source_sha256": pin, "edits": [
            {"tensor": TENSOR, "index": [0, 1], "op": "delta", "value": 0.5},
        ]})
        options = {"tensor": TENSOR, "indices": [[0, 0], [1, 1]], "max_update": 1,
                   "source_sha256": pin}
        auto = job("auto-review", "weight_tune", [source_id, cal, held], options)
        tuning = auto["result"]["report"]["tune"]
        assert tuning["calibration"] == {"before_mse": 0.5, "after_mse": 0}
        assert tuning["held_out"] == {"before_mse": 0, "after_mse": 1.75, "regression": True}
        declined = job("overlap", "weight_tune", [source_id, cal, overlap], options, expect="failed")
        # The service intentionally does not expose private helper diagnostics.
        # Check the job failed in the real helper, then replay the isolated
        # inputs to distinguish split overlap from unrelated helper errors.
        private_log = root / "worker/jobs" / declined["id"] / "engine-stderr.log"
        assert "weight helper rejected the job" in private_log.read_text()
        replay = subprocess.run([root / "python-weight-tune", source, "--source-sha256", digest,
                                 "--tensor", TENSOR, "--indices", json.dumps(options["indices"]),
                                 "--max-update", "1", "--calibration", inputs/"calibration.json",
                                 "--held-out", inputs/"overlap.json"], capture_output=True,
                                text=True, timeout=10)
        assert replay.returncode and "overlap" in replay.stderr.lower(), replay.stderr
        assert models() == initial and source.read_bytes() == original
        assert not list((root / "worker").glob("jobs/*/work/edited.*"))

        for label, ident, proposal_id, reviewed in (
            ("unreviewed", source_id, manual["id"], False),
            ("wrong-source", other_id, manual["id"], True),
            ("inspect-is-not-proposal", source_id, inspected["id"], True),
        ):
            refuse(label, [ident], {"proposal_job_id": proposal_id, "reviewed": reviewed})

        results = {}
        for label, review, edits, expected in (
            ("manual", manual, [([0, 1], 2, 2.5)], (1, 2.5, 3, 4)),
            ("auto", auto, [([0, 0], 1, 2), ([1, 1], 4, 3)], (2, 2, 3, 3)),
        ):
            reviewed = review["result"]["report"]["proposal"]
            assert reviewed["schema"] == "brewery.weight-edit-proposal.v1"
            assert reviewed["source_sha256"] == digest
            assert [(e["index"], e["before"], e["after"]) for e in reviewed["edits"]] == edits
            dtype = "BF16" if fmt == "hgn" else "F16"
            assert all(e["tensor"] == TENSOR and e["dtype"] == dtype for e in reviewed["edits"])
            final = job(f"{label}-apply", "weight_apply", [source_id],
                        {"proposal_job_id": review["id"], "reviewed": True})
            model = models()[final["result"]["library_model_id"]]
            path = Path(model["path"])
            assert model["host"] == HOST and model["kind"] == fmt
            assert path == root / "worker" / "jobs" / final["id"] / f"work/edited.{fmt}"
            assert final["result"]["verified"] is True
            assert path.read_bytes() == make_matrix(expected)
            assert source.read_bytes() == original
            results[label] = {"job_id": final["id"], "weights": expected, "model_id": model["id"],
                              "path": str(path), "sha256": checksum(path)}
        assert len(models()) == len(initial) + 2
        return {"scratch": str(root), "source": str(source), "source_sha256": digest,
                "outputs": results, "calibration": tuning["calibration"], "held_out": tuning["held_out"],
                "rejected": ["overlap", "unreviewed", "wrong-source", "inspect-is-not-proposal"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--gguf-python", required=True, type=Path)
    parser.add_argument("--format", choices=("gguf", "hgn"), default="gguf")
    args = parser.parse_args()
    if not (args.gguf_python / "gguf/gguf_reader.py").is_file():
        parser.error("--gguf-python must name the gguf-py directory")
    # Ignore all inherited service configuration; controller() sets the
    # production-variable names to fresh scratch data and an ephemeral port.
    for key in tuple(os.environ):
        if key.startswith("BREWERY_"):
            del os.environ[key]
    os.umask(0o077)
    os.environ["TMPDIR"] = "/arboretum"
    root = Path(tempfile.mkdtemp(prefix="managed-weight-smoke.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    result = run(root, args.bin_dir.resolve(strict=True), args.gguf_python.resolve(strict=True), args.format)
    save(root / "summary.json", result)
    print(json.dumps(result))
