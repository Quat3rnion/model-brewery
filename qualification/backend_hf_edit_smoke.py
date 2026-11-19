#!/usr/bin/env python3
"""Download, review, apply, and re-open real backend-hf Library artifacts.

Requires matched Brewery binaries, pinned Rust hfr, and Python with NumPy and
gguf-py. Runs only in a private Linux network/PID namespace, with a synthetic
Hub and no credentials. Keeps job receipts and wire evidence under /arboretum.
"""

import argparse
import ctypes
from hashlib import file_digest
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from tempfile import mkdtemp
from urllib.error import HTTPError
from urllib.request import Request

import managed_hfr_smoke as hf
from managed_safetensors_smoke import ORIGINAL, SHARD, TENSOR, describe, digest, fixture
from managed_weight_smoke import matrix
from managed_packed_weight_smoke import fixture as packed_fixture

HOST = "backend-hf"
EDITED = (1, 2.5, 3, 4)
REPO_ROOT = Path(__file__).resolve().parents[1]


def submit(smoke, label, kind, options, inputs=()):
    body = {"kind": kind, "options": options, "input_ids": list(inputs)}
    # Hostless download selects the reserved backend lane. The editor infers
    # that same lane from the trusted downloaded (or edited) Library ID.
    hf.save(smoke.root / f"{label}-request.json", body)
    job = smoke.call("jobs", body, status=201, key=f"backend-hf-edit.{label}")
    hf.need(job["host"] == HOST, f"{label}: job did not use the backend lane")
    return smoke.wait(label, job["id"], "succeeded")


def edit_case(smoke, lane, kind):
    filename = SHARD if kind == "safetensors" else None
    options = {"repo_id": hf.REPO, "revision": "main"}
    if filename:
        options["filename"] = filename
    download = submit(smoke, f"{kind}-download", "hf_download", options)
    receipt = download["result"]
    source_id = receipt["library_model_id"]
    model = smoke.models()[source_id]
    source = lane / "worker/jobs" / download["id"] / "work/download"
    if filename:
        source /= filename
    expected = dict(hf.FILES)
    if filename:
        expected = {"": expected[filename]}
    source_pin = digest(expected)
    hf.need(receipt["verified"] is True and receipt["commit"] == hf.COMMIT
            and receipt["artifact"]["sha256"] == source_pin
            and (model["host"], model["kind"], model["path"]) == (HOST, kind, str(source)),
            "download did not register the expected verified backend artifact")
    before = describe(source)
    hf.need({name: row[0] for name, row in before.items()} == expected, "wrong downloaded bytes")

    def tile(label, model_id, pin, values):
        detail = submit(smoke, f"{kind}-{label}", "weight_inspect", {
            "mode": "tile", "tensor": TENSOR, "origin": [0, 0], "shape": [2, 2],
            "source_sha256": pin,
        }, [model_id])
        result = detail["result"]
        report = result["report"]
        hf.need(result["source_format"] == kind and report["source_sha256"] == pin
                and report["values"] == [list(values[:2]), list(values[2:])],
                f"{label}: wrong tile or source identity")
        return detail["id"]

    inspected = tile("inspect", source_id, source_pin, ORIGINAL)
    models_before = smoke.models()
    proposal = submit(smoke, f"{kind}-propose", "weight_propose", {
        "source_sha256": source_pin, "edits": [
            {"tensor": TENSOR, "index": [0, 1], "op": "delta", "value": 0.5},
        ],
    }, [source_id])
    review = proposal["result"]["report"]["proposal"]
    hf.need(review["source_sha256"] == source_pin and len(review["edits"]) == 1, "wrong review")
    edit = review["edits"][0]
    hf.need((edit["tensor"], edit["index"], edit["dtype"], edit["before"], edit["after"])
            == (TENSOR, [0, 1], "BF16", 2, 2.5), "proposal changed the wrong value")
    hf.need(smoke.models() == models_before and describe(source) == before,
            "propose wrote weights or registered a revision")
    applied = submit(smoke, f"{kind}-apply", "weight_apply",
                     {"proposal_job_id": proposal["id"], "reviewed": True}, [source_id])
    result = applied["result"]
    edited_id = result["library_model_id"]
    models = smoke.models()
    revised = models[edited_id]
    output = lane / "editor/jobs" / applied["id"] / "work" / f"edited.{kind}"
    expected = {**hf.FILES, **fixture("hf", EDITED)}
    if filename:
        expected = {"": expected[filename]}
    output_pin = digest(expected)
    hf.need(edited_id not in models_before and set(models) == set(models_before) | {edited_id}
            and all(models[id] == row for id, row in models_before.items())
            and (revised["host"], revised["kind"], revised["path"]) == (HOST, kind, str(output)),
            "apply did not register exactly one separate same-format artifact")
    hf.need(result["verified"] is True and result["source_format"] == kind
            and result["source_sha256"] == source_pin and result["artifact"]["sha256"] == output_pin
            and result["proposal_sha256"] == proposal["result"]["artifact"]["sha256"],
            "apply did not verify the reviewed edit")
    after = describe(output)
    hf.need({name: row[0] for name, row in after.items()} == expected
            and not ({row[1] for row in before.values()} & {row[1] for row in after.values()}),
            "revision changed unexpected bytes or reused original inodes")
    reopened = tile("reopen", edited_id, output_pin, EDITED)
    hf.need(describe(source) == before, "original download changed")
    return {"download_job_id": download["id"], "inspect_job_id": inspected,
            "download_model_id": source_id, "proposal_job_id": proposal["id"],
            "apply_job_id": applied["id"], "revision_model_id": edited_id,
            "reopen_job_id": reopened, "source_sha256": source_pin, "output_sha256": output_pin}


def refuse_paths(smoke, lane):
    # This is NOT the configured token path. The probes have no credentials.
    # Watch both directory and file to detect an attempted open or read.
    credentials = lane / "credentials"
    canary = credentials / "synthetic.safetensors"
    canary.write_bytes(fixture("hf")[SHARD])
    unrelated = smoke.root / "unregistered.safetensors"
    unrelated.write_bytes(fixture("hf")[SHARD])
    probes = ((credentials, "hf"), (canary, "safetensors"), (unrelated, "safetensors"))
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    hf.need(fd >= 0, "cannot monitor canaries")
    before = smoke.call("state")
    results = []
    try:
        for path, _ in probes:
            hf.need(libc.inotify_add_watch(fd, os.fsencode(path), 0x21) >= 0,
                    f"cannot watch synthetic probe {path.name}")  # IN_ACCESS | IN_OPEN
        for path, kind in probes:
            request = Request(f"{smoke.url}/api/models", json.dumps({
                "name": f"untrusted {path.name}", "host": HOST, "kind": kind, "path": str(path),
            }).encode(), {"Content-Type": "application/json"})
            try:
                with smoke.opener.open(request, timeout=4) as response:
                    raise ValueError(f"untrusted path registered (HTTP {response.status})")
            except HTTPError as error:
                with error:
                    hf.need(error.code in (400, 409), f"unexpected rejection status {error.code}")
                    results.append({"probe": path.name, "status": error.code,
                                    "detail": error.read(4096).decode()})
        try:
            opened = os.read(fd, 4096)
        except BlockingIOError:
            opened = b""
        hf.need(not opened, "backend opened or read a synthetic untrusted path")
    finally:
        os.close(fd)
    hf.need(smoke.call("state") == before, "refusal changed controller state")
    hf.save(smoke.root / "untrusted-paths.json", results)
    return results


def gguf_case(smoke, lane, packed):
    label = "packed" if packed else "float"
    filename = f"{label}.gguf"
    source = submit(smoke, f"{label}-download", "hf_download",
                    {"repo_id": hf.REPO, "filename": filename})
    source_pin = source["result"]["artifact"]["sha256"]
    original_path = Path(source["result"]["artifact"]["path"])
    original = hf.FILES[filename]
    source_id = source["result"]["library_model_id"]
    if packed:
        one = [1] + [0] * 31
        two = [0] * 16 + [1] + [0] * 15
        train = ([one, two], [[1], [1.5]])
        held = ([[a+b for a, b in zip(one, two)]], [[0]])
        indices = [[0, 0], [0, 16]]
        expected = packed_fixture("Q8_0", (2, 3))
    else:
        train = ([[1, 0], [0, 1]], [[2, 3], [2, 3]])
        held = ([[1, 1], [2, -1]], [[3, 7], [0, 2]])
        indices = [[0, 0], [1, 1]]
        expected = matrix((2, 2, 3, 3))
    data = [smoke.call("models/weight-calibration",
                       {"name": f"{label}-{n}", "activations": batch[0], "targets": batch[1]},
                       status=201) for n, batch in (("train", train), ("held", held))]
    tuning = {"tensor": TENSOR, "indices": indices, "max_update": 1.5,
              "source_sha256": source_pin}
    if packed:
        tuning["packed"] = True
    tune = submit(smoke, f"{label}-tune", "weight_tune", tuning,
                  [source_id, data[0]["id"], data[1]["id"]])
    r = tune["result"]
    for key, model in zip(("calibration_sha256", "held_out_sha256"), data):
        expected_pin = Path(model["path"]).stem
        hf.need(r["report"]["tune"][key] == expected_pin, "tune did not bind calibration bytes")
    other = next(m["id"] for m in smoke.models().values() if m["kind"] == "safetensors")
    smoke.call("jobs", {"kind": "weight_apply", "options": {
        "proposal_job_id": tune["id"], "reviewed": True}, "input_ids": [other]}, status=400)
    applied = submit(smoke, f"{label}-apply", "weight_apply",
                     {"proposal_job_id": tune["id"], "reviewed": True}, [source_id])
    result = applied["result"]
    revision = Path(result["artifact"]["path"])
    hf.need(revision == lane / "editor/jobs" / applied["id"] / "work/edited.gguf"
            and original_path.read_bytes() == original
            and revision.read_bytes() == expected and result["artifact"]["kind"] == "gguf"
            and result["inputs"][1]["sha256"] == r["artifact"]["sha256"]
            and (not packed or result["proposal_sha256"] == r["artifact"]["sha256"]),
            "reviewed GGUF apply did not publish exactly the expected native bytes")
    opts = {"mode": "tile", "tensor": TENSOR, "origin": [0, 0], "shape": [1, 2],
            "source_sha256": result["artifact"]["sha256"]}
    if packed:
        opts["packed"] = True
    submit(smoke, f"{label}-reopen", "weight_inspect", opts, [result["library_model_id"]])
    return {"tune": tune["id"], "apply": applied["id"], "revision": result["library_model_id"]}


def run(root, args):
    for name in ("home", "tmp", "web"):
        (root / name).mkdir()
    for kind, jobs in (("edit", ["weight_inspect", "weight_propose", "weight_apply"]),
                       ("tune", ["weight_tune"])):
        helper = root / f"brewery-weight-{kind}"
        packaged = args.bin_dir / helper.name
        if packaged.is_file():
            helper.symlink_to(packaged)
            continue
        command = ["/usr/bin/env", f"PYTHONPATH={args.gguf_python}", sys.executable,
                   "-B", str(REPO_ROOT / f"tools/weight_{kind}.py")]
        receipt = json.dumps({"schema": "brewery.weight-runtime.v1", "jobs": jobs,
                              "formats": ["gguf", "hf", "safetensors"], "visual": True, "packed": True})
        helper.write_text(f'#!/bin/sh\nif [ "$1" = --capabilities ]; then printf \'%s\\n\' {shlex.quote(receipt)}; exit; fi\n'
                          f'exec {shlex.join(command)} "$@"\n')
        helper.chmod(0o700)
    # Do not add regular workers or expose any host-local model path.
    hf.save(root / "dispatch.json", {"hosts": {}})
    helpers = [(root / f"brewery-weight-{kind}").resolve(strict=True) for kind in ("edit", "tune")]
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(root / "home"),
           "TMPDIR": str(root / "tmp"), "BREWERY_WEB_DIST": str(root / "web"),
           "BREWERY_HF_HFR": str(args.hfr), "BREWERY_HF_WEIGHT_EDIT": str(helpers[0]),
           "BREWERY_HF_WEIGHT_TUNE": str(helpers[1])}
    hf.FILES = {**fixture("hf"), "float.gguf": matrix(ORIGINAL),
                "packed.gguf": packed_fixture("Q8_0", (0, 0))}
    with hf.hub(root) as server, hf.controller(root, args.bin_dir, env) as url:
        server.holds[hf.REPO][1].set()
        smoke = hf.Smoke(root, url, server)
        state = smoke.call("state")
        hf.save(root / "initial-state.json", state)
        hf.need(state["jobs"] == state["models"] == [] and [h["id"] for h in state["hosts"]] == [HOST],
                "not a fresh generated backend download lane")
        caps = state["hosts"][0]["capabilities"]
        hf.need(all(caps.get(k) is True for k in ("hf_download", "weight_inspect", "weight_propose",
                                                 "weight_apply", "weight_safetensors"))
                and caps.get("weight_tune") is True, "needs integrated local editor build")
        lane = root / "controller" / HOST
        config = lane / "worker.json"
        policy = json.loads(config.read_text())
        # The only generated config override selects the fixture origin.
        # The controller and workers retain their real receipt and edit policy.
        policy["hf_download"]["fixture_port"] = server.server_port
        hf.save(config, policy)
        hf.need(policy["root"] == str(lane / "worker"), "unexpected backend worker root")
        results = {kind: edit_case(smoke, lane, kind) for kind in ("hf", "safetensors")}
        results.update(gguf_float=gguf_case(smoke, lane, False), gguf_packed=gguf_case(smoke, lane, True))
        results["untrusted_paths"] = refuse_paths(smoke, lane)
        hf.save(root / "final-state.json", smoke.call("state"))
        hf.need(not any(row["credential"] for row in server.requests), "fixture received credentials")
        return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("bin-dir", "hfr", "gguf-python"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--hfr-sha256", required=True)
    cli.add_argument("--isolated", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if not args.isolated:
        raise SystemExit(subprocess.run([
            "/usr/bin/unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
            "--kill-child", "--mount-proc", sys.executable, "-E", "-s", "-B",
            str(Path(__file__).resolve()), *sys.argv[1:], "--isolated",
        ], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}).returncode)
    hf.need(os.getpid() == 1 and socket.if_nameindex() == [(1, "lo")],
            "requires a private PID/network namespace with only loopback")
    subprocess.run(["/usr/bin/ip", "link", "set", "lo", "up"], check=True, timeout=3)
    for name in ("bin_dir", "hfr", "gguf_python"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    hf.need(all(os.access(args.bin_dir / b, os.X_OK) for b in hf.BINARIES), "missing matched binaries")
    hf.need((args.gguf_python / "gguf/gguf_reader.py").is_file(), "expected pinned gguf-py directory")
    with args.hfr.open("rb") as source:
        hf.need(source.read(4) == b"\x7fELF" and os.access(args.hfr, os.X_OK), "expected real hfr ELF")
        source.seek(0)
        pin = file_digest(source, "sha256").hexdigest()
    hf.need(pin == args.hfr_sha256, "hfr differs from pinned build receipt")
    os.umask(0o077)
    root = Path(mkdtemp(prefix="backend-hf-edit.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    result = {"scratch": str(root), "hfr_sha256": pin, **run(root, args)}
    hf.save(root / "summary.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
