#!/usr/bin/env python3
"""Exercise managed worker policy on an isolated real HTTP controller.

Usage: python -B qualification/managed_infra_smoke.py --bin-dir PATH [--hfr PATH]
Requires Linux user/PID/network namespaces. Uses two local worker roots and a
small gated fixture engine, with no catalog, SSH, network egress, or model load.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from tempfile import mkdtemp
import time

from managed_hfr_smoke import Smoke, controller, need, save
from managed_weight_smoke import matrix

A, B, BACKEND = "fixture-alpha", "fixture-beta", "backend-hf"
JOB = "inspect"


def fixture_engine(gate):
    job = json.load(sys.stdin)
    need(job["kind"] == JOB and len(job["inputs"]) == 1, "unexpected fixture job")
    # The real worker has admitted and started the request. Block specifically
    # inside the engine so the controller can queue and observe other requests.
    if gate:
        (gate / "started").touch()
        until = time.monotonic() + 30
        while not (gate / "release").exists():
            need(time.monotonic() < until, "fixture engine timed out")
            time.sleep(0.05)
    print(json.dumps({"kind": JOB, "fixture": True}))


def setup(root, bins):
    homes, sources = root / "home", root / "inputs"
    for path in (homes, sources, root / "web", root / "tmp"):
        path.mkdir()
    inputs, hosts = {}, {}
    for host in (A, B):
        worker = root / host
        worker.mkdir()
        source = sources / f"{host}.gguf"
        source.write_bytes(matrix((1, 2, 3, 4)))
        inputs[host] = source
        engine = worker / "engine"
        command = [sys.executable, "-E", "-s", "-B", str(Path(__file__).resolve()), "--fixture-engine"]
        if host == A:
            command.append(str(worker))
        engine.write_text(f'#!/bin/sh\nexec {shlex.join(command)} "$@"\n')
        engine.chmod(0o700)
        cfg = worker / "worker.json"
        save(cfg, {"root": str(worker), "source_roots": [str(sources)], "engine": str(engine)})
        hosts[host] = {"local": True, "worker": str(bins / "brewery-worker"), "config": str(cfg),
                       "capabilities": {JOB: True}}
    save(root / "dispatch.json", {"hosts": hosts})
    return inputs


def poll(predicate, label):
    until = time.monotonic() + 12
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(0.06)
    raise ValueError(f"timed out: {label}")


def policy(smoke, host, *, revision, paused, disabled_jobs, status=200):
    return smoke.call("admin/actions", {"action": "set_host_policy", "worker": host,
                                      "revision": revision, "paused": paused,
                                      "disabled_jobs": disabled_jobs}, status=status)


def hosts(admin):
    return {row["id"]: row for row in admin["hosts"]}


def capabilities(smoke):
    return {row["id"]: row["capabilities"] for row in smoke.call("state")["hosts"]}


def submit(smoke, label, host, model, status=201):
    data = {"kind": JOB, "host": host, "input_ids": [model], "options": {}}
    return smoke.call("jobs", data, status=status, key=f"infra.{label}")


def run(root, args):
    inputs = setup(root, args.bin_dir)
    cfgs = (root / "dispatch.json", root / A / "worker.json", root / B / "worker.json")
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in cfgs}
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(root / "home"),
           "TMPDIR": str(root / "tmp"), "BREWERY_WEB_DIST": str(root / "web"),
           # Don't discover an unrequested sibling hfr.
           "BREWERY_HF_HFR": str(args.hfr or (root / "no-hfr"))}
    with controller(root, args.bin_dir, env) as url:
        smoke = Smoke(root, url, None)
        state = smoke.call("state")
        need(state["models"] == state["jobs"] == [], "fixture controller is not fresh")
        initial = smoke.call("admin/state")
        save(root / "admin-before.json", initial)
        rows = hosts(initial)
        need(rows.keys() == ({A, B, BACKEND} if args.hfr else {A, B})
             and not initial.get("workers"), "Infra does not expose expected workers without a catalog")
        for host in (A, B):
            need(JOB in rows[host]["allowed_jobs"] and not rows[host]["paused"],
                 f"{host} cannot accept the fixture job")
        caps = capabilities(smoke)
        if args.hfr:
            need("hf_download" in rows[BACKEND]["allowed_jobs"] and caps[BACKEND].get("hf_download") is True,
                 "reserved backend service not visible")

        models = {}
        for host, path in inputs.items():
            models[host] = smoke.call("models", {"name": host, "host": host, "kind": "gguf",
                                                "path": str(path)}, status=201)["id"]
        running = submit(smoke, "running", A, models[A])["id"]
        try:
            poll(lambda: (root / A / "started").exists(), "fixture engine start")
            need(smoke.call(f"jobs/{running}")["state"] == "running", "fixture job did not run")
            queued = submit(smoke, "queued", A, models[A])["id"]
            need(smoke.call(f"jobs/{queued}")["state"] == "queued", "second bound job was not queued")
            policy(smoke, A, revision=rows[A]["revision"], paused=True, disabled_jobs=[JOB])
            now = smoke.call("admin/state")
            save(root / "admin-draining.json", now)
            updated = hosts(now)[A]
            need(updated["paused"] and updated["draining"] and updated["disabled_jobs"] == [JOB]
                 and updated["revision"] != rows[A]["revision"], "paused policy or draining status not applied")
            need(hosts(now)[B] == rows[B], "other worker policy changed")
            need(capabilities(smoke)[A][JOB] is False and capabilities(smoke)[B][JOB] is True,
                 "effective capabilities did not reflect policy")
            need(smoke.call(f"jobs/{running}")["state"] == "running", "running job did not drain")
            rejected = {
                "stale": policy(smoke, A, revision=rows[A]["revision"], paused=False,
                                disabled_jobs=[], status=409),
                "new-job": submit(smoke, "paused", A, models[A], status=409),
            }
            save(root / "refusals.json", rejected)
        finally:
            (root / A / "release").touch()
        first = smoke.wait("running", running, "succeeded")
        second = smoke.wait("queued", queued, "failed")
        third = submit(smoke, "other-host", B, models[B])["id"]
        other = smoke.wait("other-host", third, "succeeded")
        need(first["host"] == second["host"] == A and other["host"] == B,
             "bound job was reassigned")
        need(not (root / A / "jobs" / queued).exists() and not (root / B / "jobs" / queued).exists(),
             "paused queued job reached a worker")
        durable = smoke.call("admin/state")
        save(root / "admin-after.json", durable)
        want = hosts(durable)[A]
        need(want["paused"] and not want["draining"] and want["revision"] == updated["revision"]
             and want["disabled_jobs"] == [JOB], "drain did not end or stale write applied")
        audit = durable["audit"]
        need(audit[1:] == initial["audit"] and audit[0]["action"] == "set_host_policy"
             and audit[0]["worker"] == A and audit[0]["outcome"] == "applied",
             "audit did not record exactly the accepted policy write")
        if args.hfr:
            need(hosts(durable)[BACKEND] == rows[BACKEND] and capabilities(smoke)[BACKEND] == caps[BACKEND],
                 "policy leaked onto reserved backend service")

    # Restart the same test controller DB with byte-identical static worker configs.
    with controller(root, args.bin_dir, env) as url:
        smoke = Smoke(root, url, None)
        state = smoke.call("admin/state")
        save(root / "admin-restarted.json", state)
        need(hosts(state)[A] == want and state["audit"] == durable["audit"],
             "controller restart lost policy or audit")
        need(capabilities(smoke)[A][JOB] is False and capabilities(smoke)[B][JOB] is True,
             "controller restart lost effective capabilities")
        need({str(p): sha256(p.read_bytes()).hexdigest() for p in cfgs} == before,
             "restart changed static worker configs")
    return {"running": running, "rejected_queued": queued, "other_worker": third,
            "revision": want["revision"], "audit": durable["audit"], "backend": bool(args.hfr)}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--fixture-engine":
        fixture_engine(Path(sys.argv[2]) if len(sys.argv) == 3 else None)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path, help="built controller/dispatcher/worker directory")
    parser.add_argument("--hfr", type=Path, help="optional trusted hfr to expose the reserved backend service")
    parser.add_argument("--isolated", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.isolated:
        raise SystemExit(subprocess.run([
            "/usr/bin/unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
            "--kill-child", "--mount-proc", sys.executable, "-E", "-s", "-B",
            str(Path(__file__).resolve()), *sys.argv[1:], "--isolated",
        ], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}).returncode)
    need(os.getpid() == 1 and socket.if_nameindex() == [(1, "lo")],
         "requires a fresh private PID/network namespace with only loopback")
    subprocess.run(["/usr/bin/ip", "link", "set", "lo", "up"], check=True, timeout=3)
    args.bin_dir = args.bin_dir.resolve(strict=True)
    required = ["brewery", "brewery-dispatch", "brewery-worker"]
    if args.hfr:
        args.hfr = args.hfr.resolve(strict=True)
        need(os.access(args.hfr, os.X_OK), "missing executable hfr")
        required.append("brewery-engine")
    need(all(os.access(args.bin_dir / name, os.X_OK) for name in required), "missing Brewery binaries")
    os.umask(0o077)
    root = Path(mkdtemp(prefix="managed-infra.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    report = {"scratch": str(root), **run(root, args)}
    save(root / "summary.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
