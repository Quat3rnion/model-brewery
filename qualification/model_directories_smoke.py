#!/usr/bin/env python3
"""Check Settings directory access end to end on a private local-eval worker."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import mkdtemp

from managed_hfr_smoke import BINARIES, Smoke, controller, need, save
from managed_weight_smoke import TENSOR, matrix

HOST = "local-eval"


def admin(smoke):
    return next(h for h in smoke.call("admin/state")["hosts"] if h["id"] == HOST)


def plan(smoke, revision, action, path, *, previous=None):
    request = {"worker": HOST, "revision": revision, "action": action, "path": str(path)}
    if previous is not None:
        request["previous_path"] = str(previous)
    response = smoke.call("admin/model-directories/plan", request)
    need(all(response[k] == v for k, v in request.items()) and response["confirmation"],
         "review did not describe the requested directory change")
    return {"worker": HOST, "revision": revision, "confirmation": response["confirmation"]}


def confirm(smoke, request, status=200):
    return smoke.call("admin/model-directories/confirm", request, status=status)


def register(smoke, path):
    return smoke.call("models", {"name": str(path.relative_to(smoke.root)), "host": HOST,
                                "path": str(path), "kind": "gguf"}, status=201)["id"]


def inspect(smoke, label, model, raw, expected, status=201):
    request = {"host": HOST, "kind": "weight_inspect", "input_ids": [model],
               "options": {"tensor": TENSOR, "index": [0, 1]}}
    response = smoke.call("jobs", request, status=status, key=f"directory-smoke.{label}")
    if status != 201:
        save(smoke.root / f"{label}.json", response)
        return response
    job = smoke.wait(label, response["id"], "succeeded")
    report = job["result"]["report"]
    need(job["host"] == HOST and report["value"] == expected
         and report["source_sha256"] == sha256(raw).hexdigest(),
         f"{label}: wrong value or source fingerprint")
    return job["id"]


def fingerprint(path):
    stat = path.stat()
    return (path.read_bytes(), stat.st_dev, stat.st_ino, stat.st_mtime_ns)


def run(root, args):
    for name in ("home", "tmp", "web", "worker", "original", "replacement"):
        (root / name).mkdir()
    old_dir, new_dir = root / "original", root / "replacement"
    old, new = old_dir / "model.gguf", new_dir / "model.gguf"
    old_bytes, new_bytes = matrix((1, 2, 3, 4)), matrix((5, 6, 7, 8))
    old.write_bytes(old_bytes)
    new.write_bytes(new_bytes)
    worker = root / "worker.json"
    dispatch = root / "dispatch.json"
    bins = args.bin_dir
    save(worker, {"root": str(root / "worker"), "source_roots": [],
                  "engine": str(bins / "brewery-engine")})
    save(dispatch, {"hosts": {HOST: {"local": True, "worker": str(bins / "brewery-worker"),
                                     "config": str(worker), "capabilities": {}}}})
    unchanged = {p: fingerprint(p) for p in (old, new, worker, dispatch)}
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(root / "home"),
           "TMPDIR": str(root / "tmp"), "BREWERY_WEB_DIST": str(args.web_dist or root / "web"),
           "BREWERY_HF_HFR": str(root / "no-hfr")}
    for key, name in (("HELPER", "brewery-weight-edit"), ("TUNER", "brewery-weight-tune")):
        env[f"BREWERY_LOCAL_EDIT_{key}"] = str(args.helper_dir / name)

    with controller(root, bins, env) as url:
        smoke = Smoke(root, url, None)
        before = admin(smoke)
        save(root / "initial-admin.json", before)
        need(before["model_directories"] == {"available": True, "configured": [], "granted": []},
             "rootless local-eval cannot plan directory access")
        need(not before["weight_editor"]["enabled"] and smoke.call("state")["jobs"] == [],
             "editor must start disabled with no jobs")
        original = register(smoke, old)
        inspect(smoke, "rootless", original, old_bytes, 2, status=409)
        if args.browser_driver:
            with (root / "browser.log").open("w") as log:
                subprocess.run(["node", args.browser_driver, url, HOST, old_dir],
                               env={**env, "CHROMIUM_BIN": str(args.chromium)},
                               stdout=log, stderr=log, timeout=60, check=True)
            before = admin(smoke)
            need(before["model_directories"]["granted"] == [] and not before["weight_editor"]["enabled"],
                 "browser round trip did not restore rootless, disabled state")
        add = plan(smoke, before["revision"], "add", old_dir)
        save(root / "add-plan.json", add)
        need(admin(smoke) == before, "planning changed active directory access")
        confirm(smoke, add)
        granted = admin(smoke)
        confirm(smoke, add, status=409)
        save(root / "granted-admin.json", granted)
        need(granted["model_directories"]["granted"] == [str(old_dir)]
             and granted["model_directories"]["configured"] == []
             and granted["revision"] != before["revision"] and not granted["weight_editor"]["enabled"],
             "confirmation did not grant only the model directory")
        inspect(smoke, "grant-not-enable", original, old_bytes, 2, status=409)
        smoke.call("admin/actions", {"action": "set_weight_editor", "worker": HOST,
                                      "revision": granted["revision"], "enabled": True})
        enabled = admin(smoke)
        first = inspect(smoke, "after-enable", original, old_bytes, 2)
        audit = smoke.call("admin/state")["audit"]

    with controller(root, bins, env) as url:
        smoke = Smoke(root, url, None)
        restarted = admin(smoke)
        save(root / "restart-admin.json", restarted)
        need(restarted == enabled and smoke.call("admin/state")["audit"] == audit,
             "grant, editor enablement or audit did not persist")
        second = inspect(smoke, "after-restart", original, old_bytes, 2)
        replacement = register(smoke, new)
        inspect(smoke, "before-replace", replacement, new_bytes, 6, status=409)
        replace = plan(smoke, restarted["revision"], "replace", new_dir, previous=old_dir)
        need(admin(smoke) == restarted, "replacement plan changed active access")
        confirm(smoke, replace)
        replaced = admin(smoke)
        need(replaced["model_directories"]["granted"] == [str(new_dir)], "replacement left old access")
        inspect(smoke, "replaced-original", original, old_bytes, 2, status=409)
        third = inspect(smoke, "after-replace", replacement, new_bytes, 6)
        remove = plan(smoke, replaced["revision"], "remove", new_dir)
        confirm(smoke, remove)
        final = admin(smoke)
        need(final["model_directories"]["granted"] == final["model_directories"]["configured"] == [],
             "directory removal did not leave the worker rootless")
        inspect(smoke, "after-remove", replacement, new_bytes, 6, status=409)
        state = smoke.call("state")
        need({row["id"] for row in state["jobs"]} == {first, second, third}
             and set(smoke.models()) == {original, replacement}, "rejected request changed jobs or Library")
        final_audit = smoke.call("admin/state")["audit"]
        changes = ["remove", "replace", "add"] + (["remove", "add"] if args.browser_driver else [])
        need([r["change"] for r in final_audit
              if r["worker"] == HOST and r["action"] == "set_model_directories"] == changes,
             "directory actions did not leave the expected audit trail")
        save(root / "final-admin.json", final)
        save(root / "audit.json", final_audit)
    need(all(fingerprint(p) == pin for p, pin in unchanged.items()),
         "directory policy changed original source bytes, inodes, or static worker config")
    return {"inspected": {"enabled": first, "restart": second, "replacement": third},
            "source_sha256": sha256(old_bytes).hexdigest(), "browser_grant": bool(args.browser_driver)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--helper-dir", type=Path, help="actual packaged helpers, defaults to bin-dir")
    parser.add_argument("--web-dist", type=Path)
    parser.add_argument("--browser-driver", type=Path)
    parser.add_argument("--chromium", type=Path)
    parser.add_argument("--isolated", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.isolated:
        raise SystemExit(subprocess.run([
            "/usr/bin/unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
            "--kill-child", "--mount-proc", sys.executable, "-E", "-s", "-B",
            str(Path(__file__).resolve()), *sys.argv[1:], "--isolated",
        ], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}).returncode)
    need(os.getpid() == 1 and socket.if_nameindex() == [(1, "lo")], "private loopback namespace required")
    subprocess.run(["/usr/bin/ip", "link", "set", "lo", "up"], check=True, timeout=3)
    args.helper_dir = args.helper_dir or args.bin_dir
    for name in ("bin_dir", "helper_dir", "web_dist", "browser_driver", "chromium"):
        if path := getattr(args, name):
            setattr(args, name, path.resolve(strict=True))
    need(all(os.access(args.bin_dir / b, os.X_OK) for b in BINARIES), "missing Brewery binaries")
    need(os.access(args.helper_dir / "brewery-weight-edit", os.X_OK), "missing packaged editor")
    need(not args.browser_driver or (args.web_dist and args.chromium), "browser requires UI and Chromium")
    os.umask(0o077)
    root = Path(mkdtemp(prefix="model-directories.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    report = {"scratch": str(root), **run(root, args)}
    save(root / "summary.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
