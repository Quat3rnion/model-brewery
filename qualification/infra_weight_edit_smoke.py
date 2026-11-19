#!/usr/bin/env python3
"""Exercise local weight-editor opt-in against a real isolated HTTP controller."""

import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import mkdtemp
from urllib.parse import urlsplit

import managed_hfr_smoke as hf
from managed_hfr_smoke import BINARIES, Smoke, controller, need, save
from managed_safetensors_smoke import SHARD, describe, digest, fixture
from managed_weight_smoke import TENSOR, hgn_matrix, matrix

HOST = "local-edit-fixture"
KINDS = ("weight_inspect", "weight_propose", "weight_apply", "weight_tune")
BASE = (1, 2, 3, 4)
INVENTORY = {"mode": "inventory", "offset": 0, "limit": 16}


def setup(root, args):
    for name in ("home", "tmp", "inputs", "worker", "web"):
        (root / name).mkdir(mode=0o700)
    worker = root / "worker.json"
    save(worker, {"root": str(root / "worker"), "source_roots": [str(root / "inputs")],
                  "engine": str(args.bin_dir / "brewery-engine")})
    save(root / "dispatch.json", {"hosts": {HOST: {"local": True, "config": str(worker),
         "worker": str(args.bin_dir / "brewery-worker")}}})
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(root / "home"),
           "TMPDIR": str(root / "tmp"), "BREWERY_WEB_DIST": str(args.web_dist or root / "web"),
           "BREWERY_HF_HFR": str(root / "no-hfr")}
    for role, binary in (("DISPATCH", "brewery-dispatch"), ("WORKER", "brewery-worker"),
                         ("ENGINE", "brewery-engine")):
        env[f"BREWERY_LOCAL_EDIT_{role}"] = str(args.bin_dir / binary)
    for role, name in (("HELPER", "brewery-weight-edit"), ("TUNER", "brewery-weight-tune")):
        env[f"BREWERY_LOCAL_EDIT_{role}"] = str((args.helper_dir or args.bin_dir) / name)
    return env


def host(smoke, route="admin/state"):
    return next(row for row in smoke.call(route)["hosts"] if row["id"] == HOST)


def editor(smoke, revision, enabled, status=200):
    return smoke.call("admin/actions", {"action": "set_weight_editor", "worker": HOST,
                                      "revision": revision, "enabled": enabled}, status=status)


def transition(smoke, args, root, enabled, revision):
    if not args.transition_driver:
        editor(smoke, revision, enabled)
        return
    action = "enable" if enabled else "disable"
    with (root / f"browser-{action}.log").open("w") as log:
        subprocess.run(["node", args.transition_driver, smoke.url, HOST, action, root / "browser"],
                       env={"PATH": "/usr/bin:/bin", "HOME": str(root / "home"),
                            "TMPDIR": str(root / "tmp"), "CHROMIUM_EXECUTABLE": str(args.chromium)},
                       stdout=log, stderr=log, timeout=30, check=True)


def browser_edit(smoke, args, source, model_id):
    original = describe(source)
    models = smoke.models()
    root = smoke.root
    with (root / "browser-edit.log").open("w") as log:
        subprocess.run(["node", Path(__file__).with_name("infra_weight_edit_smoke_browser.mjs"),
                        smoke.url, model_id, root / "browser", args.playwright, args.chromium],
                       env={"PATH": "/usr/bin:/bin", "HOME": str(root / "home"),
                            "TMPDIR": str(root / "tmp")}, stdout=log, stderr=log, timeout=70, check=True)
    evidence = json.loads((root / "browser/edit.json").read_text())
    job = smoke.wait("browser-apply", evidence["job_id"], "succeeded")
    revision = evidence["model_id"]
    rows = smoke.models()
    path = Path(rows[revision]["path"])
    want = expected("gguf", (1.0009765625, 2, 3, 4))  # Next representable F16 above 1.
    need(evidence["value"] == 1.0009765625 and revision not in models
         and set(rows) == set(models) | {revision} and rows[revision]["kind"] == "gguf"
         and path.is_relative_to(root) and snapshot(path) == want and describe(source) == original
         and job["result"]["library_model_id"] == revision and job["result"]["verified"] is True,
         "browser edit did not apply and reopen exactly one F16 weight")
    return evidence


def register(smoke, path, kind):
    return smoke.call("models", {"name": path.name, "host": HOST, "kind": kind, "path": str(path)},
                      status=201)["id"]


def submit(smoke, label, kind, ids, options, status=201):
    body = {"kind": kind, "host": HOST, "input_ids": ids, "options": options}
    save(smoke.root / f"{label}-request.json", body)
    row = smoke.call("jobs", body, status=status, key=f"infra-edit.{label}")
    return smoke.wait(label, row["id"], "succeeded") if status == 201 else row


def snapshot(path):
    return {key: data for key, (data, _) in describe(path).items()}


def expected(kind, values):
    if kind == "hf":
        return fixture("hf", values)
    if kind == "safetensors":
        return {"": fixture("hf", values)[SHARD]}
    return {"": (matrix if kind == "gguf" else hgn_matrix)(values)}


def download_sources(root, args, env):
    """Use real pinned hfr with an immutable loopback Hub, not live Library data."""
    hf.FILES = {**fixture("hf"), "matrix.gguf": matrix(BASE)}
    files = hf.FILES.copy()
    inputs = root / "inputs"
    if not args.hfr:
        for name, data in files.items():
            (inputs / "indexed-hf").mkdir(exist_ok=True)
            target = inputs / "indexed-hf" / name if name != "matrix.gguf" else inputs / name
            target.write_bytes(data)
        (inputs / SHARD).write_bytes(files[SHARD])
    else:
        with hf.hub(root) as hub:
            hub.holds[hf.REPO][1].set()
            for destination, selection in (("exact-gguf", ("matrix.gguf",)), ("exact-st", (SHARD,)),
                                           ("indexed-hf", tuple(n for n in files if n != "matrix.gguf"))):
                path = inputs / destination
                endpoint = f"http://127.0.0.1:{hub.server_port}"
                need(urlsplit(endpoint).hostname == "127.0.0.1", "fixture escaped loopback")
                cmd = [args.hfr, "--endpoint", endpoint, "download", hf.REPO, *selection,
                       "--revision", hf.COMMIT, "--local-dir", path, "--transfer", "range", "--verify"]
                with (root / f"{destination}-hfr.log").open("w") as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=log, timeout=12, check=True)
                need(snapshot(path) == {name: files[name] for name in selection}, "download inventory/bytes differ")
            need(not any(row["credential"] for row in hub.requests), "fixture received credentials")
        need((inputs / "exact-gguf/matrix.gguf").is_file(), "missing exact GGUF")
    return {
        "gguf": inputs / ("exact-gguf/matrix.gguf" if args.hfr else "matrix.gguf"),
        "safetensors": inputs / ("exact-st/" + SHARD if args.hfr else SHARD),
        "hf": inputs / "indexed-hf",
    }


def edit_case(smoke, kind, source, model_id, calibration, held):
    original = describe(source)
    pin = digest(expected(kind, BASE))
    need(snapshot(source) == expected(kind, BASE), f"{kind}: unexpected source bytes")
    inventory = submit(smoke, f"{kind}-inventory", "weight_inspect", [model_id], INVENTORY)
    report = inventory["result"]["report"]
    tensor = next(t for t in report["tensors"] if t["name"] == TENSOR)
    dtype = "F16" if kind == "gguf" else "BF16"
    need(report["source_sha256"] == pin and tensor["shape"] == [2, 2] and tensor["dtype"] == dtype
         and tensor.get("editable", kind not in ("hf", "safetensors")) is True,
         f"{kind}: wrong inventory")

    def tile(label, ident, sha, values):
        response = submit(smoke, f"{kind}-{label}", "weight_inspect", [ident], {
            "mode": "tile", "tensor": TENSOR, "origin": [0, 0], "shape": [2, 2], "source_sha256": sha,
        })
        detail = response["result"]["report"]
        need(detail["source_sha256"] == sha and detail["values"] == [list(values[:2]), list(values[2:])],
             f"{kind}: wrong tile or source pin")
        return response["id"]

    tile("source-tile", model_id, pin, BASE)
    before = smoke.models()
    manual = submit(smoke, f"{kind}-manual", "weight_propose", [model_id], {
        "source_sha256": pin, "edits": [{"tensor": TENSOR, "index": [0, 1], "op": "delta", "value": 0.5}],
    })
    automatic = submit(smoke, f"{kind}-auto", "weight_tune", [model_id, calibration, held], {
        "source_sha256": pin, "tensor": TENSOR, "indices": [[0, 0], [1, 1]], "max_update": 1,
    })
    metrics = automatic["result"]["report"]["tune"]
    need(metrics["calibration"] == {"before_mse": 0.5, "after_mse": 0}
         and metrics["held_out"] == {"before_mse": 0, "after_mse": 1.75, "regression": True},
         f"{kind}: wrong calibration/held-out metrics")
    need(smoke.models() == before and describe(source) == original, f"{kind}: proposal published data")
    refusal = submit(smoke, f"{kind}-unreviewed", "weight_apply", [model_id],
                     {"proposal_job_id": manual["id"], "reviewed": False}, status=400)
    results = {"source_sha256": pin, "inventory_job_id": inventory["id"], "unreviewed": refusal}
    for mode, proposal, values, changes in (
        ("manual", manual, (1, 2.5, 3, 4), [([0, 1], 2, 2.5)]),
        ("auto", automatic, (2, 2, 3, 3), [([0, 0], 1, 2), ([1, 1], 4, 3)]),
    ):
        review = proposal["result"]["report"]["proposal"]
        need(review["source_sha256"] == pin and
             [(e["index"], e["before"], e["after"]) for e in review["edits"]] == changes,
             f"{kind}: {mode} review differs")
        previous = smoke.models()
        job = submit(smoke, f"{kind}-{mode}-apply", "weight_apply", [model_id],
                     {"proposal_job_id": proposal["id"], "reviewed": True})
        result = job["result"]
        new_id = result["library_model_id"]
        state = smoke.models()
        row = state[new_id]
        output = Path(row["path"])
        want = expected(kind, values)
        output_sha = digest(want)
        need(new_id not in previous and set(state) == set(previous) | {new_id}
             and all(state[k] == v for k, v in previous.items())
             and row["kind"] == kind and row["host"] == HOST and output.is_relative_to(smoke.root)
             and output != source and result["verified"] is True and result["artifact"]["kind"] == kind
             and result["source_sha256"] == pin and result["artifact"]["sha256"] == output_sha
             and result["inputs"][1]["sha256"] == proposal["result"]["artifact"]["sha256"],
             f"{kind}: apply did not publish one reviewed same-format revision")
        revised = describe(output)
        need({name: data for name, (data, _) in revised.items()} == want
             and not ({x[1] for x in revised.values()} & {x[1] for x in original.values()}),
             f"{kind}: output had collateral bytes or shared original file inodes")
        reopened = tile(f"{mode}-reopen", new_id, output_sha, values)
        need(describe(source) == original, f"{kind}: original changed")
        results[mode] = {"proposal_job_id": proposal["id"], "apply_job_id": job["id"],
                         "reopen_job_id": reopened, "model_id": new_id, "sha256": output_sha}
    return results


def run(root, args):
    env = setup(root, args)
    sources = download_sources(root, args, env)
    native = root / "inputs/native.hgn"
    native.write_bytes(hgn_matrix(BASE))
    sources["hgn"] = native
    cal, held = root / "inputs/cal.json", root / "inputs/held.json"
    save(cal, {"activations": [[1, 0], [0, 1]], "targets": [[2, 3], [2, 3]]})
    save(held, {"activations": [[1, 1], [2, -1]], "targets": [[3, 7], [0, 2]]})
    cfgs = (root / "worker.json", root / "dispatch.json", cal, held)
    baseline = {p: sha256(p.read_bytes()).hexdigest() for p in cfgs}
    with controller(root, args.bin_dir, env) as url:
        smoke = Smoke(root, url, None)
        world = smoke.call("state")
        before = host(smoke)
        save(root / "admin-before.json", before)
        caps = host(smoke, "state")["capabilities"]
        need(not world["jobs"] and not any(caps.get(k) for k in KINDS),
             "fixture was not initially disabled")
        ids = {kind: register(smoke, path, kind) for kind, path in sources.items()}
        calibration, hold = (register(smoke, path, "weight-calibration") for path in (cal, held))
        submit(smoke, "initial-disabled", "weight_inspect", [ids["gguf"]], INVENTORY, status=409)
        need(not smoke.call("state")["jobs"], "disabled edit unexpectedly queued")
        transition(smoke, args, root, True, before["revision"])
        after = host(smoke)
        need(after["weight_editor"]["enabled"] is True, "controller did not enable the editor")
        save(root / "admin-enabled.json", after)
        active = host(smoke, "state")["capabilities"]
        need(all(active.get(k) is True for k in KINDS), "manual, visual, or auto edit lane unavailable")
        stale = editor(smoke, before["revision"], False, status=409)
        cases = {kind: edit_case(smoke, kind, sources[kind], ids[kind], calibration, hold)
                 for kind in (args.format or sources)}
        browser = browser_edit(smoke, args, sources["gguf"], ids["gguf"]) if args.playwright else None
        revision = host(smoke)["revision"]
        transition(smoke, args, root, False, revision)
        stopped = host(smoke)
        save(root / "admin-disabled.json", stopped)
        need(stopped["weight_editor"]["enabled"] is False and stopped["revision"] != revision,
             "controller did not disable edit jobs")
        final = smoke.call("state")
        denied = submit(smoke, "after-disabled", "weight_inspect", [ids["hgn"]], INVENTORY, status=409)
        need(smoke.call("state") == final and host(smoke, "state")["capabilities"] == caps,
             "disabled job changed state or existing capabilities")
        for key in ("resources", "resource_bounds", "effective_resources", "paused", "disabled_jobs"):
            need(before.get(key) == after.get(key) == stopped.get(key), f"opt-in changed host {key}")
        need({p: sha256(p.read_bytes()).hexdigest() for p in cfgs} == baseline, "settings/worker/source pins changed")
        save(root / "final-state.json", final)
        return {"before": before, "enabled": after, "disabled": stopped, "matrix": cases,
                "stale": stale, "disabled_refusal": denied,
                "browser_settings": bool(args.transition_driver),
                "browser_edit": browser,
                "input_provenance": "pinned hfr outside Library" if args.hfr else "local fixtures"}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--bin-dir", type=Path, required=True)
    cli.add_argument("--helper-dir", type=Path, help="packaged weight edit/tune launchers (defaults to bin-dir)")
    cli.add_argument("--web-dist", type=Path)
    cli.add_argument("--transition-driver", type=Path)
    cli.add_argument("--chromium", type=Path)
    cli.add_argument("--playwright", type=Path, help="playwright-core index.mjs for the browser edit")
    cli.add_argument("--hfr", type=Path)
    cli.add_argument("--hfr-sha256")
    cli.add_argument("--format", choices=("gguf", "hgn", "safetensors", "hf"), action="append",
                     help="narrow the fixture matrix; default exercises all four formats")
    cli.add_argument("--isolated", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if not args.isolated:
        raise SystemExit(subprocess.run([
            "/usr/bin/unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
            "--kill-child", "--mount-proc", sys.executable, "-E", "-s", "-B",
            str(Path(__file__).resolve()), *sys.argv[1:], "--isolated",
        ], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}).returncode)
    need(os.getpid() == 1 and socket.if_nameindex() == [(1, "lo")], "private loopback namespace required")
    subprocess.run(["/usr/bin/ip", "link", "set", "lo", "up"], check=True, timeout=3)
    for name in ("bin_dir", "helper_dir", "web_dist", "transition_driver", "chromium", "playwright", "hfr"):
        path = getattr(args, name)
        if path:
            setattr(args, name, path.resolve(strict=True))
    need(all(os.access(args.bin_dir / b, os.X_OK) for b in BINARIES), "missing matched Brewery binaries")
    if args.transition_driver or args.playwright:
        need(args.web_dist and args.chromium and os.access(args.chromium, os.X_OK),
             "browser Settings check needs --web-dist and --chromium")
    if args.hfr:
        with args.hfr.open("rb") as binary:
            need(binary.read(4) == b"\x7fELF", "expected real pinned hfr ELF")
            binary.seek(0)
            need(file_digest(binary, "sha256").hexdigest() == args.hfr_sha256, "hfr SHA mismatch")
    os.umask(0o077)
    root = Path(mkdtemp(prefix="infra-weight-edit.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    report = {"scratch": str(root), **run(root, args)}
    save(root / "summary.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
