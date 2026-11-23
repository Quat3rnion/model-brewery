#!/usr/bin/env python3
"""Exercise behavior scoring twice on an isolated Qwen CPU worker."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.request import ProxyHandler, Request, build_opener

from weight_space_acceptance import ARB, ASSETS, HF, PINS, REPO, RUNNER, SAFE, SNAPSHOT_PIN
from weight_space_acceptance import require, save, setup, sha


def exercise(base, run, pins, runner):
    http = build_opener(ProxyHandler({}))

    def call(path, body=None, key=None, status=200):
        headers = {"Origin": base}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if key:
            headers["Idempotency-Key"] = "a2-" + key
        request = Request(base + path, headers=headers,
                          data=None if body is None else json.dumps(body).encode())
        with http.open(request, timeout=120) as reply:
            require(reply.status == status, f"{path}: expected {status}, got {reply.status}")
            return json.load(reply)

    def wait(job):
        for _ in range(600):
            current = call("/api/jobs/" + job["id"])
            if current["state"] not in ("queued", "running"):
                require(current["state"] == "succeeded", str(current))
                return current
            time.sleep(2)
        raise TimeoutError(job["id"])

    require(not call("/api/scratch"), "expected an empty isolated scratch store")
    call("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                               "revision": 0, "enabled": True})
    parent = call("/api/models", {"host": "local", "kind": "safetensors", "path": str(SAFE)},
                  status=201)
    assets = call("/api/models", {"host": "local", "kind": "assets",
                                 "path": str(run / "assets")}, status=201)
    root = call("/api/scratch", {"parent_id": parent["id"], "parent_sha256": PINS[SAFE]},
                key="root", status=201)
    path = "/api/scratch/" + root["id"]
    receipt = wait(call(path + "/gguf", {"sha256": root["sha256"], "assets_id": assets["id"]},
                        key="gguf", status=202))["result"]["gguf"]
    reports, diffs, jobs = [], [], []
    for run in (1, 2):
        job = wait(call(path + "/checks", {
            "sha256": root["sha256"], "behavior_direction": {"prompt_sets": pins},
        }, key=f"check-{run}", status=202))
        print(f"check {run}: succeeded ({job['id']})", flush=True)
        bundle = call(path + "/checks")
        require(bundle["measured_job_id"] == job["id"], "check bundle did not advance")
        require(all(bundle[status]["status"] == "passed"
                    for status in ("integrity", "smoke", "behavior")), "W4 checks did not pass")
        measured = bundle["measured_report"]
        require(measured["runner_sha256"] == sha(runner) and
                measured["parent_gguf_sha256"] == receipt["sha256"] ==
                measured["revision_gguf_sha256"], "runner or GGUF pin mismatch")
        report = bundle["behavior_direction"]
        require(report["config"]["prompt_sets"] == pins and report["cpu_threads"] == 4,
                "unexpected replay pins or CPU settings")
        flags = report["benign_refusal_flags"]
        require(len(report["metrics"]) == 6 and report["counts"]["benign_refusal"] == len(flags) == 16,
                "expected all four scorers, including the 16 benign holdout prompts")
        require("benign_refusal" not in report["samples"] and
                all(set(pair) == {"parent", "child"} and
                    all(type(flag) is bool for flag in pair.values()) for pair in flags),
                "benign refusal results must contain flags only")
        refusal = next(m for m in report["metrics"] if m["name"] == "benign_refusal_rate")
        require(all(refusal[role] == sum(pair[role] for pair in flags) / len(flags)
                    for role in ("parent", "child")), "rate does not match the flags")
        require(report["kl_continuation_positions"] > 0, "KL did not score continuation positions")
        for metric in report["metrics"]:
            require(metric["parent"] == metric["child"], str(metric))
            if metric["name"].startswith("kl_"):
                require(metric["child"] == 0, str(metric))
        diff = call(f"/api/diff/behavior?a={parent['id']}&b={root['id']}")
        require(len(diff["metrics"]) == 23 and
                all(m["delta"] == 0 and m["flag"] == "unchanged" for m in diff["metrics"]),
                "W5 did not return all 23 metrics as unchanged")
        reports.append(report)
        diffs.append(diff)
        jobs.append(job["id"])
    require(jobs[0] != jobs[1] and reports[0] == reports[1] and diffs[0] == diffs[1],
            "independent check jobs did not give identical results")
    measured = {key: value for key, value in reports[0].items() if key != "samples"}
    return {"qualification": "not release-qualified", "jobs": jobs,
            "safetensors_sha256": root["sha256"], "gguf_sha256": receipt["sha256"],
            "runner_sha256": sha(runner), "behavior_direction": measured,
            "behavior_diff": diffs[0], "repeat_identical": True}


def prepare_cache(run, source_cache, index):
    cache = run / "contrast-cache"
    cache.mkdir(mode=0o700)
    pins = dict(line.split() for line in index.read_text().splitlines() if line.strip())
    require(set(pins) == {"slop", "sycophancy", "response_length", "benign_refusal"},
            "unexpected approved set index")
    for pin in pins.values():
        require(re.fullmatch("[a-f0-9]{64}", pin), "invalid approved prompt pin")
        source = source_cache / f"{pin}.json"
        require(sha(source) == pin, "approved prompt cache pin mismatch")
        shutil.copyfile(source, cache / source.name)
    return cache, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "bins", "runner", "contrast-cache", "contrast-index"):
        parser.add_argument("--" + arg, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--web-dist", type=Path, default=ARB / "w8-accept/web")
    args = parser.parse_args()
    scratch = (ARB / "xa-scratch").resolve()
    require(ARB in args.root.parents or scratch in args.root.parents,
            "use an isolated run root under /arboretum")
    args.root.mkdir(parents=True, exist_ok=True)
    evidence = args.root / "evidence"
    evidence.mkdir()
    require(all(sha(path) == pin for path, pin in PINS.items() if path != RUNNER), "input pin mismatch")
    manifest = "".join(f"{sha(HF / name)}  {name}\n" for name in sorted((*ASSETS, SAFE.name)))
    require(hashlib.sha256(manifest.encode()).hexdigest() == SNAPSHOT_PIN, "HF snapshot pin mismatch")
    subprocess.run(["sha256sum", "-c", ARB / "pins/lanternfish-0ea30b3.libs.sha256"], check=True)
    with tempfile.TemporaryDirectory(prefix="isolated-", dir=args.root) as tmp:
        run = Path(tmp)
        env = setup(args, run)
        env["TMPDIR"] = os.environ["TMPDIR"]
        cache, pins = prepare_cache(run, args.contrast_cache, args.contrast_index)
        config = run / "worker.json"
        worker = json.loads(config.read_text())
        worker["engine_env"].update({
            "BREWERY_CONTRAST_CACHE": str(cache),
            "BREWERY_SCRATCH_CHECKS_RUNNER": str(args.runner),
            "BREWERY_SCRATCH_CHECKS_RUNNER_SHA256": sha(args.runner),
        })
        save(run, "worker", worker)
        config.chmod(0o600)
        log_path = evidence / "controller.log"
        with log_path.open("w") as log, subprocess.Popen(
                [args.bins / "brewery"], env=env, cwd=REPO, stdout=log, stderr=log,
                stdin=subprocess.DEVNULL) as controller:
            try:
                base = None
                for _ in range(100):
                    require(controller.poll() is None, "controller exited; see controller.log")
                    match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$",
                                      log_path.read_text(), re.M)
                    if match:
                        require(int(match[2]) not in (0, 3000, 4095, 4141), "unexpected port")
                        base = match[1]
                        break
                    time.sleep(.1)
                require(base is not None, "isolated controller did not listen")
                # The API registers assets only within the isolated run's source roots.
                result = exercise(base, run, pins, args.runner)
                result["binary_sha256"] = {
                    name: sha(args.bins / name)
                    for name in ("brewery", "brewery-worker", "brewery-engine", "brewery-dispatch")
                }
                save(evidence, "report", result)
            finally:
                if controller.poll() is None:
                    controller.terminate()
                    try:
                        controller.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        controller.kill()


if __name__ == "__main__":
    os.environ.setdefault("TMPDIR", str(ARB / "tmp"))
    main()
