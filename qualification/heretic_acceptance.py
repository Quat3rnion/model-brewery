"""Run the A6 experiment in an isolated local stack, never in a live controller."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from weight_space_acceptance import ARB, PINS, REPO, SAFE, require, save, setup, sha

sys.path.insert(0, str(REPO / "tools"))
from direction_prompt_cache import normalize


PROMPTS = {
    "sycophancy": "3b255d8b4a6fceeaa8311202fe35f1ff3cb6cf85d5453c2bdffa576510d21021",
    "response_length": "f42e2aac3d2c0faad8d0f0e5e2a2c2a5c8580bb1f51ece3e6a30b26eac3c24b6",
}

DOLLY = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
A2_RUNNER = "cc9ca0889c004524e2c371fae73276e6e7dca6f30e52473b03710cf5d9f61f36"


def install_acceptance_profile(runtime, cache):
    # Input is already the verified A1 cache. Never fetch in acceptance.
    profile_builder = r"""
import hashlib, json, sys, tomllib, tomli_w
from pathlib import Path
runtime, path = map(Path, sys.argv[1:])
data = path.read_bytes()
assert hashlib.sha256(data).hexdigest() + ".jsonl" == path.name
rows = (json.loads(line) for line in data.splitlines() if line)
prompts = list(dict.fromkeys(
    " ".join(row["instruction"].split()) for row in rows
    if row["category"] == "creative_writing" and not row["context"].strip()
))
data = ("\n".join(prompts[:16]) + "\n").encode()
digest = hashlib.sha256(data).hexdigest()
(runtime / "prompts" / f"{digest}.txt").write_bytes(data)
print(json.dumps([hashlib.sha256(" ".join(prompt.casefold().split()).encode()).hexdigest()
                  for prompt in prompts[:16]]))
config = tomllib.loads((runtime / "configs/noslop.toml").read_text())
config.update(max_response_length=12, batch_size=4, response_prefix="", upload_tags=[])
for group in ("scorer", "modifier"):
    for plugin in config[group].values():
        for spec in plugin.values():
            if isinstance(spec, dict) and "dataset" in spec:
                spec.update(dataset="databricks/dolly-15k:creative_writing",
                            brewery_sha256=digest, split="[8:12]" if group == "scorer" else "[:8]")
(runtime / "configs/noslop-dolly.toml").write_text(tomli_w.dumps(config))
"""
    result = subprocess.run([runtime / "venv/bin/python", "-IB", "-c", profile_builder,
                             runtime, cache / f"{DOLLY}.jsonl"], check=True, capture_output=True)
    return set(json.loads(result.stdout))


def extended_cache(source, dest, excluded):
    dest.mkdir(exist_ok=True)
    for digest in PROMPTS.values():
        path = source / f"{digest}.json"
        require(sha(path) == digest, "unverified A2 prompt set")
        shutil.copyfile(path, dest / path.name)
    data = (source / f"{DOLLY}.jsonl").read_bytes()
    require(hashlib.sha256(data).hexdigest() == DOLLY, "unverified A1 Dolly cache")
    rows = (json.loads(line) for line in data.splitlines() if line)
    rows = [row for row in rows if hashlib.sha256(
        " ".join(row["instruction"].casefold().split()).encode()
    ).hexdigest() not in excluded]
    extended = normalize("slop", rows, train=12, held_out=32)
    digest = hashlib.sha256(extended).hexdigest()
    (dest / f"{digest}.json").write_bytes(extended)
    return {"slop": digest, **PROMPTS}


def wilson(rate, count):
    z = 1.959963984540054
    denom = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denom
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denom
    return [max(0., center - half), min(1., center + half)]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "bins", "runtime", "resource-queue", "contrast-cache", "runner"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--resume-checks", action="store_true", help="Reuse a completed Heretic child and its private GGUF pair.")
    args = cli.parse_args()
    excluded = install_acceptance_profile(args.runtime.resolve(), args.contrast_cache.resolve())
    run = args.root.resolve()
    approved = (ARB / "xa-scratch").resolve()
    require(run.is_relative_to(approved) and run != approved,
            "use an isolated directory under /arboretum/xa-scratch")
    require(sha(args.runner) == A2_RUNNER, "A2 runner hash mismatch")
    if not args.resume_checks:
        run.mkdir(mode=0o700, parents=True)
    cache = run / "contrast-cache"
    prompt_pins = extended_cache(args.contrast_cache, cache, excluded)
    evidence = run / "evidence"
    dist = run / "web"
    if args.resume_checks:
        require((evidence / "child.json").is_file() and (evidence / "child-gguf.json").is_file()
                and (evidence / "baseline-gguf.json").is_file(), "missing acceptance child or GGUF evidence")
        for name in ("baseline-gguf", "child-gguf"):
            prior = json.loads((evidence / f"{name}.json").read_text())
            require(prior["state"] == "succeeded", f"{name} did not succeed")
        env = json.loads((run / "controller-env.json").read_text())
    else:
        evidence.mkdir()
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><title>Isolated A6 acceptance</title>\n")
        env = setup(SimpleNamespace(bins=args.bins.resolve(), web_dist=dist), run)
        save(run, "controller-env", env)
    wrapper = run / "gpu-admission"
    wrapper.write_text(
        '#!/bin/bash\nset -euo pipefail\n'
        '[[ $1 == --wait && $2 == -- ]]\nshift 2\n'
        f'exec "{args.resource_queue.absolute()}" run --dir "{Path.home() / ".local/state/resource-queue"}" '
        '--resource gpu --holder brewery-a6 '
        '--max-hold 1800 --wait 600 --purpose "A6 Heretic local export" -- "$@"\n'
    )
    wrapper.chmod(0o700)
    worker = json.loads((run / "worker.json").read_text())
    worker["admission"] = str(wrapper)
    worker["engine_env"].update({
        "BREWERY_ALLOW_GPU_LOAD": "1",
        "BREWERY_HERETIC_RUNTIME": str(args.runtime.resolve()),
        "BREWERY_CONTRAST_CACHE": str(cache),
        "BREWERY_SCRATCH_CHECKS_RUNNER": str(args.runner.resolve()),
        "BREWERY_SCRATCH_CHECKS_RUNNER_SHA256": A2_RUNNER,
    })
    save(run, "worker", worker)
    (run / "worker.json").chmod(0o600)
    bins = {p.name: sha(p) for p in args.bins.glob("brewery-*") if p.is_file()}
    save(evidence, "binary-pins", bins)
    require(sha(SAFE) == PINS[SAFE], "Qwen SafeTensors base hash mismatch")
    with (evidence / "controller.log").open("wb") as log:
        with subprocess.Popen([args.bins / "brewery"], env=env, stdout=log, stderr=log) as proc:
            try:
                exercise(run, evidence, proc, prompt_pins, args.resume_checks)
            finally:
                proc.terminate()


def exercise(run, evidence, proc, prompt_pins, resume):
    for _ in range(200):
        text = (evidence / "controller.log").read_text()
        match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$", text, re.M)
        if match:
            require(int(match[2]) not in {0, 3000, 4095, 4141}, "unexpected local controller port")
            break
        require(proc.poll() is None, f"controller stopped: {text}")
        time.sleep(0.1)
    else:
        raise RuntimeError("controller did not start")
    base = match[1]
    http = build_opener(ProxyHandler({}))

    def request(path, body=None, key=None, allow=frozenset({200, 201, 202})):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Idempotency-Key"] = "a6-" + key
        if path == "/mcp":
            headers["Accept"] = "application/json, text/event-stream"
        req = Request(base + path, None if body is None else json.dumps(body).encode(), headers)
        try:
            response = http.open(req, timeout=120)
        except HTTPError as error:
            response = error
        with response:
            value = json.load(response)
            require(response.status in allow, f"{path}: {response.status}: {value}")
            return value

    def submit(route, label, **body):
        value = request(route, body, label)
        print(f"{label}: {value.get('id', '')}", flush=True)
        return value

    def wait(job, label):
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            state = request("/api/jobs/" + job["id"])
            if state["state"] not in ("queued", "running"):
                save(evidence, label, state)
                require(state["state"] == "succeeded", f"{label} failed: {state.get('error')}")
                return state["result"]
            time.sleep(1)
        raise TimeoutError(label)

    def call(name, **arguments):
        arguments["idempotency_key"] = "a6-" + name
        response = request("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": name, "arguments": arguments}})
        result = response["result"]
        require(not result["isError"], f"{name}: {result}")
        return result["structuredContent"]

    if resume:
        queued = json.loads((evidence / "heretic.json").read_text())
        child_id = queued["result"]["scratch_id"]
        child = request("/api/scratch/" + child_id)
        parent, = [m for m in request("/api/models") if m["id"] == child["provenance"]["parent_id"]]
        assets, = [m for m in request("/api/models") if m["kind"] == "assets"]
    else:
        require(not request("/api/models"), "controller Library is not empty")
        request("/api/admin/actions", {"action": "set_weight_editor", "worker": "local",
                                "revision": 0, "enabled": True}, "editor")
        parent = submit("/api/models", "base", host="local", kind="safetensors", path=str(SAFE))
        assets = submit("/api/models", "assets", host="local", kind="assets", path=str(run / "assets"))
        rejection = request("/api/scratch/heretic_reproduce", {
            "parent_id": parent["id"], "base_sha256": PINS[SAFE],
            "manifest": {"version": "4", "settings": {"model": "Qwen/Qwen2.5-0.5B-Instruct",
                                                       "model_commit": "a" * 40}},
        }, "unattested-manifest", allow={409})
        require("verified Hugging Face base download" in rejection["error"], "accepted an unattested base")
        save(evidence, "unattested-manifest", rejection)
        queued = call("run_heretic", parent_id=parent["id"], base_sha256=PINS[SAFE],
                      profile="noslop-dolly", trials=2, trial_index=0, seed=83,
                      assets_id=assets["id"])
        result = wait(queued, "heretic")
        child_id = result["scratch_id"]
        child = request("/api/scratch/" + child_id)
        require(child["provenance"]["parent_id"] == parent["id"]
                and child["state"] == "scratch" and child["delta"]["changed_tensors"] > 0,
                "Heretic output is not an edited Library child")
        save(evidence, "child", child)
        baseline = submit("/api/scratch", "baseline-copy", parent_id=parent["id"], parent_sha256=PINS[SAFE])
        for name, row in (("baseline", baseline), ("child", child)):
            conversion = submit(f"/api/scratch/{row['id']}/gguf", name + "-gguf",
                                assets_id=assets["id"], sha256=row["sha256"])
            wait(conversion, name + "-gguf")
    checks = submit(f"/api/scratch/{child_id}/checks", "a2-w4-held-out", sha256=child["sha256"],
                    behavior_direction={"prompt_sets": prompt_pins})
    result = wait(checks, "a2-w4-held-out")
    bundle = result["check_bundle"]
    require(all(bundle[k]["status"] == "passed" for k in ("integrity", "smoke", "behavior")),
            "W4 checks did not complete")
    behavior = bundle["behavior_direction"]
    require(behavior["config"]["prompt_sets"] == prompt_pins and behavior["counts"]["slop"] == 32,
            "A2 prompt pins mismatch")
    require({m["id"] for m in request("/api/models")} == {parent["id"], assets["id"]},
            "Heretic changed the Library")
    marker = next(row for row in behavior["metrics"] if row["name"] == "slop_marker_rate")
    save(evidence, "report", {
        "qualification": "not release-qualified", "heretic_job": queued["id"], "scratch_id": child_id,
        "base_sha256": PINS[SAFE], "child_sha256": child["sha256"],
        "changed_tensors": child["delta"]["changed_tensors"],
        "w4": bundle["measured_report"]["metrics"], "a2": behavior["metrics"],
        "slop": {"count": 32, "pin": prompt_pins["slop"], "interval": "Wilson 95% per marginal rate",
                 **{role: {"rate": marker[role], "flags": round(32 * marker[role]),
                           "interval": wilson(marker[role], 32)} for role in ("parent", "child")}},
    })


if __name__ == "__main__":
    main()
