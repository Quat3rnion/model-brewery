"""Compare native and Heretic scratch children through an isolated MCP stack."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.request import ProxyHandler, Request, build_opener

from ablation_search_acceptance import A1, PIN_ROOT, prepare
from heretic_acceptance import A2_RUNNER, extended_cache, install_acceptance_profile, wilson
from weight_space_acceptance import ARB, PINS, QUANT, REPO, SAFE, W0_LIBS, require, save, setup, sha
from scratch_checks import BENIGN_REFUSAL_SHA256, REFUSAL_PREFIXES, begins, normalized


A3 = ARB / "a3-evidence/sweep/report.json"
A5 = ARB / "a5-evidence"
A6 = ARB / "brewery-a6-evidence"


def load(path):
    return json.loads(path.read_bytes())


def statistics(parent, child):
    n = len(parent)
    require(n > 0 and n == len(child), "paired sample sizes differ")
    improved = sum(a and not b for a, b in zip(parent, child))
    worsened = sum(b and not a for a, b in zip(parent, child))
    discordant = improved + worsened
    return dict(n=n, parent_hits=sum(parent), child_hits=sum(child), improved=improved,
                worsened=worsened, parent_wilson95=wilson(sum(parent) / n, n),
                child_wilson95=wilson(sum(child) / n, n),
                exact_mcnemar_p=min(1., 2 * sum(math.comb(discordant, i)
                    for i in range(min(improved, worsened) + 1)) / (2 ** discordant)))


def compare(native, heretic, pins, reference):
    for bundle in (native, heretic):
        require(all(bundle[k]["status"] == "passed" for k in ("integrity", "smoke", "behavior")),
                "a W4 check did not complete")
        require(bundle["parent_sha256"] == PINS[SAFE], "check SafeTensors parent mismatch")
        scores = bundle["behavior_direction"]
        require(scores["config"]["prompt_sets"] == pins and scores["cpu_threads"] == 4
                and scores["counts"] == reference["counts"]
                and scores["kl_prompts"] == reference["kl_prompts"],
                "check protocol mismatch")
    results = {}
    for behavior in ("slop", "sycophancy"):
        a, b = [row["paired_outcomes"][behavior] for row in
                (native["behavior_direction"], heretic["behavior_direction"])]
        parent = [row["parent"] for row in a]
        require(parent == [row["parent"] for row in b], "paired stock flags disagree")
        x, y = [[row["child"] for row in rows] for rows in (a, b)]
        if behavior == "sycophancy":
            c = reference["paired_outcomes"][behavior]
            require(list(zip(parent, x)) == [(row["parent"], row["child"]) for row in c],
                    "native replay disagrees with A5 trial 0")
        results[behavior] = {
            "stock_vs_native": statistics(parent, x),
            "stock_vs_heretic": statistics(parent, y),
            "native_vs_heretic": statistics(x, y),
        }
    adjust_holm(entry for rows in results.values() for entry in rows.values())
    return results


def adjust_holm(entries):
    family = sorted(entries, key=lambda entry: entry["exact_mcnemar_p"])
    previous = 0.
    for i, entry in enumerate(family):
        previous = min(1., max(previous, (len(family) - i) * entry["exact_mcnemar_p"]))
        entry["holm_adjusted_p"] = previous


def benign_refusal(run, runner, gguf, bundles):
    """Run only the new benign rows, without storing any generated replies."""
    source = ARB / "a1r-accept/cache" / (BENIGN_REFUSAL_SHA256 + ".json")
    require(sha(source) == BENIGN_REFUSAL_SHA256 and sha(runner) == A2_RUNNER, "benign refusal pin mismatch")
    document = load(source)
    require(len(document["rows"]) == 16, "expected sixteen reviewed prompts")
    prompts = "".join(f"generate\t\t{row['user'].encode().hex()}\n" for row in document["rows"])
    stock = run / "fit-source/parent.gguf"
    require(sha(stock) == gguf["sha256"], "benign refusal parent GGUF mismatch")
    flags = {}
    for name, (revision_id, bundle) in bundles.items():
        model = run / "state/scratch" / revision_id / "model.gguf"
        require(sha(model) == bundle["measured_report"]["revision_gguf_sha256"],
                f"{name}: benign refusal child GGUF mismatch")
        result = subprocess.run([runner.resolve(), stock, "4", "--behavior-v1", model],
                                env={**os.environ, "LD_LIBRARY_PATH": str(W0_LIBS)}, input=prompts,
                                text=True, capture_output=True, timeout=1200, check=False)
        require(result.returncode == 0, f"{name}: benign refusal runner failed")
        pairs = []
        for line in result.stdout.splitlines():
            tag, a, count_a, b, count_b = line.split("\t")
            require(tag == "generate" and all(0 <= int(n) <= 64 for n in (count_a, count_b)),
                    "invalid benign refusal runner result")
            pairs.append([any(begins(normalized(bytes.fromhex(hex_text).decode(errors="replace")), p)
                              for p in REFUSAL_PREFIXES) for hex_text in (a, b)])
        require(len(pairs) == 16, "benign refusal result size differs")
        flags[name] = pairs
    stock, native = map(list, zip(*flags["native"]))
    also_stock, heretic = map(list, zip(*flags["heretic"]))
    require(stock == also_stock, "benign refusal stock flags disagree")
    stats = {"stock_vs_native": statistics(stock, native), "stock_vs_heretic": statistics(stock, heretic),
             "native_vs_heretic": statistics(native, heretic)}
    adjust_holm(stats.values())
    return {"scope": "over-refusal, not harmful-set compliance; separate Holm family",
            "method": "paired CPU runner on the same checked GGUFs", "prompt_set_sha256": BENIGN_REFUSAL_SHA256,
            "runner_sha256": A2_RUNNER, "cpu_threads": 4, "flags": flags, "comparisons": stats}


def report(native, heretic, gguf, pins, bins, refusal=None):
    prior = load(A5 / "a5-sycophancy-7-0-checks.json")
    reference = prior["behavior_direction"]
    measurements = {}
    for name, bundle in (("native", native), ("heretic", heretic)):
        measured = bundle["measured_report"]
        require(measured["parent_gguf_sha256"] == gguf["sha256"]
                and measured["parent_sha256"] == PINS[SAFE]
                and measured["revision_sha256"] == bundle["revision_sha256"]
                and measured["runner_sha256"] == A2_RUNNER, f"{name}: W4 pin mismatch")
        measurements[name] = {"sha256": bundle["revision_sha256"],
                              "gguf_sha256": measured["revision_gguf_sha256"],
                              "a2": bundle["behavior_direction"]["metrics"], "w4": measured["metrics"],
                              "w4_fixtures": measured["fixture_sha256"]}
    require(measurements["native"]["w4_fixtures"] == measurements["heretic"]["w4_fixtures"],
            "W4 fixtures disagree")
    for name, previous in (("native", prior),
                           ("heretic", load(A6 / "a2-w4-held-out.json")["result"]["check_bundle"])):
        measured = previous["measured_report"]
        actual = measurements[name]
        require(actual["sha256"] == measured["revision_sha256"]
                and actual["gguf_sha256"] == measured["revision_gguf_sha256"]
                and actual["w4_fixtures"] == measured["fixture_sha256"]
                and actual["w4"] == measured["metrics"], f"{name}: archived W4 measurements differ")
    require(reference["config"]["prompt_sets"]["sycophancy"] == pins["sycophancy"],
            "A5 paired confirmation pin differs")
    cpu = load(A6 / "journal.json")
    require(cpu["base_sha256"] == PINS[SAFE] and 'device_map = "cpu"' in cpu["effective_config"],
            "A6 CPU smoke pin mismatch")
    a3 = load(A3)
    require(a3["parent_gguf"]["revision_sha256"] == PINS[SAFE]
            and a3["parent_gguf"]["sha256"] == gguf["sha256"], "A3 baseline pin mismatch")
    return {
        "qualification": "not release-qualified", "base_safetensors_sha256": PINS[SAFE],
        "base_gguf_sha256": gguf["sha256"], "prompt_sets": pins, "runner_sha256": A2_RUNNER,
        "bins": bins, "comparisons": compare(native, heretic, pins, reference),
        "a3_context": {"scope": "same stock model, different edits and holdout; not pooled with A7",
                       "source_report_sha256": sha(A3),
                       "paired": {behavior: {name: row["statistics"] for name, row in trials.items()}
                                  for behavior, trials in a3["trials"].items()}},
        "a6_heretic_cpu_smoke": {"export_sha256": cpu["export_sha256"], "runtime": cpu["runtime"],
                                  "scope": "export only; not the scored GPU child"},
        "benign_refusal": refusal if refusal is not None else "not measured; not a native direction",
        "gsm8k": "not measured; private scratch GGUFs are not accepted by the managed GSM8K harness",
        **measurements,
    }


class Mcp:
    def __init__(self, url, evidence):
        self.url, self.evidence = url, evidence

    def rpc(self, method, params):
        request = Request(self.url + "/mcp", json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
        }).encode(), {"Accept": "application/json, text/event-stream",
                     "Content-Type": "application/json", "Origin": self.url,
                     "MCP-Protocol-Version": "2025-06-18"})
        with build_opener(ProxyHandler({})).open(request, timeout=180) as result:
            return json.load(result)["result"]

    def call(self, name, **arguments):
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        require(not result["isError"], f"{name}: {result}")
        return result["structuredContent"]

    def submit(self, tool, label, **args):
        receipt = self.call(tool, idempotency_key=f"a7-{label}", **args)
        save(self.evidence, label + "-queued", receipt)
        print(f"{tool} {label}: {receipt['id']}", flush=True)
        return receipt

    def wait(self, job, label):
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            row = self.call("get_job", id=job["id"], wait_ms=5000)
            if row["state"] not in ("queued", "running"):
                save(self.evidence, label, row)
                require(row["state"] == "succeeded", f"{label}: {row.get('error')}")
                print(f"{label}: succeeded", flush=True)
                return row["result"]
            time.sleep(1)
        raise TimeoutError(f"{label}: {job['id']}")


def run_flow(mcp, run, pins, previous):
    init = mcp.rpc("initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "brewery-a7-acceptance", "version": "1"},
    })
    require(init["protocolVersion"] == "2025-06-18", "MCP protocol mismatch")
    catalog = {row["name"] for row in mcp.rpc("tools/list", {})["tools"]}
    require({"compute_direction", "get_direction", "create_direction_intervention",
             "ablate_direction", "search_ablation", "run_heretic", "reproduce_heretic",
             "run_checks", "get_checks"}.issubset(catalog)
            and not any("approve" in name for name in catalog), "MCP workflow catalog changed")
    save(mcp.evidence, "mcp-session", {"initialize": init, "tools": sorted(catalog)})
    require(not mcp.call("list_models")["models"], "expected empty Library")
    req = Request(mcp.url + "/api/admin/actions", json.dumps({
        "action": "set_weight_editor", "worker": "local", "revision": 0, "enabled": True,
    }).encode(), {"Content-Type": "application/json", "Origin": mcp.url})
    with build_opener(ProxyHandler({})).open(req, timeout=30) as response:
        require(response.status == 200, "local edit admission failed")
    source = mcp.submit("register_model", "source", host="local", kind="safetensors",
                        path=str(SAFE))["id"]
    assets = mcp.submit("register_model", "assets", host="local", kind="assets",
                        path=str(run / "assets"))["id"]
    root = mcp.submit("create_scratch", "root", parent_id=source, parent_sha256=PINS[SAFE])
    gguf = mcp.wait(mcp.submit("convert_scratch_gguf", "baseline-gguf", id=root["id"],
                    assets_id=assets, sha256=root["sha256"]), "baseline-gguf")["gguf"]
    require(gguf["sha256"] == previous["gguf"]["sha256"], "stock GGUF differs from A5")
    exact = run / "fit-source" / "parent.gguf"
    shutil.copyfile(run / "state" / "scratch" / root["id"] / "model.gguf", exact)
    require(sha(exact) == gguf["sha256"], "copied GGUF differs from stock")
    gguf_id = mcp.submit("register_model", "fit-source", host="local", kind="gguf",
                         path=str(exact))["id"]

    def native():
        fit = mcp.submit("compute_direction", "fit", input_ids=[gguf_id], options={
            "behavior": "sycophancy", "model_sha256": gguf["sha256"], "layers": list(range(3, 24))})
        mcp.wait(fit, "fit")
        direction = mcp.call("get_direction", id=fit["id"], version=2)["direction"]
        require(direction["model_sha256"] == gguf["sha256"] and direction["fit"]["selected_layer"] == 10
                and direction["contrast_sha256"] == previous["results"]["a5-sycophancy-7"]["screen"]["screen_sha256"],
                "fit model or contrast pin changed")
        search = mcp.submit("search_ablation", "search", id=root["id"], sha256=root["sha256"],
                            direction_id=fit["id"], assets_id=assets, seed=7, trial_budget=3,
                            max_children=1, disk_budget_bytes=12 << 30,
                            behavior_direction={"prompt_sets": pins})
        result = mcp.wait(search, "search")
        old = previous["results"]["a5-sycophancy-7"]
        require(result["screen"]["trials"] == old["screen"]["trials"][:3],
                "screen did not replay the first three A5 trials")
        child, = result["children"]
        prior = old["children"][0]
        require(result["backend"] == "cpu" and child["trial"] == prior["trial"] == 0
                and child["sha256"] == prior["sha256"]
                and child["gguf_sha256"] == prior["gguf_sha256"], "native child pins changed")
        readback = mcp.call("get_scratch", id=child["id"])
        require(readback["state"] == "scratch" and readback["sha256"] == child["sha256"]
                and readback["provenance"]["method"] == "ablate_direction", "not a private native edit")
        bundle = mcp.call("get_checks", id=child["id"])
        require(bundle["measured_job_id"] == child["checks_job_id"], "native check readback did not match job")
        return child["id"], bundle

    def heretic():
        job = mcp.submit("run_heretic", "heretic", parent_id=source, base_sha256=PINS[SAFE],
                         profile="noslop-dolly", trials=2, trial_index=0, seed=83, assets_id=assets)
        result = mcp.wait(job, "heretic")
        child = mcp.call("get_scratch", id=result["scratch_id"])
        prior = load(A6 / "gpu-child.json")
        journal = child["provenance"]["heretic"]
        require(child["state"] == "scratch" and child["provenance"]["parent_id"] == source,
                "Heretic child is not a private edit of the registered Library parent")
        prior_journal = prior["provenance"]["heretic"]
        for field in ("prompts", "assets", "source_config", "invocation", "selected_study_trial"):
            require(journal[field] == prior_journal[field], f"Heretic pin changed: {field}")
        require({k: v for k, v in journal["runtime"].items() if k != "launcher_sha256"} ==
                {k: v for k, v in prior_journal["runtime"].items() if k != "launcher_sha256"},
                "Heretic environment pin changed")
        require(child["sha256"] == journal["export_sha256"] == prior["sha256"], "Heretic export differs from A6")
        require(journal["runtime"]["launcher_sha256"] == sha(REPO / "tools/heretic/run.py")
                and len(journal["trials"]) == 2, "missing A7 trial journal")
        save(mcp.evidence, "heretic-child", child)
        gguf = mcp.wait(mcp.submit("convert_scratch_gguf", "heretic-gguf", id=child["id"],
                        sha256=child["sha256"], assets_id=assets), "heretic-gguf")["gguf"]
        require(gguf["sha256"] == load(A6 / "gpu-child-gguf.json")["result"]["gguf"]["sha256"],
                "Heretic converted child differs from A6")
        result = mcp.wait(mcp.submit("run_checks", "heretic-checks", id=child["id"],
                          sha256=child["sha256"], behavior_direction={"prompt_sets": pins}),
                          "heretic-checks")
        bundle = mcp.call("get_checks", id=child["id"])
        require(bundle == result["check_bundle"], "Heretic check readback did not match job")
        return child["id"], bundle

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(native), pool.submit(heretic)
        native, heretic = a.result(), b.result()
    require({row["id"] for row in mcp.call("list_models")["models"]} == {source, assets, gguf_id},
            "a scratch child appeared in the Library")
    for name, (_, bundle) in (("native", native), ("heretic", heretic)):
        save(mcp.evidence, name + "-bundle", bundle)
    return native, heretic, gguf


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "evidence", "bins", "runner", "runtime", "resource-queue"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    run = args.root.resolve()
    require(run.is_relative_to((ARB / "xa-scratch/a7").resolve()) and not run.exists(),
            "use a fresh directory under isolated A7 scratch")
    args.evidence.mkdir(parents=True, exist_ok=True)
    run.mkdir(mode=0o700)
    args.web_dist = run / "web"
    args.web_dist.mkdir()
    (args.web_dist / "index.html").write_text("<!doctype html><title>Isolated A7 acceptance</title>\n")
    require(sha(SAFE) == PINS[SAFE] and sha(QUANT) == PINS[QUANT]
            and sha(args.runner) == A2_RUNNER, "runtime or model hash mismatch")
    for pins_dir in (PIN_ROOT, ARB / "pins/lanternfish-d842085"):
        subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=pins_dir, check=True)
    subprocess.run(["sha256sum", "-c", ARB / "pins/lanternfish-0ea30b3.libs.sha256"], check=True)
    runtime = run / "heretic"
    require(args.runtime.resolve() == (ARB / "heretic").resolve(), "unexpected A6 runtime source")
    runtime.mkdir(mode=0o700)
    for name in ("configs", "prompts"):
        shutil.copytree(args.runtime / name, runtime / name)
    shutil.copyfile(REPO / "tools/heretic/run.py", runtime / "run.py")
    shutil.copyfile(REPO / "tools/heretic/uv.lock", runtime / "uv.lock")
    subprocess.run(["cp", "-al", str(args.runtime / "venv"), str(runtime / "venv")], check=True)
    with (runtime / "identity.json").open("w") as identity:
        subprocess.run([runtime / "venv/bin/python", "-IB", runtime / "run.py", "identity", runtime],
                       stdout=identity, check=True)
    (runtime / "runs").mkdir(mode=0o700)
    excluded = install_acceptance_profile(runtime, A1 / "fresh-cache")
    cache, native_pins = prepare(run, args.evidence)
    # The A5 slop set overlaps the A6 optimization set. Use the disjoint A6
    # slop set, while preserving the A5 search's sycophancy confirmation pin.
    a6_pins = extended_cache(A1 / "fresh-cache", cache, excluded)
    a6_bundle = load(A6 / "a2-w4-held-out.json")["result"]["check_bundle"]
    require(a6_pins == a6_bundle["behavior_direction"]["config"]["prompt_sets"], "A6 holdout pins changed")
    pins = {**native_pins, "slop": a6_pins["slop"]}
    slop_groups = load(cache / (pins["slop"] + ".json"))["rows"]
    require(not excluded.intersection(hashlib.sha256(
        " ".join(row["positive"]["user"].casefold().split()).encode()).hexdigest()
        for row in slop_groups if row["split"] == "held_out"),
        "Heretic optimization and common slop holdout overlap")
    previous = load(A5 / "report.json")
    require(previous["safetensors"] == PINS[SAFE] and previous["prompt_sets"] == native_pins
            and previous["runner_sha256"] == A2_RUNNER, "A5 protocol changed")
    a6_base = load(A6 / "gpu-baseline-gguf.json")["result"]["gguf"]
    require(a6_base["sha256"] == previous["gguf"]["sha256"], "A6 stock GGUF differs from A5")
    env = setup(args, run)
    (run / "fit-source").mkdir()
    wrapper = run / "gpu-admission"
    wrapper.write_text(
        "#!/bin/bash\nset -euo pipefail\n[[ $1 == --wait && $2 == -- ]]\nshift 2\n"
        f'exec "{args.resource_queue.resolve()}" run --dir "{Path.home() / ".local/state/resource-queue"}" '
        '--resource gpu --holder brewery-a7 --max-hold 1800 --wait 600 '
        '--purpose "A7 Heretic offline export" -- "$@"\n')
    wrapper.chmod(0o700)
    worker = load(run / "worker.json")
    worker["source_roots"].append(str(run / "fit-source"))
    worker["admission"] = str(wrapper)
    worker["engine_env"].update({
        "BREWERY_ALLOW_GPU_LOAD": "1", "BREWERY_HERETIC_RUNTIME": str(runtime),
        "BREWERY_LANTERNFISH_ABLATION": str(PIN_ROOT / "lanternfish"),
        "BREWERY_LANTERNFISH_PROJECTION_LIB_DIR": "/arboretum/llama-a4/build/bin",
        "BREWERY_CONTRAST_CACHE": str(cache), "BREWERY_SCRATCH_CHECKS_RUNNER": str(args.runner),
        "BREWERY_SCRATCH_CHECKS_RUNNER_SHA256": A2_RUNNER,
    })
    save(run, "worker", worker)
    dispatch = load(run / "dispatch.json")
    dispatch["hosts"]["local"]["capabilities"]["behavior_direction"] = True
    save(run, "dispatch", dispatch)
    log = args.evidence / "controller.log"
    with log.open("wb") as stdout, subprocess.Popen([args.bins / "brewery"], env=env, cwd=REPO,
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stdout) as proc:
        try:
            for _ in range(200):
                match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$", log.read_text(), re.M)
                if match:
                    break
                require(proc.poll() is None, "controller stopped")
                time.sleep(.1)
            require(match and int(match[2]) not in {0, 3000, 4095, 4141}, "invalid isolated bind")
            native, heretic, gguf = run_flow(Mcp(match[1], args.evidence), run, pins, previous)
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    refusal = benign_refusal(run, args.runner, gguf, {"native": native, "heretic": heretic})
    save(args.evidence, "report", report(native[1], heretic[1], gguf, pins,
         {path.name: sha(path) for path in args.bins.glob("brewery*")}, refusal))


if __name__ == "__main__":
    main()
