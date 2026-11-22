"""Run native HT3/HT4 conversion against an isolated local controller and worker."""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from urllib.error import HTTPError

from qualification.local_sm120 import controller, request, save
from qualification.test_layer_recipe_hgn import (
    decode_trellis, indexed_digest, llama_fixture, read_hgn_v2, sha, snapshot, write_safetensors,
)


def wait(url, body, expected, worker):
    try:
        flow, _ = request(url, "conversions", body)
    except HTTPError as error:
        raise AssertionError((error.code, error.read().decode())) from error
    until = time.monotonic() + 90
    while flow["state"] == "running" and time.monotonic() < until:
        assert "artifact_model_id" not in flow
        time.sleep(.15)
        flow, _ = request(url, "conversions/" + flow["id"])
    jobs = {stage: request(url, "jobs/" + ident)[0] for stage, ident in flow["job_ids"].items()}
    if flow["state"] != expected:
        logs = [p.read_text()[-4096:] for p in (worker / "jobs" / flow["job_ids"][flow["stage"]]).rglob("engine-stderr.log")]
        raise AssertionError((flow, jobs[flow["stage"]].get("error"), logs))
    return flow, jobs


PIN = "BREWERY_EXPECTED_SOURCE_SHA256"


def check_pins(bins, root, config, engine_job, mode):
    env = os.environ.copy()
    env.pop(PIN, None)
    if mode == "engine-pin":
        env.update(config["engine_env"], **{PIN: "0" * 64})
        proc = subprocess.run([bins / "brewery-engine"], input=json.dumps(engine_job), capture_output=True,
                              text=True, cwd=root / "worker", env=env, timeout=10)
        assert proc.returncode != 0 and not proc.stdout and "source-pinned" in proc.stderr, proc
        return
    env.update(BREWERY_WORKER_CONFIG=str(root / "worker.json"))
    env[PIN] = "0" * 64
    for variant in ("inherited", "configured"):
        if variant == "configured":
            env.pop(PIN)
            config["engine_env"][PIN] = "0" * 64
            save(root / "worker.json", config)
        rpc = {"op": "prepare", "id": f"native-{variant}-pin", "action": "start", "job": engine_job}
        proc = subprocess.run([bins / "brewery-worker", "rpc"], input=json.dumps(rpc), capture_output=True,
                              check=True, text=True, cwd=root, env=env, timeout=10)
        reply = json.loads(proc.stdout)["ok"]
        assert reply["state"] != "queued", (variant, reply)
        assert "source-pinned" in reply["error"], (variant, reply)


def run(bins, helper, indexed, mode, weight_edit):
    with tempfile.TemporaryDirectory(prefix="native-hgn-smoke-", dir="/arboretum") as tmp:
        root = Path(tmp)
        (root / "bin").symlink_to(bins)
        (root / "worker").mkdir(mode=0o700)
        source_root = root / "sources"
        source_root.mkdir()
        assets = source_root / ("indexed" if indexed else "assets")
        tensors, _ = llama_fixture(assets, trellis=True)
        weights = (assets if indexed else source_root) / "model.safetensors"
        write_safetensors(weights, tensors)
        if indexed:
            save(assets / "model.safetensors.index.json",
                 {"weight_map": {name: weights.name for name in tensors}})
        original, original_assets = weights.read_bytes(), snapshot(assets)
        source_digest = indexed_digest(original_assets) if indexed else sha(original)
        recipe = {"schema": "brewery.layer-quant.v1", "source_sha256": source_digest,
                  "default_type": "BF16", "layers": {"0": "HT3", "1": "HT4"}}
        config = {
            "root": str(root / "worker"), "source_roots": [str(source_root)],
            "engine": str(bins / "brewery-engine"),
            "engine_env": {
                "BREWERY_LAYER_HGN_HELPER": str(helper), "BREWERY_LAYER_HGN_HELPER_SHA256": sha(helper.read_bytes()),
                "BREWERY_CPU_THREADS": "2",
            },
        }
        caps = {"hgn_layer_recipe": True}
        if weight_edit is not None:
            caps.update(weight_inspect=True, weight_visual=True, weight_safetensors=True)
            config["engine_env"].update(BREWERY_WEIGHT_EDIT=str(weight_edit), BREWERY_WEIGHT_VISUAL="1",
                                        BREWERY_WEIGHT_SAFETENSORS="1",
                                        BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES="67108864")
        save(root / "worker.json", config)
        if mode != "workflow":
            def resolved(kind, path, digest):
                return {"kind": kind, "path": str(path), "host": "tiny", "source_host": "tiny",
                        "sha256": digest, "bundle_sha256": digest if path.is_dir() else
                        indexed_digest({path.name: path.read_bytes()})}

            inputs = ([resolved("hf", assets, source_digest)] if indexed else
                      [resolved("safetensors", weights, source_digest),
                       resolved("assets", assets, indexed_digest(original_assets))])
            job = {"kind": "inspect", "inputs": inputs,
                   "options": {"output_format": "hgn-layer-recipe", "layer_recipe": recipe}}
            check_pins(bins, root, config, job, mode)
            return
        save(root / "dispatch.json", {"hosts": {"tiny": {
            "local": True, "worker": str(bins / "brewery-worker"),
            "config": str(root / "worker.json"), "capabilities": caps,
        }}})
        with controller(root, 0) as url:
            assert request(url, "state")[0]["hosts"][0]["capabilities"]["hgn_layer_recipe"]
            def register(kind, path):
                return request(url, "models", {"kind": kind, "path": str(path), "host": "tiny"})[0]["id"]

            ids = ([register("hf", assets)] if indexed
                   else [register("safetensors", weights), register("assets", assets)])
            if weight_edit is not None:
                job, _ = request(url, "jobs", {"kind": "weight_inspect", "host": "tiny", "input_ids": ids[:1],
                                               "options": {"mode": "inventory", "offset": 0, "limit": 128}})
                route = "jobs/" + job["id"]
                until = time.monotonic() + 25
                while job["state"] in ("queued", "running") and time.monotonic() < until:
                    time.sleep(.15)
                    job, _ = request(url, route)
                assert job["state"] == "succeeded", job.get("error")
                inventory = job["result"]["report"]
                assert inventory["source_sha256"] == source_digest
                assert inventory["next_offset"] is None
                assert {t["name"] for t in inventory["tensors"]} == tensors.keys()
                recipe["source_sha256"] = inventory["source_sha256"]
            body = {"format": "hgn", "input_ids": ids, "options": {"layer_recipe": recipe}}
            # Direct jobs must not be able to select the managed adapter.
            try:
                request(url, "jobs", {"kind": "inspect", "host": "tiny", "input_ids": ids,
                                      "options": {"layer_recipe": recipe, "output_format": "hgn-layer-recipe"}})
            except HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError("direct jobs accepted managed HGN")

            bad = {**body, "options": {"layer_recipe": {**recipe, "source_sha256": "0" * 64}}}
            rejected, jobs = wait(url, bad, "failed", root / "worker")
            assert set(jobs) == {"inspect"} and "artifact_model_id" not in rejected
            assert len(request(url, "state")[0]["models"]) == len(ids)

            def check(flow, jobs, expected):
                assert set(jobs) == {"inspect", "plan", "convert", "verify"}
                assert all(j["state"] == "succeeded" for j in jobs.values())
                reports = {stage: job["result"] for stage, job in jobs.items()}
                assert all(r["layer_plan_sha256"] == reports["inspect"]["layer_plan_sha256"]
                           and r["layer_recipe_plan"]["recipe"] == expected
                           and r["runtime_load_verified"] is False
                           and r["release_qualified"] is False for r in reports.values())
                verify = reports["verify"]
                receipt = verify["layer_recipe_receipt"]
                assert receipt["structural_reader_verified"] is True and receipt["runtime_load_verified"] is False
                models = request(url, "state")[0]["models"]
                output = next(m for m in models if m["id"] == flow["artifact_model_id"])
                assert output["kind"] == "hgn-archive" and output["host"] == "tiny"
                path = Path(output["path"])
                archive = snapshot(path)
                assert indexed_digest(archive) == verify["inputs"][1]["sha256"]
                raw = (path / "model.hgn").read_bytes()
                assert sha(raw) == receipt["model"]["sha256"]
                published_bytes = sum(map(len, archive.values()))
                asset_bytes = published_bytes - len(raw) - len(archive["layer-recipe.receipt.json"])
                preflight = reports["plan"]["report"]["disk"]
                encode = reports["convert"]["report"]["disk"]
                verify_disk = verify["report"]["disk"]
                assert preflight["phase"] == "preflight", preflight
                assert preflight["retained_archive_bytes"] >= published_bytes
                assert preflight["verify_copy_bytes"] == len(raw) + asset_bytes
                assert preflight["payload_bytes"] == max(encode["payload_bytes"],
                    preflight["retained_archive_bytes"] + preflight["verify_copy_bytes"])
                assert encode["phase"] == "convert" and encode["source_stream_bytes"] > 0
                assert encode["payload_bytes"] >= encode["source_stream_bytes"] + len(raw) + asset_bytes
                assert verify_disk["phase"] == "verify" and verify_disk["retained_archive_bytes"] == published_bytes
                assert verify_disk["payload_bytes"] == verify_disk["verify_copy_bytes"] == len(raw) + asset_bytes
                assert verify_disk["payload_bytes"] < encode["payload_bytes"]
                assert verify["file_verification"] == {"status": "passed", "scope": "structure_only",
                    "archive_sha256": verify["inputs"][1]["sha256"], "model_sha256": sha(raw)}
                table = read_hgn_v2(raw)
                for layer, kind in expected["layers"].items():
                    bits = int(kind[-1])
                    name = f"model.layers.{layer}.self_attn.q_proj"
                    code, variant, shape, payload, _ = table[name + ".weight"]
                    assert (code, variant, shape) == (16, 0x1200 + 2 * bits, [128, 128])
                    su, sv = [struct.unpack("<128e", table[name + "." + tag][3]) for tag in ("suh", "svh")]
                    decoded = [v for row in decode_trellis(payload, 128, 128, bits, su, sv) for v in row]
                    source = tensors[name + ".weight"][2]
                    values = struct.unpack("<16384f", b"".join(b"\0\0" + source[i:i+2] for i in range(0, len(source), 2)))
                    nrmse = math.sqrt(sum((a-b)**2 for a,b in zip(decoded, values)) / sum(b*b for b in values))
                    assert nrmse < .35, (layer, kind, nrmse)
                assert weights.read_bytes() == original and snapshot(assets) == original_assets
                return {"workflow": flow["id"], "archive_sha256": verify["inputs"][1]["sha256"],
                        "plan_sha256": verify["layer_plan_sha256"], "jobs": flow["job_ids"]}

            first = check(*wait(url, body, "succeeded", root / "worker"), recipe)
            first_path = root / "worker" / "jobs" / first["jobs"]["convert"] / "work" / "converted.hgn-bundle"
            first_archive = snapshot(first_path)
            revised = {**recipe, "layers": {"0": "HT4", "1": "HT3"}}
            second = check(*wait(url, {**body, "options": {"layer_recipe": revised}}, "succeeded", root / "worker"), revised)
            assert all(first[key] != second[key] for key in ("workflow", "archive_sha256", "plan_sha256"))
            assert snapshot(first_path) == first_archive
            assert len(request(url, "state")[0]["models"]) == len(ids) + 2
            print(json.dumps({"layout": "indexed" if indexed else "standalone",
                              "original_preserved": True, "first": first, "revised": second}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--weight-edit", type=Path, help="packaged read-only inventory helper used by Convert")
    parser.add_argument("--check", choices=("workflow", "worker-pin", "engine-pin"), default="workflow")
    args = parser.parse_args()
    for indexed in (False, True) if args.check == "workflow" else (False,):
        run(args.bin_dir.resolve(strict=True), args.helper.resolve(strict=True), indexed, args.check,
            args.weight_edit.resolve(strict=True) if args.weight_edit else None)
