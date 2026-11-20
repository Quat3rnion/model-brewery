"""Exercise both SafeTensors layouts against an isolated, real HTTP workbench."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time
from urllib.request import ProxyHandler, Request, build_opener

from local_sm120 import NoRedirect, controller, request, save
from test_layer_recipe_gguf import GgufLayerRecipe, read_gguf, sha256 as file_sha256, tree_sha256


def submit(url, body, key):
    req = Request(url + "/api/conversions", json.dumps(body).encode(), {
        "Content-Type": "application/json", "Idempotency-Key": key,
    })
    with build_opener(ProxyHandler({}), NoRedirect()).open(req, timeout=10) as reply:
        return reply.status, json.load(reply)


def run(args, indexed):
    fixture = GgufLayerRecipe()
    fixture.setUp()
    try:
        flags, original_sha = fixture.make_source(indexed, args.assets)
        source = flags[1]
        source_kind = "hf" if indexed else "safetensors"
        assets = None if indexed else flags[3]
        assets_sha = tree_sha256(assets) if assets else None
        fixture.write_recipe(original_sha, {"0": "Q8_0", "1": "F16"})
        recipe = json.loads(fixture.recipe.read_text())

        with tempfile.TemporaryDirectory(dir="/arboretum") as temp:
            root = Path(temp)
            (root / "bin").symlink_to(args.bin_dir)
            (root / "worker").mkdir(mode=0o700)
            save(root / "worker.json", {
                "root": str(root / "worker"), "source_roots": [str(fixture.root)],
                "engine": str(args.bin_dir / "brewery-engine"),
                "engine_env": {
                    "BREWERY_GGUF_LAYER_RECIPE": "1", "BREWERY_GGUF_SAFETENSORS": "1",
                    "BREWERY_GGUF_PYTHON": str(args.python), "BREWERY_LLAMA_CPP": str(args.llama),
                    "BREWERY_GGUF_HF_EXPORTER": str(args.exporter),
                    "BREWERY_GGUF_QUANTIZER": str(args.quantizer), "BREWERY_CPU_THREADS": "2",
                },
            })
            save(root / "dispatch.json", {"hosts": {"tiny": {
                "local": True, "worker": str(args.bin_dir / "brewery-worker"),
                "config": str(root / "worker.json"),
                "capabilities": {"gguf_output": True, "gguf_layer_recipe": True,
                                 "gguf_hf_output": True, "gguf_safetensors_output": True},
            }}})
            with controller(root, 0) as url:
                assert request(url, "state")[0]["hosts"][0]["capabilities"]["gguf_layer_recipe"]
                models = [(source_kind, source)] + ([("assets", assets)] if assets else [])
                ids = [request(url, "models", {"host": "tiny", "path": str(path), "kind": kind})[0]["id"]
                       for kind, path in models]
                body = {"format": "gguf", "input_ids": ids, "options": {"layer_recipe": recipe}}
                status, flow = submit(url, body, "layer-smoke.v1")
                assert status == 201
                status, replay = submit(url, body, "layer-smoke.v1")
                assert status == 200 and replay["id"] == flow["id"]
                until = time.monotonic() + 60
                while flow["state"] == "running" and time.monotonic() < until:
                    assert "artifact_model_id" not in flow
                    time.sleep(.15)
                    flow, _ = request(url, "conversions/" + flow["id"])
                results = {stage: request(url, "jobs/" + ident)[0]
                           for stage, ident in flow["job_ids"].items()}
                assert flow["state"] == "succeeded", (flow, results[flow["stage"]])
                assert set(results) == {"inspect", "plan", "convert", "verify"}
                assert all(r["state"] == "succeeded" for r in results.values())
                results = {stage: detail["result"] for stage, detail in results.items()}
                plan = results["inspect"]["layer_recipe_plan"]
                plan_sha = sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                assert all(r["layer_plan_sha256"] == plan_sha and r["layer_recipe_plan"] == plan
                           for r in results.values())
                assert plan["recipe"] == recipe and plan["assets_sha256"] == assets_sha
                source_bundle = (original_sha if indexed else sha256(
                    f"{source.name}\0{source.stat().st_size}\0{original_sha}\n".encode()
                ).hexdigest())
                role_manifest = f"source\0{source_kind}\0{source_bundle}\n"
                if assets:
                    role_manifest += f"assets\0assets\0{assets_sha}\n"
                source_sha = sha256(("brewery-canonical-source-v1\n" + role_manifest).encode()).hexdigest()
                assert plan["source_bundle_sha256"] == source_sha
                assert all(r["source_sha256"] == source_sha and r["inputs"][0]["sha256"] == original_sha
                           for r in results.values())
                state = request(url, "state")[0]
                assert len(state["models"]) == len(ids) + 1 and len(state["jobs"]) == 4
                artifact = next(m for m in state["models"] if m["id"] == flow["artifact_model_id"])
                path = Path(artifact["path"])
                assert artifact["kind"] == "gguf" and path.stat().st_size < 64 << 20
                assert (file_sha256(path) == results["convert"]["artifact"]["sha256"]
                        == results["verify"]["inputs"][1]["sha256"])
                stored = read_gguf(path)
                fixture.assert_types({name: ty for name, (_, ty, _) in stored.items()}, (8, 1))
                assert {name: {"shape": list(shape), "ggml_type": ty} for name, (shape, ty, _) in stored.items()
                        } == results["verify"]["report"]["stored_tensors"] == plan["tensors"]
                assert all(stored[name][2] == data for name, data in fixture.vectors.items())
                assert (tree_sha256(source) if indexed else file_sha256(source)) == original_sha
                assert assets is None or tree_sha256(assets) == assets_sha
                print(json.dumps({"source": source_kind, "original_sha256": original_sha,
                                  "source_sha256": source_sha, "layer_plan_sha256": plan_sha,
                                  "artifact_sha256": file_sha256(path), "stored_ggml_types": [0, 1, 8, 30],
                                  "workflow_id": flow["id"], "job_ids": flow["job_ids"]}), flush=True)
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exporter", type=Path, default=Path(__file__).resolve().parents[1] / "tools/gguf-export-hf")
    for name in ("bin-dir", "assets", "quantizer", "llama"):
        parser.add_argument("--" + name, required=True, type=lambda path: Path(path).resolve())
    # Preserve the Python entrypoint in the isolated runtime, not its symlink target.
    parser.add_argument("--python", required=True, type=lambda path: Path(path).absolute())
    args = parser.parse_args()
    for indexed in (False, True):
        run(args, indexed)
