"""Run one isolated four-stage GGUF workflow with a small real GGUF and quantizer."""
import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
import tempfile
import time

from local_sm120 import controller, request, save


def checksum(p):
    with p.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def run(bins, quantizer, source, preset, matrix):
    assert source.stat().st_size < (64 << 20), "smoke accepts only a bounded fixture"
    before = checksum(source)
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        (r / "bin").symlink_to(bins)
        (r / "worker").mkdir(mode=0o700)
        roots = {source.parent}
        if matrix:
            roots.add(matrix.parent)
        save(r / "worker.json", {"root": str(r/"worker"), "source_roots": sorted(map(str, roots)),
                                 "engine": str(bins/"brewery-engine"), "engine_env": {
                                     "BREWERY_GGUF_QUANTIZER": str(quantizer),
                                     "BREWERY_CPU_THREADS": "2"}})
        save(r / "dispatch.json", {"hosts":{"tiny":{"local":True, "worker":str(bins/"brewery-worker"),
                                   "config":str(r/"worker.json"),"capabilities":{"gguf_output":True}}}})
        with controller(r, 0) as url:
            catalog, _ = request(url, "profiles/gguf-catalog")
            assert preset in [v["id"] for v in catalog["presets"]]
            inputs = []
            for kind, path in [("gguf",source),("imatrix",matrix)]:
                if path:
                    m, _ = request(url,"models",{"name":path.name,"host":"tiny","kind":kind,"path":str(path)})
                    inputs.append(m["id"])
            flow, _ = request(url, "conversions", dict(format="gguf", host="tiny", input_ids=inputs,
                                                       options={"preset":preset}))
            timeout = time.monotonic() + 60
            while flow["state"] == "running" and time.monotonic() < timeout:
                time.sleep(.15)
                flow, _ = request(url,"conversions/"+flow["id"])
            if flow["state"] != "succeeded":
                job, _ = request(url,"jobs/"+flow["job_ids"][flow["stage"]])
                raise RuntimeError((flow,job.get("error"),job.get("result",{}).get("report")))
            assert set(flow["job_ids"]) == {"inspect","plan","convert","verify"}
            final, _ = request(url,"jobs/"+flow["job_ids"]["verify"])
            state, _ = request(url,"state")
            artifact = next(m for m in state["models"] if m["id"]==flow["artifact_model_id"])
            assert artifact["kind"] == "gguf" and final["state"]=="succeeded"
            with Path(artifact["path"]).open("rb") as f:
                assert f.read(4)==b"GGUF"
            assert before==checksum(source)
            print(json.dumps({"preset":preset,"format":artifact["kind"],"bytes":Path(artifact["path"]).stat().st_size,
                              "stages":list(flow["job_ids"]),"source_unchanged":True}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("bin-dir","quantizer","source"):
        p.add_argument("--"+flag,required=True,type=Path)
    p.add_argument("--preset",default="BF16")
    p.add_argument("--imatrix",type=Path)
    a = p.parse_args()
    run(a.bin_dir.resolve(),a.quantizer.resolve(),a.source.resolve(),a.preset,a.imatrix and a.imatrix.resolve())
