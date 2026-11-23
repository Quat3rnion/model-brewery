"""Verify a Qwen2.5-0.5B CPU projection, trace diff, and subtraction export."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from behavior_direction_acceptance import ARB, MODEL, MODEL_SHA, W0_PIN, run, save, sha


def exercise(evidence, base, commit, pin):
    def request(path, body=None, binary=False):
        payload = None if body is None else json.dumps(body).encode()
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        with urlopen(Request(base + path, payload, headers), timeout=30) as response:
            return response.read() if binary else json.load(response)

    def mcp(tool, **args):
        reply = request("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                "params": {"name": tool, "arguments": args}})
        assert not reply.get("error") and reply["result"]["isError"] is False, reply
        return reply["result"]["structuredContent"]

    def submit(tool, key, **args):
        return mcp(tool, idempotency_key=key, **args)

    def await_job(queued):
        stop = time.monotonic() + 300
        while time.monotonic() < stop:
            job = mcp("get_job", id=queued["id"], wait_ms=5000)
            if job["state"] not in {"queued", "running"}:
                assert job["state"] == "succeeded", job
                return job
        raise TimeoutError(queued["id"])

    assert sha(MODEL) == MODEL_SHA
    model = request("/api/models", {"host": "local", "kind": "gguf", "path": str(MODEL)})["id"]
    job = await_job(submit("compute_direction", "fit-slop", input_ids=[model],
                           options={"behavior": "slop", "model_sha256": MODEL_SHA,
                                    "layers": list(range(24))}))
    fit = job["result"]["fit"]
    layer = fit["selected_layer"]
    assert layer is not None and layer > 0, fit
    source = mcp("get_direction", id=job["id"], version=2, layer=layer)["direction"]
    vector = source["fit"]["layers"][0]["unit_vector"]
    assert len(vector) == 896
    save(evidence / "selected-direction.json", source)
    prompt = "The capital of France is"
    capture = {"layers": [layer], "roles": ["l_out"], "positions": [4], "top_k": 20}
    target = {"layer": layer, "role": "l_out", "positions": [4], "source_layer": layer,
              "direction_id": job["id"], "direction_version": 2}
    binding = {"model_id": model, "model_sha256": MODEL_SHA}
    projected = submit("create_direction_intervention", "project", name="Project out slop",
                       direction_rules=[dict(target, op="project_out")], **binding)
    subtracted = submit("create_direction_intervention", "subtract", name="Subtract slop",
                        direction_rules=[dict(target, op="subtract", scale=0.5)], **binding)

    url = mcp("export_control_vector", id=subtracted["id"], version=1)["url"]
    gguf = request(url, binary=True)
    assert gguf.startswith(b"GGUF\x03\x00\x00\x00")
    assert f"direction.{layer}".encode() in gguf
    assert gguf[-4 * len(vector):] == struct.pack("<896f", *(-0.5 * x for x in vector))
    try:
        request(f"/api/interventions/{projected['id']}/versions/1/gguf", binary=True)
        raise AssertionError("projection exported as a control vector")
    except HTTPError as error:
        assert error.code == 400 and "cannot express a projection" in json.load(error)["error"]

    def trace(key, saved=None):
        options = capture if saved is None else {
            **capture, "intervention_artifact": {"id": saved["id"], "version": 1},
        }
        trace_job = await_job(submit("start_trace", key, input_ids=[model], prompt=prompt, options=options))
        assert trace_job["result"]["model_sha256"] == MODEL_SHA
        assert trace_job["result"]["executable_sha256"] == sha(pin if saved else W0_PIN)
        return trace_job["id"]

    a, b = trace("baseline"), trace("projected", projected)

    def component(trace_id, edits):
        manifest = mcp("read_trace", id=trace_id, view="manifest")
        assert manifest["intervention_count"] == edits
        view = mcp("read_trace", id=trace_id, view="tensor", layer=layer, role="l_out", start=4, end=5)
        row, = view["rows"]
        data = base64.b64decode(row["data_base64"], validate=True)
        assert row["position"] == 4 and hashlib.sha256(data).hexdigest() == row["sha256"]
        activation = struct.unpack("<896f", data)
        return math.fsum(x * d for x, d in zip(activation, vector)) / math.hypot(*vector)

    before, after = component(a, 0), component(b, 1)
    assert abs(before) > 1e-3 and abs(after) < 1e-6, (before, after)
    diff = mcp("diff_traces", a=a, b=b)
    logit_shift = max(row["linf"] or 0 for row in diff["model_logits"])
    assert logit_shift > 1e-3, diff["model_logits"]
    save(evidence / "trace-diff.json", diff)
    report = {
        "release_qualified": False, "brewery_commit": commit,
        "model": MODEL.name, "model_sha256": MODEL_SHA, "lanternfish_sha256": sha(pin),
        "baseline_lanternfish_sha256": sha(W0_PIN),
        "behavior": "slop", "direction_id": job["id"], "direction_version": 2,
        "layer": layer, "role": "l_out", "position": 4, "prompt": prompt,
        "baseline_trace": a, "projected_trace": b, "component_before": before,
        "component_after": after, "max_abs_shared_top20_logit_shift": logit_shift,
        "subtraction_gguf_sha256": hashlib.sha256(gguf).hexdigest(),
    }
    save(evidence / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("evidence", "bins", "cache", "pin", "libs"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    scratch = (ARB / "xa-scratch").resolve()
    for key, value in vars(args).items():
        path = value.resolve()
        if ARB not in path.parents and scratch not in path.parents:
            parser.error(f"--{key} must be inside /arboretum")
        setattr(args, key, path)
    args.evidence.mkdir(mode=0o700, parents=True, exist_ok=False)
    run(args, pin=args.pin, libs=args.libs, exercise_stack=exercise)
