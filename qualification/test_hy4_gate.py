"""Exercise the real HTTP/controller/worker/codec path with a tiny Hy4 fixture."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = "blk.1.attn_gate.weight"
SOURCE = "model.layers.1.self_attn.linear_gate.weight"
SHARD = "model-00065-of-00148.safetensors"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data))
    return sha(path.read_bytes())


def wait(fn, proc):
    for _ in range(200):
        assert proc.poll() is None
        result = fn()
        if result:
            return result
        time.sleep(.05)
    raise AssertionError("Fixture timed out")


def exercise(root, binaries, tools, codec):
    spec = importlib.util.spec_from_file_location("hy4_plan", ROOT / "tools/hy4_plan.py")
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    hy, schema = planner.halo_modules(tools)
    q = sys.modules["gguf_common"]
    source = root / "source"
    source.mkdir()
    pins = {}
    pins["config.json"] = dump(source / "config.json",
                              dict(model_type="hy_v4_flash", text_config=dict(model_type="hy_v4_flash")))
    pins["model.safetensors.index.json"] = dump(source / "model.safetensors.index.json",
                                               dict(weight_map={SOURCE: SHARD}))
    # Known Q8_0 encoding: all +127.0 gives FP16 scale 1 and signed bytes 127.
    bf16 = struct.pack("<H", 0x42fe) * (256 * 4096)
    shard_header = json.dumps({SOURCE: dict(dtype="BF16", shape=[4096, 256],
                                            data_offsets=[0, len(bf16)])}).encode()
    shard_head = struct.pack("<Q", len(shard_header)) + shard_header
    (source / SHARD).write_bytes(shard_head + b"\0" * len(bf16))
    tensors = [
        q.TensorPlan("before.weight", (96,), 30, "text", nbytes=192),
        q.TensorPlan(TARGET, (256, 4096), 30, "text", source=SOURCE, nbytes=len(bf16)),
        q.TensorPlan("after.weight", (96,), 30, "text", nbytes=192),
        q.TensorPlan("blk.1.engram_embd.weight", (96, 3), 30, "engram_disk", nbytes=576),
    ]
    cursor = 0
    for t in tensors:
        t.offset = cursor
        cursor += q.align(t.nbytes)
    head, disk = hy.build_header(schema.ARCHITECTURE, {}, tensors)
    size = hy.tensor_end(head, tensors[-1])
    control = root / "control.gguf"
    original = bytearray(size)
    original[:len(head)] = head
    for t, payload in zip(tensors, (b"A"*192, bf16, b"B"*192, b"E"*576), strict=True):
        start = len(head) + t.offset
        original[start:start+len(payload)] = payload
    control.write_bytes(original)
    control.chmod(0o444)
    os.link(control, root / "control.partial")
    pins["control"] = sha(original)
    preflight = root / "preflight.json"
    pins["preflight"] = dump(preflight, dict(architecture=schema.ARCHITECTURE,
        source_revision=schema.REVISION, tensors=list(map(asdict, tensors)), file_bytes=size,
        data_offset=len(head), engram_file_offset=disk))
    pins["quantizer"] = sha(codec.read_bytes())
    runner = root / "fixture-engine.py"
    runner.write_text(
        "import importlib.util\n"
        f"spec=importlib.util.spec_from_file_location('hy4_gate', {str(ROOT / 'tools/hy4_gate.py')!r})\n"
        "m=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(m)\n"
        f"m.main(pins={pins!r})\n"
    )
    worker = root / "worker.json"
    dump(worker, dict(root=str(root / "worker"), engine=str(runner),
        hy4_plan=dict(host="fixture", source=str(source), halo_tools=str(tools),
                      encode=dict(control=str(control), preflight=str(preflight), quantizer=str(codec)))))
    dispatch = root / "dispatch.json"
    dump(dispatch, dict(hosts=dict(fixture=dict(local=True, config=str(worker),
         worker=str(binaries / "brewery-worker"), capabilities=dict(hy4_encode=True)))))
    dist = root / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("fixture")
    log = root / "controller.log"
    with log.open("wb") as output:
        proc = subprocess.Popen([binaries / "brewery"], stdout=output, stderr=subprocess.STDOUT,
            env={**os.environ, "BREWERY_LISTEN":"127.0.0.1:0", "BREWERY_DATA":str(root / "db"),
                 "BREWERY_WEB_DIST":str(dist), "BREWERY_DISPATCH_CONFIG":str(dispatch),
                 "BREWERY_DISPATCH":str(binaries / "brewery-dispatch"), "BREWERY_POLL_MS":"25"})
    try:
        base = wait(lambda: next((line.split()[-1] for line in log.read_text().splitlines()
                                  if line.startswith("brewery listening on ")), None), proc)

        def request(path, body=None):
            req = urllib.request.Request(base + "/api/" + path,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.load(r)

        contract = request("openapi.json")
        schemas = contract["components"]["schemas"]
        assert schemas["ConvertJob"]["properties"]["options"]["$ref"] == "#/components/schemas/ConvertJobOptions"
        assert {variant["$ref"] for variant in schemas["ConvertJobOptions"]["oneOf"]} == {
            "#/components/schemas/ConversionOptions", "#/components/schemas/Hy4EncodeOptions"}
        declared = schemas["Hy4EncodeOptions"]
        assert declared["properties"]["hy4_overrides"]["const"] == {TARGET: "Q8_0"}
        assert declared["required"] == ["hy4_overrides"] and not declared["additionalProperties"]
        assert "hy4_encode" in schemas["HostCapabilityMap"]["propertyNames"]["anyOf"][1]["enum"]
        assert contract["paths"]["/api/admin/ssh-host-key"]["post"]["operationId"] == "probeSshHostKey"
        assert request("state")["hosts"][0]["capabilities"]["hy4_encode"] is True
        model = request("models", dict(host="fixture", kind="hf", path=str(source)))["id"]
        job = dict(kind="convert", host="fixture", input_ids=[model],
                   options=dict(hy4_overrides={TARGET: "Q8_0"}))

        def run():
            jid = request("jobs", job)["id"]

            def terminal():
                detail = request("jobs/" + jid)
                return detail if detail["state"] in ("failed", "succeeded") else None

            return wait(terminal, proc), root / "worker/jobs" / jid

        bad, job_dir = run()
        assert bad["state"] == "failed", bad
        assert "Source BF16 gate does not match control" in (job_dir / "engine-stderr.log").read_text()
        assert not list((job_dir / "work").iterdir())
        (source / SHARD).write_bytes(shard_head + bf16)
        control.chmod(0o644)
        with control.open("r+b") as f:
            f.seek(len(head))
            f.write(b"Z")
        control.chmod(0o444)
        bad, job_dir = run()
        assert bad["state"] == "failed", bad
        assert "Immutable control SHA-256 mismatch" in (job_dir / "engine-stderr.log").read_text()
        assert not list((job_dir / "work").iterdir())
        control.chmod(0o644)
        control.write_bytes(original)
        control.chmod(0o444)
        good, job_dir = run()
        assert good["state"] == "succeeded", (good, (job_dir / "engine-stderr.log").read_text())
        result = good["result"]
        art = Path(result["artifact"]["path"])
        assert json.loads(Path(result["receipt_path"]).read_text()) == result
        actual = art.read_bytes()
        want_q8 = (b"\x00\x3c" + b"\x7f" * 32) * (256 * 4096 // 32)
        at = len(head) + tensors[1].offset
        end = at + len(bf16)
        delta = len(want_q8) - len(bf16)
        assert len(actual) == size + delta == result["artifact"]["bytes"]
        assert sha(actual) == result["artifact"]["sha256"]
        assert actual[len(head):at] == original[len(head):at]
        assert actual[at:at+len(want_q8)] == want_q8
        assert actual[at+len(want_q8):] == original[end:]
        assert result["verified_tensor_spans"] == 4 and result["unchanged_tensor_payloads"] == 3
        assert result["engram"]["file_offset"] == disk + delta
        assert result["engram"]["tables"][0]["sha256"] == sha(b"E"*576)
        assert control.read_bytes() == original and control.stat().st_nlink == 2
        assert (source / SHARD).read_bytes() == shard_head + bf16
        again, job_dir = run()
        assert again["state"] == "failed" and not (job_dir / "work").exists(), again
        print("PASS: public job rejects source mismatch, exact Q8_0 known answer, unchanged spans, Engram, control hardlink, and one-output limit")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    binaries, tools, codec = (Path(p).resolve(strict=True) for p in sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="brewery-hy4-gate-", dir="/arboretum") as d:
        exercise(Path(d), binaries, tools, codec)
