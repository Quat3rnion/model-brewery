#!/usr/bin/env python3
"""Controller test double at the dispatcher process boundary. No engine or SSH."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).parent
if sys.argv[1] == "suggest-paths":
    action, host = sys.argv[1:]
    payload = json.load(sys.stdin)
    assert set(payload) == {"prefix"}
    assert host in ("alpha", "beta")
    prefix = payload["prefix"]
    if prefix in ("/timeout/", "/disconnect/"):
        transport = subprocess.Popen(["sleep", "20"])
        (root / f"suggest-{prefix.strip('/')}-pid").write_text(f"{os.getpid()} {transport.pid}")
        time.sleep(20)
    if prefix == "/unreachable/":
        print("private transport diagnostics", file=sys.stderr)
        print('{"error":"private remote detail"}')
        sys.exit(1)
    if prefix == "/malformed/":
        print(json.dumps({"items": [{"path": "/malformed/../private", "kind": "file"}],
                          "truncated": False}))
        sys.exit(0)
    if prefix == "/oversized/":
        print(" " * (64 * 1024 + 1))
        sys.exit(0)
    if not prefix:
        print(json.dumps({"items": [{"path": "/model roots", "kind": "directory"}],
                          "truncated": False}))
        sys.exit(0)
    print(json.dumps({"items": [{"path": prefix + "demo model.gguf", "kind": "file"},
                                {"path": prefix + "assets", "kind": "directory"}], "truncated": True}))
    sys.exit(0)

action, host, job_id = sys.argv[1:]
job = root / f"{job_id}.json"
partial = {"artifact": "/not-published/artifact"}


def state(value, progress, **kwargs):
    return {"state": value, "progress": progress, **kwargs}


def write(value):
    tmp = job.with_suffix(".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(job)
    return value


def out(value):
    print(json.dumps(value))


def hf_manifest(files):
    data = "".join(f"{f['path']}\0{f['bytes']}\0{f['sha256']}\n"
                   for f in sorted(files, key=lambda f: f["path"]))
    digest = hashlib.sha256(data.encode()).hexdigest()
    return dict(files=files, sha256=digest, bundle_sha256=digest,
                bytes=sum(f["bytes"] for f in files))


if (root / "offline").exists():
    out(state("missing", "unrelated remote message", error="test transport failure"))
    sys.exit(255)

prior = json.loads(job.read_text()) if job.exists() else None
if action == "trace-read":
    query = json.load(sys.stdin)
    assert prior and prior["state"] == "succeeded"
    if (root / "trace-pruned").exists():
        out({"status": "missing"})
    elif (root / "trace-interventions").exists():
        import base64
        import struct
        assert query["view"] == "tensor" and query["prompt"] == 0
        rows = []
        for pos, val in enumerate([1.0, -2.0]):
            if query["start"] <= pos < query["end"]:
                data = struct.pack("<f", val)
                rows.append(dict(position=pos, shape=[1], sha256=hashlib.sha256(data).hexdigest(),
                                 data_base64=base64.b64encode(data).decode(), node="l_out-0"))
        if (root / "trace-corrupt").exists() and rows:
            rows[0]["data_base64"] = "AAAAAA=="
        out(dict(status="ok", data=dict(dtype="f32_le", rows=rows, missing_positions=[],
                                        layer=query["layer"], role=query["role"], prompt=0)))
    elif query["view"] == "manifest":
        out({"status": "ok", "data": {"schema_version": 1}})
    elif query["view"] == "diff_input":
        options = prior["result"]["envelope"]["options"]
        spec = {k: options[k] for k in ("positions", "roles", "layers", "top_k", "byte_cap")}
        edited = bool(options.get("interventions"))
        mismatch = root / "trace-token-mismatch"
        token = 12 if mismatch.exists() and mismatch.read_text() == job_id else 11
        top = [{"token_id": i, "logit": logit} for i, logit in
               ([(1, 3.), (2, 1.), (8, 0.)] if edited else [(1, 2.), (2, 1.), (7, 0.)])]
        out({"status": "ok", "data": {
            "spec": spec, "tokens": [{"position": 0, "token_id": token}], "missing": [],
            "tensors": [{"layer": 0, "role": "l_out", "position": 0, "shape": [1],
                         "data_base64": "AAAAAA==" if edited else "AACAPw=="}],
            "logit_lens": [{"position": 0, "layer": 0, "top": top[:spec["top_k"]]}],
            "model_logits": [{"position": 0, "top": top[:spec["top_k"]]}],
        }})
    elif query["end"] is None:
        out({"status": "too_large"})
    else:
        assert query == {"view": "tensor", "prompt": 0, "layer": 0,
                         "role": "l_out", "start": 0, "end": 1}
        out({"status": "ok", "data": {"rows": [{"data_base64": "AACAPw=="}]}})
elif action == "status":
    sampling = root / f"{job_id}.sampling"
    if sampling.exists() and prior and prior["state"] == "queued":
        polls = root / f"{job_id}.polls"
        number = int(polls.read_text()) + 1 if polls.exists() else 1
        polls.write_text(str(number))
        if number == 1:
            out(state("missing", "not yet visible"))
        elif number == 2:
            out(state("succeeded", "premature status", result=partial))
        elif number == 3:
            out(state("queued", "Copying: 120 / 360 GB", result=partial, progress_detail={
                "phase": "copy_input", "unit": "bytes",
                "completed": 120000000000, "total": 360000000000,
                "current_file": "model-001.safetensors", "files_completed": 1, "files_total": 3,
                "input_index": 1, "input_count": 1, "updated_at_ms": int(time.time() * 1000),
            }))
        else:
            (root / f"{job_id}.status-pid").write_text(str(os.getpid()))
            time.sleep(20)
    elif (root / f"{job_id}.forget").exists() or prior is None:
        out(state("missing", "not launched"))
    elif prior["state"] == "running" and (root / "progress-reply.json").exists():
        print((root / "progress-reply.json").read_text())
    elif prior["state"] == "running" and not (root / "hold").exists():
        if (root / "fail").exists():
            out(write(state("failed", "failed validation", result=partial, error="validation failed")))
        else:
            payload = prior["payload"]
            if (root / "changed-weight-source").exists() and payload.get("kind") == "weight_propose":
                payload["source_sha256"] = "d" * 64
            out(write(state("succeeded", "published", result=payload)))
    else:
        out({k: v for k, v in prior.items() if k != "payload"})
elif action == "cancel":
    if prior is not None and prior["state"] in ("succeeded", "failed"):
        out(prior)
    elif (root / "hf-races-cancel").exists() and prior and prior.get("payload", {}).get("kind") == "hf_download":
        out(write(state("succeeded", "published before worker saw cancellation", result=prior["payload"])))
    elif (root / "weight-apply-races-cancel").exists() and prior and prior.get("payload", {}).get("kind") == "weight_apply":
        out(write(state("succeeded", "published", result=prior["payload"])))
    else:
        out(write(state("cancelled", "confirmed cancelled", result=partial)))
else:
    with (root / "launches").open("a") as f:
        print(job_id, action, host, file=f)
    if prior is not None:
        out({k: v for k, v in prior.items() if k != "payload"})
        sys.exit(0)
    envelope = json.load(sys.stdin)
    if (root / "failstart").exists():
        (root / "failstart").unlink()
        print("test interrupted launch", file=sys.stderr)
        sys.exit(1)
    result = {"action": action, "host": host, "envelope": envelope}
    if envelope["kind"] == "trace" and (root / "trace-interventions").exists():
        result.update(kind="trace", schema_version=1, model_id=envelope["inputs"][0]["id"],
                      model_sha256="a" * 64, prompts=[{}])
    if action == "start" and envelope["kind"] == "hf_download":
        assert envelope["inputs"] == [] and "prompt" not in envelope
        options = envelope["options"]
        filename = options.get("filename")
        kind = "hf" if filename is None else "gguf" if filename.endswith(".gguf") else "safetensors"
        result = dict(kind="hf_download", repo_id=options["repo_id"],
                      revision=options.get("revision", "main"),
                      commit="0123456789abcdef0123456789abcdef01234567",
                      verified=True, artifact=dict(kind=kind,
                          path=f"/worker/jobs/{job_id}/work/download" + (f"/{filename}" if filename else ""),
                          sha256="a" * 64, bytes=17))
        if filename is not None:
            result["filename"] = filename
        fixture = root / "hf-receipt-override.json"
        if fixture.exists():
            overrides = json.loads(fixture.read_text())
            result["artifact"].update(overrides.pop("artifact", {}))
            result.update(overrides)
    if action == "start" and envelope["kind"].startswith("weight_") and (root / "weight-fixture").exists():
        kind = envelope["kind"]
        result["kind"] = kind
        source_format = envelope["inputs"][0]["kind"]
        if source_format in ("hgn", "safetensors", "hf"):
            result["source_format"] = source_format
        source_sha = "a" * 64
        if source_format == "hf":
            files = [
                {"path": "config.json", "bytes": 3, "sha256": "a" * 64},
                {"path": "model-00001-of-00001.safetensors", "bytes": 12, "sha256": "b" * 64},
                {"path": "model.safetensors.index.json", "bytes": 9, "sha256": "c" * 64},
            ]
            source_manifest = hf_manifest(files)
            source_sha = source_manifest["sha256"]
            result["inputs"] = [dict(envelope["inputs"][0], **source_manifest)]
        if source_format == "safetensors":
            source_manifest = hf_manifest([
                {"path": Path(envelope["inputs"][0]["path"]).name, "bytes": 18, "sha256": source_sha}
            ])
            source_manifest["sha256"] = source_sha
            result["inputs"] = [dict(envelope["inputs"][0], **source_manifest)]
        options = envelope["options"]
        packed = options.get("packed") is True and (root / "packed-weight-receipts").exists()
        if packed:
            result["packed"] = True
        dtype = "Q8G64" if source_format == "hgn" else (
            "Q8_0" if (root / "weight-q8").exists() else "Q4_0")
        if (root / "wrong-weight-codec").exists():
            dtype = "Q4_0" if source_format == "hgn" else "Q8G64"
        if kind == "weight_inspect" and options.get("mode") == "inventory":
            result.update(source_sha256=source_sha, report={"mode":"inventory","source_sha256":source_sha,
                "offset":0,"total":1,"next_offset":None,
                "tensors":[{"name":"blk.0.weight","shape":[8,8],"dtype":"F32","elements":64}]})
        if kind == "weight_inspect" and options.get("mode") == "tile":
            result.update(source_sha256=source_sha, report={"mode":"tile","source_sha256":source_sha,
                "tensor":options["tensor"],"origin":options["origin"],"tile_shape":options["shape"],
                "shape":[8,8],"dtype":"F32","values":[[1.0]]})
        if kind in ("weight_propose", "weight_tune"):
            edits = options.get("edits", [{}])
            if packed or source_format in ("safetensors", "hf"):
                edits = [dict(edit, dtype=dtype, before=1.0, after=1.5) for edit in edits]
                if not packed:
                    for edit in edits:
                        edit["dtype"] = "F32"
            result.update(
                source_sha256=source_sha,
                report={"proposal": {"schema": "brewery.weight-edit-proposal.v1",
                    "source_sha256": source_sha, "edits": edits}},
                artifact={"kind": "weight-edit-proposal", "path": f"/worker/jobs/{job_id}/work/weight-proposal.json",
                          "sha256": "b" * 64, "bytes": 12},
            )
        if kind == "weight_apply":
            expected = {"source_sha256": source_sha, "proposal_sha256": "b" * 64}
            if packed:
                expected["packed"] = True
            if packed or source_format in ("safetensors", "hf"):
                result["proposal_sha256"] = ("d" if (root / "wrong-approved-weight-digest").exists() else "b") * 64
            assert options == expected
            assert envelope["inputs"][1]["kind"] == "weight-edit-proposal"
            verified = not (root / "bad-weight-verify").exists()
            artifact_kind = "gguf" if (root / "wrong-weight-output-kind").exists() else source_format
            suffix = "gguf" if (root / "wrong-weight-output-leaf").exists() else source_format
            if (root / "wrong-weight-source-format").exists():
                result["source_format"] = "gguf"
            output = {"sha256": "c" * 64, "bytes": 18}
            if source_format == "hf":
                revised = [dict(f) for f in files]
                revised[1]["sha256"] = "e" * 64
                if (root / "missing-hf-member").exists():
                    revised.pop(0)  # A plausible, digest-consistent but incomplete checkpoint.
                output = hf_manifest(revised)
                if (root / "wrong-hf-manifest-hash").exists():
                    output["sha256"] = output["bundle_sha256"] = "f" * 64
            result.update(verified=verified, source_sha256=source_sha,
                report={"schema": "brewery.weight-edit-result.v1", "source_sha256": source_sha,
                        "output_sha256": output["sha256"], "edit_count": 1, "verified": verified},
                artifact=dict(output, kind=artifact_kind, path=f"/worker/jobs/{job_id}/work/edited.{suffix}"))
    if action == "start" and (root / "conversion-fixture").exists():
        kind = envelope["kind"]
        result.update(kind=kind, source_sha256="a" * 64)
        gguf = envelope["options"].get("output_format") == "gguf"
        if gguf:
            result.update(output_format="gguf", gguf_type=envelope["options"]["gguf_type"], release_qualified=False)
            if envelope["options"].get("encoder_family") == "rocmfpx":
                result["backend_receipt"] = {
                    "encoder_family": "rocmfpx",
                    "encoder_revision": envelope["options"]["encoder_revision"],
                    "executable_sha256": ("d" if kind == "plan" and (root / "changed-rocmfpx-tool").exists() else "c") * 64,
                }
        if recipe := envelope["options"].get("profile_recipe"):
            result["custom_profile"] = {"revision": recipe["revision"], "release_qualified": False}
        if kind in ("inspect", "plan"):
            result["disk_preflight"] = {"fits": not (root / f"no-fit-{kind}").exists()}
        if kind == "convert":
            result["report"] = {"published": True}
            result["artifact"] = {"kind": "hgn", "bytes": 9, "sha256": "b" * 64,
                                  "path": f"/worker/jobs/{job_id}/work/converted"}
            if gguf:
                result["artifact"].update(kind="gguf", path=f"/worker/jobs/{job_id}/work/converted.gguf")
                result["report"]["passed"] = True
                result["file_verification"] = {"status":"passed"}
            if (root / "bad-path").exists():
                result["artifact"]["path"] = "é" + result["artifact"]["path"]
        if kind == "verify":
            result["inputs"] = list(envelope["inputs"])
            result["inputs"][1].update(bytes=9, sha256="b" * 64)
            result["file_verification"] = {"status": "passed"}
            result["report"] = {"passed": not (root / "bad-verify").exists(),
                                "manifest_schema": 2, "provenance_attested": True,
                                "source_tensors": 1, "output_tensors": 1, "payload_bytes": 9,
                                "errors": {}}
    if action == "start":
        if (root / "blocking_start").exists():
            write(state("queued", "resolving inputs"))
            time.sleep(20)
        out(write(state("running", "worker progress", result=partial, payload=result)))
    elif action in ("transfer", "preview_transfer"):
        if (root / "sample_launch").exists():
            (root / f"{job_id}.sampling").write_text(str(os.getpid()))
            write(state("queued", "resolving inputs"))
            for _ in range(3000):
                if (root / f"{job_id}.release").exists():
                    break
                time.sleep(.01)
        result["inputs"] = [
            dict(item, source_host=item["host"], host=host, path=f"/verified/job-cache/{job_id}/{i}", bytes=7)
            for i, item in enumerate(envelope["inputs"])
        ]
        result["missing_bytes"] = len(result["inputs"]) * 7
        result["available_bytes"] = 4096
        out(write(state("succeeded", "staging complete", result=result)))
    elif action == "cleanup":
        result["removed"] = []
        out(write(state("succeeded", "cache cleaned", result=result)))
    else:
        raise ValueError(action)
