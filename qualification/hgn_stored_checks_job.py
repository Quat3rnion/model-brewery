#!/usr/bin/env python3
"""Run the signed HGN checks through an isolated dispatcher, worker and engine."""
import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hgn_trace_fixture import fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exercise(bins, root, evidence):
    os.umask(0o077)
    with tempfile.TemporaryDirectory(dir=root, prefix="h5-stored-") as tmp:
        home = Path(tmp)
        worker, source = home / "worker", home / "source"
        worker.mkdir()
        source.mkdir()
        fixture(source, embedding_rows=[(1, 1)] * 4, editable=True)
        model = source / "model.hgn"
        pin = sha(model)
        key = os.urandom(32)
        key_file = home / "key"
        key_file.write_bytes(key)
        registry = home / "assets.json"
        registry.write_text(json.dumps({pin: {
            kind: {"path": str(source / file), "sha256": sha(source / file)}
            for kind, file in (("sidecar", "sidecar.hgn"), ("tokenizer", "tokenizer.json"),
                               ("config", "config.json"))
        }}))
        # The driver runs under an exclusive reservation. Do not nest one.
        wrapper = home / "build-wrapper"
        wrapper.write_text('#!/bin/sh\nset -eu\ntest "$1" = --\nshift\nexec "$@"\n')
        wrapper.chmod(0o700)
        host = "h5-fixture-worker"
        config = home / "worker.json"
        config.write_text(json.dumps(dict(
            root=str(worker), source_roots=[str(source)], engine=str(bins / "brewery-engine"),
            disk_reserve_bytes=0,
            hgn_edit=dict(host=host, grant_key_file=str(key_file), build_admission=str(wrapper)),
            engine_env=dict(BREWERY_HGN_TRACE_PYTHON=sys.executable,
                            BREWERY_HGN_TRACE_PYTHON_SHA256=sha(Path(sys.executable)),
                            BREWERY_HGN_TRACE_ASSETS=str(registry), BREWERY_CPU_THREADS="1"),
        )))
        dispatch = home / "dispatch.json"
        dispatch.write_text(json.dumps({"hosts": {host: {
            "local": True, "worker": str(bins / "brewery-worker"), "config": str(config)}}}))
        env = dict(os.environ, BREWERY_DISPATCH_CONFIG=str(dispatch))
        inp = dict(id="fixture-stock", host=host, kind="hgn-v2", path=str(model))

        def call(action, jid, body=None):
            proc = subprocess.run(
                [bins / "brewery-dispatch", action, host, jid], env=env, text=True,
                input=json.dumps(body) if body is not None else None, capture_output=True, timeout=15,
            )
            reply = json.loads(proc.stdout)
            assert proc.returncode == 0 or (
                proc.returncode == 2 and reply["state"] == "failed"
            ), (proc.stdout, proc.stderr)
            return reply

        def run(jid, kind, grant_key, domain, req, *, corrupt=False):
            claim = dict(job_id=jid, host=host, input_id=inp["id"], path=str(model),
                         kind=inp["kind"], request=req)
            raw = json.dumps(claim, separators=(",", ":")).encode()
            sig = hmac.digest(key, domain + b"\0" + raw, "sha256")
            encode = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")
            if corrupt:
                sig = bytes(32)
            reply = call("start", jid, dict(kind=kind, inputs=[inp],
                                           options={grant_key: f"{encode(raw)}.{encode(sig)}"}))
            deadline = time.monotonic() + 30
            while reply["state"] not in ("failed", "succeeded", "cancelled") and time.monotonic() < deadline:
                time.sleep(.05)
                reply = call("status", jid)
            if reply["state"] not in ("failed", "succeeded", "cancelled"):
                call("cancel", jid)
                raise TimeoutError(jid)
            return reply

        edit = run("h5-edit", "scratch_hgn_ablation", "hgn_edit_grant", b"brewery.hgn-edit.v1",
                   dict(parent_sha256=pin, producer=None, first_layer=0, last_layer=0,
                        direction=[1.] + [0.] * 127))
        assert edit["state"] == "succeeded", (
            edit, (worker / "jobs/h5-edit/engine-stderr.log").read_text()
        )
        output = worker / "jobs/h5-edit/work/brewery-private-hgn-scratch"
        child = output / "model.hgn"
        feedback = output / "hgn-feedback.json"
        receipt = edit["result"]["report"]
        child_sha = receipt["revision_sha256"]
        req = dict(parent_sha256=pin, revision_sha256=child_sha, root_sha256=pin, parent_producer=None,
                   child_producer="h5-edit", first_layer=0, last_layer=0)

        def check(jid, **kwargs):
            return run(jid, "scratch_hgn_checks", "hgn_check_grant", b"brewery.hgn-checks.v1",
                       req, **kwargs)

        good = check("h5-check")
        log = worker / "jobs/h5-check/engine-stderr.log"
        assert good["state"] == "succeeded", (good, log.read_text() if log.exists() else "")
        result = good["result"]
        assert result["revision_sha256"] == child_sha == sha(child)
        assert result["parent_sha256"] == pin == sha(model)
        assert len(result["tensors"]) == 3
        assert all(row["passed"] for row in result["tensors"].values()), result["tensors"]
        assert all(row["kl_parent_child"] > 0.0 and row["max_abs_logit_delta"] > 0.0
                   for row in result["prompts"]), result["prompts"]
        assert len(result["prompts"]) == 2 and str(home) not in repr(result)
        assert not {"parent_generation", "child_generation"} & result.keys()
        bad_grant = check("bad-grant", corrupt=True)
        assert bad_grant["state"] == "failed" and "invalid controller HGN grant" in bad_grant["error"]

        content = child.read_bytes()
        child.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        bad_model = check("bad-model")
        assert bad_model["state"] == "failed" and str(home) not in repr(bad_model)
        assert "stored HGN SHA-256 mismatch" in (
            worker / "jobs/bad-model/engine-stderr.log").read_text()
        child.write_bytes(content)
        broken = json.loads(feedback.read_bytes())
        broken["tensors"].pop(next(iter(broken["tensors"])))
        feedback.write_text(json.dumps(broken))
        bad_feedback = check("bad-feedback")
        assert bad_feedback["state"] == "failed" and str(home) not in repr(bad_feedback)
        assert "HGN feedback receipt changed" in bad_feedback["error"]
        evidence.write_text(json.dumps(dict(result=result, bad_model_rejected=True,
                                            bad_feedback_rejected=True, bad_grant_rejected=True,
                                            release_qualified=False), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("bins", "scratch", "evidence"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    exercise(args.bins.resolve(), args.scratch.resolve(), args.evidence.resolve())
