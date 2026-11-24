#!/usr/bin/env python3
"""Check the real HGN worker and engine in an isolated, private CPU fixture."""
import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time


def checkpoint(path):
    parts = []
    names = [f"layers.{layer}.{stem}.weight" for layer in range(2)
             for stem in ("linear_attn.out_proj", "mlp.experts.down_proj")]
    start = (104 + 160 * len(names) + 63) & ~63
    weights = struct.pack("<4H", 0x4080, 0x4000, 0x3f80, 0xc040)
    checksum = struct.unpack("<II", weights)
    for i, name in enumerate(names):
        shape = [2, 2, 0, 0] if i % 2 == 0 else [1, 2, 2, 0]
        parts.append(struct.pack("<96sII4qQQII", name.encode(), 0, 2 + i % 2, *shape,
                                 start + i * 64, len(weights), checksum[0] ^ checksum[1], 0))
    header = struct.pack("<4sIQQQQ64s", b"HGN1", 2, len(parts), 104, start,
                         start + 64 * len(parts), b"h4-native-fixture")
    path.write_bytes((header + b"".join(parts)).ljust(start, b"\0")
                     + weights.ljust(64, b"\0") * len(parts))
    return start


def exercise(bins, directory, evidence):
    os.umask(0o077)
    with tempfile.TemporaryDirectory(dir=directory, prefix="h4-native-") as tmp:
        root = Path(tmp)
        worker, source = root / "worker", root / "source"
        worker.mkdir()
        source.mkdir()
        path = source / "stock.hgn"
        offset = checkpoint(path)
        stock = path.read_bytes()
        stock_sha = hashlib.sha256(stock).hexdigest()
        identity = "fixture-worker"
        key = os.urandom(32)
        (root / "key").write_bytes(key)
        # The test driver already holds the desktop exclusive build reservation.
        wrapper = root / "build-wrapper"
        wrapper.write_text('#!/bin/sh\nset -eu\ntest "$1" = --\nshift\nexec "$@"\n')
        wrapper.chmod(0o700)
        config = root / "worker.json"
        config.write_text(json.dumps(dict(
            root=str(worker), source_roots=[str(source)], engine=str(bins / "brewery-engine"),
            disk_reserve_bytes=0, hgn_edit=dict(host=identity, grant_key_file=str(root / "key"),
                                               build_admission=str(wrapper)))))
        dispatch = root / "dispatch.json"
        dispatch.write_text(json.dumps({"hosts": {identity: {
            "local": True, "worker": str(bins / "brewery-worker"), "config": str(config)}}}))
        env = dict(os.environ, BREWERY_DISPATCH_CONFIG=str(dispatch), BREWERY_WORKER_CONFIG=str(config))
        inp = dict(id="fixture-stock", host=identity, kind="hgn-v2", path=str(path))

        def call(action, identity_, job_id, job=None):
            child = subprocess.run([bins / "brewery-dispatch", action, identity_, job_id],
                                   input=json.dumps(job) if job is not None else None,
                                   env=env, capture_output=True, text=True, timeout=10)
            assert child.returncode == 0, (child.stdout, child.stderr)
            return json.loads(child.stdout)

        def run_edit(job_id, layer, pin, producer=None):
            claim = dict(job_id=job_id, host=identity, input_id=inp["id"], path=str(path),
                         kind=inp["kind"], request=dict(parent_sha256=pin, producer=producer,
                         first_layer=layer, last_layer=layer, direction=[1., 0.]))
            raw = json.dumps(claim, separators=(",", ":")).encode()
            sig = hmac.digest(key, b"brewery.hgn-edit.v1\0" + raw, "sha256")
            encode = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")
            request = dict(kind="scratch_hgn_ablation", inputs=[inp],
                           options=dict(hgn_edit_grant=f"{encode(raw)}.{encode(sig)}"))
            call("start", identity, job_id, request)
            limit = time.monotonic() + 10
            def terminal():
                while time.monotonic() < limit:
                    receipt = call("status", identity, job_id)
                    if receipt["state"] in ("failed", "succeeded", "cancelled"):
                        return receipt
                    time.sleep(.05)
                return None

            receipt = terminal()
            if receipt is not None:
                return receipt
            call("cancel", identity, job_id)
            limit = time.monotonic() + 10
            if terminal() is None:
                receipt = call("status", identity, job_id)
                raise RuntimeError(f"isolated worker did not stop: {receipt}")
            raise TimeoutError(job_id)

        first = run_edit("h4-first", 0, stock_sha)
        assert first["state"] == "succeeded", first
        one = first["result"]["report"]
        second = run_edit("h4-second", 1, one["revision_sha256"], "h4-first")
        assert second["state"] == "succeeded", second
        two = second["result"]["report"]
        assert one["parent_sha256"] == stock_sha == hashlib.sha256(path.read_bytes()).hexdigest()
        assert two["parent_sha256"] == one["revision_sha256"]
        for jid, report, layer_count in [("h4-first", one, 1), ("h4-second", two, 2)]:
            work = worker / "jobs" / jid / "work" / "brewery-private-hgn-scratch"
            model = work / "model.hgn"
            data = model.read_bytes()
            assert len(data) == len(stock)
            assert hashlib.sha256(data).hexdigest() == report["revision_sha256"]
            assert json.loads((work / "hgn-feedback.json").read_text()) == report
            for i in range(4):
                at = offset + i * 64
                expected = (b"\0" * 4 if i < 2 * layer_count else stock[at:at+4])
                assert data[at:at+4] == expected and data[at+4:at+64] == stock[at+4:at+64]
            unpublished = subprocess.run(
                [bins / "brewery-worker", "rpc"],
                input=json.dumps({"op": "describe", "input": {**inp, "path": str(model)}}),
                env=env, capture_output=True, text=True, timeout=10)
            assert "private HGN scratch is not published" in unpublished.stdout, unpublished.stdout
        bad = run_edit("h4-bad-pin", 0, "0" * 64)
        assert bad["state"] == "failed", bad
        assert not (worker / "jobs" / "h4-bad-pin" / "work").exists()
        evidence.write_text(json.dumps(dict(parent=stock_sha, first=one, second=two,
                                            private_describe_rejected=True, bad_pin_rejected=True,
                                            release_qualified=False), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    opts = parser.parse_args()
    exercise(opts.bins.resolve(), opts.scratch.resolve(), opts.evidence)
