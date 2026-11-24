#!/usr/bin/env python3
"""Check HGN screen and trace through an isolated dispatcher, worker, and engine."""

import argparse
import base64
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Replace, Sequence
from tokenizers.pre_tokenizers import Whitespace

from hgn_direction_job import digest, fixture

CONTRAST = "f42e2aac3d2c0faad8d0f0e5e2a2c2a5c8580bb1f51ece3e6a30b26eac3c24b6"


def exercise(bins, cache):
    pin = cache / f"{CONTRAST}.json"
    assert digest(pin) == CONTRAST
    with TemporaryDirectory(prefix="hgn-screen-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir(mode=0o700)
        (root / "worker").mkdir(mode=0o700)
        tokenizer = Tokenizer(WordLevel({"<unk>": 0, "a": 1, "b": 2, "c": 3}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        replacements = []
        for pair in json.loads(pin.read_bytes())["rows"]:
            for side, token in (("positive", "a"), ("negative", "b")):
                turns = pair[side]
                text = (f"<|im_start|>system\n{turns['system']}<|im_end|>\n"
                        f"<|im_start|>user\n{turns['user']}<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
                replacements.append(Replace(text, token))
        tokenizer.normalizer = Sequence(replacements)
        fixture(source, tokenizer=tokenizer, embedding_rows=[(0, 0), (1, 1), (0, 1), (0, 1)],
                marker=True, writers=True, qsa=True)

        def save(name, value):
            path = root / name
            path.write_text(json.dumps(value))
            path.chmod(0o600)
            return str(path)

        model = source / "model.hgn"
        sha = digest(model)
        assets = {name: {"path": str(source / leaf), "sha256": digest(source / leaf)}
                  for name, leaf in [("config", "config.json"), ("sidecar", "sidecar.hgn"),
                                     ("tokenizer", "tokenizer.json")]}
        pins = {name: pin["sha256"] for name, pin in assets.items()}
        python = Path(sys.executable)
        worker = save("worker.json", {
            "root": str(root / "worker"), "source_roots": [str(source)],
            "engine": str(bins / "brewery-engine"), "disk_reserve_bytes": 0,
            "engine_env": {"BREWERY_CONTRAST_CACHE": str(cache),
                           "BREWERY_HGN_TRACE_PYTHON": str(python),
                           "BREWERY_HGN_TRACE_PYTHON_SHA256": digest(python),
                           "BREWERY_HGN_TRACE_ASSETS": save("assets.json", {sha: assets}),
                           "BREWERY_CPU_THREADS": "1"},
        })
        env = {"PATH": "/usr/bin:/bin", "BREWERY_DISPATCH_CONFIG": save("dispatch.json", {
            "hosts": {"local": {"local": True, "worker": str(bins / "brewery-worker"), "config": worker}},
        })}
        inputs = [{"host": "local", "path": str(model), "id": "tiny", "kind": "hgn-v2"}]

        def call(action, job, payload=None):
            cli = subprocess.run([bins / "brewery-dispatch", action, "local", job], env=env,
                                 input=json.dumps(payload) if payload is not None else "",
                                 capture_output=True, text=True, timeout=15)
            assert cli.returncode == 0, (action, cli.stdout, cli.stderr)
            return json.loads(cli.stdout)

        def run(job, kind, options, prompt=None):
            payload = {"kind": kind, "inputs": inputs, "options": options}
            if prompt is not None:
                payload["prompt"] = prompt
            status = call("start", job, payload)
            deadline = time.monotonic() + 120
            try:
                while status["state"] in ("running", "queued") and time.monotonic() < deadline:
                    time.sleep(0.25)
                    status = call("status", job)
                assert status["state"] == "succeeded", status
            finally:
                if status["state"] in ("running", "queued"):
                    call("cancel", job)
            return status["result"]

        vector = [1] + [0] * 127
        screen = run("synthetic-screen", "screen_hgn_ablation", {
            "model_sha256": sha, "direction_id": "fixture-v2", "source_layer": 1,
            "targets": ["residual", "dense", "experts", "all"], "behavior": "response_length",
            "seed": 7, "vector": vector, "assets_sha256": pins, "held_out_pairs": 8,
        })
        assert not screen["release_qualified"]
        assert screen["model_sha256"] == sha and screen["assets_sha256"] == pins
        assert screen["contrast_sha256"] == CONTRAST
        scores = screen["scores"]
        assert scores["marker_tokens"] == [3] and scores["parent_hits"] == scores["prompts"] == 8
        assert abs(scores["control_mean"] - 0.75) < 1e-6
        assert scores["parent_mean"] > 0.99 and scores["trials"][0]["reduction"] > 0.24
        assert scores["trials"][0]["wins_over_random"] == 8
        assert scores["trials"][0]["control_shift"] == 0
        assert 0 in screen["selected"] and 2 not in screen["selected"], screen["selected"]

        trace = run("synthetic-trace", "trace", {
            "layers": [1], "positions": [512], "roles": ["l_out"], "top_k": 0,
            "retain_jobs": 4, "byte_cap": 32768, "expected_model_sha256": sha,
            "hgn_projection": {"target": "residual", "first_layer": 1, "last_layer": 1,
                               "vector": vector, "assets_sha256": pins},
        }, " ".join(["a"] * 513))
        summary = call("trace-read", "synthetic-trace", {"view": "manifest", "prompt": 0})
        assert len(summary["data"]["tokens"]) == 513
        assert summary["data"]["hgn_projection"] == {"target": "residual", "first_layer": 1, "last_layer": 1}
        row = call("trace-read", "synthetic-trace", {
            "view": "tensor", "prompt": 0, "layer": 1, "role": "l_out", "start": 512, "end": 513,
        })["data"]["rows"][0]
        values = struct.unpack("<512f", base64.b64decode(row["data_base64"]))
        assert values[::128] == (0.0,) * 4 and values[1::128] == (1.0,) * 4
        return {"screen": screen, "trace_tokens": 513, "trace_stream_components": values[::128],
                "trace_manifest_sha256": trace["prompts"][0]["manifest_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(exercise(args.bins.resolve(), args.cache.resolve()), indent=2))
