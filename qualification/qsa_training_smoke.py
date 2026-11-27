#!/usr/bin/env python3
"""Test QSA dispatch and Library publication with synthetic artifacts, not a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

TRAINER = r'''
import argparse
import fcntl
import hashlib
import json
from math import prod
import os
from pathlib import Path
import struct
import sys

# Check the actual outer reservation from both probe invocations and the
# training subprocess, without opening a GPU or using the shared host queue.
with open(os.environ["GPU_FLOCK"], "rb") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise AssertionError("QSA trainer or probe ran outside the GPU lock")

if sys.argv[1:] == ["--probe"]:
    print(json.dumps({"schema": "brewery.qsa-probe.v1", "ready": True}))
    sys.exit(0)

parser = argparse.ArgumentParser()
for arg in ("source", "train-data", "eval-data", "output-dir"):
    parser.add_argument("--" + arg, type=Path, required=True)
for arg in ("steps", "seq-len", "seed"):
    parser.add_argument("--" + arg, type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
args = parser.parse_args()
assert args.seq_len == 8

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

layers = list(range(3, 64, 4))
teacher = digest(args.source / "config.json")
score = {
    "kl": 0.05,
    "queries": len(layers) * 5,
    "capture_queries": 0,
    "capture": None,
    "capture_unavailable_reason":
        "No held-out query has more fully observed blocks than the capture budget.",
}
metrics = dict(status="passed", steps=args.steps, seq_len=args.seq_len,
               seed=args.seed, lr=args.lr, indexer_update_l2=1.0,
               train_loss=0.04, heldout_loss=0.05, before=score, after=score,
               full_attention_layers=layers, budget=2048, source_config_sha256=teacher,
               train_data_sha256=digest(args.train_data), eval_data_sha256=digest(args.eval_data))
args.output_dir.mkdir(mode=0o700)
(args.output_dir / "metrics.json").write_text(json.dumps(metrics))
header = {"__metadata__": dict(format="pt", teacher="dense_causal_qwen3_5",
    reference_revision="9eb7192828ebc6234a892d718fc775b3df648596", source_config_sha256=teacher)}
shapes = {"q_proj.weight": [512, 5120], "k_proj.weight": [128, 5120],
          "q_norm": [128], "k_norm": [128], "k_block_norm": [128]}
offset = 0
for layer in layers:
    for key, shape in shapes.items():
        end = offset + 4 * prod(shape)
        name = f"model.language_model.layers.{layer}.self_attn._qsa_indexer.{key}"
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[offset, end])
        offset = end
data = json.dumps(header).encode()
with (args.output_dir / "qsa-indexers.safetensors").open("wb") as stream:
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.truncate(stream.tell() + offset)
'''


def write_json(path, data):
    path.write_text(json.dumps(data))


def smoke(root, bins, queue):
    source = root / "sources" / "teacher"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_text("synthetic teacher identity")
    train, heldout = source.parent / "train.txt", source.parent / "heldout.txt"
    train.write_text("training example")
    heldout.write_text("held-out example")
    trainer = root / "trainer.py"
    trainer.write_text(TRAINER)
    worker_state = root / "worker-state"
    config = root / "worker.json"
    write_json(config, {
        "root": str(worker_state),
        "engine": str(bins / "brewery-engine"),
        "source_roots": [str(source.parent)],
        "engine_env": {"RESOURCE_QUEUE_DIR": str(root / "queue"),
                       "GPU_FLOCK": str(root / "gpu.lock")},
        "qsa_training": {"python": sys.executable, "trainer": str(trainer), "gpu_queue": queue},
    })
    dispatch = root / "dispatch.json"
    write_json(dispatch, {"hosts": {"fixture": {
        "local": True, "worker": str(bins / "brewery-worker"), "config": str(config),
        "capabilities": {"qsa_train": True},
    }}})
    dist = root / "web"
    dist.mkdir()
    app = subprocess.Popen(
        [bins / "brewery"],
        env={**os.environ, "BREWERY_LISTEN": "127.0.0.1:0", "BREWERY_DATA": str(root / "state"),
             "BREWERY_WEB_DIST": str(dist), "BREWERY_DISPATCH": str(bins / "brewery-dispatch"),
             "BREWERY_DISPATCH_CONFIG": str(dispatch), "BREWERY_POLL_MS": "25"},
        stdout=subprocess.PIPE, text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(app.stdout, selectors.EVENT_READ)
            assert selector.select(15), "controller did not start"
        line = app.stdout.readline().strip()
        assert line.startswith("brewery listening on "), line
        url = line.removeprefix("brewery listening on ")

        def request(path, data=None):
            req = urllib.request.Request(
                url + path, data=json.dumps(data).encode() if data is not None else None,
                headers={"Content-Type": "application/json", "Brewery-Allow-Duplicate": "1"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.load(response)

        def register(path, kind):
            return request("/api/models", {"host": "fixture", "path": str(path), "kind": kind})["id"]

        def library():
            return [model for model in request("/api/state")["models"] if model["kind"] == "qsa-indexer"]

        def run_job():
            job_id = request("/api/jobs", body)["id"]
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                result = request("/api/jobs/" + job_id)
                if result["state"] in {"succeeded", "failed", "cancelled"}:
                    return result
                time.sleep(0.1)
            raise AssertionError(f"job did not finish: {job_id}")

        teacher = register(source, "hf")
        body = {"kind": "qsa_train", "input_ids": [teacher, register(train, "qsa-corpus"),
                                                  register(heldout, "qsa-corpus")],
                "options": {"steps": 1, "seq_len": 8, "seed": 42, "lr": 0.001}}
        completed = run_job()
        if completed["state"] != "succeeded":
            for log in (worker_state / "jobs" / completed["id"]).glob("*.log"):
                print(log.name, log.read_text(errors="replace")[-4000:], file=sys.stderr)
            raise AssertionError(completed)
        result = completed["result"]
        models = library()
        assert len(models) == 1 and models[0]["id"] == result["library_model_id"]
        assert models[0]["qsa_provenance"]["teacher_model_id"] == teacher
        assert result["metrics"]["after"]["queries"] == 80
        assert result["metrics"]["after"]["capture"] is None
        with Path(result["artifact"]["path"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == result["artifact"]["sha256"]
        heldout.write_bytes(train.read_bytes())
        rejected = run_job()
        assert rejected["state"] == "failed" and "different content" in rejected["error"], rejected
        assert len(library()) == 1
        heldout.write_text("held-out example")
        weights = source / "model.safetensors"
        shard = source / "model-00001-of-00001.safetensors"
        weights.unlink()
        write_json(source / "model.safetensors.index.json", {"weight_map": {"weight": shard.name}})
        missing = run_job()
        assert missing["state"] == "failed", f"missing teacher shard was accepted: {missing['state']}"
        assert len(library()) == 1
        shard.write_text("synthetic teacher shard identity")
        assert run_job()["state"] == "succeeded", "valid teacher shard was rejected"
        assert len(library()) == 2
        print("QSA smoke passed: verified publication, separate corpora and teacher shard membership")
    finally:
        app.terminate()
        app.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    args = parser.parse_args()
    if "TMPDIR" not in os.environ:
        parser.error("Set TMPDIR=/arboretum or a private directory below it")
    temp = Path(os.environ["TMPDIR"]).resolve(strict=True)
    if not temp.is_relative_to("/arboretum"):
        parser.error("TMPDIR must be under /arboretum")
    queue = shutil.which("resource-queue")
    if queue is None:
        parser.error("resource-queue must be on PATH")
    # The worker receives an absolute queue path but uses an isolated store
    # and flock; the synthetic trainer does not use any GPU resource.
    with tempfile.TemporaryDirectory(prefix="brewery-qsa-smoke-", dir=temp) as tmp:
        smoke(Path(tmp), args.bin_dir.resolve(strict=True), os.path.abspath(queue))


if __name__ == "__main__":
    main()
