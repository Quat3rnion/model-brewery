#!/usr/bin/env python3
"""Summarize observed expert routing in local Lanternfish replay v1 traces."""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys


SCHEMA = "brewery-expert-usage-v1"
MIB = 1024 * 1024
MAX_TRACE_BYTES = 64 * MIB
MAX_POOL_BYTES = 512 * MIB
MAX_TRACES = 64
MAX_TOKENS = 512
MAX_SCALARS = 16_777_216
MAX_POOL_SCALARS = 4 * MAX_SCALARS
MAX_LAYERS = 1024
MAX_EXPERT_ROWS = 32_768
U64_MAX = (1 << 64) - 1
I32_MAX = (1 << 31) - 1
# Half an f32 ULP above the maximum finite f32, where conversion overflows.
F32_OVERFLOW = 3.4028235677973366e38
PHASES = ("prefill", "decode")
CAVEATS = [
    "Counts describe only these fixed-token workloads; they do not measure accuracy, "
    "causal importance, or deployment-wide expert use.",
    "Selection fractions divide by observed routing positions, not by selections. "
    "Experts with no selections are omitted; missing routing is not evidence of non-use.",
    "Dense models have no routed experts. Their positions have no routing observations.",
    "Weight summaries use captured selected router coefficients, not probabilities "
    "or individual scalar model weights. A recorded zero coefficient is a sample.",
    "Replay v1 has no separate tokenizer digest; tokenizer identity comes from the exact "
    "GGUF SHA-256 and vocabulary size. Trace hashes identify the source bytes, not authenticity.",
]


class TraceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise TraceError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def invalid_constant(_value):
    raise TraceError("non-finite JSON number")


def parse_float(value):
    result = float(value)
    require(math.isfinite(result), "non-finite JSON number")
    return result


def parse_int(value):
    require(len(value) <= 21, "integer outside replay v1 range")
    return int(value)


def read_trace(path, remaining_bytes, seen):
    # O_NONBLOCK lets us reject pipes before they can block the reader.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        meta = os.fstat(fd)
        require(stat.S_ISREG(meta.st_mode), "expected a regular file")
        limit = min(MAX_TRACE_BYTES, remaining_bytes)
        require(meta.st_size <= limit, "trace or pooled byte limit exceeded")
        # Do not reserve the full budget for small files; allow at most one
        # extra byte and reject a file that grows while being read.
        with os.fdopen(fd, "rb", closefd=False) as file:
            data = file.read(meta.st_size + 1)
        require(len(data) == meta.st_size, "trace size changed while reading")
    finally:
        os.close(fd)
    digest = hashlib.sha256(data).hexdigest()
    require(digest not in seen, "duplicate trace SHA-256")
    try:
        trace = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                           parse_constant=invalid_constant, parse_float=parse_float,
                           parse_int=parse_int)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise TraceError("invalid replay JSON") from error
    return trace, digest, len(data)


def obj(value, label):
    require(isinstance(value, dict), f"{label}: expected an object")
    return value


def array(value, label, length=None):
    require(isinstance(value, list), f"{label}: expected an array")
    require(length is None or len(value) == length, f"{label}: incorrect length")
    return value


def integer(value, label, lo=0, hi=U64_MAX):
    require(type(value) is int and lo <= value <= hi, f"{label}: invalid integer")
    return value


def text(value, label, limit=4096):
    require(isinstance(value, str) and len(value) <= limit and bool(value.strip()),
            f"{label}: invalid text")
    return value


def sha256(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value),
            f"{label}: invalid SHA-256")
    return value.lower()


def finite_f32(value):
    return type(value) in (int, float) and abs(value) < F32_OVERFLOW


def compact_json(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def model_identity(raw):
    model = obj(raw, "model")
    result = {
        "name": text(model["name"], "model.name"),
        "architecture": text(model["architecture"], "model.architecture"),
        "sha256": sha256(model["sha256"], "model.sha256"),
        "bytes": integer(model["bytes"], "model.bytes", lo=1),
        "n_layers": integer(model["n_layers"], "model.n_layers", lo=1, hi=MAX_LAYERS),
        "hidden": integer(model["hidden"], "model.hidden", lo=1),
        "vocab": integer(model["vocab"], "model.vocab", lo=1),
        "n_experts": integer(model["n_experts"], "model.n_experts"),
        "top_k": integer(model["top_k"], "model.top_k"),
    }
    top_k, n_experts = result["top_k"], result["n_experts"]
    require(top_k <= n_experts and (top_k == 0) == (n_experts == 0),
            "invalid model routing dimensions")
    return result


def runtime_identity(raw_build, raw_execution, prefill, decode):
    build = obj(raw_build, "build")
    version = text(build["lanternfish_version"], "build.lanternfish_version")
    lanternfish = sha256(build["lanternfish_sha256"], "build.lanternfish_sha256")
    system = text(build["llama_system_info"], "build.llama_system_info", limit=16_384)
    libraries = obj(build["runtime_libraries"], "build.runtime_libraries")
    require(len(libraries) <= 128, "too many runtime libraries")
    library_hashes = {}
    for path, digest in libraries.items():
        require(len(path) <= 4096 and Path(path).is_absolute(), "invalid runtime library path")
        library_hashes[path] = sha256(digest, "runtime library")

    execution = obj(raw_execution, "execution")
    backend = execution["backend"]
    layers = integer(execution["requested_gpu_layers"], "execution.requested_gpu_layers",
                     lo=-1, hi=I32_MAX)
    device = execution["device"]
    if backend == "cpu":
        require(layers == 0 and device is None, "invalid CPU execution identity")
    else:
        require(backend == "gpu" and layers != 0, "invalid execution backend or GPU layers")
        obj(device, "execution.device")
        device = {key: text(device[key], f"execution.device.{key}")
                  for key in ("name", "description", "backend")} | {"id": device["id"]}
        if device["id"] is not None:
            text(device["id"], "execution.device.id")
        require(device["backend"].strip().lower() not in ("cpu", "rpc"), "invalid GPU device")

    steps = array(execution["steps"], "execution.steps", 1 + decode)
    for i, raw in enumerate(steps):
        label = f"execution.steps[{i}]"
        step = obj(raw, label)
        expected = (0, prefill, "prefill") if i == 0 else (prefill + i - 1, 1, "decode")
        coverage = integer(step["first"], label), integer(step["n"], label), step["phase"]
        require(coverage == expected, f"{label}: invalid phase or position coverage")
        gpu = integer(step["gpu_compute_nodes"], label)
        gpu_mm = integer(step["gpu_matmul_nodes"], label)
        cpu_mm = integer(step["cpu_matmul_nodes"], label)
        require(
            gpu == gpu_mm == 0 and cpu_mm > 0 if backend == "cpu" else gpu >= gpu_mm > 0,
            f"{label}: invalid backend compute evidence",
        )

    identity = {
        "backend": backend, "requested_gpu_layers": layers,
        "lanternfish_version": version, "lanternfish_sha256": lanternfish,
    }
    # Keep potentially host-specific paths and device details out of the report.
    identity["sha256"] = hashlib.sha256(compact_json(identity | {
        "llama_system_info": system, "runtime_libraries": library_hashes, "device": device,
    }).encode("ascii")).hexdigest()
    return identity


@dataclass
class Expert:
    selections: int = 0
    weight_sum: float = 0.0


@dataclass
class Phase:
    observed: int = 0
    missing: int = 0
    selections: int = 0
    experts: dict[int, Expert] = field(default_factory=dict)

    def report(self):
        return {
            "observed_positions": self.observed,
            "missing_positions": self.missing,
            "selections": self.selections,
            "experts": [
                {
                    "expert": index, "selections": entry.selections,
                    "observed_positions": self.observed,
                    "selection_fraction": entry.selections / self.observed,
                    "weight_samples": entry.selections, "weight_sum": entry.weight_sum,
                    "weight_mean": entry.weight_sum / entry.selections,
                }
                for index, entry in sorted(self.experts.items())
            ],
        }


class Pool:
    def __init__(self):
        self.model = self.runtime = self.tokenizer = None
        self.traces = []
        self.seen = set()
        self.input_bytes = self.scalar_slots = self.expert_rows = 0
        self.layers = []

    def add(self, path):
        trace, digest, size = read_trace(path, MAX_POOL_BYTES - self.input_bytes, self.seen)
        obj(trace, "trace")
        require(integer(trace["schema_version"], "schema_version") == 1,
                "expected Lanternfish replay schema_version 1")
        model = model_identity(trace["model"])
        n_layers, width, vocab, top_k = (
            model["n_layers"], model["hidden"], model["vocab"], model["top_k"]
        )
        tokens = [array(trace[key], key) for key in ("prefill_tokens", "decode_tokens")]
        prefill, decode = map(len, tokens)
        count = prefill + decode
        require(prefill > 0 and count <= MAX_TOKENS,
                f"expected nonempty prefill and at most {MAX_TOKENS} total tokens")
        slots = count * (vocab + n_layers * (width + 2 * top_k + 3))
        require(slots <= MAX_SCALARS and self.scalar_slots + slots <= MAX_POOL_SCALARS,
                "trace or pooled scalar slot limit exceeded")
        for phase_tokens in tokens:
            for token in phase_tokens:
                integer(token, "token ID", hi=min(vocab - 1, I32_MAX))

        tokenizer = {"identity_source": "embedded_gguf", "model_sha256": model["sha256"], "vocab": vocab}
        runtime = runtime_identity(trace["build"], trace["execution"], prefill, decode)
        if self.traces:
            require(model == self.model and tokenizer == self.tokenizer,
                    "different model or tokenizer identity; cannot pool traces")
            require(runtime == self.runtime, "different runtime identity; cannot pool traces")
        else:
            self.model, self.runtime, self.tokenizer = model, runtime, tokenizer
            self.layers = [{phase: Phase() for phase in PHASES} for _ in range(n_layers)]

        for i, position in enumerate(array(trace["positions"], "positions", count)):
            phase = "prefill" if i < prefill else "decode"
            label = f"position {i}"
            obj(position, label)
            require(integer(position["position"], label) == i and position["phase"] == phase,
                    f"{label}: incorrect index or phase")
            logits = array(position["logits"], f"{label} logits", vocab)
            require(all(map(finite_f32, logits)), f"{label}: invalid logit")
            for j, layer in enumerate(array(position["layers"], f"{label} layers", n_layers)):
                self.observe(layer, self.layers[j][phase], i, j, width, top_k, model["n_experts"])
        self.input_bytes += size
        self.scalar_slots += slots
        self.seen.add(digest)
        self.traces.append({"sha256": digest, "prefill_tokens": prefill, "decode_tokens": decode})

    def observe(self, layer, phase, i, j, width, top_k, n_experts):
        label = f"position {i} layer {j}"
        obj(layer, label)
        require(integer(layer["layer"], label) == j, f"{label}: incorrect layer index")
        residual = array(layer["residual"], f"{label} residual", width)
        require(all(map(finite_f32, residual)), f"{label}: invalid residual")
        for key in ("attn_rms", "ffn_rms", "shared_gate"):
            value = layer[key]
            require(value is None or finite_f32(value), f"{label}: invalid {key}")
        ids, weights = layer["expert_ids"], layer["expert_weights"]
        if ids is None and weights is None:
            phase.missing += 1
            return
        require(top_k > 0 and ids is not None and weights is not None,
                f"{label}: routing IDs and weights must both be present on a routed model")
        array(ids, f"{label} expert_ids", top_k)
        array(weights, f"{label} expert_weights", top_k)
        for index in ids:
            integer(index, f"{label} expert ID", hi=min(n_experts - 1, I32_MAX))
        require(len(set(ids)) == top_k, f"{label}: duplicate expert ID")
        phase.observed += 1
        phase.selections += top_k
        for index, weight in zip(ids, weights):
            require(finite_f32(weight), f"{label}: invalid expert weight")
            if index not in phase.experts:
                require(self.expert_rows < MAX_EXPERT_ROWS, "pooled expert row limit exceeded")
                self.expert_rows += 1
                phase.experts[index] = Expert()
            entry = phase.experts[index]
            entry.selections += 1
            entry.weight_sum += weight

    def report(self):
        return {
            "schema": SCHEMA,
            "model": self.model,
            "runtime": self.runtime,
            "tokenizer": self.tokenizer,
            "traces": self.traces,
            "totals": {
                "trace_count": len(self.traces),
                **{key: sum(trace[key] for trace in self.traces)
                   for key in ("prefill_tokens", "decode_tokens")},
                "observed_positions": sum(phase.observed for layer in self.layers for phase in layer.values()),
                "missing_positions": sum(phase.missing for layer in self.layers for phase in layer.values()),
                "selections": sum(phase.selections for layer in self.layers for phase in layer.values()),
            },
            "layers": [
                {"layer": index, **{key: phase.report() for key, phase in layer.items()}}
                for index, layer in enumerate(self.layers)
            ],
            "caveats": CAVEATS,
        }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            f"Limits: {MAX_TRACES} files, {MAX_TRACE_BYTES // MIB} MiB and {MAX_TOKENS} tokens per trace, "
            f"{MAX_POOL_BYTES // MIB} MiB and {MAX_EXPERT_ROWS:,} expert rows per pool. "
            "Limit violations fail rather than truncate."
        ),
    )
    parser.add_argument("traces", type=Path, metavar="TRACE", nargs="+")
    args = parser.parse_args()
    if len(args.traces) > MAX_TRACES:
        parser.error(f"at most {MAX_TRACES} trace files are supported")
    pool = Pool()
    for path in args.traces:
        try:
            pool.add(path)
        except (OSError, TraceError, KeyError) as error:
            message = f"missing replay field {error}" if isinstance(error, KeyError) else str(error)
            print(f"expert_usage: {path}: {message}", file=sys.stderr)
            return 1
    print(compact_json(pool.report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
