#!/usr/bin/env python3
"""Offline expert reorder/prune for canonical single-file Qwen2/3 MoE GGUF v3.

Maps list old IDs in new order for every layer. Retention must be uniform and
at least the native top-k. Packed weights are never decoded or requantized.
Byte verification is not a quality check; exact router ties can depend on
expert order, and removing experts changes the model.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile

if __package__:
    from .cross_container.ggml_in_hgn import Cursor, KV_WIDTHS, TYPES, align, need
else:
    sys.dont_write_bytecode = True  # Read-only CLI commands must not create an import cache.
    from cross_container.ggml_in_hgn import Cursor, KV_WIDTHS, TYPES, align, need


PLAN_SCHEMA = "brewery.expert-map.v1"
MAX_PLAN = 8 << 20
MAX_HEADER = 32 << 20
MAX_TENSORS = 16_384
MAX_LAYERS = 512
MAX_EXPERTS = 1024
MAX_FILE = (1 << 63) - 1
CHUNK = 4 << 20
SHA256 = re.compile("[0-9a-f]{64}")

ROUTED = ("ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
SHARED = ("ffn_gate_inp_shexp", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")


class HeaderCursor(Cursor):
    """Use a larger bound than the HGN shelf so standard Qwen vocabularies fit."""

    def __init__(self, data):
        self.data = memoryview(data)
        self.pos = 0

    def string(self, limit=MAX_HEADER):
        length, = self.unpack("Q")
        need(length <= limit, "oversized GGUF string")
        return bytes(self.take(length)).decode("utf-8")

    def kv(self, kind):
        if kind != 9:
            return super().kv(kind)
        subtype, count = self.unpack("IQ")
        need(count <= 2_000_000, "oversized GGUF metadata array")
        if subtype == 8:
            for _ in range(count):
                self.string()
        else:
            need(subtype in KV_WIDTHS, "unsupported GGUF metadata array type")
            self.take(count * KV_WIDTHS[subtype])


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]  # GGML order; dim 0 is contiguous within a row.
    kind: int
    relative: int
    size: int


@dataclass(frozen=True)
class GGUF:
    fields: dict
    tensors: dict[str, Tensor]
    alignment: int
    header_end: int
    base: int
    size: int


def regular_file(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    file = os.fdopen(fd, "rb")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        file.close()
        raise ValueError(f"expected a regular file: {path}")
    return file


def identity(file):
    info = os.fstat(file.fileno())
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def no_duplicates(pairs):
    mapping = {}
    for key, value in pairs:
        need(key not in mapping, f"duplicate plan key: {key}")
        mapping[key] = value
    return mapping


def reject_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def read_plan(path, reviewed_sha):
    need(SHA256.fullmatch(reviewed_sha), "expected a lowercase reviewed plan SHA-256")
    with regular_file(path) as file:
        before = identity(file)
        need(0 < before[2] <= MAX_PLAN, "invalid plan size")
        raw = file.read(MAX_PLAN + 1)
        need(len(raw) == before[2] and identity(file) == before, "plan changed during read")
    need(hashlib.sha256(raw).hexdigest() == reviewed_sha, "reviewed plan SHA-256 mismatch")
    plan = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates, parse_constant=reject_json_constant)
    required = {"schema", "source_sha256", "layers"}
    need(type(plan) is dict and required <= plan.keys() <= required | {"top_k", "review"}
         and plan["schema"] == PLAN_SCHEMA, "unsupported plan schema or fields")
    source_sha = plan["source_sha256"]
    need(type(source_sha) is str and SHA256.fullmatch(source_sha), "invalid plan source SHA-256")
    need(type(plan["layers"]) is dict, "invalid plan layer map")
    need(type(plan.get("review", {})) is dict, "expected review object")
    return plan


def require_top_k(plan, native_top_k):
    for label, container in (("plan", plan), ("review", plan.get("review", {}))):
        if "top_k" in container:
            value = container["top_k"]
            need(type(value) is int and value > 0, f"{label}.top_k must be a positive integer")
            need(value == native_top_k, f"{label}.top_k disagrees with GGUF expert_used_count")


def read_gguf(file):
    file.seek(0)
    size = os.fstat(file.fileno()).st_size
    need(24 <= size <= MAX_FILE, "invalid GGUF file size")
    cursor = HeaderCursor(file.read(min(size, MAX_HEADER)))
    magic, version, count, num_fields = cursor.unpack("4sIQQ")
    need(magic == b"GGUF" and version == 3, "expected little-endian GGUF v3")
    need(0 < count <= MAX_TENSORS and num_fields <= MAX_TENSORS, "invalid GGUF header counts")
    fields = {}
    alignment = 32
    for _ in range(num_fields):
        key = cursor.string(1024)
        need(key.isascii() and key.isprintable() and key not in fields, "invalid or duplicate GGUF key")
        need(not key.startswith("split."), "sharded GGUF is unsupported")
        kind, = cursor.unpack("I")
        start = cursor.pos
        cursor.kv(kind)
        fields[key] = (kind, cursor.data[start:cursor.pos])
        if key == "general.alignment":
            need(kind == 4, "GGUF alignment must be uint32")
            alignment, = struct.unpack("<I", fields[key][1])
            need(0 < alignment <= 4096 and alignment & (alignment - 1) == 0, "invalid GGUF alignment")
    tensors = {}
    expected_offset = 0
    for _ in range(count):
        name = cursor.string(63)
        need(name.isascii() and name.isprintable() and name not in tensors, "invalid or duplicate GGUF tensor name")
        rank, = cursor.unpack("I")
        need(1 <= rank <= 4, f"invalid GGUF rank: {name}")
        dims = cursor.unpack("Q" * rank)
        kind, relative = cursor.unpack("IQ")
        need(kind in TYPES, f"unsupported GGML type {kind}: {name}")
        block, width = TYPES[kind][:2]
        elements = math.prod(dims)
        need(0 < elements <= MAX_FILE and dims[0] % block == 0, f"invalid packed dimensions: {name}")
        num_bytes = elements // block * width
        need(relative == expected_offset, f"invalid GGUF tensor offset: {name}")
        tensors[name] = Tensor(name, dims, kind, relative, num_bytes)
        expected_offset += align(num_bytes, alignment)
        need(expected_offset <= MAX_FILE, "GGUF tensor size overflow")
    base = align(cursor.pos, alignment)
    need(base <= MAX_HEADER, "oversized GGUF header")
    # A file may omit the last tensor's trailing alignment padding.
    last = next(reversed(tensors.values()))
    end = base + last.relative + last.size
    need(end <= size <= base + expected_offset, "GGUF tensor data truncated or has extra bytes")
    if size != end:
        file.seek(end)
        need(file.read(size - end) == bytes(size - end), "invalid GGUF trailing padding")
    return GGUF(fields, tensors, alignment, cursor.pos, base, size)


def string_field(fields, key):
    kind, data = fields.get(key, (None, None))
    need(kind == 8, f"expected string metadata: {key}")
    return bytes(data[8:]).decode("utf-8")


def uint32_field(fields, key, minimum, maximum):
    kind, data = fields.get(key, (None, None))
    need(kind == 4, f"expected scalar uint32 metadata: {key}")
    value, = struct.unpack("<I", data)
    need(minimum <= value <= maximum, f"metadata value out of range: {key}")
    return value


def model_layout(model):
    fields, tensors = model.fields, model.tensors
    arch = string_field(fields, "general.architecture")
    need(arch in {"qwen2moe", "qwen3moe"}, f"unsupported GGUF MoE architecture: {arch}")
    if "general.type" in fields:
        need(string_field(fields, "general.type") == "model", "expected full GGUF model")
    layers = uint32_field(fields, f"{arch}.block_count", 1, MAX_LAYERS)
    emb = uint32_field(fields, f"{arch}.embedding_length", 1, MAX_FILE)
    ff = uint32_field(fields, f"{arch}.expert_feed_forward_length", 1, MAX_FILE)
    experts = uint32_field(fields, f"{arch}.expert_count", 1, MAX_EXPERTS)
    top_k = uint32_field(fields, f"{arch}.expert_used_count", 1, experts)
    expert_keys = {f"{arch}.{suffix}" for suffix in (
        "expert_count", "expert_used_count", "expert_feed_forward_length",
        "expert_weights_scale", "expert_weights_norm", "expert_gating_func",
    )}
    if arch == "qwen2moe":
        shared_key = f"{arch}.expert_shared_feed_forward_length"
        shared = uint32_field(fields, shared_key, 1, MAX_FILE)
        expert_keys.add(shared_key)
    for key in fields:
        if key in {f"{arch}.expert_group_count", f"{arch}.expert_group_used_count",
                   f"{arch}.nextn_predict_layers"}:
            uint32_field(fields, key, 0, 0)
        elif re.search(r"(^|[._])(?:experts?|router|routing)([._]|$)", key.casefold()):
            need(key in expert_keys, f"unsupported expert/router metadata: {key}")

    # llama.cpp models/qwen{2,3}moe.cpp: only these four tensors depend on n_expert.
    # An exact name allowlist rejects biases/scales, fused expert weights and other companions.
    required = {"token_embd.weight", "output_norm.weight"}
    optional = {"output.weight"} if arch == "qwen3moe" else set()
    if arch == "qwen2moe":
        required.add("output.weight")
    expert_layers = {}
    for layer in range(layers):
        prefix = f"blk.{layer}."
        for role in ("attn_norm", "attn_output", "ffn_norm"):
            required.add(prefix + role + ".weight")
        if arch == "qwen3moe":
            required.update(prefix + role + ".weight" for role in ("attn_q_norm", "attn_k_norm"))
        # Fused or separate attention tensors do not participate in routing.
        fused = prefix + "attn_qkv.weight"
        separate = {prefix + f"attn_{role}.weight" for role in ("q", "k", "v")}
        attention = {fused} if fused in tensors else separate
        required.update(attention)
        biases = {prefix + f"attn_{role}.bias" for role in ("q", "k", "v")}
        if fused in tensors and prefix + "attn_qkv.bias" in tensors:
            biases = {prefix + "attn_qkv.bias"}
        optional.update(biases)
        dims = dict(zip(ROUTED, ((emb, experts), (emb, ff, experts), (emb, ff, experts), (ff, emb, experts))))
        if arch == "qwen2moe":
            dims.update(zip(SHARED, ((emb,), (emb, shared), (emb, shared), (shared, emb))))
        for role, shape in dims.items():
            name = prefix + role + ".weight"
            required.add(name)
            need(name in tensors and tensors[name].dims == shape, f"missing or unexpected GGUF shape: {name}")
            if role in ROUTED:
                expert_layers[name] = layer
    missing = required - tensors.keys()
    extra = tensors.keys() - required - optional
    need(not missing, f"missing model tensors: {', '.join(sorted(missing)[:4])}")
    need(not extra, f"unsupported GGUF tensor/companion: {', '.join(sorted(extra)[:4])}")
    for name, tensor in tensors.items():
        if name not in expert_layers:
            need(len(tensor.dims) <= 2, f"unexpected non-routed tensor rank: {name}")
    return arch, layers, experts, top_k, expert_layers


def reviewed_maps(mapping, layers, experts, top_k):
    need(set(mapping) == {str(i) for i in range(layers)}, "plan must include exactly every routed layer")
    maps = [mapping[str(i)] for i in range(layers)]
    for i, ids in enumerate(maps):
        need(type(ids) is list and top_k <= len(ids) <= experts, f"invalid retention count at layer {i}")
        need(all(type(old) is int and 0 <= old < experts for old in ids)
             and len(ids) == len(set(ids)), f"expected unique in-range integer expert IDs at layer {i}")
        need(len(ids) == len(maps[0]), "GGUF expert count must be uniform across layers")
    return maps


def gguf_string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def project(model, fields, expert_layers, kept):
    # All edited fields, dimensions and offsets are fixed-width: the data base is unchanged.
    relative = 0
    tensors = {}
    for name, tensor in model.tensors.items():
        dims, size = tensor.dims, tensor.size
        if name in expert_layers:
            size = size // dims[-1] * kept
            dims = (*dims[:-1], kept)
        tensors[name] = Tensor(name, dims, tensor.kind, relative, size)
        relative += align(size, model.alignment)
    return GGUF(fields, tensors, model.alignment, model.header_end, model.base, model.base + relative)


def rewrite_header(out, model):
    out.write(struct.pack("<4sIQQ", b"GGUF", 3, len(model.tensors), len(model.fields)))
    for key, (kind, value) in model.fields.items():
        out.write(gguf_string(key))
        out.write(struct.pack("<I", kind))
        out.write(value)
    for name, tensor in model.tensors.items():
        out.write(gguf_string(name))
        dims = tensor.dims
        out.write(struct.pack("<I" + "Q" * len(dims) + "IQ", len(dims), *dims, tensor.kind, tensor.relative))
    need(out.tell() == model.header_end, "unexpected GGUF header size")
    out.write(bytes(model.base - model.header_end))


def spans(model, tensor, mapping):
    start = model.base + tensor.relative
    if mapping is None:
        yield start, tensor.size
        return
    stride = tensor.size // tensor.dims[-1]
    for old in mapping:
        yield start + stride * old, stride


def chunks(file, offset, size):
    file.seek(offset)
    while size:
        block = file.read(min(CHUNK, size))
        need(block, "short GGUF data read")
        yield block
        size -= len(block)


def rewrite_payload(source, out, model, maps, expert_layers, expected):
    for name, tensor in model.tensors.items():
        mapping = maps[expert_layers[name]] if name in expert_layers else None
        for offset, size in spans(model, tensor, mapping):
            for block in chunks(source, offset, size):
                out.write(block)
        out.write(bytes(-expected.tensors[name].size % model.alignment))


def verify_output(source, out, model, maps, expert_layers, expected):
    """Read back the staged GGUF and compare every tensor byte to the intended source slice."""
    actual = read_gguf(out)
    need(actual == expected and list(actual.tensors) == list(expected.tensors)
         and list(actual.fields) == list(expected.fields), "output GGUF header failed verification")
    out.seek(0)
    header = out.read(actual.base)
    need(len(header) == actual.base and not any(header[actual.header_end:]), "invalid output header padding")
    digest = hashlib.sha256(header)
    for name, tensor in actual.tensors.items():
        mapping = maps[expert_layers[name]] if name in expert_layers else None
        need(out.tell() == actual.base + tensor.relative, f"staged data offset mismatch: {name}")
        for offset, size in spans(model, model.tensors[name], mapping):
            for block in chunks(source, offset, size):
                actual_block = out.read(len(block))
                need(actual_block == block, f"staged tensor bytes mismatch: {name}")
                digest.update(actual_block)
        pad = bytes(-tensor.size % actual.alignment)
        need(out.read(len(pad)) == pad, f"staged padding mismatch: {name}")
        digest.update(pad)
    need(not out.read(1), "staged GGUF has extra data")
    return digest.hexdigest()


def run(command, source_path, plan_path, reviewed_sha, output=None):
    plan = read_plan(plan_path, reviewed_sha)
    if command == "apply":
        need(not os.path.lexists(output), "output already exists")
    with regular_file(source_path) as source:
        before = identity(source)
        model = read_gguf(source)
        arch, layers, experts, top_k, expert_layers = model_layout(model)
        require_top_k(plan, top_k)
        maps = reviewed_maps(plan["layers"], layers, experts, top_k)
        digest = hashlib.sha256()
        for block in chunks(source, 0, before[2]):
            digest.update(block)
        need(digest.hexdigest() == plan["source_sha256"], "physical source SHA-256 mismatch")
        need(identity(source) == before, "source changed during verification")
        kept = len(maps[0])
        fields = dict(model.fields)
        fields[f"{arch}.expert_count"] = (4, struct.pack("<I", kept))
        expected = project(model, fields, expert_layers, kept)
        receipt = {
            "architecture": arch, "layers": layers,
            "source_sha256": plan["source_sha256"], "plan_sha256": reviewed_sha,
            "source_bytes": model.size, "output_bytes": expected.size,
            "source_expert_count": experts, "output_expert_count": kept, "expert_used_count": top_k,
            "operation": "prune" if kept < experts else "reorder", "release_qualified": False,
        }
        if command == "preview":
            return receipt
        if command == "verify":
            with regular_file(output) as out:
                preverify = identity(out)
                receipt["output_sha256"] = verify_output(source, out, model, maps, expert_layers, expected)
                need(identity(out) == preverify, "output changed during verification")
                need(identity(source) == before, "source changed during verification")
            return receipt
        # Publish only after verification. A same-filesystem hard link cannot replace a name.
        with tempfile.TemporaryDirectory(prefix=".expert-gguf-", dir=output.parent) as stage:
            staged = Path(stage) / "model.gguf"
            with staged.open("x+b") as out:
                os.fchmod(out.fileno(), 0o600)
                rewrite_header(out, expected)
                rewrite_payload(source, out, model, maps, expert_layers, expected)
                out.flush()
                os.fsync(out.fileno())
                receipt["output_sha256"] = verify_output(source, out, model, maps, expert_layers, expected)
                need(identity(source) == before, "source changed during rewrite")
                os.link(staged, output, follow_symlinks=False)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preview", "validate a reviewed expert map and report the projected size without writing"),
        ("apply", "apply a reviewed expert map to a new GGUF"),
        ("verify", "read-only verification of an existing output against source and reviewed map"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--plan", type=Path, required=True, help=f"{PLAN_SCHEMA} JSON file")
        command.add_argument("--plan-sha256", required=True, help="SHA-256 of the exact reviewed plan file bytes")
        if name != "preview":
            output_help = "new file; existing paths are never replaced" if name == "apply" else "existing rewritten GGUF"
            command.add_argument("--output", type=Path, required=True, help=output_help)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.command, args.source, args.plan, args.plan_sha256, getattr(args, "output", None)),
                         sort_keys=True))
    except (OSError, ValueError, RecursionError, struct.error) as exc:
        print(f"expert_gguf: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
