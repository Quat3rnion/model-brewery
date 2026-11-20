#!/usr/bin/env python3
"""Apply a reviewed expert map to a bounded native HGN v2 file, offline.

Supports canonical fused layers.N.mlp gate/gate_up/down tensors, including
I4R and Q4CP planes and Q8G64/Q6G64 rows. No decode/re-encode or runtime/config
changes. Source limit: 32 MiB, 4096 tensors, 8 Mi elements per tensor. This is
not a loader for published model bundles or proof of runtime compatibility.
Pruning needs a reviewed top-level top_k.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

from cross_container.ggml_in_hgn import ENTRY, HEADER, align, read_file, xor32
from weight_hgn import inventory, packed_size, require

SCHEMA = "brewery.expert-map.v1"
SHA256 = re.compile(r"[0-9a-f]{64}")
LAYER = re.compile(r"0|[1-9][0-9]{0,5}")
MLP_PREFIX = re.compile(r"(?:mtp\.)?layers\.(?:0|[1-9][0-9]{0,5})\.mlp\.")
ROUTING = re.compile(r"(?:^|[._])(?:router|routing|experts?|moe)(?:$|[._])", re.I)
ROUTER = "gate.weight"
UP = "experts.gate_up_proj.weight"
DOWN = "experts.down_proj.weight"
FLOAT_WIDTH = {"BF16": 2, "F16": 2, "F32": 4}
SHARED = {
    f"shared_expert.{proj}.{part}"
    for proj in ("gate_proj", "up_proj", "down_proj")
    for part in ("weight", "suh", "svh")
} | {"shared_expert_gate.weight"}
STOCK_TOP_K = {b"qwen3.8-flash-next-v2": 10, b"qwen3.8-flash-next-ht43": 10}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate expert map key: {key!r}")
        result[key] = value
    return result


def bad_constant(value):
    raise ValueError(f"invalid expert map JSON constant: {value}")


def reviewed_map(path, reviewed):
    require(SHA256.fullmatch(reviewed), "expected lowercase --reviewed-sha256")
    # read_file also requires a regular non-symlink file and a stable read.
    raw = read_file(path)
    require(len(raw) <= 1 << 20, "expert map exceeds 1 MiB")
    require(hashlib.sha256(raw).hexdigest() == reviewed, "reviewed map SHA-256 mismatch")
    mapping = json.loads(raw.tobytes().decode("utf-8"), object_pairs_hook=unique_object,
                         parse_constant=bad_constant)
    require(isinstance(mapping, dict) and mapping.get("schema") == SCHEMA,
            f"expected {SCHEMA}")
    digest, layers = mapping.get("source_sha256"), mapping.get("layers")
    require(isinstance(digest, str) and SHA256.fullmatch(digest), "invalid map source_sha256")
    require(isinstance(layers, dict) and 0 < len(layers) <= 4096, "invalid or empty expert layer map")
    for key, order in layers.items():
        require(LAYER.fullmatch(key), f"invalid expert layer: {key!r}")
        require(isinstance(order, list) and order
                and all(type(i) is int and i >= 0 for i in order)
                and len(set(order)) == len(order), f"layer {key}: invalid or duplicate expert IDs")
    review = mapping.get("review")
    top_ks = [item["top_k"] for item in (mapping, review)
              if isinstance(item, dict) and "top_k" in item]
    require(all(type(k) is int and k > 0 for k in top_ks), "invalid map top_k")
    require(len(set(top_ks)) <= 1, "conflicting map top_k values")
    return digest, layers, mapping.get("top_k")


def take_experts(raw, start, stride, order):
    return b"".join(raw[start + i * stride:start + (i + 1) * stride] for i in order)


def expert_payload(raw, t, order):
    experts, rows, cols = t.shape
    if t.dtype in FLOAT_WIDTH or t.dtype in {"Q8G64", "Q6G64"}:
        # Native affine stores keep codes and scale/bias inside each row.
        return take_experts(raw, 0, t.size // experts, order)
    if t.dtype == "I4R":
        codes = rows * cols // 2
        scales = rows * cols // 64
        return (take_experts(raw, 0, codes, order)
                + take_experts(raw, experts * codes, scales, order))
    if t.dtype == "Q4CP":
        # Native Q4CP has one global codebook, two explicit planes, and
        # alignment between the planes that depends on the expert count.
        codes = rows * cols // 2
        stride = align(cols // 16, 16)
        scale_start = 64 + align(experts * codes)
        require(not any(raw[64 + experts * codes:scale_start]), f"nonzero Q4CP plane padding: {t.name}")
        require(all(math.isfinite(v) for (v,) in struct.iter_unpack("<f", raw[:64])),
                f"non-finite Q4CP codebook: {t.name}")
        meaningful_scale = cols // 16
        if stride != meaningful_scale:
            for start in range(scale_start, len(raw), stride):
                require(not any(raw[start + meaningful_scale:start + stride]),
                        f"nonzero Q4CP scale-row padding: {t.name}")
        return (bytes(raw[:64]) + take_experts(raw, 64, codes, order)
                + bytes(align(len(order) * codes) - len(order) * codes)
                + take_experts(raw, scale_start, rows * stride, order))
    raise ValueError(f"unsupported routed expert packing {t.dtype}: {t.name}")


def layer_changes(data, tensors, layer, order, top_k):
    prefix = f"layers.{layer}."
    mlp = prefix + "mlp."
    local = {name[len(mlp):]: t for name, t in tensors.items() if name.startswith(mlp)}
    require({ROUTER, UP, DOWN} <= local.keys(),
            f"layer {layer}: expected fused native router, gate/up, and down tensors")
    gate, up, down = (local[role] for role in (ROUTER, UP, DOWN))
    require(len(gate.shape) == 2 and gate.dtype in FLOAT_WIDTH
            and len(up.shape) == len(down.shape) == 3,
            f"layer {layer}: unsupported router or fused expert layout")
    experts, width = gate.shape
    intermediate = down.shape[-1]
    require(up.shape == (experts, 2 * intermediate, width)
            and down.shape == (experts, width, intermediate),
            f"layer {layer}: inconsistent router and expert shapes")
    require(max(width, intermediate) <= 16384
            and (up.dtype != "I4R" or intermediate % 128 == 0),
            f"layer {layer}: unsupported native expert row width or I4R gate/up halves")
    require(max(order) < experts, f"layer {layer}: expert ID out of range (count {experts})")
    if len(order) < experts:
        require(top_k is not None, f"layer {layer}: pruning requires a reviewed top_k")
    if top_k is not None:
        require(top_k <= len(order), f"layer {layer}: retained experts fewer than top_k ({top_k})")

    rotations = {}
    if up.dtype == "I4R":
        rotations["experts.su"] = width
        rotations["experts.gate_up_proj.svh"] = 2 * intermediate
    if down.dtype == "I4R":
        rotations["experts.down_proj.suh"] = intermediate
        rotations["experts.down_proj.svh"] = width
    # The stock I4R converter shares these vectors across experts. Native
    # HGN HT instead has per-expert planes and, with pruning, unsupported ranks/shapes.
    for name, count in rotations.items():
        v = local.get(name)
        require(v is not None and v.dtype == "F16" and v.shape == (count,),
                f"unproven I4R companion layout: {mlp}{name}")
        raw = data[v.offset:v.offset + v.size]
        require(all(word in (0x3C00, 0xBC00) for (word,) in struct.iter_unpack("<H", raw)),
                f"unproven I4R companion (expected shared +/-1 signs): {mlp}{name}")
    for t in (up, down):
        require(t.dtype not in {"HT3", "HT4"}, f"unsupported routed expert packing {t.dtype}: {t.name}")
    unknown = local.keys() - {ROUTER, UP, DOWN} - SHARED - rotations.keys()
    require(not unknown, f"layer {layer}: unproven MoE sidecar/layout {mlp}{min(unknown, default='')}")
    for name in tensors:
        if name.startswith(prefix) and not name.startswith(mlp):
            suffix = name[len(prefix):]
            require(suffix.startswith(("self_attn.", "linear_attn.", "attn_hyper_connection.",
                                       "mlp_hyper_connection.", "ple."))
                    or suffix in {"input_layernorm.weight", "post_attention_layernorm.weight"},
                    f"layer {layer}: unproven layer sidecar/layout {name}")
    changes = {}
    for t in (gate, up, down):
        raw = data[t.offset:t.offset + t.size]
        shape = (len(order), *t.shape[1:])
        payload = (take_experts(raw, 0, t.size // experts, order)
                   if t is gate else expert_payload(raw, t, order))
        expected = (math.prod(shape) * FLOAT_WIDTH[t.dtype] if t.dtype in FLOAT_WIDTH
                    else packed_size(t.dtype, shape, t.name))
        require(len(payload) == expected, f"rewritten HGN tensor size mismatch: {t.name}")
        changes[t.name] = (shape, payload)
    return changes


def write_new(destination, data, header, tensors, changes):
    magic, version, count, table, base, _, identity = header
    entries, payloads = [], []
    end = base
    for i, t in enumerate(tensors):
        raw_name, store, _, *fields = ENTRY.unpack_from(data, table + i * ENTRY.size)
        checksum, variant = fields[-2:]
        if t.name in changes:
            shape, payload = changes[t.name]
            checksum = xor32(payload)
        else:
            shape, payload = t.shape, data[t.offset:t.offset + t.size]
        offset = align(end)
        entries.append(ENTRY.pack(raw_name, store, len(shape), *shape, *([0] * (4 - len(shape))),
                                  offset, len(payload), checksum, variant))
        payloads.append(payload)
        end = offset + len(payload)
    require(end <= len(data), "new HGN exceeds source size")
    fd, tmp = tempfile.mkstemp(prefix=".expert-hgn-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(HEADER.pack(magic, version, count, table, base, end, identity))
            f.writelines(entries)
            for payload in payloads:
                f.write(bytes(align(f.tell()) - f.tell()))
                f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        # Atomic no-clobber, even if an existing path is a symlink/hard link.
        os.link(tmp, destination)
    finally:
        os.unlink(tmp)


def apply(source, map_path, destination, reviewed):
    require(not os.path.lexists(destination), "output already exists; choose a new file")
    require(not source.is_dir(), "HGN model bundles are not supported; expected one native HGN v2 file")
    digest, layers, top_k = reviewed_map(map_path, reviewed)
    data = read_file(source)
    require(hashlib.sha256(data).hexdigest() == digest, "source HGN SHA-256 mismatch")
    tensors = inventory(data)
    header = HEADER.unpack_from(data)
    native_top_k = STOCK_TOP_K.get(header[-1].rstrip(b"\0"))
    if native_top_k is not None and top_k is not None:
        require(top_k == native_top_k, f"map top_k disagrees with known native top_k ({native_top_k})")
    by_name = {t.name: t for t in tensors}
    for name in by_name:
        require(MLP_PREFIX.match(name) or not ROUTING.search(name), f"unproven routing sidecar: {name}")
    changes = {}
    for layer, order in layers.items():
        changes.update(layer_changes(data, by_name, layer, order, top_k))
    write_new(destination, data, header, tensors, changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="read-only native HGN v2 file")
    parser.add_argument("map", type=Path, help=f"reviewed {SCHEMA} JSON")
    parser.add_argument("output", type=Path, help="new HGN file; must not exist")
    parser.add_argument("--reviewed-sha256", required=True, help="SHA-256 of the exact reviewed map bytes")
    args = parser.parse_args()
    try:
        apply(args.source, args.map, args.output, args.reviewed_sha256)
    except (OSError, ValueError, RecursionError) as exc:
        parser.exit(1, f"expert-hgn: {exc}\n")


if __name__ == "__main__":
    main()
