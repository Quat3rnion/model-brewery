"""Inspect native EXL3 SafeTensors metadata without loading a model.

Pinned to ExLlamaV3 v1.6.7 (see UPSTREAM). Linear tile layout and marker
values come from modules/quant/exl3_lib/quantize.py, supported rates from
conversion/allocation.py, and grouped embedding rows from
conversion/ngram.py. Only headers, JSON, and four bytes per marker are read;
this does not test weight contents, model completeness, encoding, or inference.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys

from weight_safetensors import (
    INDEX, MAX_BYTES, MAX_TENSORS, OPEN_DIR, OPEN_FILE, WIDTHS, need, open_at, open_path,
    parse_header, parse_json, read_at, stamp,
)


UPSTREAM = "https://github.com/turboderp-org/exllamav3/tree/8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0"
HEADER_BUDGET = 32 << 20
JSON_BUDGET = 16 << 20
MAX_DIRECTORY_ENTRIES = 4096
MODEL_FILE = "model.safetensors"
SHARD = re.compile(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors")
MARKERS = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
INTEGER_WORDS = frozenset(range(16, 129, 16))
HALF_INTEGER_WORDS = {24, 40, 56}
TORCH_DTYPES = {
    "I8": "int8", "I16": "int16", "I32": "int32", "I64": "int64", "U8": "uint8",
    "F16": "float16", "BF16": "bfloat16", "F32": "float32",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "F8_E8M0": "uint8",
}


def member_header(fd, member, budget, expected):
    st = os.fstat(fd)
    need(stat.S_ISREG(st.st_mode) and 8 <= st.st_size <= MAX_BYTES,
         f"{member}: expected a bounded regular SafeTensors file")
    need(stamp(st) == expected, f"{member}: file changed before inspection")
    tensors, length, payload = parse_header(
        lambda start, count: read_at(fd, start, count), member, st.st_size, budget,
    )
    markers = {}
    for name, tensor in tensors.items():
        suffix = name.rsplit(".", 1)[-1]
        if "." in name and suffix in MARKERS:
            need(tensor.dtype == "I32" and tensor.shape == (),
                 f"{name}: expected an I32 scalar codebook marker")
            value, = struct.unpack("<I", read_at(fd, tensor.start, 4))
            need(value == MARKERS[suffix], f"{name}: incorrect {suffix} multiplier")
            markers[name] = value
    need(stamp(os.fstat(fd)) == expected, f"{member}: file changed during inspection")
    return tensors, markers, length, payload


def linear_layout(key, trellis, tensors, markers):
    need(trellis.dtype == "I16" and len(trellis.shape) == 3,
         f"{trellis.name}: expected rank-3 I16 linear trellis")
    tiles_in, tiles_out, words = trellis.shape
    need(tiles_in > 0 and tiles_out > 0 and tiles_in % 8 == tiles_out % 8 == 0,
         f"{trellis.name}: padded linear axes must be positive multiples of 128 features")
    marker_names = [f"{key}.{suffix}" for suffix in MARKERS if f"{key}.{suffix}" in markers]
    need(len(marker_names) <= 1, f"{key}: ambiguous codebook markers")
    marker = marker_names[0] if marker_names else None
    codebook = marker.rsplit(".", 1)[-1] if marker else "3inst"
    need(words in INTEGER_WORDS or (codebook == "mul1" and words in HALF_INTEGER_WORDS),
         f"{trellis.name}: unsupported words per tile ({words}) for {codebook}")

    for suffix in ("weight", "scale", "signs"):
        need(f"{key}.{suffix}" not in tensors, f"{key}: conflicting .{suffix} and linear trellis")
    padded_in, padded_out = 16 * tiles_in, 16 * tiles_out
    rotations = {}
    for axis, unpacked, packed, features in (
        ("input", "suh", "su", padded_in), ("output", "svh", "sv", padded_out),
    ):
        h, p = tensors.get(f"{key}.{unpacked}"), tensors.get(f"{key}.{packed}")
        need(h is not None or p is not None, f"{key}: missing {axis} rotation tensor")
        need(h is None or p is None, f"{key}: ambiguous {axis} rotation tensors")
        t = h or p
        need((t.dtype, t.shape) == (("F16", (features,)) if h else ("I16", (features // 16,))),
             f"{t.name}: invalid {axis} rotation dtype or shape")
        rotations[axis] = {"tensor": t.name, "packing": "fp16" if h else "i16-sign-bits"}
    bias = tensors.get(f"{key}.bias")
    need(bias is None or (bias.dtype in {"F16", "F32"} and bias.shape == (padded_out,)),
         f"{key}.bias: expected F16 or F32 padded output vector")
    return {
        "layout": "linear",
        "padded_in_features": padded_in,
        "padded_out_features": padded_out,
        "words_per_tile": words,
        "bits_per_weight": words // 16 if words % 16 == 0 else words / 16,
        "codebook": codebook,
        "marker": None if marker is None else {"tensor": marker, "multiplier": markers[marker]},
        "rotations": rotations,
    }


def embedding_layout(key, trellis, tensors):
    need(trellis.dtype == "I16",
         f"{trellis.name}: expected I16 token embedding trellis")
    signs = tensors.get(f"{key}.signs")
    need(signs is not None and signs.dtype == "F16" and len(signs.shape) == 2 and
         signs.shape[0] > 0 and signs.shape[1] == 256,
         f"{key}.signs: expected F16 [groups, 256] token embedding signs")
    for suffix in ("weight", "scale", "su", "sv", "suh", "svh", "bias", *MARKERS):
        need(f"{key}.{suffix}" not in tensors, f"{key}: conflicting .{suffix} and embedding trellis")
    groups = signs.shape[0]
    rows, row_words = trellis.shape
    need(rows > 0 and row_words % groups == 0, f"{trellis.name}: invalid embedding row shape")
    words = row_words // groups
    need(words - 1 in INTEGER_WORDS, f"{trellis.name}: unsupported words per embedding group ({words})")
    return {
        "layout": "token-embedding",
        "rows": rows,
        "padded_embedding_features": 256 * groups,
        "words_per_group": words,
        "bits_per_weight": (words - 1) // 16,
        "codebook": "mul1",
        "marker": None,
        "sign_tensor": signs.name,
    }


def layouts(tensors, markers):
    result = {}
    for name, trellis in sorted(tensors.items()):
        if not name.endswith(".trellis"):
            continue
        key = name.removesuffix(".trellis")
        need(bool(key), "empty EXL3 module prefix")
        layout = (embedding_layout(key, trellis, tensors) if len(trellis.shape) == 2
                  else linear_layout(key, trellis, tensors, markers))
        result[key] = {
            "module": key, "packed_tensor": name, "packed_shape": trellis.shape,
            **layout,
        }
    for name in markers:
        need(name.rsplit(".", 1)[0] in result, f"{name}: orphan codebook marker")
    need(result, "no supported EXL3 trellis tensors")
    return result


def read_json_member(root, entries, name):
    expected = entries[name]
    with open_at(root, (name,), OPEN_FILE) as fd:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and 0 < st.st_size <= JSON_BUDGET,
             f"{name}: expected a bounded regular JSON file")
        need(stamp(st) == expected, f"{name}: file changed before inspection")
        obj = parse_json(read_at(fd, 0, st.st_size))
        need(stamp(os.fstat(fd)) == expected, f"{name}: file changed during inspection")
    need(isinstance(obj, dict), f"{name}: expected JSON object")
    return obj


def check_method(obj, source):
    need(isinstance(obj, dict) and obj.get("quant_method", "exl3") == "exl3",
         f"{source}: conflicting or malformed quantization method")


def check_sidecar(sidecar, modules, tensors):
    name = "quantization_config.json"
    check_method(sidecar, name)
    if "tensor_storage" not in sidecar:
        return 0
    storage = sidecar["tensor_storage"]
    need(isinstance(storage, dict) and len(storage) <= MAX_TENSORS,
         f"{name}: invalid tensor_storage")
    checked = 0
    for key, record in storage.items():
        need(isinstance(record, dict), f"{name}: malformed storage row for {key!r}")
        info = modules.get(key)
        claimed_exl3 = record.get("quant_format") == "exl3"
        if not claimed_exl3 and (info is None or info["layout"] != "linear"):
            continue
        need(info is not None and info["layout"] == "linear" and claimed_exl3,
             f"{key}: sidecar quant_format disagrees with the linear tensors")
        bits = record.get("bits_per_weight")
        need(type(bits) in {int, float} and bits == info["bits_per_weight"],
             f"{key}: sidecar bits_per_weight disagrees with the trellis")
        for codebook, value in MARKERS.items():
            field = f"{codebook}_multiplier"
            if info["codebook"] == codebook:
                need(type(record.get(field)) is int and record[field] == value,
                     f"{key}: sidecar codebook disagrees with the {codebook} marker")
            else:
                need(field not in record, f"{key}: unexpected sidecar codebook {codebook}")
        stored = record.get("stored_tensors")
        required = {info["packed_tensor"], *(v["tensor"] for v in info["rotations"].values())}
        if info["marker"]:
            required.add(info["marker"]["tensor"])
        if f"{key}.bias" in tensors:
            required.add(f"{key}.bias")
        need(isinstance(stored, dict) and required <= stored.keys(),
             f"{key}: sidecar missing linear tensor descriptors")
        for tensor_name, descriptor in stored.items():
            t = tensors.get(tensor_name)
            need(t is not None and (tensor_name == key or tensor_name.startswith(f"{key}.")),
                 f"{key}: sidecar describes an unknown module tensor")
            need(isinstance(descriptor, dict) and isinstance(descriptor.get("shape"), list) and
                 all(type(n) is int for n in descriptor["shape"]) and
                 tuple(descriptor["shape"]) == t.shape and
                 t.dtype in TORCH_DTYPES and descriptor.get("dtype") == f"torch.{TORCH_DTYPES[t.dtype]}" and
                 type(descriptor.get("n_bytes")) is int and
                 descriptor["n_bytes"] == math.prod(t.shape) * WIDTHS[t.dtype],
                 f"{t.name}: sidecar descriptor disagrees with the SafeTensors header")
        checked += 1
    return checked


def make_report(members, tensors, markers, header_bytes, modules, uninspected=(), metadata=(),
                sidecar_rows=0):
    return {
        "representation": "exl3", "validation": "metadata-only", "upstream": UPSTREAM,
        "members": list(members),
        "uninspected_safetensors": list(uninspected),
        "metadata_members": list(metadata),
        "sidecar_linear_rows_checked": sidecar_rows,
        "tensor_count": len(tensors),
        "header_bytes_read": header_bytes,
        "marker_payload_bytes_read": 4 * len(markers),
        "modules": list(modules.values()),
    }


def inspect_directory(root):
    start = stamp(os.fstat(root))
    entries = {}
    with os.scandir(root) as scan:
        for i, entry in enumerate(scan):
            need(i < MAX_DIRECTORY_ENTRIES, "checkpoint directory exceeds entry limit")
            if entry.name.endswith(".safetensors") or entry.name in {
                INDEX, "config.json", "quantization_config.json",
            }:
                entries[entry.name] = stamp(entry.stat(follow_symlinks=False))
    mapping = None
    metadata = []
    shard_names = sorted(name for name in entries if SHARD.fullmatch(name))
    if INDEX in entries:
        need(MODEL_FILE not in entries, "ambiguous unsharded model and indexed shards")
        index = read_json_member(root, entries, INDEX)
        metadata.append(INDEX)
        mapping = index.get("weight_map")
        need(isinstance(mapping, dict) and 0 < len(mapping) <= MAX_TENSORS,
             "invalid SafeTensors weight_map")
        need(all(isinstance(n, str) and SHARD.fullmatch(n) for n in mapping.values()),
             "index must reference canonical flat SafeTensors shards")
        members = sorted(set(mapping.values()))
        need(members == shard_names and len(members) >= 2 and all(
            n == f"model-{i:05}-of-{len(members):05}.safetensors" for i, n in enumerate(members, 1)
        ), "index does not match a complete set of canonical shards")
    else:
        need(MODEL_FILE in entries and not shard_names,
             "expected model.safetensors or canonical shards with a SafeTensors index")
        members = [MODEL_FILE]
    tensors, markers = {}, {}
    used, payload_bytes = 0, 0
    for name in members:
        with open_at(root, (name,), OPEN_FILE) as fd:
            local, local_markers, length, payload = member_header(
                fd, name, HEADER_BUDGET - used - 8, entries[name],
            )
        used += length + 8
        payload_bytes += payload
        need(len(tensors) + len(local) <= MAX_TENSORS and payload_bytes <= MAX_BYTES,
             "checkpoint exceeds aggregate limits")
        need(not tensors.keys() & local.keys(), "duplicate tensor across indexed shards")
        tensors.update(local)
        markers.update(local_markers)
    if mapping is not None:
        need(mapping == {key: t.member for key, t in tensors.items()},
             "index does not match SafeTensors headers")
        if "metadata" in index:
            meta = index["metadata"]
            need(isinstance(meta, dict), "invalid index metadata")
            if "total_size" in meta:
                need(type(meta["total_size"]) is int and meta["total_size"] == payload_bytes,
                     "index total_size does not match SafeTensors payload size")
    modules = layouts(tensors, markers)
    if "config.json" in entries:
        config = read_json_member(root, entries, "config.json")
        metadata.append("config.json")
        if "quantization_config" in config:
            check_method(config["quantization_config"], "config.json")
    sidecar_rows = 0
    if "quantization_config.json" in entries:
        sidecar = read_json_member(root, entries, "quantization_config.json")
        metadata.append("quantization_config.json")
        sidecar_rows = check_sidecar(sidecar, modules, tensors)
    for name in (*members, *metadata):
        need(stamp(os.stat(name, dir_fd=root, follow_symlinks=False)) == entries[name],
             f"{name}: file changed during inspection")
    need(stamp(os.fstat(root)) == start, "checkpoint directory changed during inspection")
    uninspected = sorted(n for n in entries.keys() - set(members) if n.endswith(".safetensors"))
    return make_report(members, tensors, markers, used, modules, uninspected, metadata, sidecar_rows)


def inspect(path):
    """Return schema evidence, not checkpoint provenance or runnable model evidence."""
    path = Path(path)
    st = path.lstat()
    if stat.S_ISDIR(st.st_mode):
        with open_path(path, OPEN_DIR) as root:
            need(stamp(os.fstat(root)) == stamp(st), "checkpoint directory changed before inspection")
            return inspect_directory(root)
    need(path.suffix == ".safetensors" and stat.S_ISREG(st.st_mode),
         "expected a regular .safetensors file or checkpoint directory")
    with open_path(path, OPEN_FILE) as fd:
        tensors, markers, length, _ = member_header(fd, path.name, HEADER_BUDGET - 8, stamp(st))
    return make_report([path.name], tensors, markers, length + 8, layouts(tensors, markers))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("checkpoint", help="standalone SafeTensors file or canonical checkpoint directory")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.checkpoint), indent=2))
    except (OSError, ValueError, RecursionError) as exc:
        print(f"EXL3 inspection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
