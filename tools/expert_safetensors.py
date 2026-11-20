#!/usr/bin/env python3
"""Apply a reviewed Flash-Next expert map to a new offline SafeTensors revision.

Layer keys are decimal strings: trunk 0..N-1, then MTP N. Each array maps a
new expert ID to an old expert ID. A reorder may name a subset of the layers;
a prune must name all of them and keep the same number in each.
"""

import argparse
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

from weight_safetensors import (
    CHUNK, HEADER, INDEX, MAX_BYTES, OPEN_DIR, OPEN_FILE, WIDTHS, Files,
    hash_fd, need, open_path, parse_json, read_at, rename_noreplace, stamp,
)


SCHEMA = "brewery.expert-map.v1"
CONFIG = "config.json"
STANDALONE = "model.safetensors"
PREFIX = "model.language_model.layers."
MTP = "mtp.layers.0.mlp."
FLOATS = {"BF16", "F16", "F32"}
SHARED = {
    "shared_expert_gate.weight",
    *("shared_expert." + p + "." + t
      for p in ("gate_proj", "up_proj", "down_proj") for t in ("weight", "bias")),
}


def json_bytes(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256(value, label):
    need(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value),
         f"{label} must be a lowercase SHA-256")
    return value


def pinned(path, expected, label):
    sha256(expected, label)
    with open_path(path, OPEN_FILE) as fd:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and 0 < st.st_size <= HEADER,
             f"{label} requires a bounded regular file without links")
        raw = read_at(fd, 0, st.st_size)
        need(stamp(os.fstat(fd)) == stamp(st), f"{label} changed during read")
        need(hashlib.sha256(raw).hexdigest() == expected, f"{label} SHA-256 mismatch")
    return raw, stamp(st)


def tensor_bytes(t):
    return math.prod(t.shape) * WIDTHS[t.dtype]


class Source(Files):
    """Use the shared checked inventory, shard parser, and source digest convention.

    Unlike range-based weight reads, surgery hashes every member before reading
    any tensor and then checks source stamps throughout the copy.
    """

    def __init__(self, path):
        self.path = Path(path)
        meta = self.path.lstat()
        self.directory = stat.S_ISDIR(meta.st_mode)
        self.parent = self.path if self.directory else self.path.parent
        self.stat, self.dirs, self.chunks = {}, {}, {}
        if self.directory:
            with open_path(self.parent, OPEN_DIR) as fd:
                need(stamp(meta) == stamp(os.fstat(fd)), "indexed HF directory changed")
                self._walk(fd)
            need(INDEX in self.stat and CONFIG in self.stat,
                 "indexed HF requires the SafeTensors index and config.json")
        else:
            need(self.path.suffix == ".safetensors" and stat.S_ISREG(meta.st_mode)
                 and meta.st_nlink == 1, "expected a standalone regular .safetensors file without links")
            self.stat[self.path.name] = meta
        self.names = sorted(self.stat)
        need(0 < sum(st.st_size for st in self.stat.values()) <= MAX_BYTES,
             "SafeTensors file set exceeds size limit")

    def verify(self, expected=None):
        hashes = {}
        for member in self.names:
            with self.open_member(member) as fd:
                hashes[member] = hash_fd(fd, self.stat[member].st_size)
        self.check_current()
        digest = self.digest(hashes)
        need(expected is None or digest == expected, "SafeTensors source SHA-256 mismatch")
        return digest

    def metadata(self, member):
        length = self.stat[member].st_size
        need(0 < length <= HEADER, f"{member} exceeds metadata bounds")
        return self.range(member, 0, length)

    def group_shards(self):
        groups = {}
        for tensor in self._headers().values():
            groups.setdefault(tensor.member, {})[tensor.name] = tensor
        need(set(groups) == {n for n in self.names if n.lower().endswith(".safetensors")},
             "unindexed SafeTensors shard")
        return groups


def positive(obj, key, bound=MAX_BYTES):
    value = obj.get(key)
    need(type(value) is int and 0 < value <= bound, f"invalid Flash-Next {key}")
    return value


def flash_config(raw):
    config = parse_json(raw)
    need(isinstance(config, dict) and config.get("model_type") == "qwen4_exp"
         and config.get("architectures") == ["Qwen4ExpForConditionalGeneration"],
         "unsupported Flash-Next architecture")
    text = config.get("text_config")
    need(isinstance(text, dict) and text.get("model_type") == "qwen4_exp_text",
         "expected Flash-Next text_config")
    mtp = text.get("mtp")
    need(positive(text, "mtp_num_hidden_layers") == 1 and isinstance(mtp, dict)
         and positive(mtp, "num_hidden_layers") == 1 and mtp.get("layer_types") == ["full_attention"],
         "expected one integrated Flash-Next MTP block")
    experts = positive(text, "num_experts", 16384)
    top_k = positive(text, "num_experts_per_tok", experts)
    layers = positive(text, "num_hidden_layers", 256)
    hidden = positive(text, "hidden_size")
    inner = positive(text, "moe_intermediate_size")
    fields = {"num_experts": experts, "num_experts_per_tok": top_k,
              "hidden_size": hidden, "moe_intermediate_size": inner}
    for obj in (config, text, mtp):
        need(obj.get("quantization_config") is None, "unsupported quantized Flash-Next config")
        need(not obj.keys() & {"n_routed_experts", "num_local_experts", "expert_map"},
             "unsupported expert count or map override")
        need(not obj.keys() & {"n_group", "topk_group", "num_expert_group", "num_expert_groups"},
             "unsupported grouped expert routing")
        for key, value in fields.items():
            need(key not in obj or (type(obj[key]) is int and obj[key] == value),
                 f"ambiguous Flash-Next {key}")
    return config, layers, experts, top_k, hidden, inner


def plan_layers(plan, layers, experts, top_k):
    review = plan.get("review", {})
    need(isinstance(review, dict), "expert plan review must be an object")
    for obj in (plan, review):
        need("top_k" not in obj or (type(obj["top_k"]) is int and obj["top_k"] == top_k),
             "expert plan top_k must match native top-k")
    maps = plan["layers"]
    need(isinstance(maps, dict) and maps, "expert plan requires a nonempty layer map")
    layer_ids = {str(i) for i in range(layers + 1)}
    need(maps.keys() <= layer_ids, "unknown expert layer ID (MTP follows the trunk)")
    for layer, indices in maps.items():
        need(isinstance(indices, list) and top_k <= len(indices) <= experts
             and all(type(i) is int and 0 <= i < experts for i in indices)
             and len(set(indices)) == len(indices),
             f"invalid expert IDs or count below top-k for layer {layer}")
    counts = {len(indices) for indices in maps.values()}
    need(len(counts) == 1, "expert counts must agree between mapped layers")
    remaining = next(iter(counts))
    need(remaining == experts or maps.keys() == layer_ids,
         "pruning requires every routed trunk and MTP layer")
    return remaining


def expert_edits(plan, groups, layers, experts, hidden, inner):
    prefixes = {PREFIX + str(i) + ".mlp.": str(i) for i in range(layers)}
    prefixes[MTP] = str(layers)
    projections = {"gate_up_proj": (2 * inner, hidden), "down_proj": (hidden, inner)}
    required = {"gate.weight": {(experts, hidden)}}
    optional = {"gate.bias": {(experts,)}, "gate.e_score_correction_bias": {(experts,)}}
    for proj, (rows, cols) in projections.items():
        key = "experts." + proj
        required[key] = {(experts, rows, cols)}
        optional[key + "_bias"] = {(experts, rows)}
        optional[key + "_scale"] = {
            (experts,), (experts, rows), (experts, rows, 1), (experts, rows, cols),
        }
    seen = {prefix: set() for prefix in prefixes}
    edits = {}
    identity = list(range(experts))
    for group in groups.values():
        for name, t in group.items():
            base, delimiter, suffix = name.partition(".mlp.")
            prefix = base + delimiter
            if prefix not in prefixes:
                need(not any("expert" in p or "router" in p for p in name.split("."))
                     and not (delimiter and (
                         base.startswith((PREFIX, "mtp.", "model.layers."))
                         or suffix.startswith("gate."))),
                     f"unsupported routed tensor or layer: {name}")
                continue
            if suffix in SHARED:
                continue
            shapes = required.get(suffix) or optional.get(suffix)
            need(shapes and t.shape in shapes and t.dtype in FLOATS,
                 f"unsupported expert tensor layout: {name} ({t.dtype}, {t.shape})")
            seen[prefix].add(suffix)
            indices = plan["layers"].get(prefixes[prefix])
            if indices is not None and indices != identity:
                edits[name] = indices
    for prefix, names in seen.items():
        need(required.keys() <= names, f"missing Flash-Next routed tensors in {prefix}")
    return edits


def copy_range(src, dst, start, count):
    for pos in range(start, start + count, CHUNK):
        dst.write(read_at(src, pos, min(CHUNK, start + count - pos)))


def rewrite_header(source, member, tensors, edits):
    with source.open_member(member) as fd:
        length, = struct.unpack("<Q", read_at(fd, 0, 8))
        original = parse_json(read_at(fd, 8, length))
    header = {}
    if "__metadata__" in original:
        header["__metadata__"] = original["__metadata__"]
    end = 0
    ordered = sorted(tensors.values(), key=lambda t: (t.start, tensor_bytes(t), t.name))
    for t in ordered:
        indices = edits.get(t.name)
        shape = t.shape if indices is None else (len(indices), *t.shape[1:])
        size = math.prod(shape) * WIDTHS[t.dtype]
        header[t.name] = {"dtype": t.dtype, "shape": shape, "data_offsets": [end, end + size]}
        end += size
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    raw += b" " * (-len(raw) % 8)
    need(len(raw) <= HEADER, "rewritten SafeTensors header exceeds bounds")
    return struct.pack("<Q", len(raw)) + raw, ordered, end


def write_member(source, member, dest, edits, rewrite=None, data=None):
    def create(path, flags):
        return os.open(path, flags | os.O_NOFOLLOW, 0o600)

    with open(dest, "xb", opener=create) as out:
        if data is not None:
            out.write(data)
        else:
            with source.open_member(member) as fd:
                if rewrite is None:
                    copy_range(fd, out, 0, source.stat[member].st_size)
                else:
                    header, tensors, _ = rewrite
                    out.write(header)
                    for t in tensors:
                        size = tensor_bytes(t)
                        indices = edits.get(t.name)
                        if indices is None:
                            copy_range(fd, out, t.start, size)
                        else:
                            stride = size // t.shape[0]
                            for old in indices:
                                copy_range(fd, out, t.start + old * stride, stride)
        out.flush()
        os.fsync(out.fileno())


def publish(source, dest, config_raw, config_check, groups, edits, pruning):
    need(not os.path.lexists(dest), "destination already exists")
    need(not source.directory or not Path(os.path.abspath(dest)).is_relative_to(
         Path(os.path.abspath(source.path))), "destination cannot be inside the source directory")
    rewrites = {member: rewrite_header(source, member, tensors, edits)
                for member, tensors in groups.items() if tensors.keys() & edits.keys()}
    replacements = {}
    if pruning and source.directory:
        index = parse_json(source.metadata(INDEX))
        index.setdefault("metadata", {})["total_size"] = sum(
            rewrites[member][2] if member in rewrites else sum(map(tensor_bytes, tensors.values()))
            for member, tensors in groups.items()
        )
        replacements.update({INDEX: json_bytes(index), CONFIG: config_raw})
    if not source.directory:
        replacements[CONFIG] = config_raw
    members = {member: member for member in source.names} if source.directory else {
        STANDALONE: source.names[0], CONFIG: None,
    }
    lengths = {}
    for name, member in members.items():
        if name in replacements:
            lengths[name] = len(replacements[name])
        elif member in rewrites:
            header, _, payload = rewrites[member]
            lengths[name] = len(header) + payload
        else:
            lengths[name] = source.stat[member].st_size
    with open_path(dest.parent, OPEN_DIR) as fd:
        space = os.fstatvfs(fd)
        need(space.f_bavail * space.f_frsize >= sum(lengths.values()),
             "insufficient disk space for new expert revision")
    with tempfile.TemporaryDirectory(prefix=".expert-safetensors-", dir=dest.parent) as temp:
        work = Path(temp) / "revision"
        work.mkdir(mode=0o700)
        for name in sorted(source.dirs):
            if name:
                (work / name).mkdir(mode=0o700)
        for name, member in members.items():
            write_member(source, member, work / name, edits, rewrites.get(member), replacements.get(name))
            need((work / name).stat().st_size == lengths[name], "expert revision output size differs")
        for name in sorted({"", *source.dirs}, reverse=True):
            with open_path(work / name, OPEN_DIR) as fd:
                os.fsync(fd)
        source.check_current()
        if config_check is not None:
            config_check()
        rename_noreplace(work, dest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    digest = commands.add_parser("source-digest", help="print the physical source SHA-256")
    apply = commands.add_parser("apply", help="publish a new directory using a reviewed expert map")
    for command in (digest, apply):
        command.add_argument("--source", type=Path, required=True, help="indexed HF directory or standalone file")
        command.add_argument("--config", type=Path, help="required for standalone; not for indexed HF")
        command.add_argument("--config-sha256", help="SHA-256 of the exact standalone config bytes")
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument("--plan-sha256", required=True, help="SHA-256 of the exact reviewed map bytes")
    apply.add_argument("--output", type=Path, required=True, help="new output directory, never overwritten")
    args = parser.parse_args(argv)
    try:
        source = Source(args.source)
        config_check = None
        if source.directory:
            need(args.config is None and args.config_sha256 is None,
                 "indexed HF config must come from the pinned source directory")
            config_raw = source.metadata(CONFIG)
        else:
            need(args.config is not None and args.config_sha256 is not None,
                 "standalone SafeTensors requires --config and --config-sha256")
            config_raw, checked = pinned(args.config, args.config_sha256, "standalone config")

            def config_check():
                need(pinned(args.config, args.config_sha256, "standalone config")[1] == checked,
                     "standalone config changed since verification")

        if args.command == "source-digest":
            digest = source.verify()
            if config_check is not None:
                config_check()
            print(digest)
            return 0

        raw, _ = pinned(args.plan, args.plan_sha256, "reviewed plan")
        plan = parse_json(raw)
        need(isinstance(plan, dict) and {"schema", "source_sha256", "layers"} <= plan.keys()
             and plan["schema"] == SCHEMA, f"expected a {SCHEMA} plan")
        expected = sha256(plan["source_sha256"], "plan source")
        config, layers, experts, top_k, hidden, inner = flash_config(config_raw)
        remaining = plan_layers(plan, layers, experts, top_k)
        source.verify(expected)
        groups = source.group_shards()
        edits = expert_edits(plan, groups, layers, experts, hidden, inner)
        if remaining != experts:
            text = config["text_config"]
            for obj in (config, text, text["mtp"]):
                if "num_experts" in obj:
                    obj["num_experts"] = remaining
            config_raw = json_bytes(config)
        publish(source, args.output, config_raw, config_check, groups, edits, remaining != experts)
        print(args.output)
        return 0
    except (OSError, ValueError, RecursionError, struct.error) as exc:
        print(f"expert-safetensors: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
