#!/usr/bin/env python3
"""Bounded, read-only inspection of Agention APR Gyro GGUF v3 (explicit V1 rotation).

Disk IDs follow ik_llama.cpp 7c00f8788e303bfba2886138e288effb0fd5fbbf.
This qualifies format only, not producer authenticity, file integrity, architecture
completeness, runtime execution, or eligibility for conversion.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
import sys

if __package__:
    from .expert_gguf import HeaderCursor, MAX_FILE, MAX_HEADER, MAX_TENSORS, Tensor, identity, regular_file
    from .cross_container.ggml_in_hgn import align, need
else:
    sys.dont_write_bytecode = True
    from expert_gguf import HeaderCursor, MAX_FILE, MAX_HEADER, MAX_TENSORS, Tensor, identity, regular_file
    from cross_container.ggml_in_hgn import align, need


IK_PIN = "7c00f8788e303bfba2886138e288effb0fd5fbbf"
# ggml_init_from_file remaps these external IDs before computing row sizes.
# Entries: internal ID, name, bytes per 128 weights, bits per circular step.
TRELLIS = {
    144: (164, "TQ2_T", 34, 0),
    145: (162, "TQK6", 26, 6),
    146: (163, "TQK7", 30, 7),
    147: (165, "TQP5", 42, 5),
    148: (166, "TQP6", 50, 6),
    149: (167, "TQP7", 58, 7),
}
# Ordinary types observed in the published qwen4exp Gyro S/M/L files:
# F32, F16, Q8_0, Q5_K, Q6_K, IQ4_NL, BF16.
OTHER = {0: (1, 4), 1: (1, 2), 8: (32, 34), 13: (256, 176),
         14: (256, 210), 20: (32, 18), 30: (1, 2)}
PROFILE_TYPES = {"S": {145, 146}, "M": {144}, "L": {147, 148, 149}}
ENCODER = re.compile(r"Agention rotor encoder \(APR\), Gyro-([SML]) release [0-9]{4}-[0-9]{2}-[0-9]{2}")
ROTATION = "prism.hadamard."
CODEBOOK_BYTES = 8192
# Normative LE binary16 table from Agention Gyro SPEC.md v1.1, section 4.
# Decode `_CODEBOOK_B64` in Agention's `python/tq_ref.py` at revision
# 587a5269848eb07996c9c1433f9ad638604eb4d8 to get the raw input; no model
# or reference code execution is required to extract the table.
CODEBOOK_SHA = "ee481f151d8947ce8b08817fdf89d0a56d647b6435ceb3df0f079ea748e0e588"


@dataclass(frozen=True)
class Layout:
    tensors: dict[str, Tensor]
    data_start: int
    data_end: int
    profile: str
    encoder: str
    sign_widths: tuple[int, ...]
    rotated_count: int


def expect(fields, key, kind):
    actual, data = fields.get(key, (None, None))
    need(actual == kind, f"missing or incorrectly typed metadata: {key}")
    return HeaderCursor(data)


def text(fields, key):
    return expect(fields, key, 8).string()


def uint32(fields, key):
    return expect(fields, key, 4).unpack("I")[0]


def typed_array(fields, key, subtype, limit):
    cursor = expect(fields, ROTATION + key, 9)
    actual, count = cursor.unpack("IQ")
    need(actual == subtype and 0 < count <= limit, f"invalid {ROTATION}{key}")
    return cursor, count


def rotation(fields, tensors, trellis_names):
    fixed = {"version": (4, 1), "block_size": (4, 128),
             "transform": (8, "normalized-sylvester-walsh-hadamard"),
             "axis": (8, "input-last-dimension"), "sign_mode": (8, "explicit")}
    allowed = fixed.keys() | {"sign_widths", "sign_values", "weight_names"}
    for key in fields:
        if key.startswith(ROTATION):
            need(key[len(ROTATION):] in allowed, f"unsupported rotation metadata: {key}")
    for suffix, (kind, value) in fixed.items():
        key = ROTATION + suffix
        actual = (uint32 if kind == 4 else text)(fields, key)
        need(actual == value, f"unsupported {key}: {actual!r}")

    cursor, count = typed_array(fields, "sign_widths", 5, MAX_TENSORS)
    widths = cursor.unpack("i" * count)
    need(all(w > 0 and w % 128 == 0 for w in widths) and len(set(widths)) == len(widths),
         "invalid or duplicate prism.hadamard.sign_widths")
    cursor, count = typed_array(fields, "sign_values", 5, MAX_HEADER // 4)
    need(sum(widths) == count and all(x in (-1, 1) for x, in struct.iter_unpack("<i", cursor.take(count * 4))),
         "prism.hadamard.sign_values must be exactly one +/-1 vector per width")
    cursor, count = typed_array(fields, "weight_names", 8, MAX_TENSORS)
    names = {cursor.string(63) for _ in range(count)}
    need(len(names) == count and names <= tensors.keys() and trellis_names <= names,
         "invalid prism.hadamard.weight_names (duplicate, unknown or missing trellis weight)")
    need({tensors[name].dims[0] for name in names} == set(widths),
         "prism.hadamard.sign_widths do not match the named weight dimensions")
    return widths, count


def read_layout(prefix, size):
    """Validate bounded header bytes against a logical file size; no payload read."""
    need(len(prefix) <= MAX_HEADER and 24 <= size <= MAX_FILE, "invalid GGUF size")
    c = HeaderCursor(prefix)
    magic, version, count, field_count = c.unpack("4sIQQ")
    need(magic == b"GGUF" and version == 3, "expected little-endian GGUF v3")
    need(0 < count <= MAX_TENSORS and field_count <= MAX_TENSORS, "invalid GGUF header counts")
    fields = {}
    for _ in range(field_count):
        key = c.string(1024)
        need(key.isascii() and key.isprintable() and key not in fields, "invalid or duplicate GGUF key")
        need(not key.startswith("split."), "sharded GGUF is unsupported")
        kind, = c.unpack("I")
        pos = c.pos
        c.kv(kind)
        fields[key] = (kind, c.data[pos:c.pos])
    need(uint32(fields, "agention.trellis.state_bits") == 15, "unsupported agention.trellis.state_bits")
    for key, value in (("general.architecture", "qwen4exp"), ("general.type", "model"),
                       ("general.quantized_by", "Agention")):
        need(text(fields, key) == value, f"unsupported {key}")
    encoder = text(fields, "agention.encoder")
    match = ENCODER.fullmatch(encoder)
    need(match is not None, "unsupported agention.encoder")
    profile = match[1]
    alignment = uint32(fields, "general.alignment") if "general.alignment" in fields else 32
    need(0 < alignment <= 4096 and alignment & (alignment - 1) == 0, "invalid GGUF alignment")
    tensors, trellis_names = {}, set()
    offset = 0
    for _ in range(count):
        name = c.string(63)
        need(name.isascii() and name.isprintable() and name not in tensors, "invalid or duplicate GGUF tensor name")
        rank, = c.unpack("I")
        need(1 <= rank <= 4, f"invalid GGUF rank: {name}")
        dims = c.unpack("Q" * rank)
        kind, relative = c.unpack("IQ")
        if kind in TRELLIS:
            need(kind in PROFILE_TYPES[profile], f"unexpected Gyro-{profile} disk ID {kind}: {name}")
            block, width = 128, TRELLIS[kind][2]
            trellis_names.add(name)
        else:
            need(kind in OTHER, f"unsupported non-Gyro disk ID {kind}: {name}")
            block, width = OTHER[kind]
        elements = math.prod(dims)
        need(0 < elements <= MAX_FILE and dims[0] % block == 0, f"invalid packed dimensions: {name}")
        need(relative == offset, f"invalid GGUF tensor offset: {name}")
        size_bytes = elements // block * width
        tensors[name] = Tensor(name, dims, kind, relative, size_bytes)
        offset += align(size_bytes, alignment)
        need(offset <= MAX_FILE, "GGUF tensor size overflow")
    need(trellis_names, "no remapped Gyro disk IDs")
    base = align(c.pos, alignment)
    need(base <= len(prefix), "truncated or oversized GGUF header")
    end = base + relative + size_bytes
    need(end <= size <= base + offset, "GGUF tensor data truncated or has extra bytes")
    widths, rotated_count = rotation(fields, tensors, trellis_names)
    return Layout(tensors, base, end, profile, encoder, widths, rotated_count)


def read_codebook(path):
    with regular_file(path) as file:
        raw = file.read(CODEBOOK_BYTES + 1)
    need(len(raw) == CODEBOOK_BYTES and hashlib.sha256(raw).hexdigest() == CODEBOOK_SHA,
         "invalid Gyro codebook (require the normative 8192-byte LE binary16 table)")
    return struct.unpack("<4096e", raw)


def decode_block(kind, block, codebook):
    """Return the 128 values in the stored (rotated) basis, without inverse rotation."""
    _, _, width, bits = TRELLIS[kind]
    need(len(block) == width, "truncated Gyro block")
    scale, = struct.unpack("<e", block[:2])
    need(math.isfinite(scale), "non-finite Gyro block scale")
    payload = block[2:]
    steps = 32 if kind <= 146 else 64
    if bits == 0:
        states = ((payload[t - 1] << 8) | payload[t] for t in range(steps))
    else:
        # Duplicate the stream to make the circular window cross the end.
        stream = int.from_bytes(payload * 2, "little")
        states = (stream >> ((steps - 1 - t) * bits) for t in range(steps))
    values = []
    pairs = 2 if steps == 32 else 1
    for state in states:
        hashed = ((state & 0x7fff) * 0x9e3779b1) & 0xffffffff
        for index in (hashed >> 21, (hashed >> 10) & 2047)[:pairs]:
            values.extend((scale * codebook[2 * index], scale * codebook[2 * index + 1]))
    return values


def inspect_file(path, tensor=None, block=0, codebook_path=None):
    with regular_file(path) as file:
        before = identity(file)
        layout = read_layout(file.read(min(before[2], MAX_HEADER)), before[2])
        file.seek(layout.data_end)
        need(file.read(before[2] - layout.data_end) == bytes(before[2] - layout.data_end),
             "invalid GGUF trailing padding")
        counts = Counter(t.kind for t in layout.tensors.values() if t.kind in TRELLIS)
        out = {"family": "agention-gyro-gguf", "evidence": "format-identified", "ik_pin": IK_PIN,
               "profile": f"Gyro-{layout.profile}", "encoder_metadata": layout.encoder,
               "data_offset": layout.data_start, "tensor_count": len(layout.tensors),
               "rotation": {"version": 1, "block_size": 128, "sign_widths": layout.sign_widths,
                            "weight_count": layout.rotated_count},
               "trellis_types": [{"disk_id": kind, "ik_id": TRELLIS[kind][0], "type": TRELLIS[kind][1], "count": count}
                                 for kind, count in sorted(counts.items())]}
        if tensor is not None:
            spec = layout.tensors.get(tensor)
            need(spec is not None and spec.kind in TRELLIS, "expected a named Gyro tensor")
            width = TRELLIS[spec.kind][2]
            need(0 <= block < spec.size // width, "block index outside Gyro tensor")
            offset = layout.data_start + spec.relative + width * block
            file.seek(offset)
            raw = file.read(width)
            values = decode_block(spec.kind, raw, read_codebook(codebook_path))
            out["sample"] = {"tensor": tensor, "block": block, "byte_offset": offset,
                             "packed_sha256": hashlib.sha256(raw).hexdigest(),
                             "basis": "stored-rotated", "values": values}
        need(identity(file) == before, "GGUF file changed during read")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("--tensor", help="decode only one 128-weight block of this tensor")
    parser.add_argument("--block", type=int, help="flat block index (default 0)")
    parser.add_argument("--codebook", help="path to the normative 8192-byte LE binary16 table")
    args = parser.parse_args()
    if bool(args.tensor) != bool(args.codebook) or (args.tensor is None and args.block is not None):
        parser.error("--tensor and --codebook are required together; --block requires both")
    try:
        print(json.dumps(inspect_file(args.gguf, args.tensor, args.block or 0, args.codebook), allow_nan=False))
    except (ValueError, OSError, struct.error) as exc:
        parser.exit(1, f"gyro_gguf: {exc}\n")


if __name__ == "__main__":
    main()
