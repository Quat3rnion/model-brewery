#!/usr/bin/env python3
"""Lossless, bounded GGUF v3 to HGN tensor shelf; not a runtime model loader."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from types import MappingProxyType

SCHEMA = "brewery.cross-container.v1"
VERSION = 3  # Stock HGN readers accept only versions 1 and 2.
HEADER = struct.Struct("<4sIQQQQ64s")
ENTRY = struct.Struct("<96sII4qQQII")
META_HEADER = struct.Struct("<8sQQ32s")
META_MAGIC = b"GGMLHGN1"
GGML_STORE = 0x42524701
METADATA_STORE = 0x42524702
MAX_FILE = 32 << 20
MAX_METADATA = 4 << 20
MAX_TENSORS = 4096
MAX_ELEMENTS = 8 << 20
MAX_KV_ARRAY = 262144

# Source type: (elements/block, bytes/block, HGN stock store or None).
# GGML block layouts are pinned to llama.cpp 184f6536024668e8908ab8c86033d56708e62b40.
TYPES = {
    0: (1, 4, 1),        # F32
    1: (1, 2, 2),        # F16
    2: (32, 18, None),   # Q4_0
    3: (32, 20, None),   # Q4_1
    6: (32, 22, None),   # Q5_0
    7: (32, 24, None),   # Q5_1
    8: (32, 34, None),   # Q8_0
    10: (256, 84, None),  # Q2_K
    11: (256, 110, None), # Q3_K
    12: (256, 144, None), # Q4_K
    13: (256, 176, None), # Q5_K
    14: (256, 210, None), # Q6_K
    16: (256, 66, None),  # IQ2_XXS
    17: (256, 74, None),  # IQ2_XS
    18: (256, 98, None),  # IQ3_XXS
    19: (256, 50, None),  # IQ1_S
    20: (32, 18, None),   # IQ4_NL
    21: (256, 110, None), # IQ3_S
    22: (256, 82, None),  # IQ2_S
    23: (256, 136, None), # IQ4_XS
    27: (1, 8, 4),        # I64
    29: (256, 56, None),  # IQ1_M
    30: (1, 2, 0),        # BF16
    34: (256, 54, None),  # TQ1_0
    35: (256, 66, None),  # TQ2_0
    39: (32, 17, None),   # MXFP4
    40: (64, 36, None),   # NVFP4
    41: (128, 18, None),  # Q1_0
    42: (64, 18, None),   # Q2_0
}
KV_WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def align(offset, alignment=64):
    return (offset + alignment - 1) & -alignment


def digest(data):
    return hashlib.sha256(data).digest()


def read_file(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        before = os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode), "expected a regular file")
        need(0 < before.st_size <= MAX_FILE, "input exceeds offline file limit or is empty")
        data = f.read(before.st_size + 1)
        after = os.fstat(f.fileno())
        need(len(data) == before.st_size and
             (before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "input changed during read")
    return memoryview(data)


class Cursor:
    def __init__(self, data):
        self.data = data[:MAX_METADATA]
        self.pos = 0

    def take(self, length):
        start = self.pos
        self.pos += length
        need(self.pos <= len(self.data), "truncated or oversized GGUF preamble")
        return self.data[start:self.pos]

    def unpack(self, fmt):
        fmt = struct.Struct("<" + fmt)
        return fmt.unpack(self.take(fmt.size))

    def string(self):
        return bytes(self.take(self.unpack("Q")[0])).decode("utf-8")

    def kv(self, kind):
        if kind == 8:
            self.string()
        elif kind == 9:
            subtype, count = self.unpack("IQ")
            need(count <= MAX_KV_ARRAY, "GGUF array exceeds offline limit")
            if subtype == 8:
                for _ in range(count):
                    self.string()
            else:
                need(subtype in KV_WIDTHS, "unsupported GGUF array subtype")
                self.take(count * KV_WIDTHS[subtype])
        else:
            need(kind in KV_WIDTHS, "unknown GGUF metadata type")
            self.take(KV_WIDTHS[kind])


@dataclass(frozen=True)
class Field:
    gguf_type: int
    raw: bytes  # includes original key, kind, and value bytes


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]  # HGN order (last dimension is the GGML row width)
    ggml_type: int
    offset: int
    size: int

    def storage(self):
        stock = TYPES[self.ggml_type][2]
        return (GGML_STORE, self.ggml_type) if stock is None else (stock, 0)


def gguf_preamble(data, file_size, *, layouts=None):
    if layouts is None:
        layouts = TYPES
    c = Cursor(data)
    magic, version, count, fields = c.unpack("4sIQQ")
    need(magic == b"GGUF" and version == 3, "expected little-endian GGUF v3")
    need(1 <= count <= MAX_TENSORS and fields <= MAX_TENSORS, "invalid GGUF counts")
    metadata = {}
    alignment = 32
    for _ in range(fields):
        start = c.pos
        key = c.string()
        kind, = c.unpack("I")
        value_start = c.pos
        c.kv(kind)
        need(key and key not in metadata, "empty or duplicate GGUF metadata key")
        need(not key.startswith("split."), "sharded GGUF is outside this slice")
        if key == "general.alignment":
            need(kind == 4, "invalid GGUF alignment type")
            alignment, = struct.unpack_from("<I", c.data, value_start)
            need(0 < alignment <= 4096 and not alignment & (alignment - 1),
                 "invalid GGUF alignment")
        metadata[key] = Field(kind, bytes(c.data[start:c.pos]))
    tensors = []
    names = {SCHEMA}
    for _ in range(count):
        name = c.string()
        need(name.isascii() and name.isprintable() and 0 < len(name) <= 95 and name not in names,
             "GGUF name cannot be represented as a unique HGN tensor name")
        names.add(name)
        rank, = c.unpack("I")
        need(1 <= rank <= 4, "unsupported GGUF tensor rank")
        dims = c.unpack("Q" * rank)
        kind, offset = c.unpack("IQ")
        need(kind in layouts, f"unsupported GGML type {kind} for {name}")
        block, width = layouts[kind][:2]
        elements = math.prod(dims)
        need(0 < elements <= MAX_ELEMENTS and dims[0] % block == 0, "invalid GGML block shape")
        need(offset % alignment == 0, "unaligned GGUF tensor")
        tensors.append(Tensor(name, dims[::-1], kind, offset, elements // block * width))
    base = align(c.pos, alignment)
    previous = base
    for tensor in sorted(tensors, key=lambda t: t.offset):
        start = base + tensor.offset
        need(previous <= start and start + tensor.size <= file_size, "invalid GGUF tensor extents")
        previous = start + tensor.size
    tensors = [Tensor(t.name, t.shape, t.ggml_type, base + t.offset, t.size) for t in tensors]
    return metadata, tensors, c.pos


def xor32(payload):
    end = len(payload) & ~3
    xor = int.from_bytes(payload[end:], "little")
    for (word,) in struct.iter_unpack("<I", payload[:end]):
        xor ^= word
    return xor


def hgn_entry(name, shape, store, variant, offset, payload):
    return ENTRY.pack(name.encode("ascii"), store, len(shape), *shape, *([0] * (4 - len(shape))),
                      offset, len(payload), xor32(payload), variant)


def payload_of(data, tensor):
    return data[tensor.offset:tensor.offset + tensor.size]


class Reader:
    """Open versioned HGN and expose named tensor metadata, blocks, and CPU values."""

    def __init__(self, path):
        self._data = data = read_file(path)
        need(len(data) >= HEADER.size, "truncated HGN header")
        magic, version, count, table, base, size, identity = HEADER.unpack_from(data)
        need(magic == b"HGN1" and version == VERSION and identity == SCHEMA.encode().ljust(64, b"\0"),
             "unsupported HGN extension/version")
        need(2 <= count <= MAX_TENSORS + 1 and table == HEADER.size and size == len(data),
             "invalid HGN header")
        table_end = table + count * ENTRY.size
        need(base == align(table_end) and base < size and not any(data[table_end:base]),
             "invalid HGN data offset or padding")
        entries = {}
        offset = base
        end = base
        for i in range(count):
            name, store, rank, *fields = ENTRY.unpack_from(data, table + i * ENTRY.size)
            name = name.rstrip(b"\0").decode("ascii")
            dims = tuple(fields[:4])
            start, length, checksum, variant = fields[4:]
            need(name and name.isprintable() and name not in entries, "invalid HGN tensor name")
            need(1 <= rank <= 4 and all(x > 0 for x in dims[:rank]) and not any(dims[rank:]),
                 "invalid HGN tensor shape")
            end = start + length
            need(start == offset and length > 0 and end <= size, "invalid HGN payload extent")
            offset = align(end)
            entries[name] = (store, variant, dims[:rank], start, length, checksum)
            if i + 1 < count:
                need(not any(data[end:offset]), "nonzero HGN alignment padding")
        need(end == size and next(iter(entries)) == SCHEMA, "invalid HGN extension placement or size")
        meta_store, variant, shape, start, length, checksum = entries.pop(SCHEMA)
        need(meta_store == METADATA_STORE and variant == 0 and shape == (length,) and
             META_HEADER.size <= length <= META_HEADER.size + MAX_METADATA + 32 * MAX_TENSORS,
             "invalid HGN extension metadata tensor")
        meta = data[start:start + length]
        need(xor32(meta) == checksum, "HGN extension metadata checksum mismatch")
        marker, preamble_len, source_size, source_sha = META_HEADER.unpack_from(meta)
        need(marker == META_MAGIC and 24 <= preamble_len <= MAX_METADATA and
             0 < source_size <= MAX_FILE and META_HEADER.size + preamble_len + 32 * len(entries) == length,
             "invalid HGN extension metadata framing")
        self.source_size = source_size
        self.source_sha256 = source_sha.hex()
        self._preamble = meta[META_HEADER.size:META_HEADER.size + preamble_len]
        kv, source, end = gguf_preamble(self._preamble, source_size)
        need(end == preamble_len and list(entries) == [t.name for t in source],
             "HGN/GGUF tensor mapping changed")
        self.metadata = MappingProxyType(kv)
        self._fingerprints = {}
        tensors = {}
        for i, original in enumerate(source):
            store, variant, shape, start, length, checksum = entries[original.name]
            need((store, variant) == original.storage() and shape == original.shape and
                 length == original.size, f"HGN/GGUF tensor mapping changed for {original.name}")
            h = META_HEADER.size + preamble_len + 32 * i
            self._fingerprints[original.name] = (checksum, bytes(meta[h:h + 32]))
            tensors[original.name] = Tensor(original.name, shape, original.ggml_type, start, length)
        self.tensors = MappingProxyType(tensors)

    def encoded(self, name):
        payload = payload_of(self._data, self.tensors[name])
        checksum, sha = self._fingerprints[name]
        need(xor32(payload) == checksum and digest(payload) == sha,
             f"HGN tensor payload integrity mismatch for {name}")
        return bytes(payload)

    def decode(self, name):
        """Return exact scalar I64, or decoded F32 values in HGN shape order."""
        import gguf
        import numpy as np

        need(sys.byteorder == "little", "CPU decode requires a little-endian host")
        tensor = self.tensors[name]
        raw = self.encoded(name)
        if tensor.ggml_type == 27:
            return np.frombuffer(raw, dtype="<i8").reshape(tensor.shape)
        width = tensor.shape[-1] // TYPES[tensor.ggml_type][0] * TYPES[tensor.ggml_type][1]
        encoded = np.frombuffer(raw, dtype=np.uint8).reshape(-1, width)
        return gguf.dequantize(encoded, gguf.GGMLQuantizationType(tensor.ggml_type)).reshape(tensor.shape)


def pack(source, dest):
    data = read_file(source)
    _, tensors, preamble_len = gguf_preamble(data, len(data))
    payloads = [payload_of(data, t) for t in tensors]
    meta = (META_HEADER.pack(META_MAGIC, preamble_len, len(data), digest(data))
            + bytes(data[:preamble_len]) + b"".join(map(digest, payloads)))
    count = len(tensors) + 1
    base = align(HEADER.size + count * ENTRY.size)
    offset = align(base + len(meta))
    entries = [hgn_entry(SCHEMA, (len(meta),), METADATA_STORE, 0, base, meta)]
    end = offset
    for tensor, payload in zip(tensors, payloads):
        entries.append(hgn_entry(tensor.name, tensor.shape, *tensor.storage(), offset, payload))
        end = offset + tensor.size
        offset = align(end)
    need(end <= MAX_FILE, "HGN output exceeds offline file limit")
    dest = Path(dest)
    fd, temporary = tempfile.mkstemp(prefix=".ggml-in-hgn-", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(HEADER.pack(b"HGN1", VERSION, count, HEADER.size, base, end, SCHEMA.encode()))
            f.writelines(entries)
            for payload in (meta, *payloads):
                f.write(b"\0" * (align(f.tell()) - f.tell()))
                f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, dest)
    finally:
        os.unlink(temporary)
    return {"schema": SCHEMA, "tensor_count": count - 1}


def verify(source, dest):
    data = read_file(source)
    _, tensors, preamble_len = gguf_preamble(data, len(data))
    reader = Reader(dest)
    need(reader.source_size == len(data) and reader.source_sha256 == digest(data).hex(),
         "GGUF source identity mismatch")
    need(reader._preamble == data[:preamble_len], "GGUF metadata or original tensor table changed")
    for tensor in tensors:
        need(reader.encoded(tensor.name) == payload_of(data, tensor),
             f"encoded GGML bytes differ for {tensor.name}")
    return {"schema": SCHEMA, "tensor_count": len(tensors)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("pack", "verify"))
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    args = ap.parse_args()
    try:
        print(json.dumps((pack if args.action == "pack" else verify)(args.source, args.dest)))
    except (OSError, ValueError, struct.error) as error:
        print(f"ggml_in_hgn: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
