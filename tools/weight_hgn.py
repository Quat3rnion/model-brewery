"""Bounded native HGN v2 layout and checksum validation for offline edits."""
from dataclasses import dataclass
import math
import struct

from cross_container.ggml_in_hgn import ENTRY, HEADER, MAX_ELEMENTS, MAX_TENSORS, align, xor32

FLOATS = {(0, 0): ("BF16", 30, 2), (1, 0): ("F32", 0, 4), (2, 0): ("F16", 1, 2)}
PACKED = {(5, 2): "Q4CP", (7, 0): "Q8G64", (24, 64): "Q6G64", (10, 0): "FP8G128",
          (16, 0x1208): "HT4", (16, 0x1206): "HT3", (23, 128): "I4R"}
ROTATED = {"HT4", "HT3", "I4R"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def ascii_field(raw):
    value = raw.rstrip(b"\0")
    require(value and len(value) < len(raw) and all(32 <= c <= 126 for c in value),
            "invalid HGN name or identity")
    return value.decode("ascii")


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    offset: int
    size: int
    checksum_offset: int
    dtype: str
    ggml_type: int | None
    reason: str | None


def auxiliary(name):
    return name.endswith((".suh", ".svh", ".mlp.experts.su"))


def packed_size(label, shape, name):
    n, k = math.prod(shape[:-1]), shape[-1]
    if label == "Q4CP":
        require(k % 32 == 0, "invalid HGN Q4CP row")
        return 64 + align(n * k // 2) + n * align(k // 16, 16)
    if label in {"Q8G64", "Q6G64"}:
        require(k % 64 == 0, "invalid HGN group row")
        return n * (k + k // 16 if label == "Q8G64" else align(k // 16 * 13, 16))
    if label == "FP8G128":
        require(name == "layers.1.ple.ngram_embedding.weight" and len(shape) == 3 and k == 160,
                "invalid HGN FP8G128 role or shape")
        return align(n * k) + 4
    require(name.endswith(".weight") and len(shape) in (2, 3) and
            shape[-2] % 128 == 0 and k % 128 == 0, "invalid HGN rotated role or shape")
    if label == "I4R":
        require(len(shape) == 3, "invalid HGN I4R expert shape")
        return n * k // 2 + n * k // 64
    require(len(shape) == 2 or shape in ((512, 1280, 2560), (512, 2560, 640)),
            "invalid native HT expert shape")
    return n * k * (4 if label == "HT4" else 3) // 8


def check_rotation(tensors):
    by_name = {tensor.name: tensor for tensor in tensors}
    used = set()
    for tensor in tensors:
        if tensor.dtype not in ROTATED:
            continue
        base = tensor.name.removesuffix(".weight")
        expert = len(tensor.shape) == 3
        if expert:
            require(base.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")),
                    "invalid HGN rotated expert role")
        gate_up = expert and base.endswith(".gate_up_proj")
        su = base.removesuffix(".gate_up_proj") + ".su" if gate_up else base + ".suh"
        if su not in by_name:
            for suffix in (".shared_expert.gate_proj", ".shared_expert.up_proj"):
                if base.endswith(suffix):
                    su = base.removesuffix(suffix) + ".experts.su"
        ht_experts = tensor.shape[0] if expert and tensor.dtype != "I4R" else 1
        for companion, size in ((su, tensor.shape[-1] * (1 if gate_up else ht_experts)),
                                (base + ".svh", tensor.shape[-2] * ht_experts)):
            vector = by_name.get(companion)
            require(vector is not None and auxiliary(companion) and vector.dtype == "F16"
                    and math.prod(vector.shape) == size, f"invalid HGN rotation companion: {companion}")
            used.add(companion)
    require({t.name for t in tensors if auxiliary(t.name)} == used,
            "unreferenced HGN rotation companion")


def inventory(data):
    require(len(data) >= HEADER.size, "truncated HGN header")
    magic, version, count, table, base, size, identity = HEADER.unpack_from(data)
    require(magic == b"HGN1" and version == 2, "expected native HGN version 2")
    ascii_field(identity)
    require(0 < count <= MAX_TENSORS and table == HEADER.size and size == len(data),
            "invalid HGN header")
    table_end = table + count * ENTRY.size
    require(base == align(table_end) and base < size and not any(data[table_end:base]),
            "invalid HGN table extent or padding")
    tensors, names = [], set()
    end = table_end
    start = base
    for i in range(count):
        entry = table + i * ENTRY.size
        raw_name, store, rank, *fields = ENTRY.unpack_from(data, entry)
        name = ascii_field(raw_name)
        require(name not in names, "duplicate HGN tensor")
        names.add(name)
        dims = tuple(fields[:4])
        offset, length, checksum, variant = fields[4:]
        require(1 <= rank <= 4 and all(x > 0 for x in dims[:rank]) and not any(dims[rank:]),
                "invalid HGN tensor shape")
        shape = dims[:rank]
        elements = math.prod(shape)
        require(elements <= MAX_ELEMENTS and offset == start and 0 < length <= size - offset,
                "invalid HGN tensor extent")
        require(not any(data[end:offset]), "nonzero HGN alignment padding")
        key, ggml = (store, variant), None
        if key in FLOATS:
            dtype, ggml, width = FLOATS[key]
            wanted = elements * width
            if auxiliary(name):
                require(dtype == "F16", "HGN rotation companion must use F16")
            reason = "Rotation companion is read-only" if auxiliary(name) else (
                None if name.endswith(".weight") else "Non-weight HGN scalar is read-only")
        elif key == (4, 0):
            dtype, wanted, reason = "I64", elements * 8, "Integer metadata is read-only"
        else:
            dtype = PACKED.get(key)
            require(dtype is not None, f"unsupported native HGN store/variant {store}/{variant}")
            require(not auxiliary(name), "HGN rotation companion must use F16")
            wanted = packed_size(dtype, shape, name)
            reason = "Packed HGN store is read-only"
        require(length == wanted, f"HGN tensor size mismatch: {name}")
        end = offset + length
        require(xor32(data[offset:end]) == checksum, f"HGN checksum mismatch: {name}")
        tensors.append(Tensor(name, shape, offset, length, entry + 152, dtype, ggml, reason))
        start = align(end)
    require(size in (end, start) and not any(data[end:]), "invalid HGN trailing extent or padding")
    check_rotation(tensors)
    return tensors


def repair(data, tensors, patches):
    """Return checksum changes for edited tensors only; never alter a payload."""
    changes = {}
    for tensor in tensors:
        edits = sorted((offset, raw) for offset, raw in patches.items()
                       if tensor.offset <= offset < tensor.offset + tensor.size)
        if not edits:
            continue
        payload = bytearray(data[tensor.offset:tensor.offset + tensor.size])
        for offset, raw in edits:
            i = offset - tensor.offset
            payload[i:i + len(raw)] = raw
        changes[tensor.checksum_offset] = struct.pack("<I", xor32(payload))
    return changes
