#!/usr/bin/env python3
"""Inspect, propose, and apply bounded scalar edits to pinned GGUF v3 or HGN v2.

Use NumPy and gguf-py from the pinned llama.cpp checkout. Coordinates are
zero-based in logical row-major shape order (the last index varies fastest).
Apply consumes an explicit proposal and publishes a new file without clobber.
"""
import argparse
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
from types import MappingProxyType

import gguf
import numpy as np

from cross_container.ggml_in_hgn import gguf_preamble, read_file
import weight_ggml_codes
import weight_hgn
import weight_q8g64

SCHEMA = "brewery.weight-edit-proposal.v1"
MAX_JSON = 128 * 1024
MAX_EDITS = 256
DTYPES = {int(kind): (kind, width) for kind, width in
          [(gguf.GGMLQuantizationType.F32, 4), (gguf.GGMLQuantizationType.F16, 2),
           (gguf.GGMLQuantizationType.BF16, 2)]}
GGUF_LAYOUTS = {int(kind): layout for kind, layout in gguf.GGML_QUANT_SIZES.items()}
PACKED = {"Q4_0", "Q8_0", "Q8G64"}
PACKED_DISABLED = "Packed weight editing is disabled for this worker"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (float, int), "expected a finite numeric value")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("expected a finite numeric value") from exc
    require(math.isfinite(value), "expected a finite numeric value")
    return value


def encode_value(dtype, value):
    value = number(value)
    try:
        if dtype == "F32":
            return struct.pack("<f", value)
        if dtype == "F16":
            return struct.pack("<e", value)
        require(dtype == "BF16", "unsupported source dtype")
        # Round binary64 directly to BF16, ties to even. Casting through
        # F32 can erase the difference between a tie and a value just above it.
        absolute = abs(value)
        exponent = max(math.frexp(absolute)[1] - 1, -126)
        if exponent > 127:
            raise OverflowError
        rounded = math.ldexp(round(math.ldexp(absolute, 7 - exponent)), exponent - 7)
        rounded = math.copysign(rounded, value)
        return struct.pack("<f", rounded)[2:]
    except OverflowError as exc:
        raise ValueError(f"value is outside finite {dtype} range") from exc


def decode_value(dtype, b):
    if dtype == "BF16":
        return struct.unpack("<f", b"\0\0" + bytes(b))[0]
    return struct.unpack("<f" if dtype == "F32" else "<e", b)[0]


def round_value(dtype, value):
    """Round to a finite representable value in the given source dtype."""
    result = decode_value(dtype, encode_value(dtype, value))
    require(math.isfinite(result), f"value is outside finite {dtype} range")
    return result


class Source:
    source_format = "gguf"

    def __init__(self, data, source_sha256, *, validate_values=True):
        _, tensors, _ = gguf_preamble(data, len(data), layouts=GGUF_LAYOUTS)
        self._init(tensors, data, source_sha256, validate_values)

    def _init(self, tensors, data, source_sha256, validate_values):
        self._data = data
        self.source_sha256 = source_sha256
        self.packed_enabled = os.environ.get("BREWERY_WEIGHT_PACKED", "1") == "1"
        self.tensors = MappingProxyType({t.name: t for t in tensors})
        for tensor in tensors if validate_values else ():
            if tensor.ggml_type not in DTYPES and self.dtype(tensor.name) not in PACKED:
                continue
            if self.dtype(tensor.name) in PACKED and not self.packed_enabled:
                continue
            # Bound temporary arrays by validating a tensor at a time.
            require(bool(np.isfinite(self._values(tensor)).all()),
                    f"source contains non-finite values in {tensor.name}")

    def tensor(self, name):
        require(isinstance(name, str) and name in self.tensors, "unknown tensor")
        return self.tensors[name]

    def dtype(self, name):
        return gguf.GGMLQuantizationType(self.tensor(name).ggml_type).name

    def reason(self, tensor):
        if self.dtype(tensor.name) in PACKED:
            return None if self.packed_enabled else PACKED_DISABLED
        if tensor.ggml_type in DTYPES:
            return None
        return "Unsupported quantized or non-floating GGUF store is read-only"

    def _check_editable(self, tensor):
        reason = self.reason(tensor)
        require(reason is None, reason)

    def _packed_cell(self, tensor, flat):
        dtype = self.dtype(tensor.name)
        if dtype == "Q8G64":
            cols = tensor.shape[-1]
            row, col = divmod(flat, cols)
            raw = self._data[tensor.offset:tensor.offset + tensor.size]
            return (raw, math.prod(tensor.shape[:-1]), cols, row, col)
        block, cell = divmod(flat, weight_ggml_codes.BLOCK_CELLS)
        size = weight_ggml_codes.block_size(dtype)
        offset = tensor.offset + block * size
        return offset, bytes(self._data[offset:offset + size]), dtype, cell

    def candidates(self, name, index):
        tensor, flat = self._position(name, index)
        dtype = self.dtype(name)
        if dtype == "Q8G64":
            return tuple(edit.after for edit in weight_q8g64.candidates(*self._packed_cell(tensor, flat)))
        if dtype in PACKED:
            _, *args = self._packed_cell(tensor, flat)
            return tuple(edit.cell_after.value for edit in weight_ggml_codes.candidates(*args))
        return None

    def format_fields(self):
        return {"source_format": "hgn"} if self.source_format == "hgn" else {}

    def patch_checksums(self, patches):
        return {}

    def validate_written(self, path):
        pass

    def _values(self, tensor):
        dtype = self.dtype(tensor.name)
        if dtype == "Q8G64":
            # Fixed-scale native affine rows: code bytes then F16 scale/bias pairs.
            cols = tensor.shape[-1]
            stride = cols + cols // 16
            raw = np.frombuffer(self._data[tensor.offset:tensor.offset + tensor.size],
                                dtype=np.uint8).reshape(-1, stride)
            values = raw[:, :cols].reshape(-1, cols // 64, 64).astype(np.float64)
            meta = raw[:, cols:].copy().view("<f2").reshape(-1, cols // 64, 2).astype(np.float64)
            require(bool(np.isfinite(meta).all() and (meta[:, :, 0] >= 0).all()),
                    "Q8G64 needs finite nonnegative scales and finite bias")
            return (values * meta[:, :, 0, None] + meta[:, :, 1, None]).astype(
                np.float32).reshape(tensor.shape)
        if dtype in PACKED:
            kind = gguf.GGMLQuantizationType(tensor.ggml_type)
            width = weight_ggml_codes.block_size(dtype)
            raw = np.frombuffer(self._data[tensor.offset:tensor.offset + tensor.size], dtype=np.uint8)
            return gguf.dequantize(raw.reshape(-1, width), kind).reshape(tensor.shape)
        kind, width = DTYPES[tensor.ggml_type]
        raw = np.frombuffer(self._data[tensor.offset:tensor.offset + tensor.size], dtype=np.uint8)
        return gguf.dequantize(raw.reshape(-1, width), kind).reshape(tensor.shape)

    def matrix(self, name):
        """Return a read-only F32 matrix for the offline proposal generator."""
        tensor = self.tensor(name)
        self._check_editable(tensor)
        require(len(tensor.shape) == 2, "expected rank-2 tensor")
        matrix = self._values(tensor)
        matrix.setflags(write=False)
        return matrix

    def _position(self, name, index):
        tensor = self.tensor(name)
        self._check_editable(tensor)
        require(type(index) is list and len(index) == len(tensor.shape),
                "index rank does not match tensor shape")
        flat = 0
        for coordinate, dimension in zip(index, tensor.shape):
            require(type(coordinate) is int and 0 <= coordinate < dimension,
                    "tensor index out of bounds")
            flat = flat * dimension + coordinate
        return tensor, flat

    def scalar(self, name, index):
        tensor, flat = self._position(name, index)
        dtype = self.dtype(name)
        if dtype == "Q8G64":
            state = weight_q8g64.read_cell(*self._packed_cell(tensor, flat))
            return dtype, tensor.offset + state.offset, state.value
        if dtype in PACKED:
            offset, *args = self._packed_cell(tensor, flat)
            return dtype, offset, weight_ggml_codes.read_cell(*args).value
        kind, width = DTYPES[tensor.ggml_type]
        offset = tensor.offset + flat * width
        old = self._data[offset:offset + width]
        return kind.name, offset, decode_value(kind.name, old)

    def edit(self, name, index, op, value, patches):
        dtype, offset, before = self.scalar(name, index)
        if dtype in PACKED:
            tensor, flat = self._position(name, index)
            if dtype == "Q8G64":
                edit = weight_q8g64.plan_patch(*self._packed_cell(tensor, flat), op, value)
                return dtype, offset, before, edit.after, bytes([edit.new_code])
            _, block, _, cell = self._packed_cell(tensor, flat)
            # Two Q4_0 cells may share a byte. Compose with the prior nibble
            # patch without changing the other nibble or the scale.
            composed = b"".join(patches.get(offset + i, bytes([byte]))
                                for i, byte in enumerate(block))
            edit = weight_ggml_codes.edit_cell(composed, dtype, cell, operation=op, value=value)
            return (dtype, offset + edit.byte_index, before, edit.cell_after.value,
                    bytes([edit.block[edit.byte_index]]))
        encoded = encode_value(dtype, value if op == "set" else before + value)
        return dtype, offset, before, decode_value(dtype, encoded), encoded

    def inventory(self, offset, limit):
        require(0 <= offset <= len(self.tensors) and 1 <= limit <= 128, "invalid inventory page")
        tensors = []
        for t in islice(self.tensors.values(), offset, offset + limit):
            row = {"name": t.name, "shape": t.shape, "dtype": self.dtype(t.name),
                   "elements": math.prod(t.shape)}
            reason = self.reason(t)
            if self.source_format == "hgn" or reason or row["dtype"] in PACKED:
                row["editable"] = reason is None
            if reason is not None:
                row["reason"] = reason
            tensors.append(row)
        end = offset + len(tensors)
        return {"mode": "inventory", "source_sha256": self.source_sha256, "offset": offset,
                "total": len(self.tensors), "next_offset": end if end < len(self.tensors) else None,
                "tensors": tensors, **self.format_fields()}

    def tile(self, name, origin, shape):
        tensor = self.tensor(name)
        rank = len(tensor.shape)
        require(isinstance(origin, list) and isinstance(shape, list)
                and len(origin) == len(shape) == rank
                and all(type(i) is int and type(n) is int and i >= 0 and 1 <= n <= dim - i
                        for i, n, dim in zip(origin, shape, tensor.shape)), "invalid tile bounds or rank")
        require((shape[0] <= 1024 if rank == 1 else
                 all(n == 1 for n in shape[:-2]) and max(shape[-2:]) <= 32), "tile exceeds size limit")
        rows, cols = (1, shape[0]) if rank == 1 else shape[-2:]
        values, steps = [], []
        packed = self.dtype(name) in PACKED
        for r in range(rows):
            row, step_row = [], []
            for c in range(cols):
                index = origin.copy()
                index[-1] += c
                if rank > 1:
                    index[-2] += r
                value = self.scalar(name, index)[2]
                require(math.isfinite(value), "tile contains non-finite stored values")
                row.append(value)
                if packed:
                    levels = self.candidates(name, index)
                    step_row.append({"lower": max((v for v in levels if v < value), default=None),
                                     "upper": min((v for v in levels if v > value), default=None)})
            values.append(row)
            steps.append(step_row)
        return {"mode": "tile", "source_sha256": self.source_sha256, "tensor": name,
                "dtype": self.dtype(name), "shape": tensor.shape, "origin": origin,
                "tile_shape": shape, "values": values,
                **({"packed_steps": steps} if packed else {}), **self.format_fields()}


class NativeHgn(Source):
    source_format = "hgn"

    def __init__(self, data, source_sha256, *, validate_values=True):
        self._init(weight_hgn.inventory(data), data, source_sha256, validate_values)

    def dtype(self, name):
        return self.tensor(name).dtype

    def reason(self, tensor):
        if tensor.dtype == "Q8G64":
            if not tensor.name.endswith(".weight"):
                return "Non-weight HGN store is read-only"
            return None if self.packed_enabled else PACKED_DISABLED
        return tensor.reason

    def patch_checksums(self, patches):
        return weight_hgn.repair(self._data, self.tensors.values(), patches)

    def validate_written(self, path):
        require(list(self.tensors.values()) == weight_hgn.inventory(read_file(path)),
                "edited HGN layout changed")


def load_source(path, source_sha256, *, validate_values=True):
    """Load an immutable bounded source snapshot after its full hash matches."""
    require(sys.byteorder == "little", "weight editing requires a little-endian CPU")
    require(isinstance(source_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", source_sha256),
            "expected lowercase source SHA-256")
    data = read_file(path)
    require(hashlib.sha256(data).hexdigest() == source_sha256, "source SHA-256 mismatch")
    reader = NativeHgn if data[:4] == b"HGN1" else Source
    return reader(data, source_sha256, validate_values=validate_values)


def _prepare(source, edits):
    require(type(edits) is list and 1 <= len(edits) <= MAX_EDITS,
            f"expected 1 to {MAX_EDITS} scalar edits")
    results = []
    patches, visited = {}, set()
    for edit in edits:
        require(type(edit) is dict and set(edit) == {"tensor", "index", "op", "value"},
                "unexpected scalar edit fields")
        tensor, index, op = edit["tensor"], edit["index"], edit["op"]
        require(op in ("set", "delta"), "expected set or delta")
        value = number(edit["value"])
        source._position(tensor, index)
        key = tensor, tuple(index)
        require(key not in visited, "duplicate tensor coordinate")
        visited.add(key)
        dtype, offset, before, after, after_bytes = source.edit(tensor, index, op, value, patches)
        require(math.isfinite(after), f"value is outside finite {dtype} range")
        require(after != before, "edit is a no-op after dtype rounding")
        patches[offset] = after_bytes
        results.append({"tensor": tensor, "index": index.copy(), "op": op, "value": value,
                        "dtype": dtype, "before": before, "after": after})
    return {"schema": SCHEMA, "source_sha256": source.source_sha256, "edits": results}, patches


def propose(source, edits):
    return _prepare(source, edits)[0]


def read_json(path):
    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), "duplicate JSON key")
        return result

    raw_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(raw_fd, "rb") as f:
        meta = os.fstat(f.fileno())
        require(stat.S_ISREG(meta.st_mode) and 0 < meta.st_size <= MAX_JSON,
                "expected a bounded regular JSON file")
        raw = f.read(MAX_JSON + 1)
        require(0 < len(raw) == meta.st_size, "JSON changed during read")
    return json.loads(raw, object_pairs_hook=unique)


def apply(source_path, proposal, destination):
    require(type(proposal) is dict and set(proposal) == {"schema", "source_sha256", "edits"}
            and proposal["schema"] == SCHEMA, "invalid proposal schema")
    source = load_source(source_path, proposal["source_sha256"])
    edits = proposal["edits"]
    identity = {"tensor", "index", "op", "value"}
    require(type(edits) is list and all(type(e) is dict and
            set(e) == identity | {"dtype", "before", "after"} for e in edits),
            "invalid proposal edit fields")
    recalculated, patches = _prepare(source, [{k: e[k] for k in identity} for e in edits])
    for actual, expected in zip(edits, recalculated["edits"]):
        require(actual["dtype"] == expected["dtype"] and all(
            struct.pack("<d", number(actual[k])) == struct.pack("<d", expected[k])
            for k in ("before", "after")), "proposal before/after or dtype does not match")

    edit_count = len(edits)
    patches.update(source.patch_checksums(patches))
    destination = Path(destination)
    fd, temp = tempfile.mkstemp(prefix=".weight-edit-", dir=destination.parent)
    try:
        regions = []
        with os.fdopen(fd, "w+b") as stream:
            start = 0
            for offset, encoded in sorted(patches.items()):
                regions.append(source._data[start:offset])
                regions.append(encoded)
                start = offset + len(encoded)
            regions.append(source._data[start:])
            for region in regions:
                stream.write(region)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            result_sha = hashlib.sha256()
            for region in regions:
                # Compare all bytes, including the header, padding and every
                # untouched region, before publishing the edited copy.
                for i in range(0, len(region), 1 << 20):
                    expected = region[i:i + (1 << 20)]
                    actual = stream.read(len(expected))
                    require(actual == expected, "edited file failed byte verification")
                    result_sha.update(actual)
            require(not stream.read(1), "edited file has unexpected trailing bytes")
        source.validate_written(temp)
        os.link(temp, destination)
    finally:
        os.unlink(temp)
    return {"schema": "brewery.weight-edit-result.v1", "source_sha256": source.source_sha256,
            "output_sha256": result_sha.hexdigest(), "edit_count": edit_count, **source.format_fields()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inventory = commands.add_parser("inventory")
    tile = commands.add_parser("tile")
    make = commands.add_parser("propose")
    write = commands.add_parser("apply")
    for subcommand in (inspect, inventory, tile, make, write):
        subcommand.add_argument("source", type=Path)
    for subcommand in (inspect, inventory, tile, make):
        subcommand.add_argument("--source-sha256", required=True)
    inspect.add_argument("--tensor", required=True)
    inspect.add_argument("--index", required=True, help="JSON array of zero-based row-major indices")
    inventory.add_argument("--offset", type=int, required=True)
    inventory.add_argument("--limit", type=int, required=True)
    tile.add_argument("--tensor", required=True)
    tile.add_argument("--origin", required=True)
    tile.add_argument("--shape", required=True)
    make.add_argument("request", type=Path)
    write.add_argument("proposal", type=Path)
    write.add_argument("destination", type=Path)
    args = cli.parse_args()
    try:
        if args.command == "apply":
            result = apply(args.source, read_json(args.proposal), args.destination)
        else:
            source = load_source(args.source, args.source_sha256,
                                 validate_values=args.command not in ("inventory", "tile"))
            if args.command == "inspect":
                require(len(args.index) <= 256, "index is too large")
                index = json.loads(args.index)
                dtype, _, value = source.scalar(args.tensor, index)
                result = {"source_sha256": source.source_sha256, "tensor": args.tensor,
                          "index": index, "dtype": dtype, "value": value,
                          "shape": source.tensor(args.tensor).shape, **source.format_fields()}
            elif args.command == "inventory":
                result = source.inventory(args.offset, args.limit)
            elif args.command == "tile":
                require(len(args.origin) <= 128 and len(args.shape) <= 128, "tile coordinate too long")
                result = source.tile(args.tensor, json.loads(args.origin), json.loads(args.shape))
            else:
                result = propose(source, read_json(args.request))
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, OSError, struct.error, RecursionError) as exc:
        cli.exit(1, f"weight_edit: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
