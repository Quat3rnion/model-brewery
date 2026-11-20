"""Reconstruct one aligned 128x128 EXL3 linear weight block on the CPU.

Reference: ExLlamaV3 v1.6.7 tests/testlib/trellis.py. The rotated basis
has binary16 codebook values. The original basis is a binary64 calculation
of the normalized Hadamards and stored sign/scale factors; it excludes bias.
This does not check the CUDA FP16 reconstruct or reconstruct_had_slice
kernels. Those require the pinned PyTorch extension on a supported CUDA GPU.
See upstream tests/kernels/gemm/test_reconstruct_had.py for the fused
integer-rate comparison (max absolute error / max absolute reference < 2e-3).
"""

import argparse
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

from exl3_inspect import HEADER_BUDGET, MARKERS, UPSTREAM, linear_layout
from weight_safetensors import MAX_BYTES, OPEN_FILE, need, open_path, parse_header, read_at, stamp


BLOCK = 128
TILE = 16


def half_bits(word):
    return struct.unpack("<e", struct.pack("<H", word))[0]


def codebook_value(state, codebook):
    # ExLlamaV3 v1.6.7 tests/testlib/trellis.py::decode and
    # exllamav3/exllamav3_ext/quant/codebook.cuh.
    # These operands are dyadic; their sum/FMA is exact in binary64, then
    # rounded once to binary16, as in the pinned codebook.
    if codebook == "mul1":
        x = (state * MARKERS["mul1"]) & 0xFFFFFFFF
        total = sum((x >> shift) & 0xFF for shift in (0, 8, 16, 24))
        value = (1024 + total) * half_bits(0x1EEE) + half_bits(0xC931)
    else:
        x = state * MARKERS["mcg"] if codebook == "mcg" else state * 89226354 + 64248484
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        value = half_bits(x & 0xFFFF) + half_bits(x >> 16)
    return struct.unpack("<e", struct.pack("<e", value))[0]


def tile_into(matrix, row, col, raw, bits, codebook):
    # The ring runs MSB-first in little-endian u32 words, with a trailing
    # 16-bit state window at each position (pack.cu and frac.cu).
    count = len(raw) // 4
    big = struct.pack(f">{count}I", *struct.unpack(f"<{count}I", raw))
    ring = int.from_bytes(big, "big")
    width = 8 * len(raw)
    twice = (ring << width) | ring
    step, alternating = int(bits), not isinstance(bits, int)
    end = 0
    for i in range(TILE * TILE):
        end += step + (i % 2 if alternating else 0)
        start = (end - 16) % width
        state = (twice >> (2 * width - start - 16)) & 0xFFFF
        lane, element = divmod(i, 8)
        r = 2 * (lane % 4) + (element & 1) + 4 * (element & 2)
        c = lane // 4 + 2 * (element & 4)
        matrix[row + r][col + c] = codebook_value(state, codebook)


def hadamard(row):
    """In-place unnormalized Sylvester transform of a single row."""
    step = 1
    while step < BLOCK:
        for base in range(0, BLOCK, 2 * step):
            for j in range(base, base + step):
                a, b = row[j], row[j + step]
                row[j], row[j + step] = a + b, a - b
        step *= 2
    return row


def rotation_factors(read, tensors, rotation, origin):
    t = tensors[rotation["tensor"]]
    if rotation["packing"] == "fp16":
        values = struct.unpack(f"<{BLOCK}e", read(t.start + origin * 2, BLOCK * 2))
        need(all(math.isfinite(value) for value in values), f"{t.name}: nonfinite rotation factor")
        return values
    words = struct.unpack(f"<{BLOCK // 16}H", read(t.start + origin // 16 * 2, BLOCK // 8))
    return tuple(-1 if word & (1 << j) else 1 for word in words for j in range(16))


def reconstruct(path, module, origin, basis):
    """Read only the selected tile stripes, marker, and optional rotation slices."""
    path = Path(path)
    need(path.suffix == ".safetensors", "expected one .safetensors file with co-located tensors")
    need(basis in {"rotated", "original"}, "unknown reconstruction basis")
    i, o = origin
    need(all(type(x) is int and x >= 0 and x % BLOCK == 0 for x in (i, o)),
         "origin must be nonnegative and 128-aligned")
    with open_path(path, OPEN_FILE) as fd:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and 8 <= st.st_size <= MAX_BYTES,
             "expected a bounded regular SafeTensors file")
        tensors, length, _ = parse_header(
            lambda offset, count: read_at(fd, offset, count), path.name, st.st_size, HEADER_BUDGET - 8,
        )
        payload_bytes = 0

        def read(offset, count):
            nonlocal payload_bytes
            payload_bytes += count
            return read_at(fd, offset, count)

        packed = tensors.get(f"{module}.trellis")
        need(module and packed is not None, "selected module has no trellis in this file")
        selectors = {name: tensors[name] for suffix in MARKERS if (name := f"{module}.{suffix}") in tensors}
        need(len(selectors) <= 1, f"{module}: ambiguous codebook markers")
        markers = {}
        for name, t in selectors.items():
            need(t.dtype == "I32" and t.shape == (), f"{name}: expected an I32 scalar marker")
            multiplier, = struct.unpack("<I", read(t.start, 4))
            need(multiplier == MARKERS[name.rsplit(".", 1)[-1]], f"{name}: incorrect multiplier")
            markers[name] = multiplier
        info = linear_layout(module, packed, tensors, markers)
        need(i + BLOCK <= info["padded_in_features"] and o + BLOCK <= info["padded_out_features"],
             "block extends beyond the padded matrix")
        codebook, bits = info["codebook"], info["bits_per_weight"]
        row_tiles = packed.shape[1]
        tile_bytes = 2 * info["words_per_tile"]
        values = [[0.0] * BLOCK for _ in range(BLOCK)]
        for row in range(0, BLOCK, TILE):
            start_tile = ((i + row) // TILE) * row_tiles + o // TILE
            stripe = read(packed.start + start_tile * tile_bytes, BLOCK // TILE * tile_bytes)
            for col in range(0, BLOCK, TILE):
                start = col // TILE * tile_bytes
                tile_into(values, row, col, stripe[start:start + tile_bytes], bits, codebook)
        if basis == "original":
            a = rotation_factors(read, tensors, info["rotations"]["input"], i)
            b = rotation_factors(read, tensors, info["rotations"]["output"], o)
            # diag(a) H W_hat H diag(b); both H / sqrt(128) factors collapse
            # to one division by 128. This is not the fused GPU FP16 kernel.
            horizontal = [hadamard(row) for row in values]
            vertical = [hadamard(list(col)) for col in zip(*horizontal)]
            values = [[v / BLOCK * a[r] * b[c] for c, v in enumerate(row)]
                      for r, row in enumerate(zip(*vertical))]
        need(stamp(st) == stamp(os.fstat(fd)), "SafeTensors file changed during reconstruction")
    return {
        "representation": "exl3", "validation": "cpu-block-reconstruction", "upstream": UPSTREAM,
        "module": module, "basis": basis, "origin": [i, o], "shape": [BLOCK, BLOCK],
        "bits_per_weight": bits, "codebook": codebook, "header_bytes_read": length + 8,
        "payload_bytes_read": payload_bytes, "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                     epilog="Does not load a model or run inference.")
    parser.add_argument("checkpoint", help="one SafeTensors file holding the complete linear tensor group")
    parser.add_argument("module", help="tensor prefix before .trellis")
    parser.add_argument("--origin", nargs=2, type=int, metavar=("INPUT", "OUTPUT"), default=(0, 0))
    parser.add_argument("--basis", choices=("rotated", "original"), default="original")
    args = parser.parse_args()
    try:
        print(json.dumps(reconstruct(args.checkpoint, args.module, args.origin, args.basis), allow_nan=False))
    except (OSError, ValueError, RecursionError, struct.error) as exc:
        print(f"EXL3 reconstruction failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
