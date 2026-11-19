"""Plan one fixed-scale native HGN Q8G64 code change without writing a file.

The caller must identify a native Q8G64 (7/0) tensor and supply its exact
payload, excluding container padding. Row codes precede their F16 scale/bias
pairs. Container validation, checksums and publication are not part of this
helper.
"""
from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class Patch:
    offset: int
    old_code: int
    new_code: int
    before: float
    requested: float
    after: float
    encoding: str = "Q8G64"
    policy: str = "fixed-scale"


@dataclass(frozen=True)
class Cell:
    offset: int
    code: int
    scale: float
    bias: float
    value: float


def _decode(code, scale, bias):
    # The exact product and sum fit binary64. Convert once to F32,
    # matching a native fused multiply-add on finite half inputs.
    return struct.unpack("<f", struct.pack("<f", code * scale + bias))[0]


def read_cell(payload, rows, cols, row, col):
    """Return the encoded byte, affine parameters, and native F32 value."""
    if type(rows) is not int or rows < 1 or type(cols) is not int or cols < 64 or cols % 64:
        raise ValueError("Q8G64 needs positive rows and columns divisible by 64")
    data = memoryview(payload)
    stride = cols + cols // 16
    if data.ndim != 1 or data.format != "B" or not data.c_contiguous or len(data) != rows * stride:
        raise ValueError("invalid Q8G64 payload length or byte layout")
    if type(row) is not int or type(col) is not int or not (0 <= row < rows and 0 <= col < cols):
        raise ValueError("Q8G64 coordinate outside the tensor")
    row_offset = row * stride
    offset = row_offset + col
    scale, bias = struct.unpack_from("<ee", data, row_offset + cols + 4 * (col // 64))
    if not math.isfinite(scale) or scale < 0 or not math.isfinite(bias):
        raise ValueError("Q8G64 needs a nonnegative finite scale and finite bias")
    code = data[offset]
    return Cell(offset, code, scale, bias, _decode(code, scale, bias))


def candidates(payload, rows, cols, row, col):
    """Return patches for distinct changed F32 levels, in ascending code order."""
    cell = read_cell(payload, rows, cols, row, col)
    if cell.scale == 0:
        return ()
    codes = {}
    for code in range(256):
        value = _decode(code, cell.scale, cell.bias)
        if value != cell.value and (value not in codes or code % 2 < codes[value] % 2):
            codes[value] = code
    return tuple(Patch(cell.offset, cell.code, code, cell.value, value, value)
                 for value, code in codes.items())


def plan_patch(payload, rows, cols, row, col, op, value):
    """Return the exact byte and decoded F32 values for a set or delta.

    Reject out-of-range requests rather than clipping them. Select the nearest
    native decoded value to the request, breaking ties in favor of an even code.
    A zero-scale group is read-only. No byte in the input is modified.
    """
    cell = read_cell(payload, rows, cols, row, col)
    if cell.scale == 0:
        raise ValueError("Q8G64 editing requires a positive scale")
    if op not in ("set", "delta"):
        raise ValueError("Q8G64 operation must be set or delta")
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("Q8G64 edit requires a finite number")

    def decode(code):
        return _decode(code, cell.scale, cell.bias)

    before = cell.value
    requested = float(value) if op == "set" else before + value
    if not math.isfinite(requested):
        raise ValueError("Q8G64 requested value must be finite")
    if not decode(0) <= requested <= decode(255):
        raise ValueError("Q8G64 requested value is outside the fixed-scale range")
    new_code = min(range(256), key=lambda code: (abs(decode(code) - requested), code % 2))
    after = decode(new_code)
    if after == before:
        raise ValueError("Q8G64 rounded value is unchanged")
    return Patch(cell.offset, cell.code, new_code, before, requested, after)
