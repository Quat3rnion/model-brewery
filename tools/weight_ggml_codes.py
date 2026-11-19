"""Fixed-scale cell access for little-endian GGML Q4_0 and Q8_0 blocks.

The caller validates the GGUF and supplies exactly one 32-cell block. The
layout follows ggml-quants.c at revision 184f6536024668e8908ab8c86033d56708e62b40.
No operation changes the FP16 scale or requantizes any neighbor.
"""

from collections.abc import Iterable
from dataclasses import dataclass
import math
from numbers import Real
import struct

BLOCK_CELLS = 32
MAX_TARGETS = 256
# Type -> (byte size, minimum decoded code, maximum decoded code).
_LAYOUT = {"Q4_0": (18, -8, 7), "Q8_0": (34, -128, 127)}


@dataclass(frozen=True)
class Cell:
    """The signed multiplier, stored scale, and exactly decoded value."""

    code: int
    scale: float
    value: float


@dataclass(frozen=True)
class Edit:
    cell_before: Cell
    cell_after: Cell
    block: bytes
    byte_index: int


def _layout(ggml_type):
    if not isinstance(ggml_type, str) or ggml_type not in _LAYOUT:
        raise ValueError(f"unsupported fixed-scale GGML type: {ggml_type!r}")
    return _LAYOUT[ggml_type]


def block_size(ggml_type: str) -> int:
    return _layout(ggml_type)[0]


def _read(block, ggml_type, index):
    size, minimum, maximum = _layout(ggml_type)
    if not isinstance(block, bytes) or len(block) != size:
        raise ValueError(f"{ggml_type} requires exactly {size} block bytes")
    if type(index) is not int or not 0 <= index < BLOCK_CELLS:
        raise ValueError("GGML cell index must be an integer from 0 to 31")
    scale, = struct.unpack_from("<e", block)
    if not math.isfinite(scale):
        raise ValueError("nonfinite GGML block scale")
    if ggml_type == "Q4_0":
        byte_index = 2 + index % 16
        shift = 4 if index >= 16 else 0
        code = ((block[byte_index] >> shift) & 15) - 8
    else:
        byte_index = 2 + index
        shift = 0
        code, = struct.unpack_from("b", block, byte_index)
    return Cell(code, scale, code * scale), byte_index, shift, minimum, maximum


def read_cell(block: bytes, ggml_type: str, index: int) -> Cell:
    return _read(block, ggml_type, index)[0]


def _write(block, ggml_type, state, code):
    before, byte_index, shift, _, _ = state
    if code == before.code:
        return Edit(before, before, block, byte_index)
    if ggml_type == "Q4_0":
        byte = (block[byte_index] & ~(15 << shift)) | ((code + 8) << shift)
    else:
        byte = code & 255
    after = Cell(code, before.scale, code * before.scale)
    return Edit(before, after, block[:byte_index] + bytes((byte,)) + block[byte_index + 1:], byte_index)


def _finite_value(value):
    if isinstance(value, Real) and not isinstance(value, bool):
        try:
            result = float(value)
        except OverflowError:
            pass
        else:
            if math.isfinite(result):
                return result
    raise ValueError("GGML edit value must be a finite real number")


def _nearest(state, target):
    before, _, _, minimum, maximum = state
    # Order can reverse when the scale is negative.
    first, last = minimum * before.scale, maximum * before.scale
    if not min(first, last) <= target <= max(first, last):
        raise ValueError("GGML target is outside the fixed-scale representable range")
    if target == before.value:
        return before.code  # Also preserves the stored code of a zero-scale block.
    lower = math.floor(target / before.scale)
    # Compare in the value domain: division can round a target just off a
    # midpoint to an exact half-integer. FP16 times a half-integer here is exact.
    midpoint = (lower + 0.5) * before.scale
    if target == midpoint:
        return lower + (lower & 1)  # Ties to the even signed code.
    return lower + ((target > midpoint) if before.scale > 0 else (target < midpoint))


def edit_cell(block: bytes, ggml_type: str, index: int, *, operation: str, value: float) -> Edit:
    """Set or add to a decoded value, then round to nearest (ties to even).

    Reject targets outside the fixed-scale interval instead of saturating.
    A no-op returns the original bytes, including the code of a zero-scale
    block. To compose edits, pass the returned block into the next edit.
    """
    state = _read(block, ggml_type, index)
    target = _finite_value(value)
    if operation == "delta":
        target += state[0].value
    elif operation != "set":
        raise ValueError(f"unsupported GGML edit operation: {operation!r}")
    return _write(block, ggml_type, state, _nearest(state, target))


def project_candidates(block: bytes, ggml_type: str, index: int, targets: Iterable[float]) -> list[Edit]:
    """Round up to 256 absolute targets; skip invalid, unchanged and duplicate codes."""
    state = _read(block, ggml_type, index)
    seen = {state[0].code}
    edits = []
    for ordinal, target in enumerate(targets):
        if ordinal == MAX_TARGETS:
            raise ValueError(f"GGML candidate projection accepts at most {MAX_TARGETS} targets")
        try:
            code = _nearest(state, _finite_value(target))
        except ValueError:
            continue
        if code not in seen:
            seen.add(code)
            edits.append(_write(block, ggml_type, state, code))
    return edits


def candidates(block: bytes, ggml_type: str, index: int) -> tuple[Edit, ...]:
    """Return all distinct changed levels in signed code order, at most 255."""
    state = _read(block, ggml_type, index)
    before, _, _, minimum, maximum = state
    if before.scale == 0:
        return ()
    return tuple(_write(block, ggml_type, state, code) for code in range(minimum, maximum + 1)
                 if code != before.code)
