#!/usr/bin/env python3
"""Propose selected GGUF or HGN weight edits from calibration; never apply them."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

import numpy as np

from weight_edit import load_source, propose, round_value

MAX_DATA_BYTES = 8 << 20
MAX_ROWS = 1024
MAX_COORDINATES = 128
MAX_MATRIX_VALUES = 1 << 20
MAX_LAYER_PRODUCTS = 64 << 20
SWEEPS = 32


def unique_keys(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate JSON keys")
    return obj


def batch(path, shape):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_DATA_BYTES:
            raise ValueError("activation data must be a regular JSON file up to 8 MiB")
        data = stream.read(MAX_DATA_BYTES + 1)
    if len(data) > MAX_DATA_BYTES:
        raise ValueError("activation data exceeds 8 MiB")
    obj = json.loads(data, object_pairs_hook=unique_keys)
    if not isinstance(obj, dict) or set(obj) != {"activations", "targets"}:
        raise ValueError("expected exactly activations and targets")
    arrays = []
    for key, width in (("activations", shape[1]), ("targets", shape[0])):
        rows = obj[key]
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS or any(
            not isinstance(row, list) or len(row) != width
            or any(type(value) not in (float, int) for value in row) for row in rows
        ):
            raise ValueError(f"{key} must have 1–1024 rows of {width} numbers")
        arr = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"{key} contains nonfinite values")
        arrays.append(arr)
    x, target = arrays
    if len(x) != len(target):
        raise ValueError("activation and target row counts differ")
    return x, target, hashlib.sha256(data).hexdigest()


def coordinates(text, shape):
    if len(text) > 8192:
        raise ValueError("coordinate selection is too large")
    values = json.loads(text)
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_COORDINATES or any(
        not isinstance(item, list) or len(item) != 2
        or any(type(i) is not int or not 0 <= i < n for i, n in zip(item, shape))
        for item in values
    ):
        raise ValueError("select 1–128 in-bounds [row,col] coordinates")
    points = sorted(map(tuple, values))
    if len(set(points)) != len(points):
        raise ValueError("duplicate selected coordinate")
    return points


def residual(weights, x, targets):
    # Fixed contraction order; no threaded BLAS solver or random search.
    return np.einsum("ni,oi->no", x, weights, optimize=False) - targets


def mse(errors):
    answer = float(np.mean(np.square(errors)))
    if not math.isfinite(answer):
        raise ValueError("nonfinite layer-output MSE")
    return answer


def bounded_round(dtype, original, ideal, cap):
    """Find the nearest rounded update within the actual stored-value cap."""
    # Bisect towards the original only if rounding the clipped optimum
    # would exceed the cap or overflow the source dtype.
    delta = max(-cap, min(cap, ideal - original))

    def rounded(fraction):
        try:
            value = round_value(dtype, original + delta * fraction)
        except (ValueError, OverflowError):
            return None
        return value if abs(value - original) <= cap else None

    direct = rounded(1.0)
    if direct is not None:
        return direct
    lo, hi = 0.0, 1.0
    best = original
    for _ in range(54):
        mid = (lo + hi) / 2
        value = rounded(mid)
        if value is None:
            hi = mid
        else:
            lo, best = mid, value
    return best


def fit(matrix, project, x, targets, points):
    """Deterministic cyclic coordinate descent on training residuals only."""
    tuned = matrix.copy()
    errors = residual(matrix, x, targets)
    before = mse(errors)
    columns = {c: x[:, c].copy() for _, c in points}
    norms = {c: float(np.sum(column * column)) for c, column in columns.items()}
    for _ in range(SWEEPS):
        changed = False
        for r, c in points:
            if norms[c] == 0:
                continue
            prior = float(tuned[r, c])
            feature = columns[c]
            old_errors = errors[:, r]
            ideal = prior - float(np.sum(feature * old_errors)) / norms[c]
            if not math.isfinite(ideal):
                raise ValueError("nonfinite coordinate optimum")
            value = project(r, c, ideal)
            if value == prior:
                continue
            updated = old_errors + (value - prior) * feature
            if np.sum(updated * updated) < np.sum(old_errors * old_errors):
                tuned[r, c], errors[:, r] = value, updated
                changed = True
        if not changed:
            break
    # Recompute the measured objective on stored values, not the incremental
    # solver residual or a prediction using unrounded updates.
    after = mse(residual(tuned, x, targets))
    if not after < before:
        raise ValueError("no representable update improves measured calibration MSE within the cap")
    return tuned, before, after


def tune(args):
    cap = args.max_update
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("max-update must be finite and positive")
    source = load_source(args.source, args.source_sha256)
    raw = source.matrix(args.tensor)
    if raw.ndim != 2 or not 1 <= raw.size <= MAX_MATRIX_VALUES:
        raise ValueError("expected one rank-2 matrix with at most 1,048,576 values")
    matrix = raw.astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("source matrix contains nonfinite values")
    points = coordinates(args.indices, matrix.shape)
    x, targets, calibration_sha = batch(args.calibration, matrix.shape)
    held_x, held_targets, held_sha = batch(args.held_out, matrix.shape)
    if (len(x) + len(held_x)) * matrix.size > MAX_LAYER_PRODUCTS:
        raise ValueError("layer measurement exceeds the 64 Mi product budget")
    if {tuple(row) for row in x}.intersection(map(tuple, held_x)):
        raise ValueError("calibration and held-out activation rows overlap")

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        dtype = source.dtype(args.tensor)
        choices = {}
        for r, c in points:
            levels = source.candidates(args.tensor, [r, c])
            if levels is not None:
                origin = float(matrix[r, c])
                choices[r, c] = (origin, *(v for v in levels if abs(v - origin) <= cap))

        def project(r, c, ideal):
            if (r, c) in choices:
                return min(choices[r, c], key=lambda v: abs(v - ideal))
            return bounded_round(dtype, float(matrix[r, c]), ideal, cap)

        tuned, before, after = fit(matrix, project, x, targets, points)
        edits = [{"tensor": args.tensor, "index": [r, c], "op": "set", "value": float(tuned[r, c])}
                 for r, c in points if tuned[r, c] != matrix[r, c]]
        proposal = propose(source, edits)
        if any(edit["after"] != tuned[tuple(edit["index"])] for edit in proposal["edits"]):
            raise ValueError("proposal differs from measured stored values")
        held_before = mse(residual(matrix, held_x, held_targets))
        held_after = mse(residual(tuned, held_x, held_targets))
    report = {
        "schema": "brewery.weight-tune-report.v1",
        "source_sha256": proposal["source_sha256"],
        "calibration_sha256": calibration_sha,
        "held_out_sha256": held_sha,
        "tensor": args.tensor,
        "method": "rounded-coordinate-descent-v1",
        "max_sweeps": SWEEPS,
        "selected": len(points),
        "proposed": len(edits),
        "max_update": cap,
        "calibration": {"before_mse": before, "after_mse": after},
        "held_out": {"before_mse": held_before, "after_mse": held_after,
                     "regression": held_after > held_before},
    }
    return proposal, report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Each split is JSON with activations and targets. Weights are [output,input]. "
               "Writes a proposal to stdout and the separate MSE report to stderr.",
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--indices", required=True, help="JSON list of [output row,input column]")
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--held-out", required=True, type=Path)
    parser.add_argument("--max-update", required=True, type=float, help="maximum absolute change per weight")
    args = parser.parse_args()
    try:
        proposal, report = tune(args)
    except (ValueError, OSError, ArithmeticError, RecursionError) as error:
        parser.exit(2, f"weight-tune: {error}\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False), file=sys.stderr)
    print(json.dumps(proposal, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
