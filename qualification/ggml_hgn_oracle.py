#!/usr/bin/env python3
"""Compare all decoded F32 values of one HGN extension to a native GGML oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GGML_REVISION = "184f6536024668e8908ab8c86033d56708e62b40"
MAX_FILE = 8 << 20
MAX_VALUES = 4 << 20
MAX_TENSORS = 256


def verify_bounds(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 4 <= info.st_size <= MAX_FILE:
        raise ValueError(f"{path}: expected a regular fixture at most 8 MiB")


def differences(expected, actual):
    if actual.dtype != np.dtype("float32") or actual.size != expected.size:
        raise ValueError(f"reader returned {actual.size} values of type {actual.dtype}, "
                         f"expected {expected.size} float32 values")
    actual = actual.ravel()
    a, b = expected.view(np.uint32), actual.view(np.uint32)
    wrong = np.flatnonzero(a != b)
    result = {
        "bit_mismatches": int(wrong.size),
        "nonfinite_native": int(np.count_nonzero(~np.isfinite(expected))),
        "nonfinite_reader": int(np.count_nonzero(~np.isfinite(actual))),
    }
    if wrong.size:
        i = int(wrong[0])
        result["first_mismatch"] = {"index": i, "native_bits": f"{a[i]:08x}", "reader_bits": f"{b[i]:08x}"}
    return result


def compare(native, source, container):
    for path in (source, container):
        verify_bounds(path)
    sys.path.insert(0, str(ROOT / "tools/cross_container"))
    from ggml_in_hgn import Reader

    # The helper has its own size/value limits. Running it before Reader
    # prevents malformed sources from requesting a large decoded buffer.
    run = subprocess.run([str(native.resolve(strict=True)), source], capture_output=True, timeout=10)
    if run.returncode:
        detail = run.stderr[:2048].decode(errors="replace").strip()
        raise ValueError(f"native oracle exited {run.returncode}: {detail}")
    reader = Reader(container)
    seen, rows = set(), []
    at = count = 0
    while at < len(run.stdout):
        end = run.stdout.find(b"\n", at, at + 1200)
        if end == -1:
            raise ValueError("malformed native tensor header")
        type_id, type_name, size, encoded_name = run.stdout[at:end].decode("ascii").split()
        name = bytes.fromhex(encoded_name).decode()
        type_id, size = int(type_id), int(size)
        if len(seen) == MAX_TENSORS or name in seen or not 1 <= size <= MAX_VALUES - count:
            raise ValueError("invalid or oversized native tensor inventory")
        at = end + 1
        expected = np.frombuffer(run.stdout, dtype=np.float32, count=size, offset=at)
        at += size * 4
        if name not in reader.tensors or reader.tensors[name].ggml_type != type_id:
            raise ValueError(f"{name}: missing reader tensor or different GGML type")
        if np.prod(reader.tensors[name].shape, dtype=object) != size:
            raise ValueError(f"{name}: reader tensor dimensions do not match native element count")
        rows.append({"name": name, "type": type_name, "values": size,
                     **differences(expected, reader.decode(name))})
        seen.add(name)
        count += size
    if not rows or set(reader.tensors) != seen:
        raise ValueError("reader and source tensor inventories differ")
    return {
        "ggml_revision": GGML_REVISION,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "hgn_sha256": hashlib.sha256(container.read_bytes()).hexdigest(),
        "values": count,
        "equal": not any(row[key] for row in rows for key in
                         ("bit_mismatches", "nonfinite_native", "nonfinite_reader")),
        "tensors": rows,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--native", type=Path, required=True, help="native helper linked to pinned GGML")
    cli.add_argument("source", type=Path)
    cli.add_argument("container", type=Path)
    args = cli.parse_args()
    try:
        report = compare(args.native, args.source, args.container)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"oracle refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
