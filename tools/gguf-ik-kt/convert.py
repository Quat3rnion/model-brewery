#!/usr/bin/env python3
"""Bounded offline GGUF v3 exact per-layer KT encoder, for ik_llama.cpp only."""

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import struct
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expert_gguf import (GGUF, HeaderCursor, Tensor, align, identity, need,
                         no_duplicates, regular_file, reject_json_constant)


COMMIT = "7c00f8788e303bfba2886138e288effb0fd5fbbf"
SCHEMA = "brewery.ik-kt.v1"
KT = {"IQ1_KT": 158, "IQ2_KT": 153, "IQ3_KT": 154, "IQ4_KT": 155}
FLOAT_BYTES = {0: 4, 1: 2, 30: 2}
MAX_FILE = 8 << 30
MAX_HEADER = 32 << 20
MAX_HELPER = 32 << 20
MAX_PLAN = 1 << 20
MAX_TENSORS = 16384
MAX_COLUMNS = 65536
MAX_ELEMENTS = 8 << 20
CHUNK = 1 << 20
SHA = re.compile(r"[a-f0-9]{64}")
LAYER_WEIGHT = re.compile(r"blk\.(0|[1-9][0-9]*)\.[A-Za-z0-9_]+\.weight")
PROVENANCE = "brewery.quantization."
IDENTIFY = f"brewery-ik-kt-v1 {COMMIT} 158 153 154 155\n".encode()


def tensor_bytes(kind, dims):
    width = dims[0]
    if kind in FLOAT_BYTES:
        return math.prod(dims) * FLOAT_BYTES[kind]
    need(kind in KT.values() and len(dims) == 2 and width % 32 == 0,
         "unsupported tensor encoding or KT dimensions")
    blocks, tail = divmod(width // 32, 8)
    if kind == 158:
        stride = blocks * 56 + (8 + 4 * tail + min(4 * tail, 16) if tail else 0)
    elif kind == 153:
        stride = blocks * 68 + (4 + 8 * tail if tail else 0)
    elif kind == 154:
        stride = blocks * 100 + (tail + 1) // 2 + 12 * tail
    else:
        stride = blocks * 128 + 16 * tail
    return align(4 + stride, 4) * dims[1]


def read_gguf(file, *, output=False):
    file.seek(0)
    size = os.fstat(file.fileno()).st_size
    need(24 <= size <= MAX_FILE, "invalid GGUF file size")
    cursor = HeaderCursor(file.read(min(size, MAX_HEADER)))
    magic, version, count, fields = cursor.unpack("4sIQQ")
    need(magic == b"GGUF" and version == 3, "expected little-endian GGUF v3")
    need(0 < count <= MAX_TENSORS and fields <= MAX_TENSORS, "invalid GGUF header counts")
    metadata = {}
    alignment = 32
    for _ in range(fields):
        start = cursor.pos
        key = cursor.string(1024)
        need(key.isascii() and key.isprintable() and key not in metadata,
             "invalid or duplicate GGUF metadata key")
        need(not key.startswith(("split.", "agention.trellis.")), "unsupported GGUF metadata namespace")
        need(output or not key.startswith(PROVENANCE), "source already has Brewery quantization provenance")
        kind, = cursor.unpack("I")
        value = cursor.pos
        cursor.kv(kind)
        metadata[key] = cursor.data[start:cursor.pos]
        if key == "general.alignment":
            need(kind == 4, "GGUF alignment must be uint32")
            alignment, = struct.unpack("<I", cursor.data[value:cursor.pos])
            need(1 <= alignment <= 4096 and alignment & (alignment - 1) == 0,
                 "invalid GGUF alignment")
    tensors = {}
    expected = 0
    for _ in range(count):
        name = cursor.string(63)
        need(re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in tensors,
             "invalid or duplicate GGUF tensor name")
        rank, = cursor.unpack("I")
        need(1 <= rank <= 4, "invalid GGUF tensor rank")
        dims = cursor.unpack("Q" * rank)
        need(all(0 < dim <= MAX_FILE for dim in dims), "invalid GGUF tensor dimensions")
        kind, offset = cursor.unpack("IQ")
        need(output or kind in FLOAT_BYTES, f"expected floating source tensor: {name}")
        length = tensor_bytes(kind, dims)
        need(offset == expected, f"noncanonical GGUF tensor offset: {name}")
        expected += align(length, alignment)
        need(expected <= MAX_FILE, "GGUF data exceeds byte limit")
        tensors[name] = Tensor(name, dims, kind, offset, length)
    base = align(cursor.pos, alignment)
    last = next(reversed(tensors.values()))
    last_byte = base + last.relative + last.size
    need(base <= MAX_HEADER and last_byte <= size <= base + expected, "invalid GGUF extent")
    if size > last_byte:
        file.seek(last_byte)
        need(not any(file.read(size - last_byte)), "invalid GGUF trailing padding")
    return GGUF(metadata, tensors, alignment, cursor.pos, base, size)


def checked_bytes(path, digest, maximum):
    need(isinstance(digest, str) and SHA.fullmatch(digest), "expected a lowercase SHA-256")
    with regular_file(path) as file:
        before = identity(file)
        need(0 < before[2] <= maximum, f"input is empty or exceeds byte limit: {path}")
        data = file.read(before[2] + 1)
        need(len(data) == before[2] and identity(file) == before, f"input changed: {path}")
    need(hashlib.sha256(data).hexdigest() == digest, f"SHA-256 mismatch: {path}")
    return data


def sealed(data, mode):
    """Snapshot verified bytes; pathname replacement cannot change the worker inputs."""
    fd = os.memfd_create("brewery-ik-kt", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
        file.flush()
        os.fchmod(fd, mode)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        return os.fdopen(os.dup(fd), "rb")


def sha_file(file, deadline):
    file.seek(0)
    digest = hashlib.sha256()
    while block := file.read(CHUNK):
        need(time.monotonic() < deadline, "file hash exceeded wall-time budget")
        digest.update(block)
    return digest.hexdigest()


def plan_selection(path, digest, tensors):
    raw = checked_bytes(path, digest, MAX_PLAN)
    plan = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=reject_json_constant)
    need(isinstance(plan, dict) and plan.keys() == {"schema", "source_sha256", "tensors"}
         and plan["schema"] == SCHEMA, "unsupported plan schema or fields")
    source_sha = plan["source_sha256"]
    need(isinstance(source_sha, str) and SHA.fullmatch(source_sha), "invalid source SHA-256 in plan")
    selection = plan["tensors"]
    need(isinstance(selection, dict) and selection, "expected at least one exact tensor recipe")
    total = 0
    for name, recipe in selection.items():
        t = tensors.get(name)
        need(t is not None and LAYER_WEIGHT.fullmatch(name) and len(t.dims) == 2,
             f"expected existing rank-2 block weight: {name}")
        width, rows = t.dims
        total += width * rows
        need(width <= MAX_COLUMNS and width % 32 == 0 and total <= MAX_ELEMENTS,
             f"selected tensor exceeds KT row alignment or element budget: {name}")
        need(isinstance(recipe, dict) and recipe.keys() == {"type", "importance_sha256"}
             and isinstance(recipe["type"], str) and recipe["type"] in KT,
             f"unsupported exact KT recipe: {name}")
        weights = recipe["importance_sha256"]
        need(weights is None or isinstance(weights, str) and SHA.fullmatch(weights),
             f"invalid importance SHA-256: {name}")
    return source_sha, selection


def limits():
    for key, value in ((resource.RLIMIT_CORE, 0), (resource.RLIMIT_AS, 768 << 20),
                       (resource.RLIMIT_CPU, 120), (resource.RLIMIT_FSIZE, MAX_HELPER)):
        resource.setrlimit(key, (value, value))


def worker(helper, args, bindings=(), data=None, importance=None, *, max_output, deadline):
    remaining = deadline - time.monotonic()
    need(remaining > 0, "KT worker wall-time budget exhausted")
    command = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent",
               "--cap-drop", "ALL", "--clearenv", "--chdir", "/"]
    helper.seek(0)
    fds = [helper.fileno()]
    command.extend(("--perms", "0500", "--file", str(fds[0]), "/helper"))
    if importance is not None:
        importance.seek(0)
        fds.append(importance.fileno())
        command.extend(("--ro-bind-data", str(fds[-1]), "/importance"))
    for path, target in bindings:
        command.extend(("--ro-bind", str(path), target))
    command.extend(("--remount-ro", "/"))
    # RLIMIT_FSIZE bounds worker output, including upstream diagnostic noise.
    with os.fdopen(os.memfd_create("kt-output"), "w+b") as stdout, \
            os.fdopen(os.memfd_create("kt-diagnostics"), "w+b") as stderr:
        result = subprocess.run([*command, "--", "/helper", *map(str, args)],
                                input=data if data is not None else b"", stdout=stdout, stderr=stderr,
                                timeout=min(remaining, 125), env={}, cwd="/",
                                pass_fds=fds, preexec_fn=limits)
        stderr.seek(0)
        need(result.returncode == 0,
             f"pinned KT worker {args[0]} failed ({result.returncode}): "
             + stderr.read(512).decode(errors="replace").strip())
        stdout.seek(0)
        result = stdout.read(max_output + 1)
        need(len(result) <= max_output, f"excess pinned KT worker output: {args[0]}")
        return result


def text(value):
    data = value.encode("ascii")
    return struct.pack("<Q", len(data)) + data


def write_header(output, header, selection, plan_sha, source_sha, helper_sha):
    fields = {key: data for key, data in header.fields.items()
              if key not in {"general.file_type", "general.quantization_version"}
              and not key.startswith("quantize.imatrix.")}
    for key, value in {
        "backend": "ik_llama.cpp", "commit": COMMIT, "plan_sha256": plan_sha,
        "source_sha256": source_sha, "helper_sha256": helper_sha,
    }.items():
        key = PROVENANCE + key
        fields[key] = text(key) + struct.pack("<I", 8) + text(value)
    key = "general.quantization_version"
    fields[key] = text(key) + struct.pack("<II", 4, 2)
    need(len(fields) <= MAX_TENSORS, "output metadata count exceeds limit")

    output.write(struct.pack("<4sIQQ", b"GGUF", 3, len(header.tensors), len(fields)))
    for data in fields.values():
        output.write(data)
    relative = 0
    for t in header.tensors.values():
        kind = KT[selection[t.name]["type"]] if t.name in selection else t.kind
        output.write(text(t.name) + struct.pack("<I", len(t.dims)))
        output.write(struct.pack("<" + "Q" * len(t.dims) + "IQ", *t.dims, kind, relative))
        relative += align(tensor_bytes(kind, t.dims), header.alignment)
    base = align(output.tell(), header.alignment)
    need(base <= MAX_HEADER and base + relative <= MAX_FILE, "output exceeds byte limit")
    output.write(bytes(base - output.tell()))


def copy_bytes(source, destination, offset, length, deadline):
    source.seek(offset)
    remaining = length
    while remaining:
        block = source.read(min(CHUNK, remaining))
        need(block and time.monotonic() < deadline, "short or timed-out source copy")
        destination.write(block)
        remaining -= len(block)


def selected_source(source, header, selection, snapshot, deadline):
    offsets = {}
    with snapshot.open("xb") as output:
        for name in selection:
            t = header.tensors[name]
            offsets[name] = output.tell()
            copy_bytes(source, output, header.base + t.relative, t.size, deadline)
    return offsets


def write_payload(source, output, header, selection, snapshot, offsets, helper, importance_dir, deadline):
    for t in header.tensors.values():
        absolute = header.base + t.relative
        if t.name in selection:
            recipe = selection[t.name]
            width, rows = t.dims
            kind = KT[recipe["type"]]
            length = tensor_bytes(kind, t.dims)
            importance_sha = recipe["importance_sha256"]
            with ExitStack() as stack:
                weights = None
                if importance_sha is not None:
                    need(importance_dir is not None, "weighted recipe requires --importance-dir")
                    path = importance_dir / (importance_sha + ".f32")
                    raw = checked_bytes(path, importance_sha, width * 4)
                    need(len(raw) == width * 4, f"wrong importance vector length: {t.name}")
                    weights = stack.enter_context(sealed(raw, 0o400))
                encoded = worker(helper, ("encode", kind, t.kind, offsets[t.name], width, rows,
                                          length, int(importance_sha is not None)),
                                 [(snapshot, "/source")], importance=weights,
                                 max_output=length, deadline=deadline)
            need(len(encoded) == length, f"wrong encoder payload size: {t.name}")
            output.write(encoded)
        else:
            length = t.size
            copy_bytes(source, output, absolute, length, deadline)
        output.write(bytes(align(length, header.alignment) - length))


def readback(helper, snapshot, offsets, original, output, staged, selection, deadline):
    target = read_gguf(output, output=True)
    descriptors = [f"{len(target.tensors)} {target.base} {target.alignment}"]
    need(target.tensors.keys() == original.tensors.keys(), "output tensor set mismatch")
    for name, t in target.tensors.items():
        old = original.tensors[name]
        recipe = selection.get(name)
        need(t.dims == old.dims and t.kind == (KT[recipe["type"]] if recipe else old.kind),
             f"unexpected encoded type or dimensions: {name}")
        dims = " ".join(map(str, t.dims + (1,) * (4 - len(t.dims))))
        descriptors.append(f"{name} {t.kind} {t.relative} {t.size} {dims} "
                           f"{old.kind} {offsets.get(name, 0)}")
    raw = worker(helper, ("readback",), [(snapshot, "/source"), (staged, "/output")],
                 ("\n".join(descriptors) + "\n").encode(), max_output=128 * len(selection), deadline=deadline)
    stats = {}
    for line in raw.decode("ascii").splitlines():
        name, rmse, max_error = line.split()
        need(name in selection and name not in stats, "unexpected KT readback result")
        rmse, max_error = float(rmse), float(max_error)
        need(math.isfinite(rmse) and math.isfinite(max_error) and 0 <= rmse <= max_error,
             f"invalid readback metrics: {name}")
        stats[name] = {**selection[name], "rmse": rmse, "max_error": max_error}
    need(stats.keys() == selection.keys(), "missing KT readback result")
    return stats


def convert(args):
    deadline = time.monotonic() + 600
    need(not os.path.lexists(args.output), "output already exists")
    with regular_file(args.input) as source:
        before = identity(source)
        header = read_gguf(source)
        source_sha, selection = plan_selection(args.plan, args.plan_sha256, header.tensors)
        need(sha_file(source, deadline) == source_sha and identity(source) == before,
             "source SHA-256 mismatch or mutation")
        helper_raw = checked_bytes(args.helper, args.helper_sha256, MAX_HELPER)
        with sealed(helper_raw, 0o500) as helper:
            need(worker(helper, ("identify",), max_output=len(IDENTIFY), deadline=deadline) == IDENTIFY,
                 "unsupported pinned helper protocol")
            with tempfile.TemporaryDirectory(prefix=".ik-kt-", dir=args.output.parent.resolve()) as staging:
                staged = Path(staging) / "artifact"
                snapshot = Path(staging) / "selected-input"
                offsets = selected_source(source, header, selection, snapshot, deadline)
                with open(staged, "x+b", opener=lambda path, flags: os.open(path, flags, 0o600)) as output:
                    write_header(output, header, selection, args.plan_sha256, source_sha,
                                 args.helper_sha256)
                    write_payload(source, output, header, selection, snapshot, offsets, helper,
                                  args.importance_dir, deadline)
                    output.flush()
                    stats = readback(helper, snapshot, offsets, header, output, staged, selection, deadline)
                    result_sha = sha_file(output, deadline)
                    os.fsync(output.fileno())
                need(identity(source) == before and time.monotonic() < deadline,
                     "source changed or conversion timed out")
                os.link(staged, args.output)  # Same filesystem; no overwrite even if raced.
    return {"schema": SCHEMA, "backend": "ik_llama.cpp", "commit": COMMIT,
            "plan_sha256": args.plan_sha256, "helper_sha256": args.helper_sha256,
            "source_sha256": source_sha, "output_sha256": result_sha, "tensors": stats}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Plan: {"schema":"brewery.ik-kt.v1","source_sha256":"...",'
               '"tensors":{"blk.0.attn_q.weight":{"type":"IQ1_KT","importance_sha256":null}}}. '
               "Use null for unweighted encoding or the SHA-256 of a little-endian float32 vector "
               "with one importance value per input column.")
    for option in ("input", "output", "plan", "helper", "importance-dir"):
        parser.add_argument("--" + option, type=Path, required=option != "importance-dir",
                            help={"helper": "trusted static helper from build-helper.sh",
                                  "importance-dir": "directory containing <sha256>.f32 vectors"}.get(option))
    parser.add_argument("--plan-sha256", required=True, help="reviewed plan digest")
    parser.add_argument("--helper-sha256", required=True, help="operator-provisioned helper digest; never take this from a job")
    args = parser.parse_args()
    try:
        print(json.dumps(convert(args), sort_keys=True, allow_nan=False))
    except (OSError, ValueError, subprocess.TimeoutExpired) as err:
        parser.exit(1, f"ik-kt: {err}\n")


if __name__ == "__main__":
    main()
