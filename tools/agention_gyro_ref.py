#!/usr/bin/env python3
"""Bounded offline Agention reference encode plus independent GGUF decode.

Use --upstream with the extracted Apache-2.0 archive (see ARCHIVE). Requires
a trusted local C compiler named cc and NumPy. Limit: 16 MiB of input and 32
selected blocks. Rotated tensors require an explicit input basis. Unchanged
metadata (including general.file_type, if present) does not describe the new
tensor types; consult the tensor table. Not a calibrated/model-scale encoder.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

if __package__:
    from .expert_gguf import HeaderCursor, align, gguf_string, identity, need, read_gguf, regular_file
else:
    sys.dont_write_bytecode = True
    from expert_gguf import HeaderCursor, align, gguf_string, identity, need, read_gguf, regular_file


REVISION = "587a5269848eb07996c9c1433f9ad638604eb4d8"
ARCHIVE = f"https://gitlab.com/agentionai-group/agention-gyro/-/archive/{REVISION}/agention-gyro-{REVISION}.tar.gz"
PINNED = {
    "src/tq_codebook.c": "0eb4beccdf0f03e9a6ea6912a50d7ac63d5ea537675646d32a196eb10b2d1c13",
    "src/tq_ref.c": "2a2c4d0bc6f7d7a4653482c791c3cc99b5a066ad90e85e7e69d0ff78f6e9ba1c",
    "src/tq_ref.h": "81a2f2b0760379794dd67aebf2db453dbf94b99a65232d8ec36080180e20333c",
    "python/tq_ref.py": "296f802d4eada3e87cc3c0a0119479f9cb790234fbdead92b98cf24d094b6d8e",
}
TYPES = {
    "TQK4": (151, 18), "TQK5": (150, 22), "TQK6": (145, 26), "TQK7": (146, 30),
    "TQ2_T": (144, 34), "TQP5": (147, 42), "TQP6": (148, 50), "TQP7": (149, 58),
}
STATE_BITS = "agention.trellis.state_bits"
STATE_FIELD = (4, struct.pack("<I", 15))
ROTATION_PREFIX = "prism.hadamard."
ROTATION_FIELDS = {
    "version": 4, "block_size": 4, "transform": 8, "axis": 8, "sign_mode": 8,
    "weight_names": (9, 8), "sign_widths": (9, 5), "sign_values": (9, 5),
}
WEIGHTS_PER_BLOCK = 128
MAX_BLOCKS = 32
MAX_INPUT = 16 << 20
DEADLINE_SECONDS = 60


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pinned_toolchain(upstream, work):
    for name, expected in PINNED.items():
        with regular_file(upstream / name) as file:
            raw = file.read((1 << 20) + 1)
        need(len(raw) <= 1 << 20 and sha256(raw) == expected, f"upstream pin mismatch: {name}")
        (work / Path(name).name).write_bytes(raw)
    (work / "helper.c").write_bytes(Path(__file__).with_name("agention_gyro_ref_helper.c").read_bytes())
    subprocess.run(
        ["cc", "-std=c99", "-O2", "-fno-fast-math", "-ffp-contract=off",
         f'-DGYRO_REVISION="{REVISION}"', "helper.c", "tq_ref.c", "tq_codebook.c", "-lm", "-o", "helper"],
        cwd=work, capture_output=True, check=True, timeout=15,
    )
    helper = work / "helper"
    with regular_file(helper) as file:
        fingerprint = sha256(file.read(1 << 20))
        need(file.read(1) == b"", "oversized reference helper")
    reported = subprocess.run([helper, "identity"], capture_output=True, check=True, timeout=2)
    need(reported.stdout == f"brewery-agention-gyro-ref-v1 {REVISION}\n".encode("ascii"),
         "reference helper identity mismatch")
    spec = importlib.util.spec_from_file_location("agention_pinned_decoder", work / "tq_ref.py")
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    return helper, fingerprint, decoder


def select(model, requests):
    selected = {}
    total = 0
    for name, label in requests:
        need(name not in selected and name in model.tensors, f"unknown or repeated tensor: {name}")
        need(label in TYPES, f"unknown Agention type: {label}")
        tensor = model.tensors[name]
        need(tensor.kind == 0, f"selected tensor must be F32: {name}")
        need(tensor.dims[0] % WEIGHTS_PER_BLOCK == 0, f"row width must be a multiple of 128: {name}")
        total += math.prod(tensor.dims) // WEIGHTS_PER_BLOCK
        need(total <= MAX_BLOCKS, f"selection exceeds {MAX_BLOCKS} Gyro blocks")
        selected[name] = label
    return selected


def rotation_plan(model, selected, input_basis):
    """Accept the pinned v1 transform; refuse extensions rather than guessing."""
    fields = {key.removeprefix(ROTATION_PREFIX): field for key, field in model.fields.items()
              if key.startswith(ROTATION_PREFIX)}
    if not fields:
        return {}, 0
    need(input_basis is not None, "rotation metadata requires explicit --input-basis")

    def field(key):
        kind, data = fields.get(key, (None, None))
        expected = ROTATION_FIELDS[key]
        if expected == 4:
            need(kind == expected, f"expected uint32 rotation field: {key}")
            return struct.unpack("<I", data)[0]
        if expected == 8:
            need(kind == expected, f"expected string rotation field: {key}")
            return HeaderCursor(data).string(128)
        need(kind == expected[0], f"expected rotation array: {key}")
        cursor = HeaderCursor(data)
        subtype, count = cursor.unpack("IQ")
        need(subtype == expected[1] and count <= 16384, f"invalid rotation array: {key}")
        if subtype == 8:
            return [cursor.string(63) for _ in range(count)]
        return cursor.unpack("i" * count)

    need(fields.keys() <= ROTATION_FIELDS.keys() and field("version") == 1,
         "only v1 Hadamard metadata without extensions is supported")
    block = field("block_size")
    need(0 < block <= MAX_BLOCKS * WEIGHTS_PER_BLOCK and block & (block - 1) == 0,
         "invalid or oversized rotation block")
    need(field("transform") == "normalized-sylvester-walsh-hadamard"
         and field("axis") == "input-last-dimension", "unsupported rotation transform or axis")
    names = field("weight_names")
    need(names and len(set(names)) == len(names) and set(names) <= model.tensors.keys(),
         "invalid rotation weight names")
    if input_basis == "original":
        need(set(names) <= selected.keys(), "original basis requires selecting every declared rotated tensor")
    mode = field("sign_mode")
    signs = {}
    if mode == "explicit":
        widths, values = field("sign_widths"), field("sign_values")
        need(widths and len(set(widths)) == len(widths) and sum(widths) == len(values)
             and all(w > 0 and w % block == 0 for w in widths)
             and all(value in (-1, 1) for value in values), "invalid explicit rotation signs")
        offset = 0
        for width in widths:
            signs[width] = struct.pack(f"{width}b", *values[offset:offset + width])
            offset += width
    else:
        need(mode == "identity" and not ({"sign_widths", "sign_values"} & fields.keys()),
             "invalid rotation sign mode")
    by_name = {}
    for name in names:
        width = model.tensors[name].dims[0]
        need(width % block == 0 and (mode == "identity" or width in signs),
             f"rotation does not fit row: {name}")
        by_name[name] = signs.get(width, b"")
    return by_name, block


def write_gguf(out, model, payloads, selected):
    fields = dict(model.fields)
    fields[STATE_BITS] = STATE_FIELD
    out.write(struct.pack("<4sIQQ", b"GGUF", 3, len(model.tensors), len(fields)))
    for key, (kind, data) in fields.items():
        out.write(gguf_string(key) + struct.pack("<I", kind))
        out.write(data)
    offset = 0
    for name, tensor in model.tensors.items():
        dims = tensor.dims
        kind = TYPES[selected[name]][0] if name in selected else tensor.kind
        out.write(gguf_string(name) + struct.pack("<I" + "Q" * len(dims) + "IQ", len(dims), *dims, kind, offset))
        offset += align(len(payloads[name]), model.alignment)
    out.write(bytes(-out.tell() % model.alignment))
    for data in payloads.values():
        out.write(data)
        out.write(bytes(-len(data) % model.alignment))


def encode(source, output, upstream, requests, input_basis):
    need(not os.path.lexists(output), f"output already exists: {output}")
    with regular_file(source) as file:
        before = identity(file)
        need(24 <= before[2] <= MAX_INPUT, f"input GGUF must be at most {MAX_INPUT} bytes")
        # The stock reader rejects custom tensor IDs, including 144–151.
        model = read_gguf(file)
        need(model.fields.get(STATE_BITS, STATE_FIELD) == STATE_FIELD, "conflicting Agention trellis state width")
        selected = select(model, requests)
        rotation, block = rotation_plan(model, selected, input_basis)
        file.seek(0)
        raw = file.read(MAX_INPUT + 1)
        need(len(raw) == before[2] and identity(file) == before, "input changed during read")
        payloads = {t.name: memoryview(raw)[model.base + t.relative:model.base + t.relative + t.size]
                    for t in model.tensors.values()}

    report = {
        "schema": "brewery.agention-gyro-reference.v1",
        "upstream_revision": REVISION,
        "input_sha256": sha256(raw),
        "input_basis": input_basis or "stored",
        "importance": "none",
        "tensors": {},
    }
    with tempfile.TemporaryDirectory(prefix="agention-gyro-codec-") as work, \
            tempfile.TemporaryDirectory(prefix=".agention-gyro-output-", dir=output.parent) as staging:
        import numpy as np

        staged = Path(staging) / "encoded.gguf"
        helper, report["helper_sha256"], decoder = pinned_toolchain(upstream, Path(work))
        deadline = time.monotonic() + DEADLINE_SECONDS
        references = {}
        for name, label in selected.items():
            t = model.tensors[name]
            type_id, block_bytes = TYPES[label]
            n = math.prod(t.dims)
            size = n // WEIGHTS_PER_BLOCK * block_bytes
            source_values = np.frombuffer(payloads[name], dtype="<f4").reshape(t.dims[::-1])
            need(np.isfinite(source_values).all(), f"non-finite F32 input: {name}")
            rotated = name in rotation
            signs = rotation.get(name, b"")
            mode = int(rotated and input_basis == "original") | (2 if signs else 0)
            seconds = deadline - time.monotonic()
            need(seconds > 0, "reference encoder time budget exhausted")
            result = subprocess.run(
                [helper, "encode", str(type_id), str(t.dims[0]), str(n), str(block if rotated else 0), str(mode)],
                input=source_values.tobytes() + signs, capture_output=True, check=True, timeout=seconds,
            ).stdout
            need(len(result) == size + (8 if rotated else 4) * n, f"invalid reference helper response: {name}")
            payloads[name] = result[:size]
            references[name] = source_values, result[size:size + 4 * n], result[size + 4 * n:]
        with staged.open("xb") as out:
            write_gguf(out, model, payloads, selected)

        independent = decoder.GGUF(staged)
        need(independent.metadata.get(STATE_BITS) == 15 and independent.metadata_types[STATE_BITS] == (4, None),
             "independent reader rejected trellis state width")
        for name, label in selected.items():
            t = model.tensors[name]
            type_id, _ = TYPES[label]
            dims, actual_type, _ = independent.tensors[name]
            need(tuple(dims) == t.dims and actual_type == type_id, f"independent tensor table mismatch: {name}")
            decoded = independent.read(name)
            original, expected, expected_unrotated = references[name]
            need(np.isfinite(decoded).all() and decoded.astype("<f4", copy=False).tobytes() == expected,
                 f"C/Python decode mismatch: {name}")
            if name in rotation:
                decoded = independent.read(name, unrotate=True)
                need(np.isfinite(decoded).all() and decoded.astype("<f4", copy=False).tobytes() == expected_unrotated,
                     f"C/Python inverse rotation mismatch: {name}")
                if input_basis == "stored":
                    signs = rotation[name]
                    sign_array = np.frombuffer(signs, dtype=np.int8) if signs else None
                    original = decoder.hadamard_unrotate(original, block, sign_array)
                    need(np.isfinite(original).all(), f"non-finite inverse-rotated input: {name}")
            original = original.astype(np.float64)
            error = float(np.linalg.norm(decoded.astype(np.float64) - original))
            norm = float(np.linalg.norm(original))
            report["tensors"][name] = {
                "type": label, "type_id": type_id, "packed_bytes": len(payloads[name]),
                "independent_decode_match": True, "rotation_verified": name in rotation,
                "relative_l2": error / norm if norm else None,
            }
        report["output_sha256"] = sha256(staged.read_bytes())
        # Publish only a proven file, and never replace a path that appeared during encode.
        os.link(staged, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=f"Pinned archive: {ARCHIVE}")
    parser.add_argument("source", type=Path, help="bounded stock GGUF v3 input")
    parser.add_argument("output", type=Path, help="new GGUF file; must not exist")
    parser.add_argument("--upstream", type=Path, required=True, help="extracted pinned source archive")
    parser.add_argument("--tensor", nargs=2, metavar=("NAME", "TYPE"), action="append", required=True)
    parser.add_argument("--input-basis", choices=("stored", "original"), help="required for GGUF with rotation metadata")
    args = parser.parse_args()
    try:
        report = encode(args.source, args.output, args.upstream, args.tensor, args.input_basis)
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
