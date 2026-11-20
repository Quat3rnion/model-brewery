#!/usr/bin/env python3
"""Exercise the pinned GGUF catalog against an existing CPU llama-quantize.

Requires Python 3.11+, NumPy and gguf-py from the pinned llama.cpp checkout.
Fixtures are synthetic and only exercise encoding, not inference or quality.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader, GGUFWriter, LlamaFileType

CATALOG = Path(__file__).resolve().parents[1] / "crates/controller/src/gguf_catalog.json"
REVISION = "184f6536024668e8908ab8c86033d56708e62b40"
CALIBRATION = "synthetic-calibration.gguf"
DATASET = "synthetic:gguf-catalog-smoke"
TARGET = "blk.0.attn_q.weight"
EXPERT = "blk.0.ffn_up_exps.weight"
NORM = "blk.0.attn_norm.weight"
CARRIER = "Q8_0"
ALTERNATE_CARRIER = "Q4_0"
NTHREADS = "2"


def need(test, message):
    if not test:
        raise ValueError(message)


def save(writer):
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def weights(shape, seed):
    index = np.arange(np.prod(shape), dtype=np.int64)
    # Deterministic, signed, nontrivial values exactly representable as F32.
    return ((((index * 127 + (index // 256) * 31 + seed) % 1021) - 510)
            .astype(np.float32) / 512).reshape(shape)


def fixtures(out, host):
    common = {
        "token_embd.weight": (128, 256),
        "output.weight": (128, 256),
        TARGET: (256, 256),
        "blk.0.ffn_down.weight": (256, 512),
        NORM: (256,),
    }
    for mode, members in (("dense", common), ("moe", {**common, EXPERT: (2, 512, 256)})):
        writer = GGUFWriter(out / f"{mode}-f32.gguf", "llama")
        writer.add_name(f"synthetic-{mode}-quantize-only")
        writer.add_file_type(LlamaFileType.ALL_F32)
        writer.add_context_length(128)
        writer.add_embedding_length(256)
        writer.add_block_count(1)
        writer.add_feed_forward_length(512)
        writer.add_head_count(8)
        writer.add_head_count_kv(8)
        writer.add_rope_dimension_count(32)
        writer.add_layer_norm_rms_eps(1e-5)
        if mode == "moe":
            writer.add_expert_count(2)
            writer.add_expert_used_count(1)
        for seed, (name, shape) in enumerate(members.items()):
            writer.add_tensor(name, weights(shape, seed))
        save(writer)

    # Registered side-input data for the smoke, not model-produced calibration.
    writer = GGUFWriter(out / CALIBRATION, "imatrix")
    writer.add_array("imatrix.datasets", [DATASET])
    writer.add_uint32("imatrix.chunk_count", 1)
    writer.add_uint32("imatrix.chunk_size", 32)
    for name, shape in {**common, EXPERT: (2, 512, 256)}.items():
        if len(shape) < 2:
            continue
        experts = shape[0] if len(shape) == 3 else 1
        sums = np.resize(np.arange(16, 48, dtype=np.float32), (experts, shape[-1]))
        writer.add_tensor(f"{name}.in_sum2", sums)
        writer.add_tensor(f"{name}.counts", np.full(experts, 32, dtype=np.float32))
    save(writer)
    (out / "inputs.json").write_text(json.dumps({
        "dense": {"kind": "gguf", "host": host, "path": str(out / "dense-f32.gguf")},
        "moe": {"kind": "gguf", "host": host, "path": str(out / "moe-f32.gguf")},
        "calibration": {"kind": "imatrix", "host": host, "path": str(out / CALIBRATION)},
    }, indent=2) + "\n", encoding="utf-8")


def inspect(path):
    file_size = path.stat().st_size
    need(file_size <= 8 * 1024 * 1024, "output exceeded fixture size bound")
    parsed = GGUFReader(path)
    data_offset = parsed.data_offset
    ranges = []
    tensors = {}
    with path.open("rb") as stream:
        for tensor in parsed.tensors:
            need(tensor.name not in tensors, f"duplicate tensor {tensor.name}")
            tensor_type = tensor.tensor_type
            block, block_bytes = GGML_QUANT_SIZES[tensor_type]
            shape = tuple(int(x) for x in tensor.shape)
            need(shape[0] % block == 0, f"misaligned shape for {tensor.name}")
            size = int(np.prod(shape)) // block * block_bytes
            begin = tensor.data_offset
            end = begin + size
            need(data_offset <= begin and (begin - data_offset) % parsed.alignment == 0
                 and end <= file_size, f"invalid payload extent for {tensor.name}")
            ranges.append((begin, end))
            stream.seek(begin)
            payload = stream.read(size)
            need(len(payload) == size and size == tensor.n_bytes, f"short payload for {tensor.name}")
            tensors[tensor.name] = {"shape": shape, "type": tensor_type.name, "type_id": int(tensor_type),
                                    "bytes": size, "sha256": hashlib.sha256(payload).hexdigest()}
    ranges.sort()
    need(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), "overlapping tensor payloads")
    calibration = {k: v.contents() for k, v in parsed.fields.items()
                   if k.startswith("quantize.imatrix.")}
    return int(parsed.fields["general.file_type"].contents()), tensors, calibration


ALIASES = {"Q3_K": "Q3_K_M", "Q4_K": "Q4_K_M", "Q5_K": "Q5_K_M"}
# Expected encoding of the aligned attention-Q probe, not a claim that all
# tensors in a mixed preset use one type.
PRESET_TARGETS = {
    "MXFP4_MOE": "Q8_0", "IQ2_S": "IQ2_XS", "IQ2_M": "IQ2_S",
    "IQ3_XXS": "IQ2_S", "IQ3_XS": "IQ3_XXS", "IQ3_M": "IQ3_S", "Q2_K_S": "Q2_K",
    "Q3_K_S": "Q3_K", "Q3_K_M": "Q3_K", "Q3_K_L": "Q3_K",
    "Q4_K_S": "Q4_K", "Q4_K_M": "Q4_K", "Q5_K_S": "Q5_K", "Q5_K_M": "Q5_K",
}
FFN_DOWN_TYPES = {
    "IQ3_XXS": "IQ3_XXS", "IQ3_XS": "IQ3_S",
    "Q2_K": "Q3_K", "Q3_K": "Q4_K", "Q3_K_M": "Q4_K", "Q3_K_L": "Q5_K",
    "Q4_K": "Q6_K", "Q4_K_M": "Q6_K", "Q5_K": "Q6_K", "Q5_K_M": "Q6_K",
}
EMBED_TYPES = {
    "Q2_0": "Q4_K", "TQ1_0": "Q4_K", "TQ2_0": "Q4_K",
    "IQ2_XXS": "Q2_K", "IQ2_XS": "Q2_K", "IQ1_S": "Q2_K", "IQ1_M": "Q2_K",
    "IQ2_S": "IQ3_S", "IQ2_M": "IQ3_S", "IQ3_XXS": "IQ3_S", "IQ3_XS": "IQ3_S",
}
OUTPUT_Q5_K = {"IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ1_S", "IQ1_M"}


def baseline_types(preset, target_type):
    output_type = ("Q5_K" if preset in OUTPUT_Q5_K else
                   preset if preset in {"F16", "BF16", "Q8_0"} else "Q6_K")
    return {
        "token_embd.weight": EMBED_TYPES.get(preset, target_type),
        "output.weight": output_type,
        "blk.0.ffn_down.weight": FFN_DOWN_TYPES.get(preset, target_type),
        NORM: "F32",
    }


def run_case(binary, out, name, mode, preset, calibration, exact, inputs):
    path = out / f"{name}.gguf"
    cmd = [str(binary), "--max-buffer-size", "16"]
    if calibration:
        cmd += ["--imatrix", str(out / CALIBRATION)]
    if exact is not None:
        cmd += ["--tensor-type", r"^blk\.0\.attn_q\.weight$=" + exact["id"].lower()]
    cmd += [str(out / f"{mode}-f32.gguf"), str(path), preset, NTHREADS]
    log = out / f"{name}.log"
    with log.open("wb") as stream:
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=30,
                              env={**os.environ, "OMP_NUM_THREADS": NTHREADS})
    need(proc.returncode == 0, f"encoder exited {proc.returncode}; see {log.name}")
    ftype, tensors, imatrix = inspect(path)
    source = inputs[mode]
    need(tensors.keys() == source.keys(), "tensor membership changed")
    for tensor, actual in tensors.items():
        need(actual["shape"] == source[tensor]["shape"], f"shape changed for {tensor}")
        if actual["type"] == "F32":
            need(actual["sha256"] == source[tensor]["sha256"], f"F32 payload changed for {tensor}")
    expected_file_type = int(LlamaFileType[f"MOSTLY_{ALIASES.get(preset, preset)}"])
    expected = exact["id"] if exact is not None else PRESET_TARGETS.get(preset, preset)
    case = {"file_type": ftype, "tensors": tensors}
    errors = []
    if calibration and (imatrix.get("quantize.imatrix.dataset") != DATASET
                        or imatrix.get("quantize.imatrix.entries_count") != 5
                        or imatrix.get("quantize.imatrix.chunks_count") != 1):
        errors.append("output did not record the synthetic importance matrix")
    if not calibration and imatrix:
        errors.append("unexpected importance matrix metadata")
    if ftype != expected_file_type:
        errors.append(f"expected file type {expected_file_type}, got {ftype}")
    if tensors[TARGET]["type"] != expected:
        errors.append(f"{TARGET}: expected {expected}, got {tensors[TARGET]['type']}")
    if exact is not None:
        block = exact["block_size"]
        expected_bytes = int(np.prod(tensors[TARGET]["shape"])) // block * exact["block_bytes"]
        if tensors[TARGET]["type_id"] != exact["type_id"] or tensors[TARGET]["bytes"] != expected_bytes:
            errors.append("tensor type ID or byte size differs from catalog encoding")
        carrier = inputs["carriers"].get(preset, {})
        for other in sorted(tensors.keys() - {TARGET}):
            if tensors[other] != carrier.get(other):
                errors.append(f"exact override changed non-target tensor {other}")
    elif preset == "MXFP4_MOE":
        for tensor, actual in tensors.items():
            wanted = "MXFP4" if tensor == EXPERT else "F32" if tensor == NORM else "Q8_0"
            if actual["type"] != wanted:
                errors.append(f"{tensor}: expected {wanted}, got {actual['type']}")
        case["note"] = ("Dense weights are Q8_0, not MXFP4" if mode == "dense"
                        else "Only aligned expert weights are MXFP4")
    else:
        for tensor, wanted in baseline_types(preset, expected).items():
            if tensors[tensor]["type"] != wanted:
                errors.append(f"{tensor}: expected {wanted}, got {tensors[tensor]['type']}")
    return case, errors


def smoke(binary, out):
    catalog = json.loads(CATALOG.read_text())
    need(catalog["encoder_revision"] == REVISION, "catalog changed encoder revision; review fixture expectations")
    # This oracle is linked to the stock reader. Fork-specific layouts
    # have a separate quantizer and must be exercised by its matching reader.
    presets, exacts = ([p for p in catalog[k] if not p.get("encoder_family")]
                       for k in ("presets", "tensor_types"))
    need(len(presets) == len({p["id"] for p in presets}) == 37, "expected 37 unique global presets")
    need(len(exacts) == len({p["id"] for p in exacts}) == 27, "expected 27 unique exact types")
    inputs = {mode: inspect(out / f"{mode}-f32.gguf")[1] for mode in ("dense", "moe")}
    inputs["carriers"] = {}
    runs = []

    def attempt(name, mode, preset, imatrix, exact=None):
        try:
            case, errors = run_case(binary, out, name, mode, preset, imatrix, exact, inputs)
        except (OSError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired) as exc:
            case, errors = {}, [str(exc)]
        row = {"case": name, "preset": preset, "fixture": mode, "calibration": imatrix,
               **case, "status": "failed" if errors else "passed", "errors": errors}
        runs.append(row)
        print(f"{row['status']:6} {name}" + (f": {'; '.join(errors)}" if errors else ""), flush=True)
        return row

    for preset in presets:
        name = preset["id"]
        row = attempt(f"global-{name}", "dense", name, preset["requires_imatrix"])
        if name in (CARRIER, ALTERNATE_CARRIER) and "tensors" in row:
            inputs["carriers"][name] = row["tensors"]
    attempt("global-MXFP4_MOE-experts", "moe", "MXFP4_MOE", False)
    for exact in exacts:
        # Always distinguish the requested type from the carrier; ignoring a
        # Q8_0 override must not pass merely because the default was also Q8_0.
        carrier = ALTERNATE_CARRIER if exact["id"] == CARRIER else CARRIER
        attempt(f"exact-{exact['id']}", "dense", carrier, exact["requires_imatrix"], exact)
    failed = sum(row["status"] == "failed" for row in runs)
    with binary.open("rb") as stream:
        binary_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    (out / "report.json").write_text(json.dumps({
        "schema": "brewery-gguf-catalog-smoke-v1", "catalog_revision": REVISION,
        "encoder_binary_sha256": binary_sha,
        "case_count": len(runs), "failed": failed, "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"{len(runs) - failed}/{len(runs)} cases passed; report: {out / 'report.json'}")
    return int(bool(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--quantize", type=Path, help="existing pinned CPU llama-quantize executable; omit for fixture only")
    parser.add_argument("--host", default="fixture", help="registered worker alias used in inputs.json")
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        out.mkdir(mode=0o700)
        fixtures(out, args.host)
        return smoke(args.quantize.resolve(strict=True), out) if args.quantize else 0
    except (ValueError, OSError) as exc:
        print(f"gguf_catalog_smoke: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
