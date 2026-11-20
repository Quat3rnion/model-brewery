#!/usr/bin/env python3
"""Bounded, offline GLQ 3INST CPU codec probe (one linear, not an HF model).

Pass a local unpacked source tree from https://github.com/cnygaard/glq/releases/tag/v0.8.38.
Use CPU PyTorch with torch.compiler.set_stance, NumPy and safetensors. The probe does not
import GLQ's model stack, compile a kernel, or use a published model or calibration dataset.
It writes a temporary config and safetensors file next to this script and removes them.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType


REVISION = "6aaaa580a5e3ff5160c94188211e7a6379724ea4"  # cnygaard/glq v0.8.38
SOURCES = {
    "ldlq": "f1dd14700286748bcfde0a3fdd7e364cf74c02ffff04e5ab6d23ba4339f4061e",
    "hadamard": "51b156605ef90853251549f7fc6dc0f63ca076fa1e15d2b64b2e10230f2d597c",
    "rht": "1a253b40e8d7a0f0ce65748fa6362fce1bc364dc498dd280c6a51e9d73be3513",
    "trellis_recipe": "9c4dda6c36de26bc54a9a91a742f420db659666f6785d34beeed7ef698f0a802",
    "trellis": "34b8bb5c4f8588eb50a9cdeecaa473ff0be0fd137f1badc2215754defbee9f1c",
}
LAYOUT = {
    "quant_method": "glq", "codebook": "trellis", "variant": "3inst",
    "trellis_layout": "kernel", "codesz": 16, "block_diagonal": True,
}
RATES = {2: (2,), 3: (3,), 4: (4,), 5: (4, 1), 6: (4, 2), 7: (4, 3), 8: (4, 4)}
M, N = 32, 64
PREFIX = "model.layers.0.self_attn.q_proj."


def read_recipe(config):
    """Check the quantization config for the block-diagonal 3INST kernel subset."""
    q = config.get("quantization_config") if isinstance(config, dict) else None
    if (not isinstance(q, dict) or q.get("block_diagonal") is not True
            or any(q.get(key) != value for key, value in LAYOUT.items())):
        raise ValueError("expected block-diagonal GLQ 3INST kernel metadata")
    bpw = q.get("bpw")
    if type(bpw) not in (int, float) or bpw not in RATES:
        raise ValueError("expected a uniform integer 3INST bitrate from 2 to 8")
    return RATES[bpw]


@contextmanager
def _upstream(source):
    """Execute only digest-pinned sources; never import GLQ's model entry point."""
    files = {}
    for name, digest in SOURCES.items():
        path = source / "glq" / f"{name}.py"
        with path.open("rb") as f:
            data = f.read(128 * 1024)
            if f.read(1) or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"{path} does not match GLQ v0.8.38 ({REVISION})")
        files[name] = (path, data)
    if any(key == "glq" or key.startswith("glq.") for key in sys.modules):
        raise ValueError("run the reference probe in a process without an imported GLQ package")
    if any(importlib.util.find_spec(dep) is not None
           for dep in ("triton", "fast_hadamard_transform")):
        raise ValueError("run with a CPU environment without optional Triton/Hadamard extensions")
    package = ModuleType("glq")
    package.__path__ = []
    sys.modules["glq"] = package
    try:
        for name, (path, data) in files.items():
            module = ModuleType(f"glq.{name}")
            module.__file__ = str(path)
            module.__package__ = "glq"
            sys.modules[module.__name__] = module
            setattr(package, name, module)
            exec(compile(data, str(path), "exec"), module.__dict__)
        yield package.trellis
    finally:
        for name in files:
            sys.modules.pop(f"glq.{name}", None)
        sys.modules.pop("glq", None)


def _components(tensors, stages, torch):
    """Validate the exact native component set, dtypes, tile widths, and scales."""
    fields = {
        "trellis_packed": (torch.int16, (M * N // 256, 16 * stages[0])),
        "SU": (torch.float16, (M,)), "SV": (torch.float16, (N,)),
        "Wscale": (torch.float32, ()),
    }
    if len(stages) == 2:
        fields["trellis_packed2"] = (torch.int16, (M * N // 256, 16 * stages[1]))
        fields["inv_resid_scale2"] = (torch.float32, ())
    if set(tensors) != {PREFIX + key for key in fields}:
        raise ValueError("native tensor set does not match the GLQ 3INST stage recipe")
    result = {}
    for key, (dtype, shape) in fields.items():
        tensor = tensors[PREFIX + key]
        if tensor.device.type != "cpu" or tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(f"{key}: invalid GLQ 3INST dtype or shape")
        if key in ("SU", "SV") and not (tensor.abs() == 1).all():
            raise ValueError(f"{key}: invalid GLQ RHT signs")
        if not shape and not (torch.isfinite(tensor) and tensor > 0):
            raise ValueError(f"{key}: invalid GLQ 3INST scale")
        result[key] = tensor
    return result


def _decode_stage(raw, K):
    """Independent scalar oracle: follow the upstream CPU layout's lane/slot addresses."""
    # See glq/csrc/cpu/glq_trellis_layout.hpp. A 32x32 region holds 4 tiles;
    # each lane owns one K-byte little-endian chunk from each tile. Read a
    # circular MSB-first window; the 1-bit residual spans two following lanes.
    weights = [0.0] * (M * N)
    for block in range(M * N // 1024):
        rbase, cbase = divmod(block, N // 32)
        for slot in range(4):
            start = (block * 32 * 4 + slot) * K
            chunks = [int.from_bytes(raw[offset:offset + K], "little")
                      for offset in range(start, start + 32 * 4 * K, 4 * K)]
            for lane, chunk in enumerate(chunks):
                continuation = chunks[(lane + 1) % 32]
                if K == 1:
                    continuation = (continuation << 8) | chunks[(lane + 2) % 32]
                else:
                    continuation >>= 8 * K - 16
                window = (chunk << 16) | continuation
                row = rbase * 32 + (slot & 1) * 16 + lane // 4
                col = cbase * 32 + (slot >> 1) * 16 + (lane % 4) * 2
                for j in range(8):
                    state = (window >> (K * (8 - j))) & 0xffff
                    pattern = ((state * 89226354 + 64248484) & 0x8fff8fff) ^ 0x3b603b60
                    low, high = struct.unpack("<ee", struct.pack("<I", pattern))
                    # The add is rounded to FP16 before promotion to FP32.
                    value, = struct.unpack("<e", struct.pack("<e", low + high))
                    i = (row + (j & 2) * 4) * N + col + (j & 4) * 2 + (j & 1)
                    weights[i] = value
    return weights


def probe(source, bpw):
    """Encode and read a deterministic new 32x64 matrix using the native component layout."""
    config = {"quantization_config": {**LAYOUT, "bpw": bpw}}
    stages = read_recipe(config)
    if sys.byteorder != "little":
        raise ValueError("the reference probe requires a little-endian CPU")
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise ValueError("CPU PyTorch, NumPy and safetensors are required") from exc
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise ValueError("use CPU-only PyTorch for this probe")
    if not hasattr(getattr(torch, "compiler", None), "set_stance"):
        raise ValueError("CPU PyTorch with torch.compiler.set_stance is required")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _upstream(source) as gt, torch.no_grad(), torch.compiler.set_stance("force_eager"):
            if list(stages) != gt.trellis_rvq_recipe(bpw):
                raise ValueError("3INST recipe differs from the pinned encoder")
            codebooks = [gt.TrellisCodebook(variant="3inst", K=K, device="cpu") for K in stages]
            rng = torch.Generator(device="cpu").manual_seed(2718)
            weights = torch.randn(M, N, generator=rng, dtype=torch.float32, device="cpu")
            features = torch.randn(2 * N, N, generator=rng, dtype=torch.float32, device="cpu")
            hessian = features.T @ features / (2 * N)  # synthetic SPD calibration, not model data
            encoded, artifacts = gt.quantize_layer_trellis_rht(weights, hessian, codebooks)
            with tempfile.TemporaryDirectory(prefix="glq_ref_", dir=Path(__file__).resolve().parent) as tmp:
                pack = Path(tmp) / "model.safetensors"
                metadata = Path(tmp) / "config.json"
                metadata.write_text(json.dumps(config), encoding="utf-8")
                save_file({PREFIX + k: v for k, v in artifacts.items()}, pack)
                reloaded = load_file(pack, device="cpu")
                decoded_stages = read_recipe(json.loads(metadata.read_text(encoding="utf-8")))
            if any(key not in reloaded or not torch.equal(value, reloaded[key])
                   for key, value in ((PREFIX + k, v) for k, v in artifacts.items())):
                raise ValueError("GLQ native component changed on disk")
            components = _components(reloaded, decoded_stages, torch)
            packed = [components["trellis_packed"]]
            scales = [1.0]
            dropped_residual_refused = False
            if len(stages) == 2:
                packed.append(components["trellis_packed2"])
                scales.append(components["inv_resid_scale2"].item())
                try:
                    _components({k: v for k, v in reloaded.items()
                                 if k != PREFIX + "trellis_packed2"}, stages, torch)
                except ValueError:
                    dropped_residual_refused = True
                if not dropped_residual_refused:
                    raise ValueError("missing residual tensor was accepted")
            raw = [tensor.numpy().tobytes() for tensor in packed]
            readers = [gt.TrellisCodebook(variant="3inst", K=K, device="cpu") for K in decoded_stages]
            reference_stages = []
            for reader, tensor, payload, K in zip(readers, packed, raw, decoded_stages):
                expected = torch.tensor(_decode_stage(payload, K), dtype=torch.float32).reshape(M, N)
                actual = gt.decode_layer(reader, tensor, M, N)
                if not torch.equal(expected.view(torch.int32), actual.view(torch.int32)):
                    raise ValueError(f"K={K}: independent 3INST scalar decode disagrees with upstream")
                reference_stages.append(expected)
            normalized = gt.decode_layer_nstage(readers, packed, scales, M, N)
            reference = reference_stages[0]
            if len(stages) == 2:
                reference = reference + reference_stages[1] * scales[1]
            if not torch.equal(reference.view(torch.int32), normalized.view(torch.int32)):
                raise ValueError("independent normalized 3INST decode disagrees with upstream")
            rht = gt.RHT(M, N, device="cpu", block_diagonal=True)
            rht.su, rht.sv = components["SU"].float(), components["SV"].float()
            restored = rht.inverse_transform_weights(normalized * components["Wscale"])
            if not torch.allclose(restored, encoded, rtol=1e-6, atol=1e-6):
                raise ValueError("restored GLQ matrix differs from encoder reconstruction")
            return {
                "scope": "single_matrix_cpu_codec", "upstream_revision": REVISION, "bpw": bpw,
                "shape": [M, N], "stage_bits": list(stages), "packed_bytes": [len(x) for x in raw],
                "normalized_decode": "bit_exact", "restored_decode": "matches_encoder",
                "dropped_residual_refused": dropped_residual_refused,
                "restored_max_abs_difference": (restored - encoded).abs().max().item(),
            }
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help=f"local GLQ source tree pinned to {REVISION}")
    parser.add_argument("--bpw", type=int, choices=RATES,
                        help="one rate to test (default: all seven legal integer rates)")
    args = parser.parse_args()
    try:
        results = [probe(args.source, bpw) for bpw in (RATES if args.bpw is None else (args.bpw,))]
        print(json.dumps(results if args.bpw is None else results[0], sort_keys=True))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
