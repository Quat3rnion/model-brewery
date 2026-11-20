#!/usr/bin/env python3
"""Bounded Cornell QTIP CPU probe; see --help. Not a model converter."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


PIN = "e90c6688c8dfae326a3a81b5eb032db7c6680ec0"
UPSTREAM = f"https://github.com/Cornell-RelaxML/qtip/tree/{PIN}"
# Only these reviewed files may be read as upstream code. Do not add the source
# directory to sys.path or execute its package initializers, scripts or models.
FILES = {
    "lib/utils/kernel_check.py": "d22902b020d44e6602d090ee10b8feba363654b4efb066fabdfc27b9d2f51ddb",
    "lib/utils/kernel_decompress.py": "01b6520c8f39982ac5f35de58364f31f79f00555350c8c1776e4fd9b1ca0a63f",
    "lib/utils/matmul_had.py": "a3fd9da34b923bed46c1dddc2995ff9a73ef646f6ec86569ec40c29a4c49dd86",
    "lib/utils/math_utils.py": "65d50936e87b2c266806de201dea89b2d74a2ed38e33ef462bd8c3aafb333844",
    "lib/utils/misc.py": "9f79025604a4ebf9ef0d4031f85b65e298fda4c068b475f8d9e7660345ac18a1",
    "lib/utils/data_utils.py": "07e830ec93c13796950d16ca72770de738cfe42f719355963ad14664131f3f3e",
    "lib/codebook/bitshift.py": "a299ae97d2ccc80a142095c3c16ed619b435b68736fd52702ab396bc37218531",
    "lib/linear/quantized_linear.py": "1660be3de95e90a802bf2d039eae0f9e80884d4fc1e3ea701e962a9c0147b21a",
    "lib/algo/ldlq.py": "793e364fbe91e5b28740d0fc81a6e8618daa6a6a8ce5adbf9b877ba2e46e5bbe",
    "lib/codebook/__init__.py": "c7691d3c62e8d1e2f25cfed66c513c42779ed7927fd1162bbb56372b5b2396ef",
}

FEATURES = 32
SAMPLES = 64
SEED = 42
SIGMA = 0.01
SCALE_OVERRIDE = 0.9
TILE = 16
# Rows are examples actually exercised, not the mathematical parameter range.
ROWS = (
    ("lut", 10, 2, 10),
    ("1mad", 16, 1, 0),
    ("2mad", 16, 1, 0),
    ("3inst", 16, 1, 0),
    ("quantlut", 16, 1, 8),
    ("quantlut", 16, 2, 8),
    ("quantlut_sym", 16, 2, 8),
)


def verify_source(root):
    source = {}
    for path, digest in FILES.items():
        candidate = root / path
        if not candidate.is_file():
            raise ValueError(f"missing pinned source: {path}")
        with candidate.open("rb") as stream:
            data = stream.read(2 << 20)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"pinned source digest mismatch: {path}")
        source[path] = data
    return source


def forbidden(*args, **kwargs):
    raise RuntimeError("network datasets and pickle are disabled in this CPU probe")


def module(name, package=False):
    result = ModuleType(name)
    if package:
        result.__path__ = []
    sys.modules[name] = result
    parent, _, local = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], local, result)
    return result


def load_qtip(source):
    # The upstream package initializers pull in CUDA, dataset and model code.
    # Execute only the verified modules used below. An empty kdict is unused
    # here; no fused operator or mock output is registered.
    module("lib", package=True)
    utils = module("lib.utils", package=True)
    module("lib.codebook", package=True).kdict = {}
    module("lib.algo", package=True)
    module("lib.linear", package=True)
    module("glog")
    module("datasets").load_dataset = forbidden
    fast_had = module("fast_hadamard_transform")

    def load(name):
        result = module(name)
        path = name.replace(".", "/") + ".py"
        exec(compile(source[path], f"{PIN}/{path}", "exec"), result.__dict__)
        return result

    utils.clean = load("lib.utils.misc").clean
    utils.has_kernel = load("lib.utils.kernel_check").has_kernel
    load("lib.utils.kernel_decompress")
    had = load("lib.utils.matmul_had")
    utils.get_hadK = had.get_hadK
    utils.matmul_hadU_cuda = had.matmul_hadU_cuda

    def cpu_hadamard(x, scale):
        # Only the K=1 (power-of-two) branch is used in this harness.
        if had.get_hadK(x.shape[-1])[1] != 1 or x.device.type != "cpu":
            raise ValueError("unsupported transform in CPU probe")
        return had.matmul_hadU(x) * (scale * math.sqrt(x.shape[-1]))

    fast_had.hadamard_transform = cpu_hadamard
    data = load("lib.utils.data_utils")
    utils.dtype_from_str = data.dtype_from_str
    return SimpleNamespace(
        utils=utils,
        had=had,
        math=load("lib.utils.math_utils"),
        data=data,
        bitshift=load("lib.codebook.bitshift"),
        linear=load("lib.linear.quantized_linear").QuantizedLinear,
        ldlq=load("lib.algo.ldlq").LDLQ,
    )


def fused_registry(source):
    # Read the shape table as data, without importing the CUDA extension.
    tree = ast.parse(source["lib/codebook/__init__.py"])
    value, = (
        stmt.value for stmt in tree.body if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "kernels" for t in stmt.targets)
    )
    return ast.literal_eval(value)


def fixture(torch, qtip):
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    weights = torch.randn(FEATURES, FEATURES, generator=rng) * 0.2
    activations = torch.randn(SAMPLES, FEATURES, generator=rng)
    # Correlate adjacent channels so LDLQ sees more than a diagonal Hessian.
    activations += 0.5 * activations.roll(1, dims=1)
    stats = {
        "flatH": qtip.data.sym_to_flat(activations.T @ activations / SAMPLES),
        "n": FEATURES,
        "ct": SAMPLES,
    }
    h = qtip.math.regularize_H(qtip.math.flat_to_sym(stats["flatH"], stats["n"]), SIGMA)
    su, sv = torch.randn(2, FEATURES, generator=rng).sign()
    inputs = torch.randn(3, FEATURES, generator=rng)  # no batch-one fused dispatch
    return weights, h, su, sv, inputs


def round_trip(torch, safe, qtip, registry, fixture_data, mode, L, K, V, Q):
    weights, h, su, sv, inputs = fixture_data
    m, n = weights.shape
    args = SimpleNamespace(td_x=TILE, td_y=TILE, V=V, K=K)
    swizzled = qtip.utils.has_kernel(mode, L, K, V, Q, TILE, TILE)
    # Always pass an explicit FP16 TLUT when needed. Do not use the upstream
    # HYB constructor's /tmp/kmeans cache (torch.load), or claim a learned LUT.
    rng = torch.Generator(device="cpu").manual_seed(SEED + 1)
    tlut = torch.randn(1 << Q, V, generator=rng).half() if Q else None
    cb = qtip.bitshift.bitshift_codebook(L, K, V, Q, mode, tlut=tlut).half()
    had = qtip.had.matmul_hadUt
    # Unsplit path of upstream lib/algo/finetune.py: rotate, normalize,
    # quantize with upstream block-LDLQ, and pack cyclic trellises.
    rotated_w = had(had(weights.T * sv).T * su)
    rotated_h = had(had(h * su).T * su)
    scale = rotated_w.square().mean().sqrt() / (cb.lut.float().square().mean().sqrt() * SCALE_OVERRIDE)
    lower, _ = qtip.math.block_LDL(rotated_h, TILE)
    lower.fill_diagonal_(0)
    normalized = rotated_w / scale
    hat_w, indices = qtip.ldlq(normalized, lower, cb, args, buf_cols=n, for_kernel=swizzled)
    packed = cb.pack_trellis(
        indices.reshape(m // TILE, TILE, n // TILE, TILE // V)
        .transpose(1, 2).reshape(-1, TILE * TILE // V)
    )
    if swizzled:
        # Checkpoint byte swizzle from upstream finetune.py, as distinct
        # from the per-tile element permutation done inside upstream LDLQ.
        packed = (
            packed.view(torch.uint8).reshape(-1, 2).flip(-1)
            .reshape(m // 32, 2, n // 32, 2, TILE * TILE // 8, K)
            .permute(0, 2, 4, 3, 1, 5).flip(-1).contiguous()
            .flatten().view(torch.int16).reshape(packed.shape)
        )
    else:
        packed = packed.view(torch.int16)
    error = normalized - hat_w
    proxy_err = torch.trace(error @ rotated_h @ error.T) / torch.trace(normalized @ rotated_h @ normalized.T)
    layer = {
        "trellis": packed, "SU": su.half(), "SV": sv.half(), "Wscale": scale,
        "rcp": torch.tensor(0), "tp_rank": torch.tensor(1), "proxy_err": proxy_err,
    }
    if tlut is not None:
        layer["tlut"] = tlut
    # SafeTensors fixture envelope, not upstream .pt or a complete HF model.
    # Upstream unpack_quip receives the restored native layer fields.
    payload = safe.save(layer, metadata={
        "upstream_commit": PIN, "decode_mode": mode, "L": str(L), "K": str(K),
        "V": str(V), "tlut_bits": str(Q), "td_x": str(TILE), "td_y": str(TILE),
    })
    restored = safe.load(payload)
    if set(restored) != set(layer) or any(not torch.equal(v, restored[k]) for k, v in layer.items()):
        raise ValueError("SafeTensors fixture did not preserve native fields")
    linear = qtip.linear(n, m, TILE, TILE, L, K, V, Q, mode, dtype=torch.float16)
    qtip.data.unpack_quip(linear, restored)
    result = linear(inputs)
    reader = linear.codebook_class
    if swizzled:
        decoded = reader.get_hatW_kernel(linear.trellis, m, n)
    else:
        states = reader.cb.unpack_trellis(linear.trellis, TILE * TILE)
        decoded = reader.get_hatW(states, m, n)
    # Scalar codebooks are regenerated in FP32 on read; compare at the
    # encoder's FP16 codebook precision. Separately check the dense linear.
    reconstruction_max_abs = (decoded.half().float() - hat_w).abs().max().item()
    expected_w = had(had(hat_w).T).T * scale * su * sv[:, None]
    linear_max_abs = (result - inputs @ expected_w.T).abs().max().item()
    if reconstruction_max_abs != 0 or not linear_max_abs < 0.003 or not 0 < proxy_err < 1:
        raise ValueError(
            f"CPU round trip mismatch: {mode}/{L}/{K}/{V}/{Q}: "
            f"reconstruction={reconstruction_max_abs}, linear={linear_max_abs}, proxy_err={proxy_err.item()}"
        )
    return {
        "decode_mode": mode, "L": L, "K": K, "V": V, "tlut_bits": Q,
        "td_x": TILE, "td_y": TILE, "weight_shape": [m, n],
        "packing": "cuda_swizzled" if swizzled else "reference",
        "trellis_bytes": packed.numel() * packed.element_size(),
        "fixture_bytes": len(payload),
        "fixture_has_fused_matvec": swizzled and (m, 1, n, K) in registry,
        "native_tensors": {
            k: {"dtype": str(v.dtype).removeprefix("torch."), "shape": list(v.shape)}
            for k, v in restored.items()
        },
        "reconstruction_max_abs": reconstruction_max_abs,
        "reconstruction_compare_dtype": "float16",
        "linear_max_abs": linear_max_abs, "proxy_err": proxy_err.item(),
    }


def probe(source):
    import torch
    from safetensors import torch as safe

    if torch.version.cuda is not None or torch.version.hip is not None:
        raise ValueError("use a CPU-only PyTorch build for this probe")
    torch.set_num_threads(1)
    torch.compiler.set_stance("force_eager")
    torch.load = forbidden
    qtip = load_qtip(source)
    registry = fused_registry(source)
    parameters = [(mode, L, K, V, Q) for mode, L, V, Q in ROWS for K in range(1, 5)]
    parameters += [("quantlut_sym", 16, K, 2, 9) for K in (2, 3, 4)]
    with torch.no_grad():
        data = fixture(torch, qtip)
        rows = [round_trip(torch, safe, qtip, registry, data, *params) for params in parameters]
    return {
        "upstream": UPSTREAM, "upstream_commit": PIN,
        "torch_version": torch.__version__, "fixture_format": "safetensors",
        "scope": "synthetic single matrix CPU reference, no tensor parallelism or finetuning",
        "calibration": {
            "kind": "synthetic_input_second_moment", "samples": SAMPLES, "seed": SEED,
            "sigma_reg": SIGMA, "scale_override": SCALE_OVERRIDE, "fields": ["flatH", "n", "ct"],
            "model_calibration": "not run; approved corpus and model inputs required",
            "upstream_hessian_script": "quantize_llama/input_hessian_llama.py",
            "upstream_corpus": "togethercomputer/RedPajama-Data-1T-Sample",
            "upstream_finetune_script": "quantize_llama/quantize_finetune_llama.py",
        },
        "gpu_kernel_executed": False,
        "upstream_fused_matvec_registry": {
            "mode": "quantlut_sym", "L": 16, "V": 2, "tlut_bits": 9, "td_x": 16, "td_y": 16,
            "shape_order": ["out_features", "batch", "in_features", "K"],
            "registered_shapes": registry,
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use isolated Python (-I) with CPU-only PyTorch >=2.6, NumPy, tqdm and SafeTensors. "
        "Accepts no model, dataset or checkpoint files. Outputs a JSON report, not a checkpoint.",
    )
    parser.add_argument("--source", required=True, type=Path, help=f"local source tree at {PIN}")
    args = parser.parse_args()
    try:
        report = probe(verify_source(args.source))
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
