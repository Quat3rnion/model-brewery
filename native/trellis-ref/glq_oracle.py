"""Generate original-basis GLQ test vectors; not a native runtime dependency."""

import argparse
from pathlib import Path
import struct
import sys

# Use the same hash-verified reference and acceptance input as tools/glq_ref.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import glq_ref  # noqa: E402


def write_vectors(source, output):
    import torch
    from safetensors.torch import load, save

    if sys.byteorder != "little" or torch.version.cuda is not None or torch.version.hip is not None:
        raise ValueError("use a little-endian host and CPU-only PyTorch")
    torch.set_num_threads(1)
    with glq_ref._upstream(source) as gt, torch.no_grad(), torch.compiler.set_stance("force_eager"):
        # Cover 1..4 bits per stage and a non-power-of-two, multi-group rotation.
        cases = [(32, 64, bpw) for bpw in range(2, 9)] + [(96, 96, 5)]
        with output.open("wb") as f:
            f.write(b"GLQ3V1\0\0")
            f.write(struct.pack("<I", len(cases)))
            for m, n, bpw in cases:
                rng = torch.Generator(device="cpu").manual_seed(2718)
                w = torch.randn(m, n, generator=rng, dtype=torch.float32, device="cpu")
                a = torch.randn(2 * n, n, generator=rng, dtype=torch.float32, device="cpu")
                h = a.T @ a / (2 * n)
                books = [gt.TrellisCodebook(variant="3inst", K=k, device="cpu")
                         for k in gt.trellis_rvq_recipe(bpw)]
                decoded, artifacts = gt.quantize_layer_trellis_rht(w, h, books)
                artifacts = load(save(artifacts))
                x = torch.randn(3, n, generator=rng, dtype=torch.float32, device="cpu")
                x[0].zero_()
                x[0, -1] = 1  # include a basis vector for the trailing block
                # Independent original-basis dense reference, not the packed matmul.
                y = x @ decoded.T
                f.write(struct.pack("<4I2f", m, n, bpw, len(x), artifacts["Wscale"].item(),
                                    artifacts["inv_resid_scale2"].item() if bpw > 4 else 0))
                for key in ("trellis_packed", "trellis_packed2", "SU", "SV"):
                    if key in artifacts:
                        f.write(artifacts[key].numpy().tobytes())
                f.write(x.numpy().tobytes())
                f.write(y.numpy().tobytes())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    write_vectors(args.source, args.output)
