"""Compare a logical coordinate and its shared block in two verified GGUF fixtures."""
import argparse
import json
import math
from pathlib import Path

import gguf
import numpy as np


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def compare(old, new, name, index):
    need(all(p.is_file() and p.stat().st_size <= 32 << 20 for p in (old, new)),
         "expected two verified GGUF fixtures of at most 32 MiB each")
    a, b = (gguf.GGUFReader(p) for p in (old, new))
    # This is a comparison of two runs of the same quantizer and recipe.
    # Include the complete header/table, so a rename, changed type, reordered
    # tensor, altered metadata or new alignment cannot pass as a scalar edit.
    need(old.stat().st_size == new.stat().st_size and a.data_offset == b.data_offset and
         np.array_equal(a.data[:a.data_offset], b.data[:b.data_offset]), "GGUF layouts differ")
    matches = [i for i, t in enumerate(a.tensors) if t.name == name]
    need(len(matches) == 1, "expected one tensor of that name")
    selected = matches[0]
    x, y = a.tensors[selected], b.tensors[selected]
    shape = tuple(map(int, reversed(x.shape)))
    need(isinstance(index, list) and len(index) == len(shape) and
         all(type(k) is int and 0 <= k < n for k, n in zip(index, shape)), "invalid coordinate")
    need(math.prod(shape) <= 2**22, "selected tensor exceeds decoded preview limit")
    pos = int(np.ravel_multi_index(tuple(index), shape))
    block, _ = gguf.GGML_QUANT_SIZES[x.tensor_type]
    start = pos // block * block
    before, after = [gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
                     for t in (x, y)]
    need(all(np.isfinite(v).all() for v in (before, after)), "nonfinite decoded weights")
    changes = before.view(np.uint32) != after.view(np.uint32)
    within = int(changes[start:start + block].sum())
    others = [t.name for i, (t, u) in enumerate(zip(a.tensors, b.tensors))
              if i != selected and t.data.tobytes() != u.data.tobytes()]
    return dict(tensor=name, index=index, encoding=x.tensor_type.name,
                selected_before=float(before[pos]), selected_after=float(after[pos]),
                block_size=block, changed_neighbors=within - int(changes[pos]),
                changed_values_outside_block=int(changes.sum()) - within,
                other_changed_tensors=others)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--index", required=True, help="zero-based JSON coordinates in logical row-major order")
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.before, args.after, args.tensor, json.loads(args.index))))
    except (OSError, ValueError, NotImplementedError) as e:
        parser.exit(1, f"weight-impact: {e}\n")
