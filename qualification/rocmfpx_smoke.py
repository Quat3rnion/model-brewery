#!/usr/bin/env python3
"""Prove five base ROCmFPX GGUF encodings and CPU decode on synthetic Qwen2.

Use PYTHONPATH=<pinned ROCmFPX source>/gguf-py and NumPy. Supply the matching
CPU-only llama-quantize and llama-simple builds and a fresh /arboretum run dir.
This is a format/runtime smoke, not a model quality or GPU qualification.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.vocab import bytes_to_unicode


# Preset -> (GGML tensor type, general.file_type, bytes per 32 weights).
PRESETS = {
    "Q3_0_ROCMFPX": (104, 112, 14),
    "Q4_0_ROCMFP4": (100, 100, 18),
    "Q4_0_ROCMFP4_FAST": (101, 103, 17),
    "Q6_0_ROCMFPX": (102, 110, 26),
    "Q8_0_ROCMFPX": (103, 111, 33),
}
Q = "blk.0.attn_q.weight"
MAX_BYTES = 8 << 20


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fixture(path):
    writer = GGUFWriter(path, "qwen2")
    try:
        width, hidden, vocab = 256, 512, 288
        writer.add_name("synthetic-rocmfpx-cpu-smoke")
        writer.add_file_type(0)
        writer.add_context_length(64)
        writer.add_embedding_length(width)
        writer.add_feed_forward_length(hidden)
        writer.add_block_count(1)
        writer.add_head_count(8)
        writer.add_head_count_kv(8)
        writer.add_rope_dimension_count(32)
        writer.add_layer_norm_rms_eps(1e-6)
        # The synthetic vocabulary is only for exercising the runtime.
        tokens = [bytes_to_unicode()[b] for b in range(256)]
        tokens += ["<unk>", "<s>", "</s>", "pr"] + [f"dummy{i}" for i in range(vocab - 260)]
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")
        writer.add_token_list(tokens)
        writer.add_token_merges(["p r"])
        writer.add_token_types([1] * 256 + [2, 3, 3] + [1] * (vocab - 259))
        writer.add_bos_token_id(257)
        writer.add_eos_token_id(258)
        writer.add_unk_token_id(256)
        rng = np.random.default_rng(2306)
        for name, rows, cols in (
            ("token_embd.weight", vocab, width),
            ("output.weight", vocab, width),
            (Q, width, width),
            ("blk.0.attn_k.weight", width, width),
            ("blk.0.attn_v.weight", width, width),
            ("blk.0.attn_output.weight", width, width),
            ("blk.0.ffn_gate.weight", hidden, width),
            ("blk.0.ffn_up.weight", hidden, width),
            ("blk.0.ffn_down.weight", width, hidden),
        ):
            writer.add_tensor(name, rng.standard_normal((rows, cols), dtype=np.float32) * np.float32(.025))
        for name in ("output_norm", "blk.0.attn_norm", "blk.0.ffn_norm"):
            writer.add_tensor(f"{name}.weight", np.ones(width, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def run(out, name, env, *args):
    with (out / f"{name}.log").open("xb") as stream:
        subprocess.run(args, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                       stderr=subprocess.STDOUT, timeout=60, check=True, cwd=out)
    return (out / f"{name}.log").read_text(errors="replace")


def inventory(path):
    need(0 < path.stat().st_size <= MAX_BYTES, f"{path}: fixture size out of bounds")
    reader = GGUFReader(path, "r")
    ftype = int(reader.fields["general.file_type"].contents())
    tensors = {t.name: (int(t.tensor_type), tuple(map(int, t.shape)), int(t.n_bytes))
               for t in reader.tensors}
    need(len(tensors) == len(reader.tensors), "duplicate tensor names")
    return ftype, tensors


def decode(out, env, reader, model, name):
    log = run(out, f"{name}.decode", env, reader, "-m", model, "-ngl", "0", "-n", "4", "probe")
    decoded = re.search(r"decoded (\d+) tokens", log)
    need(decoded and int(decoded[1]) >= 2, f"{name}: no repeated CPU decode")
    need(re.search(r"CPU(?:_Mapped)?\s+model buffer size", log), f"{name}: no CPU model buffer")
    return int(decoded[1])


def main(args):
    os.umask(0o077)
    out = args.out.resolve()
    need(out.is_relative_to("/arboretum") and out != Path("/arboretum"), "use a new /arboretum directory")
    quantizer, reader = args.quantizer.resolve(strict=True), args.reader.resolve(strict=True)
    out.mkdir(mode=0o700)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(out), "TMPDIR": str(out), "LC_ALL": "C",
           "OMP_NUM_THREADS": "2"}
    src = out / "qwen2-f32.gguf"
    fixture(src)
    before = sha(src)
    ftype, source = inventory(src)
    need(ftype == 0 and all(t[0] == 0 for t in source.values()), "expected an F32 source")
    base_decode = decode(out, env, reader, src, "f32")
    rows = {}
    for preset, (tid, file_type, block) in PRESETS.items():
        model = out / f"{preset}.gguf"
        plan = run(out, f"{preset}.plan", env, quantizer, "--dry-run", src, model, preset, "2")
        need(not model.exists() and f"({preset.lower()})" in plan and "quant size" in plan,
             f"{preset}: dry-run failed to plan the fork layout without writing")
        run(out, f"{preset}.quantize", env, quantizer, src, model, preset, "2")
        observed_type, tensors = inventory(model)
        need(observed_type == file_type, f"{preset}: wrong general.file_type")
        need({n: t[1] for n, t in tensors.items()} == {n: t[1] for n, t in source.items()},
             f"{preset}: tensor inventory changed")
        members = {n: size for n, (kind, shape, size) in tensors.items() if kind == tid}
        need(members, f"{preset}: default routing did not emit the requested layout")
        for n, size in members.items():
            shape = tensors[n][1]
            need(shape[0] % 32 == 0 and size == (int(np.prod(shape)) // 32) * block,
                 f"{preset}: invalid block bytes for {n}")
        need(model.stat().st_size < src.stat().st_size, f"{preset}: output did not shrink")
        rows[preset] = {"sha256": sha(model), "file_bytes": model.stat().st_size,
                        "file_type": file_type, "tensor_type": tid, "bytes_per_32_weights": block,
                        "fork_tensors": members, "decoded_tokens": decode(out, env, reader, model, preset)}

    # A stock Q8 carrier can hold a fork tensor override; the override does not
    # redefine general.file_type. Here regex is anchored to one exact tensor.
    override = f"^{re.escape(Q)}$=q4_0_rocmfp4_fast"
    mixed = out / "override.gguf"
    run(out, "override.plan", env, quantizer, "--dry-run", "--tensor-type", override, src, "Q8_0", "2")
    need(not mixed.exists() and not (out / "ggml-model-Q8_0.gguf").exists(),
         "implicit-path dry-run wrote an output")
    run(out, "override.quantize", env, quantizer, "--tensor-type", override, src, mixed, "Q8_0", "2")
    mixture_type, mixture = inventory(mixed)
    need(mixture_type == 7 and [n for n, t in mixture.items() if t[0] == 101] == [Q],
         "unexpected Q8_0 carrier or fork override scope")
    mixed_decode = decode(out, env, reader, mixed, "override")
    pure = out / "pure-control.gguf"
    run(out, "pure.quantize", env, quantizer, "--pure", "--tensor-type", override, src, pure, "Q8_0", "2")
    need(inventory(pure)[1][Q][0] == 8, "--pure did not suppress regex override")
    need(sha(src) == before, "source was modified")
    report = {"source_sha256": before, "source_bytes": src.stat().st_size, "source_decoded_tokens": base_decode,
              "quantizer_sha256": sha(quantizer), "reader_sha256": sha(reader), "presets": rows,
              "override": {"carrier": "Q8_0", "file_type": mixture_type, "tensor_type": 101,
                           "tensor": Q, "decoded_tokens": mixed_decode, "pure_tensor_type": 8}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out / "receipt.json")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("quantizer", "reader", "out"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    main(cli.parse_args())
