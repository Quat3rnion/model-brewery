#!/usr/bin/env python3
"""Compile allowlisted programs and run their exported transformer weights on CPU."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys


TRACR = "9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd"
ALTA = "afb0559da1dc08a3819c5ae20e0547686052b75a"
LENGTH = 4
RASP = ("sort", "histogram", "reverse", "frac_prevs")
PROGRAMS = (*RASP, "alta_parity_relative", "alta_parity_absolute")
SCHEMA = "brewery.compiled-program.v1"


def compile_tracr(program):
    import haiku as hk
    import numpy as np
    from tracr.compiler import compiling, lib
    from tracr.rasp import rasp

    factories = {
        "sort": lambda: lib.make_sort(
            rasp.tokens, rasp.Map(int, rasp.tokens), max_seq_len=LENGTH, min_key=1
        ),
        "histogram": lib.make_hist,
        "reverse": lambda: lib.make_reverse(rasp.tokens),
        "frac_prevs": lambda: lib.make_frac_prevs(rasp.tokens == "1"),
    }
    vocab = "0123"
    model = compiling.compile_rasp_to_model(factories[program](), set(vocab), LENGTH)
    config = model.model_config
    weights = {
        f"{module}/{name}": value
        for module, params in model.params.items()
        for name, value in params.items()
    }
    # Tracr does not store the output projection in Haiku params.
    unembed = hk.without_apply_rng(
        hk.transform(lambda x: model.get_compiled_model().unembed(x, False))
    )
    width = len(model.residual_labels)
    weights["unembed"] = unembed.apply({}, np.eye(width, dtype=np.float32)).reshape(width, -1)
    output_tokens = None
    if program != "frac_prevs":
        output_tokens = list(map(int, model.output_encoder.decode(
            list(range(model.output_encoder.vocab_size))
        )))
    return {
        "architecture": "tracr",
        "compiler_commit": TRACR,
        "attention": "scaled_dot_product",
        "activation": "relu",
        "layer_norm": config.layer_norm,
        "causal": config.causal,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "head_size": config.key_size,
        "mlp_hidden_size": config.mlp_hidden_size,
        "bos": model.input_encoder.encode([compiling.COMPILER_BOS])[0],
        "input_tokens": model.input_encoder.encode([compiling.COMPILER_BOS, *vocab])[1:],
        "output_tokens": output_tokens,
        "output_encoding": "prefix_fraction" if output_tokens is None else "categorical",
    }, weights


def compile_alta(program):
    from examples import parity
    from framework.compiler import compiler_config, compiler_utils

    relative = program == "alta_parity_relative"
    spec = (parity.build_sequential_program_relative() if relative else
            parity.build_sequential_program_absolute(max_input_length=LENGTH))
    model = compiler_utils.compile_transformer(spec, compiler_config.Config(expansion_scalar_1=1000))
    weights = {
        "input_embeddings": model.embeddings.input_embeddings,
        "unembed": model.output_transform,
    }
    if model.embeddings.index_embeddings is not None:
        weights["index_embeddings"] = model.embeddings.index_embeddings
    for i, head in enumerate(model.attention_heads):
        for projection in ("query", "key", "value", "output"):
            weights[f"attn/{i}/{projection}"] = getattr(head, f"{projection}_transform")
    for i, layer in enumerate(model.feed_forward_layers):
        weights[f"mlp/{i}/w"] = layer.weights
        weights[f"mlp/{i}/b"] = layer.biases
    return {
        "architecture": "alta",
        "compiler_commit": ALTA,
        "attention": "dot_product",
        "activation": "clipped_relu",
        "layer_norm": False,
        "shared_layers": True,
        "num_layers": LENGTH + 3,
        "num_heads": len(model.attention_heads),
        "mlp_layers": len(model.feed_forward_layers),
        "relative_position_masks": [sorted(h.relative_position_mask) for h in model.attention_heads],
        "bos": parity.START if relative else None,
        "output_encoding": "categorical",
    }, weights


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compile_program(program, directory):
    import numpy as np
    from safetensors.numpy import save_file

    compiler = compile_tracr if program in RASP else compile_alta
    config, weights = compiler(program)
    version = importlib.metadata.version
    packages = ("numpy", "scipy", "safetensors")
    if program in RASP:
        packages += ("jax", "jaxlib", "dm-haiku")
    python = ".".join(map(str, sys.version_info[:3]))
    runtime = " ".join([f"python={python}", *(f"{package}={version(package)}" for package in packages)])
    config.update(schema=SCHEMA, program=program, max_input_length=LENGTH,
                  vocab_size=4 if program in RASP else 2,
                  hidden_size=weights["unembed"].shape[0], runtime=runtime)
    config_path = directory / "config.json"
    model_path = directory / "model.safetensors"
    if config_path.exists() or model_path.exists():
        raise ValueError("output directory already contains a model or config")
    # Preserve ALTA's native precision. No metadata map with unstable key order.
    save_file({name: np.ascontiguousarray(value) for name, value in sorted(weights.items())}, model_path)
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    return {"sha256": digest(model_path), "config_sha256": digest(config_path)}


def run_tracr(config, w, tokens):
    import numpy as np
    from scipy.special import softmax

    ids = [config["bos"], *(config["input_tokens"][token] for token in tokens)]
    x = w["token_embed/embeddings"][ids] + w["pos_embed/embeddings"][:len(ids)]
    heads, head_size = config["num_heads"], config["head_size"]
    for layer in range(config["num_layers"]):
        prefix = f"transformer/layer_{layer}"

        def linear(name, x):
            key = f"{prefix}/{name}"
            return x @ w[f"{key}/w"] + w[f"{key}/b"]

        q, k, v = (linear(f"attn/{p}", x).reshape(-1, heads, head_size).transpose(1, 0, 2)
                   for p in ("query", "key", "value"))
        logits = (q @ k.transpose(0, 2, 1)) / np.sqrt(head_size)
        y = (softmax(logits, axis=-1) @ v).transpose(1, 0, 2).reshape(len(ids), -1)
        x = x + linear("attn/linear", y)
        x = x + linear("mlp/linear_2", np.maximum(0, linear("mlp/linear_1", x)))
    y = x[1:] @ w["unembed"]
    encoding = config["output_encoding"]
    if encoding == "categorical":
        return [config["output_tokens"][i] for i in y.argmax(axis=1)]
    result = []
    for i, value in enumerate(y[:, 0], 1):
        rounded = round(float(value) * i) / i
        if abs(value - rounded) > 1e-4:
            raise ValueError("numeric output is outside the discrete output tolerance")
        result.append(rounded)
    return result


def run_alta(config, w, tokens):
    import numpy as np
    from scipy.special import softmax

    ids = tokens if config["bos"] is None else [config["bos"], *tokens]
    x = w["input_embeddings"][ids]
    if "index_embeddings" in w:
        x = x + w["index_embeddings"][:len(ids)]
    relative = np.arange(len(ids))[None, :] - np.arange(len(ids))[:, None]
    for _ in range(config["num_layers"]):
        y = np.zeros_like(x)
        for i, mask in enumerate(config["relative_position_masks"]):
            prefix = f"attn/{i}"
            q, k, v = (x @ w[f"{prefix}/{p}"] for p in ("query", "key", "value"))
            logits = q @ k.T
            if mask:
                logits += np.where(np.isin(relative, mask), 0.0, -1e9)
            y += (softmax(logits, axis=-1) @ v) @ w[f"{prefix}/output"]
        x = x + y
        y = x
        for i in range(config["mlp_layers"]):
            y = y @ w[f"mlp/{i}/w"] + w[f"mlp/{i}/b"]
            if i + 1 < config["mlp_layers"]:
                y = np.clip(y, 0, 1)
        x = x + y
    return (x @ w["unembed"]).argmax(axis=1)[-len(tokens):].tolist()


def run(config, weights, tokens):
    if (not isinstance(tokens, list) or not 1 <= len(tokens) <= config["max_input_length"]
            or any(type(t) is not int or not 0 <= t < config["vocab_size"] for t in tokens)):
        raise ValueError("expected a nonempty token array within the compiled vocabulary and length")
    runners = {"tracr": run_tracr, "alta": run_alta}
    return runners[config["architecture"]](config, weights, tokens)


def check(config, weights):
    # Independent program oracles. Inference above only uses the loaded tensors.
    expected = {
        "sort": sorted,
        "histogram": lambda xs: [xs.count(x) for x in xs],
        "reverse": lambda xs: xs[::-1],
        "frac_prevs": lambda xs: [xs[:i].count(1) / i for i in range(1, len(xs) + 1)],
        "alta_parity_relative": lambda xs: [sum(xs[:i]) % 2 for i in range(1, len(xs) + 1)],
        "alta_parity_absolute": lambda xs: [sum(xs[:i]) % 2 for i in range(1, len(xs) + 1)],
    }[config["program"]]
    cases = 0
    for size in range(1, config["max_input_length"] + 1):
        for ts in itertools.product(range(config["vocab_size"]), repeat=size):
            tokens = list(ts)
            try:
                actual = run(config, weights, tokens)
            except ValueError:
                actual = None
            if actual != expected(tokens):
                return {"cases": cases, "mismatch": tokens}
            cases += 1
    return {"cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("compile").add_argument("program", choices=PROGRAMS)
    commands.add_parser("check")
    commands.add_parser("run")
    for command in commands.choices.values():
        command.add_argument("directory", type=Path)
    commands.choices["run"].add_argument("tokens", type=json.loads)
    args = parser.parse_args()
    if args.command == "compile":
        for project in ("tracr", "alta"):
            sys.path.insert(0, str(Path(sys.prefix).parent / project))
        result = compile_program(args.program, args.directory)
    else:
        from safetensors.numpy import load_file

        config = json.loads((args.directory / "config.json").read_text(encoding="utf-8"))
        if config.get("schema") != SCHEMA:
            raise ValueError("unsupported compiled program config")
        weights = load_file(args.directory / "model.safetensors")
        result = run(config, weights, args.tokens) if args.command == "run" else check(config, weights)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, "-B", "-s", *sys.argv],
                  {**os.environ, "PYTHONHASHSEED": "0"})
    sys.dont_write_bytecode = True
    os.environ.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        sys.exit(str(error))
