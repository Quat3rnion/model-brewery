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
RASP = ("sort", "sort_descending", "histogram", "reverse", "frac_prevs")
PROGRAMS = (*RASP, "alta_parity_relative", "alta_parity_absolute")
SCHEMA = "brewery.compiled-program.v1"


def compile_tracr(program):
    import haiku as hk
    import numpy as np
    from tracr.compiler import (
        assemble, basis_inference, compiling, craft_graph_to_model, craft_model_to_transformer,
        expr_to_craft_graph, lib, nodes, rasp_to_graph, validating,
    )
    from tracr.craft import bases, transformers
    from tracr.rasp import rasp

    factories = {
        "sort": lambda: lib.make_sort(
            rasp.tokens, rasp.Map(int, rasp.tokens), max_seq_len=LENGTH, min_key=1
        ),
        "sort_descending": lambda: lib.make_sort(
            rasp.tokens, rasp.Map(lambda t: 3 - int(t), rasp.tokens), max_seq_len=LENGTH, min_key=1
        ),
        "histogram": lib.make_hist,
        "reverse": lambda: lib.make_reverse(rasp.tokens),
        "frac_prevs": lambda: lib.make_frac_prevs(rasp.tokens == "1"),
    }
    vocab = "0123"
    # Use the pinned compiler passes directly so the same objects that emit
    # weights also supply ground truth. No inference from weight values.
    expression = factories[program]()
    if validating.static_validate(expression):
        raise ValueError("unsupported RASP expression")
    extracted = rasp_to_graph.extract_rasp_graph(expression)
    graph, sources, sink = extracted.graph, extracted.sources, extracted.sink
    basis_inference.infer_bases(graph, sink, set(vocab), LENGTH)
    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        graph, bos_dir=bases.BasisDirection(rasp.tokens.label, compiling.COMPILER_BOS),
        mlp_exactness=100,
    )
    craft = craft_graph_to_model.craft_graph_to_model(graph, sources)
    model = craft_model_to_transformer.craft_model_to_transformer(
        craft, graph, sink, LENGTH, compiling.COMPILER_BOS, compiling.COMPILER_PAD,
    )
    config = model.model_config
    _, module_names = assemble._get_model_config_and_module_names(craft)
    basis = {label: i for i, label in enumerate(model.residual_labels)}

    def indices(directions):
        return sorted(basis[str(direction)] for direction in directions)

    def layer(module_name):
        return int(module_name.split("/")[1].removeprefix("layer_"))

    heads, mlps = {}, {}
    for name, module in zip(module_names, craft.blocks, strict=True):
        if isinstance(module, transformers.MLP):
            mlps[layer(name)] = set(module.snd.output_space.basis)
        else:
            for i, head in enumerate(module.as_multi().heads()):
                heads[id(head)] = (layer(name), i)
    operations = []
    for label, node in graph.nodes.items():
        block = node.get(nodes.MODEL_BLOCK)
        if block is None:
            continue
        sites = []
        for sub in block.blocks if isinstance(block, transformers.SeriesWithResiduals) else [block]:
            if isinstance(sub, transformers.MLP):
                output = set(sub.snd.output_space.basis)
                (n,) = (n for n, space in mlps.items() if output <= space)
                sites.append(dict(layer=n, role="ffn_out", residual_indices=indices(output)))
            else:
                for head in sub.as_multi().heads():
                    n, i = heads[id(head)]
                    sites.append(dict(layer=n, role="head_out", head=i,
                                      residual_indices=indices(head.w_ov.output_space.basis)))
        expr = node[nodes.EXPR]
        operation = dict(label=label, type=type(expr).__name__,
                         residual_indices=indices(node[nodes.OUTPUT_BASIS]), sites=sites)
        if isinstance(expr, (rasp.Aggregate, rasp.SelectorWidth)):
            # A selector does not emit a separate residual value. Its consumer's
            # attention head implements the selector in the query/key product.
            operation["selector"] = expr.selector.label
        operations.append(operation)
    truth = dict(schema_version=1, residual_labels=model.residual_labels, operations=operations)
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
    }, weights, truth


def compile_alta(program):
    from examples import parity
    from framework.compiler import compiler_config, compiler_utils, dim_utils

    relative = program == "alta_parity_relative"
    spec = (parity.build_sequential_program_relative() if relative else
            parity.build_sequential_program_absolute(max_input_length=LENGTH))
    model = compiler_utils.compile_transformer(spec, compiler_config.Config(expansion_scalar_1=1000))
    layers = LENGTH + 3
    dimensions = dim_utils.get_var_mapping(spec)
    labels = [""] * dimensions.end_idx
    for name, mapping in dimensions.var_mappings.items():
        for value, i in enumerate(range(mapping.start_idx, mapping.end_idx)):
            labels[i] = f"{name}:{value}"

    def operation(name, kind, role, **extra):
        mapping = dimensions.var_mappings[name]
        indices = list(range(mapping.start_idx, mapping.end_idx))
        return dict(label=name, type=kind, residual_indices=indices,
                    sites=[dict(layer=i, role=role, residual_indices=indices, **extra)
                           for i in range(layers)])

    operations = [operation(head.output, "Attention", "head_out", head=i)
                  for i, head in enumerate(spec.head_specs)]
    writes = {(rule.rhs.variable, rule.rhs.new_value is None) for rule in spec.mlp.get_rules()}
    operations.extend(operation(name, "Reset" if reset else "Transition", "ffn_out")
                      for name, reset in sorted(writes))
    truth = dict(schema_version=1, residual_labels=labels, operations=operations,
                 shared_layers=True, output_variable=spec.outputs)
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
        "num_layers": layers,
        "num_heads": len(model.attention_heads),
        "mlp_layers": len(model.feed_forward_layers),
        "relative_position_masks": [sorted(h.relative_position_mask) for h in model.attention_heads],
        "bos": parity.START if relative else None,
        "output_encoding": "categorical",
    }, weights, truth


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compile_program(program, directory):
    import numpy as np
    from safetensors.numpy import save_file

    compiler = compile_tracr if program in RASP else compile_alta
    config, weights, truth = compiler(program)
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
    truth_path = directory / "ground-truth.v1.json"
    if any(p.exists() for p in (config_path, model_path, truth_path)):
        raise ValueError("output directory already contains a model, config or ground truth")
    # Preserve ALTA's native precision. No metadata map with unstable key order.
    save_file({name: np.ascontiguousarray(value) for name, value in sorted(weights.items())}, model_path)
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {"sha256": digest(model_path), "config_sha256": digest(config_path)}
    truth.update(program=program, compiler_commit=config["compiler_commit"],
                 model_sha256=receipt["sha256"], config_sha256=receipt["config_sha256"])
    truth_path.write_text(json.dumps(truth, sort_keys=True) + "\n", encoding="utf-8")
    receipt["ground_truth_sha256"] = digest(truth_path)
    return receipt


def forward_tracr(config, w, tokens, xp, softmax, offsets=None, capture=None):
    ids = xp.asarray([config["bos"], *(config["input_tokens"][token] for token in tokens)])
    x = w["token_embed/embeddings"][ids] + w["pos_embed/embeddings"][:len(ids)]
    heads, head_size = config["num_heads"], config["head_size"]
    for layer in range(config["num_layers"]):
        prefix = f"transformer/layer_{layer}"

        def linear(name, x):
            key = f"{prefix}/{name}"
            return x @ w[f"{key}/w"] + w[f"{key}/b"]

        q, k, v = (linear(f"attn/{p}", x).reshape(-1, heads, head_size).transpose(1, 0, 2)
                   for p in ("query", "key", "value"))
        logits = (q @ k.transpose(0, 2, 1)) / head_size ** 0.5
        values = softmax(logits, axis=-1) @ v
        projection = w[f"{prefix}/attn/linear/w"]
        head_outputs = values @ projection.reshape(heads, head_size, -1)
        if offsets is not None:
            head_outputs = head_outputs + offsets[layer, :heads]
        attention = head_outputs.sum(axis=0) + w[f"{prefix}/attn/linear/b"]
        x = x + attention
        mlp = linear("mlp/linear_2", xp.maximum(0, linear("mlp/linear_1", x)))
        if offsets is not None:
            mlp = mlp + offsets[layer, heads]
        x = x + mlp
        if capture is not None:
            capture.append((head_outputs, attention, mlp, x))
    return x[1:] @ w["unembed"]


def run_tracr(config, w, tokens, capture=None):
    import numpy as np
    from scipy.special import softmax

    y = forward_tracr(config, w, tokens, np, softmax, capture=capture)
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


def forward_alta(config, w, tokens, xp, softmax, offsets=None, capture=None):
    ids = tokens if config["bos"] is None else [config["bos"], *tokens]
    x = w["input_embeddings"][xp.asarray(ids)]
    if "index_embeddings" in w:
        x = x + w["index_embeddings"][:len(ids)]
    relative = xp.arange(len(ids))[None, :] - xp.arange(len(ids))[:, None]
    for layer in range(config["num_layers"]):
        heads = []
        for i, mask in enumerate(config["relative_position_masks"]):
            prefix = f"attn/{i}"
            q, k, v = (x @ w[f"{prefix}/{p}"] for p in ("query", "key", "value"))
            logits = q @ k.T
            if mask:
                logits = logits + xp.where(xp.isin(relative, xp.asarray(mask)), 0.0, -1e9)
            heads.append((softmax(logits, axis=-1) @ v) @ w[f"{prefix}/output"])
        head_outputs = xp.stack(heads)
        if offsets is not None:
            head_outputs = head_outputs + offsets[layer, :-1]
        attention = head_outputs.sum(axis=0)
        x = x + attention
        y = x
        for i in range(config["mlp_layers"]):
            y = y @ w[f"mlp/{i}/w"] + w[f"mlp/{i}/b"]
            if i + 1 < config["mlp_layers"]:
                y = xp.clip(y, 0, 1)
        if offsets is not None:
            y = y + offsets[layer, -1]
        x = x + y
        if capture is not None:
            capture.append((head_outputs, attention, y, x))
    return x[-len(tokens):] @ w["unembed"]


def run_alta(config, w, tokens, capture=None):
    import numpy as np
    from scipy.special import softmax

    return forward_alta(config, w, tokens, np, softmax, capture=capture).argmax(axis=1).tolist()


def run(config, weights, tokens, capture=None):
    if (not isinstance(tokens, list) or not 1 <= len(tokens) <= config["max_input_length"]
            or any(type(t) is not int or not 0 <= t < config["vocab_size"] for t in tokens)):
        raise ValueError("expected a nonempty token array within the compiled vocabulary and length")
    runners = {"tracr": run_tracr, "alta": run_alta}
    return runners[config["architecture"]](config, weights, tokens, capture)


def write_trace(config, weights, tokens, model_dir, destination):
    import numpy as np

    capture = []
    output = run(config, weights, tokens, capture)
    destination.mkdir()
    roles = ("attn_out", "ffn_out", "l_out")
    positions = list(range(len(capture[0][3])))
    if config["architecture"] == "tracr":
        ids = [config["bos"], *(config["input_tokens"][t] for t in tokens)]
    else:
        ids = tokens if config["bos"] is None else [config["bos"], *tokens]
    manifest = dict(schema_version=1, kind="trace",
                    model=dict(architecture=config["architecture"], program=config["program"],
                               sha256=digest(model_dir / "model.safetensors"),
                               config_sha256=digest(model_dir / "config.json")),
                    spec=dict(layers=list(range(len(capture))), positions=positions, roles=roles,
                              top_k=0, byte_cap=32 << 20),
                    tokens=[dict(position=p, token_id=t) for p, t in enumerate(ids)],
                    tensors=[], missing=[], interventions=[], model_logits=[], logit_lens=[],
                    compiled_heads=[])
    for layer, (heads, *outputs) in enumerate(capture):
        # v1 keys have no head index. Keep per-head outputs in a namespaced
        # extension, without altering the standard tensor keys or inventory.
        manifest["compiled_heads"].append(dict(layer=layer, heads=heads.astype("<f4").tolist()))
        for role, rows in zip(roles, outputs, strict=True):
            for p, row in enumerate(rows):
                file = f"tensor-{len(manifest['tensors']):06}.f32"
                data = np.asarray(row, dtype="<f4").tobytes()
                (destination / file).write_bytes(data)
                manifest["tensors"].append(dict(layer=layer, role=role, position=p, node=role,
                                                file=file, shape=list(row.shape), dtype="f32_le",
                                                bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (destination / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8",
    )
    return {"output": output, "manifest_sha256": digest(destination / "manifest.json")}


def check(config, weights):
    # Independent program oracles. Inference above only uses the loaded tensors.
    expected = {
        "sort": sorted,
        "sort_descending": lambda xs: sorted(xs, reverse=True),
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
    commands.add_parser("trace")
    for command in commands.choices.values():
        command.add_argument("directory", type=Path)
    commands.choices["run"].add_argument("tokens", type=json.loads)
    commands.choices["trace"].add_argument("tokens", type=json.loads)
    commands.choices["trace"].add_argument("output", type=Path)
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
        if args.command == "trace":
            result = write_trace(config, weights, args.tokens, args.directory, args.output)
        elif args.command == "run":
            result = run(config, weights, args.tokens)
        else:
            result = check(config, weights)
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
