#!/usr/bin/env python3
"""Rebuild X9 results from pinned W13 models and the W11/X6 CPU trainers."""

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

import numpy as np
from safetensors.numpy import load_file

import compiled_program as program


PAIRS = {
    "sort": ([2, 0, 2, 1], [2, 3, 2, 1]),
    "histogram": ([2, 0, 2, 1], [2, 0, 2, 2]),
    "alta_parity_relative": ([1, 0, 1, 0], [1, 0, 0, 0]),
}
TRAIN = dict(features=64, k=8, steps=700)
SPLIT_SEED = 9


def load(directory):
    config = json.loads((directory / "config.json").read_text())
    weights_file = directory / "model.safetensors"
    truth_file = directory / "ground-truth.v1.json"
    truth = json.loads(truth_file.read_text())
    if (config["schema"] != program.SCHEMA or truth["schema_version"] != 1
            or truth["program"] != config["program"]
            or truth["compiler_commit"] != config["compiler_commit"]
            or truth["model_sha256"] != program.digest(weights_file)
            or truth["config_sha256"] != program.digest(directory / "config.json")):
        raise ValueError("Ground truth does not match the compiled model")
    return config, load_file(weights_file), truth


def capture(config, weights, tokens):
    blocks = []
    result = program.run(config, weights, list(tokens), blocks)
    return result, blocks


def model_ref(truth, directory):
    return {key: truth[key] for key in ("program", "model_sha256", "config_sha256", "compiler_commit")} | {
        "ground_truth_sha256": program.digest(directory / "ground-truth.v1.json"),
    }


def circuit(config, weights, truth):
    import jax
    import jax.numpy as jnp
    from jax.nn import softmax

    jax.config.update("jax_enable_x64", True)
    clean, corrupt = PAIRS[config["program"]]
    clean_output, _ = capture(config, weights, clean)
    corrupt_output, _ = capture(config, weights, corrupt)
    heads = config["num_heads"]
    w = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in weights.items()}
    # The last differing output is deterministic and avoids selecting a
    # position by observing its attribution or patch performance.
    pos = max(p for p in range(len(clean)) if clean_output[p] != corrupt_output[p])
    target, contrast = clean_output[pos], corrupt_output[pos]
    if config["architecture"] == "tracr":
        i, j = (config["output_tokens"].index(token) for token in (target, contrast))
        forward = program.forward_tracr
    else:
        i, j = target, contrast
        forward = program.forward_alta
    clean_blocks, corrupt_blocks = [], []
    forward(config, w, clean, jnp, softmax, capture=clean_blocks)
    forward(config, w, corrupt, jnp, softmax, capture=corrupt_blocks)
    delta = jnp.stack([jnp.concatenate((a[0] - b[0], (a[2] - b[2])[None, ...]))
                       for a, b in zip(clean_blocks, corrupt_blocks, strict=True)])
    zero = jnp.zeros_like(delta)

    def metric(tokens, offsets):
        logits = forward(config, w, tokens, jnp, softmax, offsets)
        return logits[pos, i] - logits[pos, j]

    corrupt_metric, gradient = jax.value_and_grad(lambda offsets: metric(corrupt, offsets))(zero)
    actual = float(corrupt_metric)
    attribution = np.asarray(gradient * delta)
    positives = {(site["layer"], site.get("head", heads)) for op in truth["operations"]
                 for site in op["sites"]}
    ranking = []
    for layer, component in np.ndindex(delta.shape[:2]):
        score = float(np.abs(attribution[layer, component].sum(axis=-1)).sum())
        patched = float(metric(corrupt, zero.at[layer, component].set(delta[layer, component])))
        site = (dict(role="head_out", head=component) if component < heads else dict(role="ffn_out"))
        ranking.append(dict(layer=layer, **site, compiler_site=(layer, component) in positives,
                            score=score, patched_delta=patched - actual))
    ranking.sort(key=lambda site: -site["score"])
    top_k = len(positives)
    cutoff = ranking[top_k - 1]["score"]
    above = [site for site in ranking if site["score"] > cutoff]
    tied = [site for site in ranking if site["score"] == cutoff]
    found = sum(site["compiler_site"] for site in above)
    found += (top_k - len(above)) * sum(site["compiler_site"] for site in tied) / len(tied)
    yes = [site["score"] for site in ranking if site["compiler_site"]]
    no = [site["score"] for site in ranking if not site["compiler_site"]]
    auc = sum(float(a > b) + 0.5 * (a == b) for a in yes for b in no) / (len(yes) * len(no)) if no else None
    return dict(program=config["program"], clean=clean, corrupt=corrupt, clean_output=clean_output,
                corrupt_output=corrupt_output, target=target, contrast=contrast, output_position=pos,
                clean_metric=float(metric(clean, zero)), corrupt_metric=actual,
                sites=len(ranking), compiler_sites=top_k, top_k=top_k, precision=found / top_k,
                recall=found / top_k, roc_auc=auc, ranking=ranking)


def feature_operations(truth):
    # ALTA clears the attention read ports in the MLP. Measure only
    # transition state at l_out; the trace and X5 still include all heads.
    return [op for op in truth["operations"] if not truth.get("shared_layers") or op["type"] == "Transition"]


def feature_rows(specs):
    """Read the same l_out rows exported by the v1 runner, before serialization."""
    config = specs[0][0]
    inputs = list(itertools.product(range(config["vocab_size"]), repeat=config["max_input_length"]))
    order = np.random.default_rng(SPLIT_SEED).permutation(len(inputs))
    layers = sorted({max(site["layer"] for site in op["sites"]) for op in feature_operations(specs[0][2])})
    rows = {layer: [] for layer in layers}
    for index in order:
        captures = [capture(c, w, inputs[index])[1] for c, w, _ in specs]
        for layer in layers:
            # Skip the invariant compiler BOS. Split whole inputs, not rows.
            rows[layer].append(np.stack([blocks[layer][3][1:] for blocks in captures], axis=1))
    return {layer: np.concatenate(blocks).astype(np.float32) for layer, blocks in rows.items()}


def subspace_matches(operations, layer, decoder, seen):
    # Unit-vector projection onto the compiler's orthonormal coordinate axes
    # equals the maximum absolute cosine to any unit vector in that subspace.
    norms = np.linalg.norm(decoder, axis=1)
    valid = seen & (norms > 1e-8)
    matches = []
    for op in operations:
        if max(site["layer"] for site in op["sites"]) != layer:
            continue
        projected = np.linalg.norm(decoder[:, op["residual_indices"]], axis=1)
        cosines = np.divide(projected, norms, out=np.zeros_like(norms), where=valid).clip(0, 1)
        feature = int(np.argmax(np.where(valid, cosines, -1))) if valid.any() else None
        matches.append(dict(operation=op["label"], layer=layer, feature=feature,
                            cosine=float(cosines[feature]) if feature is not None else None))
    return matches


def fit_features(specs, artifact_dir):
    import torch
    import train_sae
    import train_crosscoder
    from safetensors.torch import save_file

    if torch.__version__ != train_sae.TORCH:
        raise ValueError(f"Use torch=={train_sae.TORCH}")
    torch.set_num_threads(1)
    artifact_dir.mkdir()
    methods = len(specs)
    rows_by_layer = feature_rows(specs)
    matches = [[] for _ in specs]
    metrics = []
    for layer, rows in rows_by_layer.items():
        x = torch.from_numpy(rows)
        train, heldout = x[:len(x) * 3 // 4], x[len(x) * 3 // 4:]
        if methods == 1:
            train, heldout = train[:, 0], heldout[:, 0]
            weights, norm, report = train_sae.fit(train, heldout, **TRAIN)
            dense = ((train * norm["scale"] - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"]).relu()
            strength, index = dense.topk(TRAIN["k"], dim=1)
            active = torch.zeros_like(dense).scatter(1, index, strength)
            decoders = weights["W_dec"][None, ...]
        else:
            weights, scale, report = train_crosscoder.fit(train, heldout, **TRAIN)
            active = train_crosscoder.codes(train, weights, scale, TRAIN["k"])
            decoders = weights["W_dec"]
        ref = artifact_dir / f"layer-{layer}.safetensors"
        save_file(weights, ref)
        metrics.append(dict(layer=layer, train_rows=len(train), heldout_rows=len(heldout),
                            weights_sha256=program.digest(ref), reconstruction=report))
        seen = active.gt(0).any(dim=0).numpy()
        for side, (_, _, truth) in enumerate(specs):
            matches[side].extend(subspace_matches(feature_operations(truth), layer, decoders[side].numpy(), seen))
    means = [float(np.mean([m["cosine"] for m in side if m["cosine"] is not None])) for side in matches]
    return dict(torch=torch.__version__, train=TRAIN, split_seed=SPLIT_SEED, layers=metrics,
                best_matches=matches, mean_best_cosine=means)


def build(directory):
    directory.mkdir()
    methods = dict(x5=[], w11=[], x6=[])
    models = {}
    for name in (*PAIRS, "sort_descending"):
        model_dir = directory / name
        model_dir.mkdir()
        subprocess.run([sys.executable, "-B", str(Path(program.__file__)), "compile", name, model_dir],
                       check=True, stdout=subprocess.DEVNULL)
        spec = load(model_dir)
        models[name] = (spec, model_ref(spec[2], model_dir))
        config, weights, truth = spec
        models[name][1]["runtime"] = config["runtime"]
        checks = program.check(config, weights)
        if "mismatch" in checks:
            raise ValueError(f"Compiled program failed its oracle: {name}: {checks}")
        if name in PAIRS:
            trace_hashes = {}
            for side, tokens in zip(("clean", "corrupt"), PAIRS[name], strict=True):
                trace = program.write_trace(config, weights, tokens, model_dir, model_dir / side)
                trace_hashes[side] = trace["manifest_sha256"]
            methods["x5"].append(circuit(*spec) | {"trace_sha256": trace_hashes})
            methods["w11"].append({"program": name} | fit_features([spec], model_dir / "sae"))
        models[name][1]["checked_inputs"] = checks["cases"]
    names = ("sort", "sort_descending")
    specs = [models[name][0] for name in names]
    if specs[0][2]["residual_labels"] != specs[1][2]["residual_labels"]:
        raise ValueError("Paired models do not share residual coordinates")
    methods["x6"].append({"programs": names} | fit_features(specs, directory / "crosscoder"))
    return dict(schema_version=1, qualification="not release-qualified", backend="cpu",
                models=[ref for _, ref in models.values()], methods=methods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "circuit"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = build(args.directory) if args.command == "build" else circuit(*load(args.directory))
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
