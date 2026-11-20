#!/usr/bin/env python3
"""Propose an expert map for local review. Never writes or applies model changes."""

import argparse
import hashlib
import math
from pathlib import Path
import sys

from expert_usage import (
    I32_MAX, MAX_EXPERT_ROWS, MAX_LAYERS, MAX_TOKENS, MAX_TRACE_BYTES, MAX_TRACES, PHASES,
    SCHEMA as USAGE_SCHEMA, TraceError, array, compact_json, finite_f32, integer,
    model_identity, obj, read_trace, require, sha256,
)


MAP_SCHEMA = "brewery.expert-map.v1"
ACTIVATION_SCHEMA = "brewery.expert-activations.v1"
REAP_PAPER = "https://arxiv.org/html/2510.13999v3#S4"
REAP_UPSTREAM = (
    "https://github.com/CerebrasResearch/reap/blob/1970473c51ca3caeb98c10392f15b3a08a672974"
    "/src/reap/pruning_metrics.py#L172-L211"
)

REAP_INPUT = f"""Input files ({ACTIVATION_SCHEMA}, max {MAX_TRACE_BYTES // (1 << 20)} MiB each):
  schema, source_sha256, observer_sha256, split, n_layers, routed_layers,
  n_experts, top_k, samples.
  source_sha256: exact artifact identity, not a digest of the model name.
  observer_sha256: identity of the capture instrument and settings.
  split: "calibration" or "held-out"; both must use the same source and observer.
  routed_layers: all routed layer IDs; do not include dense layers.
  samples: [{{"tokens": [TOKEN_ID, ...], "positions": [POSITION, ...]}}, ...].
  POSITION: {{"LAYER_ID": [{{"expert": OLD_ID, "applied_gate": G, "output_l2": N}}, ...], ...}}.
  Record all selected experts at every token and routed layer. G is the actual
  applied combination coefficient (not a logit); N is the L2 norm of the
  expert output before multiplication by G, not a residual or FFN/layer norm.
  Use disjoint token sequences from independent workloads. Exact repeated
  sequences are rejected; that check does not prove dataset independence.
  Every expert needs a calibration observation; held-out data never sets rank.
  The result is a proposal, not a quality result. No model is loaded or changed."""


def evidence(path):
    try:
        value, digest, _ = read_trace(path, MAX_TRACE_BYTES, set())
    except (OSError, TraceError) as error:
        raise TraceError(f"{path}: {error}") from error
    return obj(value, str(path)), digest


def frequency(path, source, layers_arg):
    usage, digest = evidence(path)
    require(usage["schema"] == USAGE_SCHEMA, f"expected {USAGE_SCHEMA}")
    model = model_identity(usage["model"])
    require(model["sha256"] == source, "usage report source SHA-256 does not match")
    n_layers, n_experts, top_k = (model[key] for key in ("n_layers", "n_experts", "top_k"))
    require(top_k > 0, "usage report has no routed experts")
    all_layers = set(range(n_layers))
    layers = all_layers if layers_arg is None else set(layers_arg)
    require(layers <= all_layers, "--layers contains an out-of-range layer")
    require(len(layers) * n_experts <= MAX_EXPERT_ROWS, "expert map exceeds expert row limit")

    tokenizer = obj(usage["tokenizer"], "usage tokenizer")
    require(tokenizer["identity_source"] == "embedded_gguf"
            and sha256(tokenizer["model_sha256"], "tokenizer model SHA-256") == source
            and integer(tokenizer["vocab"], "tokenizer vocab") == model["vocab"],
            "usage tokenizer does not match the source")
    traces = array(usage["traces"], "usage traces")
    require(1 <= len(traces) <= MAX_TRACES, "invalid usage trace count")
    tokens = dict.fromkeys(PHASES, 0)
    seen = set()
    for trace in traces:
        obj(trace, "usage trace")
        trace_sha = sha256(trace["sha256"], "trace SHA-256")
        require(trace_sha not in seen, "duplicate usage trace SHA-256")
        seen.add(trace_sha)
        prefill = integer(trace["prefill_tokens"], "prefill tokens", lo=1, hi=MAX_TOKENS)
        decode = integer(trace["decode_tokens"], "decode tokens", hi=MAX_TOKENS - prefill)
        tokens["prefill"] += prefill
        tokens["decode"] += decode

    scores, coverage = {}, {}
    totals = {"trace_count": len(traces), **{f"{p}_tokens": tokens[p] for p in PHASES},
              "observed_positions": 0, "missing_positions": 0, "selections": 0}
    for layer, row in enumerate(array(usage["layers"], "usage layers", n_layers)):
        obj(row, "usage layer")
        require(integer(row["layer"], "usage layer index") == layer, "incorrect usage layer index")
        counts = [0] * n_experts if layer in layers else None
        layer_coverage = {}
        for phase in PHASES:
            label = f"layer {layer} {phase}"
            phase_row = obj(row[phase], label)
            observed = integer(phase_row["observed_positions"], label, hi=tokens[phase])
            missing = integer(phase_row["missing_positions"], label, hi=tokens[phase])
            require(observed + missing == tokens[phase], f"{label}: inconsistent usage coverage")
            if counts is None:
                require(observed == 0, f"{label}: --layers cannot omit observed routed layers")
            else:
                require(missing == 0, f"{label}: missing routing cannot be ranked as zero use")
            selections = integer(phase_row["selections"], label)
            require(selections == observed * top_k, f"{label}: incorrect total selections")
            unique = set()
            subtotal = 0
            for expert in array(phase_row["experts"], label):
                obj(expert, label)
                index = integer(expert["expert"], label, hi=n_experts - 1)
                require(index not in unique, f"{label}: duplicate expert")
                require(integer(expert["observed_positions"], label) == observed,
                        f"{label}: inconsistent expert coverage")
                count = integer(expert["selections"], label, lo=1, hi=observed)
                unique.add(index)
                subtotal += count
                if counts is not None:
                    counts[index] += count
            require(subtotal == selections, f"{label}: expert selections do not match total")
            totals["observed_positions"] += observed
            totals["missing_positions"] += missing
            totals["selections"] += selections
            layer_coverage[phase] = {"observed_positions": observed, "missing_positions": missing}
        if counts is not None:
            scores[str(layer)] = counts
            coverage[str(layer)] = layer_coverage | {"selections": counts}

    usage_totals = obj(usage["totals"], "usage totals")
    for key, count in totals.items():
        require(integer(usage_totals[key], f"usage totals.{key}") == count, "inconsistent usage totals")
    return scores, top_k, {
        "usage_sha256": digest, "trace_sha256": sorted(seen), "coverage": coverage,
        "excluded_layers": sorted(all_layers - layers),
        "score": "selection_count",
        "caveat": "Frequency measures this workload only, not expert importance or quality.",
    }


def activation_split(path, source, split, seen_samples):
    capture, digest = evidence(path)
    require(capture["schema"] == ACTIVATION_SCHEMA, f"expected {ACTIVATION_SCHEMA}")
    require(sha256(capture["source_sha256"], f"{split} source SHA-256") == source,
            f"{split} source SHA-256 does not match")
    require(capture["split"] == split, f"expected {split} split")
    observer = sha256(capture["observer_sha256"], f"{split} observer SHA-256")
    n_layers = integer(capture["n_layers"], "n_layers", lo=1, hi=MAX_LAYERS)
    layers = sorted(integer(layer, "routed layer", hi=n_layers - 1)
                    for layer in array(capture["routed_layers"], "routed_layers"))
    require(layers and len(set(layers)) == len(layers), "invalid routed_layers")
    n_experts = integer(capture["n_experts"], "n_experts", lo=1, hi=MAX_EXPERT_ROWS)
    top_k = integer(capture["top_k"], "top_k", lo=1, hi=n_experts)
    require(len(layers) * n_experts <= MAX_EXPERT_ROWS, "expert map exceeds expert row limit")
    identity = observer, n_layers, layers, n_experts, top_k

    products = {str(layer): [[] for _ in range(n_experts)] for layer in layers}
    samples = array(capture["samples"], f"{split} samples")
    require(samples, f"{split}: expected at least one sample")
    n_positions = 0
    for index, sample in enumerate(samples):
        label = f"{split} sample {index}"
        obj(sample, label)
        tokens = array(sample["tokens"], f"{label} tokens")
        require(tokens, f"{label}: expected a nonempty token sequence")
        for token in tokens:
            integer(token, f"{label} token", hi=I32_MAX)
        token_hash = hashlib.sha256(compact_json(tokens).encode("ascii")).digest()
        require(token_hash not in seen_samples, f"{label}: duplicate or overlapping token sequence")
        seen_samples.add(token_hash)
        positions = array(sample["positions"], f"{label} positions", len(tokens))
        for position, raw in enumerate(positions):
            pos_label = f"{label} position {position}"
            observed = obj(raw, pos_label)
            require(observed.keys() == products.keys(),
                    f"{pos_label}: missing or unexpected routed layer observations")
            for layer, selected in observed.items():
                entries = products[layer]
                row_label = f"{pos_label} layer {layer}"
                unique = set()
                for entry in array(selected, row_label, top_k):
                    obj(entry, row_label)
                    expert = integer(entry["expert"], row_label, hi=n_experts - 1)
                    require(expert not in unique, f"{row_label}: duplicate expert observation")
                    unique.add(expert)
                    gate, norm = entry["applied_gate"], entry["output_l2"]
                    require(finite_f32(gate) and gate >= 0, f"{row_label}: invalid applied gate")
                    require(finite_f32(norm) and norm >= 0, f"{row_label}: invalid output L2 norm")
                    entries[expert].append(gate * norm)
        n_positions += len(positions)

    coverage = {}
    for layer, entries in products.items():
        # REAP paper §4 Eq. (9) and upstream observer: a mean of per-token
        # products, divided by the count of tokens routed to this expert.
        coverage[layer] = {
            "observed_positions": n_positions, "selections": [len(values) for values in entries],
            "scores": [math.fsum(values) / len(values) if values else None for values in entries],
        }
    return identity, {"sha256": digest, "samples": len(samples), "coverage": coverage}


def reap(calibration, held_out, source):
    seen_samples = set()
    identity, cal = activation_split(calibration, source, "calibration", seen_samples)
    held_id, held = activation_split(held_out, source, "held-out", seen_samples)
    require(held_id == identity, "calibration and held-out capture dimensions or observer differ")
    scores = {layer: row["scores"] for layer, row in cal["coverage"].items()}
    for layer, values in scores.items():
        unseen = [index for index, value in enumerate(values) if value is None]
        require(not unseen,
                f"layer {layer}: no calibration activation for experts {unseen[:10]}; undefined REAP score")
    return scores, identity[-1], {
        "calibration": cal, "held_out": held, "observer_sha256": identity[0],
        "score": "mean(applied_gate * pre_gate_expert_output_l2 | expert_selected)",
        "references": {"paper": REAP_PAPER, "upstream": REAP_UPSTREAM},
        "caveat": ("Held-out scores describe source routing, not routing or quality after pruning. "
                   "A null held-out score means no selection, not a measured zero."),
    }


def layer_arg(value):
    try:
        layers = [int(layer) for layer in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated layer indices") from error
    if min(layers) < 0 or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layers must be unique nonnegative indices")
    return layers


def map_proposal(source, method, scores, top_k, keep, review):
    n_experts = len(next(iter(scores.values())))
    keep = n_experts if keep is None else keep
    require(top_k <= keep <= n_experts, f"--keep must be between top_k={top_k} and {n_experts}")
    layers = {
        layer: sorted(range(n_experts), key=lambda index: (-values[index], index))[:keep]
        for layer, values in scores.items()
    }
    if method == "reap":
        for layer, row in review["held_out"]["coverage"].items():
            row["removed_selections"] = sum(row["selections"]) - sum(
                row["selections"][index] for index in layers[layer]
            )
    return {
        "schema": MAP_SCHEMA, "source_sha256": source, "layers": layers, "top_k": top_k,
        "review": review | {
            "required": True, "release_qualified": False, "method": method,
            "operation": "reorder" if keep == n_experts else "prune",
            "source_experts_per_layer": n_experts, "keep_per_layer": keep,
            "tie_break": "ascending_old_expert_id",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Save the map for review: > map.json. Review the map before handing it to a writer.",
    )
    commands = parser.add_subparsers(dest="method", required=True)
    freq = commands.add_parser("frequency", help="rank complete Lanternfish expert_usage reports",
                               epilog="Use the exact GGUF that Lanternfish ran. A report from a converted "
                                      "or quantized copy does not identify its original artifact.")
    freq.add_argument("--usage", required=True, type=Path, help=f"{USAGE_SCHEMA} JSON")
    freq.add_argument("--layers", type=layer_arg, metavar="0,1,...",
                      help="declare all routed layers when the source has known dense layers")
    reap_cmd = commands.add_parser("reap", help="rank observed router-weighted expert output norms",
                                   epilog=REAP_INPUT, formatter_class=argparse.RawDescriptionHelpFormatter)
    reap_cmd.add_argument("--calibration", required=True, type=Path, help="calibration activation JSON")
    reap_cmd.add_argument("--held-out", required=True, type=Path, help="independent held-out activation JSON")
    for cmd in (freq, reap_cmd):
        cmd.add_argument("--source-sha256", required=True, help="exact source SHA-256 from trusted inventory")
        cmd.add_argument("--keep", type=int, help="experts to retain per layer (default: reorder only)")
    args = parser.parse_args()
    try:
        source = sha256(args.source_sha256, "--source-sha256")
        if args.method == "frequency":
            scores, top_k, review = frequency(args.usage, source, args.layers)
        else:
            scores, top_k, review = reap(args.calibration, args.held_out, source)
        proposal = compact_json(map_proposal(source, args.method, scores, top_k, args.keep, review))
    except (TraceError, KeyError) as error:
        message = f"missing field {error}" if isinstance(error, KeyError) else str(error)
        print(f"expert_plan: {message}", file=sys.stderr)
        return 1
    print(proposal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
