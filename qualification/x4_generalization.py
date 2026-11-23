#!/usr/bin/env python3
"""Run the two-fold X4 corpus-loss check on an isolated Brewery stack."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
from statistics import median
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from sensitivity_acceptance import wait


REPO = Path(__file__).resolve().parents[1]
W4 = "9678f9afdc31b2a9be7fec2045997e2360693ee72b893287c0d539983bff4923"
DOLLY = "2c8296319e4d9c2829447fde4b1edf3203d56749625c6fa6eea675ae6ab15bb5"
DOLLY_SOURCE = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
DOLLY_SPLITS = {
    "creative_writing": "c844b9b32e5547473b0b30268d99dde456657a5c3d5f98d4e92bbeffd4ac7987",
    "open_qa": "f42e2aac3d2c0faad8d0f0e5e2a2c2a5c8580bb1f51ece3e6a30b26eac3c24b6",
}
CORPORA = {"dolly": DOLLY, "w4": W4}
DEFAULT_SCRATCH = Path("/mnt/games/model-brewery/xa-scratch/x4g")
DEFAULT_EVIDENCE = Path("/arboretum/x4g-evidence")
X4 = Path("/arboretum/x4-evidence")
X4C = Path("/arboretum/x4c-evidence")
W0 = Path("/arboretum/llama-w0")
PIN = Path("/arboretum/pins/lanternfish-x4c-e0fd1b7")
W0_LIBS = Path("/arboretum/pins/lanternfish-0ea30b3.libs.sha256")
QUANTIZER = Path("/arboretum/brewery-wbux-release/releases/8c915d083dabc9afd43dede95cbd715cae972231/runtime/gguf/llama-quantize")
EVAL = X4 / "bins/x4-kl"
REFERENCE = Path("/arboretum/models/qwen2.5-0.5b-instruct-w0/Qwen2.5-0.5B-Instruct.bf16.gguf")
QUANTIZED = REFERENCE.with_name("qwen2.5-0.5b-instruct-q4_k_m.gguf")
REFERENCE_SHA = "a33250ab73bbef0d876bf04e01517f5f43292dd8ede34e12d3a3a886e49f36f0"
QUANTIZED_SHA = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
F32_SHA = "e4a22146b4a06238d0cda3461a9d3ff9fc2e4e786893dd1bcf81865734afe0da"
TYPES = {2: "Q4_0", 6: "Q5_0", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}
TARGET_BYTES = 346757888
EVAL_POSITIONS = {"w4": 222, "dolly": 186}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual} != {expected}")
    return path


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path):
    return json.loads(path.read_bytes())


def normalize(line):
    return " ".join(line.split())


def prepare(scratch, cache):
    handoff = scratch / "handoff"
    w4 = verified(REPO / "crates/eval/data/edit-checks/heldout.txt", W4)
    source = verified(cache / f"{DOLLY_SOURCE}.jsonl", DOLLY_SOURCE)
    raw = set()
    with source.open(encoding="utf-8") as rows:
        for line in rows:
            row = json.loads(line)
            if not row["context"].strip():
                raw.add((row["category"], row["instruction"].strip()))
    lines = []
    for category, digest in DOLLY_SPLITS.items():
        split = load(verified(cache / f"{digest}.json", digest))
        held_out = [row["positive"]["user"] for row in split["rows"] if row["split"] == "held_out"]
        assert len(held_out) == 8 and all((category, line) in raw for line in held_out)
        lines.extend(normalize(line) for line in held_out)
    assert len(set(lines)) == 16
    assert not {line.casefold() for line in lines} & {
        normalize(line).casefold() for line in w4.read_text().splitlines()
    }
    dolly = handoff / "dolly.txt"
    dolly.write_bytes(("\n".join(lines) + "\n").encode())
    verified(dolly, DOLLY)
    (handoff / "w4.txt").write_bytes(w4.read_bytes())
    weights = load(handoff / "spec.json")["weights"]
    assert len(set(weights)) == 170
    for name, digest in CORPORA.items():
        save(handoff / f"{name}.json", {"weights": weights, "corpus_sha256": digest})
        print(f"{name}: {digest}")


def attribution(root, ref, quant, reference, quantized, spec, backend):
    import numpy as np
    from gguf.quants import GGMLQuantizationType, dequantize

    manifest = load(root / "gradient.json")
    assert manifest["backend"] == backend and manifest["placement"]["strict"]
    assert manifest["corpus_sha256"] == spec["corpus_sha256"]
    assert manifest["dtype"] == "f32_le" and manifest["objective"] == "mean_next_token_nll"
    assert [row["name"] for row in manifest["gradients"]] == spec["weights"]
    assert len(manifest["tokens"]) == 16
    assert manifest["observation_count"] == sum(len(ids) - 1 for ids in manifest["tokens"])
    scores = {}
    for row in manifest["gradients"]:
        name = row["name"]
        a, b = reference.tensors[name], quantized.tensors[name]
        assert a.kind == 0 and list(a.dims) == row["shape"] and a.dims == b.dims
        cols, nrows = a.dims
        row_bytes = b.size // nrows
        assert Path(row["file"]).name == row["file"]
        ref.seek(reference.base + a.relative)
        quant.seek(quantized.base + b.relative)
        digest = hashlib.sha256()
        chunks = []
        with (root / row["file"]).open("rb") as gradient:
            for lo in range(0, nrows, 512):
                rows = min(nrows - lo, 512)
                raw = gradient.read(rows * cols * 4)
                digest.update(raw)
                g = np.frombuffer(raw, "<f4")
                x = np.frombuffer(ref.read(rows * cols * 4), "<f4")
                packed = np.frombuffer(quant.read(rows * row_bytes), "u1").reshape(rows, row_bytes)
                y = dequantize(packed, GGMLQuantizationType(b.kind)).reshape(-1)
                chunks.append(float(np.dot(g.astype("f8"), (y - x).astype("f8"))))
            assert not gradient.read(1) and digest.hexdigest() == row["sha256"]
        scores[name] = math.fsum(chunks)
    return scores, manifest


def corpus_pairs(scores, quantized, base, align):
    rank = sorted(scores, key=lambda name: (-abs(scores[name]), name))
    pairs, seen = [], set()
    for upper in rank:
        if upper in seen or not upper.startswith("blk.") or upper not in base or not scores[upper]:
            continue
        high = quantized.tensors[upper]
        for lower in reversed(rank):
            if abs(scores[lower]) * 2 >= abs(scores[upper]):
                break
            if lower in seen or not lower.startswith("blk.") or lower not in base:
                continue
            low = quantized.tensors[lower]
            delta = align(low.size, quantized.alignment) - align(high.size, quantized.alignment)
            if low.dims == high.dims and delta > 0:
                pairs.append({"raise": upper, "lower": lower, "bytes_moved": delta})
                seen.update((upper, lower))
                break
        if len(pairs) == 3:
            break
    assert len(pairs) == 3
    return pairs


def random_pairs(template, seed, quantized, base, align):
    rng = random.Random(seed)
    selected, seen = [], set()
    for pair in template:
        upper, lower = pair["raise"], pair["lower"]
        high, low = quantized.tensors[upper], quantized.tensors[lower]
        assert high.dims == low.dims
        pool = [n for n in base if n.startswith("blk.") and n not in seen
                and quantized.tensors[n].dims == high.dims]
        rise = rng.choice(sorted(n for n in pool if base[n] == base[upper]))
        fall = rng.choice(sorted(n for n in pool if base[n] == base[lower]))
        a, b = quantized.tensors[rise], quantized.tensors[fall]
        delta = align(b.size, quantized.alignment) - align(a.size, quantized.alignment)
        assert delta == pair["bytes_moved"]
        selected.append({"raise": rise, "lower": fall, "bytes_moved": delta})
        seen.update((rise, fall))
    return selected


def plan(scratch, evidence):
    sys.path[:0] = [str(REPO / "tools"), str(W0 / "gguf-py")]
    from expert_gguf import align, read_gguf

    def swaps(rows):
        return [{key: row[key] for key in ("raise", "lower", "bytes_moved")} for row in rows]

    def overrides(pairs):
        result = base.copy()
        for row in pairs:
            a, b = row["raise"], row["lower"]
            result[a], result[b] = base[b], base[a]
        return result

    verified(REFERENCE, REFERENCE_SHA)
    f32 = verified(scratch / "handoff/reference.f32.gguf", F32_SHA)
    prior = load(X4C / "final/brewery-corpus-ranking.json")["recipe_proposal"]
    template = swaps(load(X4 / "cpu.json")["recipe_proposal"]["layer_pairs"])
    prior_controls = load(X4C / "brewery-random-selection.json")
    plans = {}
    with f32.open("rb") as ref, verified(QUANTIZED, QUANTIZED_SHA).open("rb") as quant:
        reference, quantized = read_gguf(ref), read_gguf(quant)
        base = {n: TYPES[t.kind] for n, t in quantized.tensors.items()
                if t.kind in TYPES and n != "output.weight"}
        assert base == prior["uniform_profile"]["overrides"]
        ordinary = {f"seed-{seed}": random_pairs(template, seed, quantized, base, align)
                    for seed in range(1, 7)}
        assert all(prior_controls[name] == pairs for name, pairs in ordinary.items())
        for train, test in (("dolly", "w4"), ("w4", "dolly")):
            spec = load(scratch / "handoff" / f"{train}.json")
            assert spec["corpus_sha256"] == CORPORA[train] != CORPORA[test]
            scores = {}
            manifests = {}
            for backend in ("cpu", "cuda"):
                root = scratch / f"{train}-{backend}"
                scores[backend], manifests[backend] = attribution(
                    root, ref, quant, reference, quantized, spec, backend
                )
                save(evidence / f"{train}-{backend}-ranking.json", scores[backend])
                save(evidence / f"{train}-{backend}-gradient.json", manifests[backend])
            cpu, gpu = manifests["cpu"], manifests["cuda"]
            assert cpu["placement"]["device"] == "CPU" and "CUDA" in gpu["placement"]["device"]
            assert cpu["tokens"] == gpu["tokens"]
            loss_error = abs(cpu["loss"] - gpu["loss"])
            assert loss_error < 5e-5, loss_error
            pairs = corpus_pairs(scores["cuda"], quantized, base, align)
            assert pairs == corpus_pairs(scores["cpu"], quantized, base, align)
            if train == "w4":
                assert pairs == swaps(prior["layer_pairs"])
                assert overrides(pairs) == prior["profile"]["overrides"]
            exchange = {"uniform": [], "corpus": pairs} | ordinary
            used = [overrides(p) for p in exchange.values()]
            # Fixed seed sequence; skip duplicate recipes before any evaluation.
            seed = 4
            while len(exchange) < 11:
                control = random_pairs(pairs, seed, quantized, base, align)
                profile = overrides(control)
                if profile not in used:
                    exchange[f"corpus-matched-seed-{seed}"] = control
                    used.append(profile)
                seed += 1
                assert seed <= 100
            if train == "w4":
                assert all(prior_controls[name] == rows for name, rows in exchange.items()
                           if name.startswith("corpus-matched"))
            plans[train] = {
                "train": train, "train_sha256": CORPORA[train], "positions": cpu["observation_count"],
                "test": test, "test_sha256": CORPORA[test], "cpu_mean_nll": cpu["loss"],
                "cuda_mean_nll": gpu["loss"], "cpu_cuda_loss_error": loss_error,
                "max_abs_attribution_error": max(abs(scores["cpu"][n] - v) for n, v in scores["cuda"].items()),
                "exchange": exchange, "profiles": {name: overrides(p) for name, p in exchange.items()},
            }
            print(f"{train} -> {test}: CPU and CUDA choose the same exchanges: {pairs}", flush=True)
    save(evidence / "plans.json", plans)


def controls_summary(rows, prefix):
    controls = [row for name, row in rows.items() if name.startswith(prefix)]
    recipe = rows["corpus"]
    result = {"count": len(controls)}
    for metric in ("perplexity", "mean_kl"):
        values = [row[metric] for row in controls]
        med = median(values)
        result[metric] = {
            "median": med,
            "recipe_beats_median": recipe[metric] < med,
            "recipe_beats": sum(recipe[metric] < value for value in values),
        }
    result["recipe_beats_on_both"] = sum(
        all(recipe[m] < row[m] for m in ("perplexity", "mean_kl")) for row in controls
    )
    return result


def evaluate(scratch, evidence, work):
    original_tools = load(X4C / "final/brewery-accept-tools.json")
    verified(REFERENCE, REFERENCE_SHA)
    verified(QUANTIZER, original_tools["quantizer_sha256"])
    verified(EVAL, original_tools["kl_sha256"])
    verified(PIN / "SHA256SUMS", original_tools["pin_manifest_sha256"])
    verified(PIN / "lanternfish", original_tools["lanternfish_sha256"])
    for name in ("brewery", "brewery-dispatch", "brewery-engine", "brewery-worker"):
        verified(scratch / "bins" / name, original_tools[name])
    for line in W0_LIBS.read_text().splitlines():
        digest, path = line.split()
        verified(Path(path), digest)
    save(evidence / "tools.json", original_tools)
    plans = load(evidence / "plans.json")
    assert {p["train"]: p["test"] for p in plans.values()} == {"w4": "dolly", "dolly": "w4"}
    for fold in plans.values():
        assert fold["train_sha256"] == CORPORA[fold["train"]]
        assert fold["test_sha256"] == CORPORA[fold["test"]]
        assert len(fold["profiles"]) == 11
    assert plans["dolly"]["profiles"]["uniform"] == plans["w4"]["profiles"]["uniform"]
    corpora = {name: verified(scratch / "handoff" / f"{name}.txt", digest).read_bytes()
               for name, digest in CORPORA.items()}
    bins = scratch / "bins"
    dist, worker = work / "dist", work / "worker"
    dist.mkdir()
    (dist / "index.html").write_text("<title>Isolated X4 generalization</title>\n")
    worker.mkdir(mode=0o700)
    cfg = work / "worker.json"
    save(cfg, {
        "root": str(worker), "source_roots": [str(REFERENCE.parent)],
        "engine": str(bins / "brewery-engine"), "disk_reserve_bytes": 0,
        "engine_env": {"BREWERY_GGUF_QUANTIZER": str(QUANTIZER), "BREWERY_CPU_THREADS": "16"},
    })
    cfg.chmod(0o600)
    dispatch = work / "dispatch.json"
    save(dispatch, {"hosts": {"local": {"local": True, "worker": str(bins / "brewery-worker"),
                                       "config": str(cfg), "capabilities": {"gguf_output": True}}}})
    env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "TMPDIR": "/arboretum/tmp"}
    stack = dict(env, BREWERY_LISTEN="127.0.0.1:0", BREWERY_DATA=str(work / "state"),
                 BREWERY_WEB_DIST=str(dist), BREWERY_DISPATCH=str(bins / "brewery-dispatch"),
                 BREWERY_DISPATCH_CONFIG=str(dispatch), BREWERY_POLL_MS="100")
    log = evidence / "controller.log"
    conversions, results = {}, {}
    opener = build_opener(ProxyHandler({}))
    with log.open("wb") as stream:
        controller = subprocess.Popen([bins / "brewery"], env=stack, stdout=stream, stderr=stream)
        try:
            def listen():
                assert controller.poll() is None, f"Controller stopped: {log}"
                prefix = "brewery listening on http://127.0.0.1:"
                return next((line.removeprefix(prefix) for line in log.read_text().splitlines()
                             if line.startswith(prefix)), None)

            port = int(wait(listen, timeout=10))
            assert 1024 <= port <= 65535 and port not in (3000, 4095, 4141)
            origin = f"http://127.0.0.1:{port}"

            def http(path, data=None):
                req = Request(origin + path, None if data is None else json.dumps(data).encode(),
                              {"Content-Type": "application/json"})
                try:
                    with opener.open(req, timeout=30) as response:
                        return json.load(response)
                except HTTPError as err:
                    raise RuntimeError(f"{path}: {err.code}: {err.read().decode()}") from err

            ref_id = http("/api/models", {"host": "local", "kind": "gguf", "path": str(REFERENCE)})["id"]

            def terminal(jid):
                receipt = http(f"/api/conversions/{jid}")
                return receipt if receipt["state"] in ("succeeded", "failed", "cancelled") else None

            def managed(label, overrides):
                key = tuple(sorted(overrides.items()))
                if key not in conversions:
                    print(f"Convert {label}", flush=True)
                    profile = http("/api/profiles", dict(format="gguf", preset="Q4_K_M",
                                                        name=f"X4g {label}", overrides=overrides))
                    assert profile["overrides"] == overrides
                    req = dict(format="gguf", input_ids=[ref_id], profile_revision=profile["revision"],
                               options={"threads": 16})
                    jid = http("/api/conversions", req)["id"]
                    receipt = wait(lambda: terminal(jid), timeout=1200)
                    save(evidence / f"{label}-receipt.json", receipt)
                    assert receipt["state"] == "succeeded" and receipt["stage"] == "verify", receipt
                    assert receipt["profile_revision"] == profile["revision"]
                    model = next(m for m in http("/api/models") if m["id"] == receipt["artifact_model_id"])
                    output = Path(model["path"]).resolve(strict=True)
                    assert output.is_relative_to(worker)
                    size, digest = output.stat().st_size, sha(output)
                    assert size == TARGET_BYTES, (label, size)
                    conversions[key] = (output, {"bytes": size, "sha256": digest, "conversion_id": jid,
                                                  "profile_revision": profile["revision"],
                                                  "receipt": f"{label}-receipt.json"})
                return conversions[key]

            uniform, info = managed("uniform", plans["w4"]["profiles"]["uniform"])
            assert info["sha256"] == load(X4 / "cpu/recipe-trace.json")["uniform"]["sha256"]
            known = load(X4C / "final/brewery-comparison.json")["rows"]
            for train, fold in plans.items():
                test = fold["test"]
                rows = {}
                for name, overrides in fold["profiles"].items():
                    label = f"{train}-to-{test}-{name}"
                    output, info = managed(label, overrides)
                    print(f"Score {label}", flush=True)
                    with (work / f"{label}.log").open("wb") as err:
                        result = subprocess.run([EVAL, REFERENCE, uniform, output, "8"],
                                                env=dict(env, LD_LIBRARY_PATH=str(W0 / "build/bin")),
                                                input=corpora[test], stdout=subprocess.PIPE, stderr=err,
                                                check=True, timeout=120)
                    raw = json.loads(result.stdout)
                    assert raw["prompts"] == 16 and raw["positions"] == EVAL_POSITIONS[test]
                    row = {"perplexity": raw["proposal_perplexity"], "mean_kl": raw["proposal_mean_kl"]} | info
                    if test == "w4" and (name == "uniform" or name.startswith("seed-")):
                        for value, expected in zip((row["perplexity"], row["mean_kl"]), known[name], strict=True):
                            assert math.isclose(value, expected, rel_tol=1e-6), (label, value, expected)
                    rows[name] = row
                    save(evidence / f"{train}-to-{test}-partial.json", rows)
                results[train] = {
                    "evaluation_corpus": test, "evaluation_sha256": CORPORA[test],
                    "evaluation_positions": EVAL_POSITIONS[test],
                    "reference_perplexity": raw["reference_perplexity"],
                    "recipes": rows, "random_original": controls_summary(rows, "seed-"),
                    "random_exchange_matched": controls_summary(rows, "corpus-matched-seed-"),
                }
                save(evidence / "results.json", results)
        finally:
            controller.terminate()
            controller.wait(timeout=10)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Prepare the corpus files and Lanternfish specifications")
    prep.add_argument("--cache", type=Path, default=Path("/arboretum/brewery-a1-evidence/fresh-cache"))
    planning = sub.add_parser("plan", help="Rank GPU gradients and freeze both folds' recipes")
    planning.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    scoring = sub.add_parser("convert", help="Convert each recipe through Brewery and score on held-out text")
    scoring.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.scratch, args.cache)
    elif args.command == "plan":
        plan(args.scratch, args.evidence)
    elif args.command == "convert":
        with tempfile.TemporaryDirectory(dir=args.scratch, prefix="isolated-") as work:
            evaluate(args.scratch, args.evidence, Path(work))
