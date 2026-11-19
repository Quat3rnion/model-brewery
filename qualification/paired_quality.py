#!/usr/bin/env python3
"""Compare paired GSM8K predictions from two saved workbench jobs offline."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path, PurePosixPath
import sys

from release_check import (
    DATASET, PROTOCOL, MAX_PREDICTIONS, Invalid, answer, array, complete,
    decode, digest, exported_job, hash_ok, integer, need, object_, questions,
    read_file, text, unique,
)

SCHEMA = "brewery-paired-quality-v1"
ARTIFACT_KINDS = {"baseline": "gsm8k-baseline", "evaluate": "gsm8k-evaluation"}
HGN_MOUNTS = {"checkpoint": "/brewery-checkpoint.hgn", "lookup": "/brewery-lookup.hgn",
              "vision": "/brewery-vision.hgn"}


def halogen(p, source):
    proof = object_(p["load_proof"])
    q = object_(proof["qualification"])
    obs = object_(proof["observation"])
    image = proof["image_id"]
    need(p["source_sha256"] == source and hash_ok(p["manifest_sha256"])
         and proof["schema"] == "halogen-linux-engine-load-v1"
         and isinstance(image, str) and image.startswith("sha256:") and hash_ok(image[7:])
         and hash_ok(proof["container_id"]) and q["schema"] == "halogen-hgn-preflight-v1"
         and q["format"] in ("hgn-v2", "ht43")
         and integer(obs["engine_pid"], 2)
         and integer(obs["start_ticks"], 1) and integer(obs["listener_inode"], 1),
         "invalid Halogen bundle or runtime identity")
    files = unique(q["files"], "role")
    mapped = unique(obs["mapped"], "path")
    need("checkpoint" in files and files.keys() <= HGN_MOUNTS.keys()
         and mapped.keys() == {HGN_MOUNTS[r] for r in files}, "inconsistent runtime member roles")
    members = []
    for role, f in sorted(files.items()):
        meta = object_(f["identity"])
        path = PurePosixPath(f["path"])
        observed = mapped[HGN_MOUNTS[role]]
        need(path.is_absolute() and ".." not in path.parts and hash_ok(f["sha256"])
             and text(f["hgn_model_id"]) and integer(meta["bytes"], 1)
             and integer(meta["inode"], 1) and integer(meta["dev_major"]) and integer(meta["dev_minor"])
             and all(type(observed[k]) is int for k in ("major", "minor", "bytes", "inode"))
             and [meta[k] for k in ("dev_major", "dev_minor", "bytes", "inode")]
             == [observed[k] for k in ("major", "minor", "bytes", "inode")],
             "qualified member and runtime mapping disagree")
        members.append({"role": role, "path": f["path"], "sha256": f["sha256"],
                        "hgn_model_id": f["hgn_model_id"], "bytes": meta["bytes"]})
    tokenizer = unique(p["tokenizer"], "path")
    parents = {str(PurePosixPath(name).parent) for name in tokenizer}
    need(1 <= len(tokenizer) <= 32 and len(parents) == 1
         and parents <= {"tokenizer", "native/tokenizer"}
         and any(PurePosixPath(n).name == "tokenizer.json" for n in tokenizer)
         and all(str(PurePosixPath(name)) == name and hash_ok(t["sha256"])
                 for name, t in tokenizer.items()), "invalid tokenizer file identities")
    return {
        "method": "halogen", "manifest_sha256": p["manifest_sha256"], "files": members,
        "tokenizer": [{"path": name, "sha256": t["sha256"]} for name, t in sorted(tokenizer.items())],
        "runtime": {"schema": proof["schema"], "image_id": image, "container_id": proof["container_id"],
                    "format": q["format"], "engine_pid": obs["engine_pid"], "start_ticks": obs["start_ticks"],
                    "listener_inode": obs["listener_inode"]},
        "import_sha256": None,
    }


def provenance(job, identity):
    p = object_(identity["provenance"])
    method = p.get("method")
    synthetic = identity["synthetic"]
    if job["kind"] == "evaluate" and not synthetic:
        return halogen(p, identity["source_sha256"])
    out = {"method": method, "manifest_sha256": None, "files": None, "tokenizer": None,
           "runtime": None, "import_sha256": None}
    if method == "imported-predictions":
        need(hash_ok(p["import_sha256"]) and len(job["input_ids"]) in (1, 2)
             and (job["kind"] != "baseline" or
                  (len(job["input_ids"]) == 1 and "baseline_source" not in job["options"])),
             "invalid imported-predictions identity")
        out["import_sha256"] = p["import_sha256"]
        return out
    expected = "synthetic-endpoint-fixture" if synthetic else "operator-bf16-endpoint"
    need(job["kind"] == "baseline" and method == expected
         and p["source_evidence"] == "matched-response-headers"
         and all(text(p[key]) for key in ("baseline_source", "runtime_id", "model"))
         and job["options"].get("baseline_source") == p["baseline_source"] and not job["input_ids"],
         "unknown or incompatible prediction provenance")
    out["runtime"] = {"identity": "declared_endpoint",
                      **{k: p[k] for k in ("baseline_source", "runtime_id", "model", "source_evidence")},
                      "image_id": None}
    return out


def read_side(job_path, predictions_path, qs):
    job = exported_job(job_path)
    need(job["kind"] in ARTIFACT_KINDS, "expected a baseline or evaluate job")
    result = complete(job)
    dataset = object_(result["dataset"])
    need(dataset == DATASET and dataset["full_split"] is True
         and result["protocol_id"] == PROTOCOL, "not the full pinned GSM8K test split and protocol")
    art = object_(result["artifact"])
    need(art["kind"] == ARTIFACT_KINDS[job["kind"]] and integer(art["bytes"], 1, MAX_PREDICTIONS)
         and hash_ok(art["sha256"]), "invalid prediction artifact identity")
    raw = read_file(predictions_path, MAX_PREDICTIONS)
    need(len(raw) == art["bytes"] and digest(raw) == art["sha256"],
         "prediction artifact SHA-256 or size mismatch")
    run = object_(decode(raw))
    identity = object_(result["identity"])
    need(run["schema"] == "brewery-gsm8k-predictions-v1"
         and run["dataset_sha256"] == DATASET["sha256"] and run["protocol_id"] == PROTOCOL
         and type(identity["synthetic"]) is bool and run["synthetic"] is identity["synthetic"]
         and hash_ok(identity["source_sha256"]) and run["source_sha256"] == identity["source_sha256"]
         and object_(run["provenance"]) == object_(identity["provenance"]),
         "prediction payload identity, protocol or synthetic flag differs from job")
    rows = array(run["predictions"])
    need(len(rows) == len(qs), "prediction artifact must cover all 1319 questions")
    parsed = []
    for i, (row, (question_sha, _)) in enumerate(zip(rows, qs), 1):
        need(integer(row["id"], i, i) and row["question_sha256"] == question_sha,
             f"prediction ID/order or question hash mismatch at row {i}")
        need(isinstance(row["text"], str) and len(row["text"].encode()) <= 65536,
             f"invalid completion at row {i}")
        parsed.append(answer(row["text"]))
    correct = sum(a == ref for a, (_, ref) in zip(parsed, qs))
    n = len(qs)
    score = {"correct": correct, "total": n, "unparsed": parsed.count(None),
             "accuracy_basis_points": (correct * 10_000 + n // 2) // n}
    need(result["score"] == score and all(type(v) is int for v in result["score"].values()),
         "reported score differs from recomputed score")
    return ({"job_id": job["id"], "kind": job["kind"], "host": job["host"], "input_ids": job["input_ids"],
             "source_sha256": identity["source_sha256"], "predictions": {"sha256": art["sha256"], "bytes": art["bytes"]},
             "score": score, "provenance": provenance(job, identity)}, parsed, identity["synthetic"])


def analyze(left_side, right_side, qs):
    left, a, synthetic = left_side
    right, b, right_synthetic = right_side
    need(left["job_id"] != right["job_id"], "choose two distinct job IDs")
    need(synthetic is right_synthetic, "synthetic and non-synthetic jobs cannot be paired")
    improvements, regressions, changes = [], [], []
    both_correct = both_incorrect = 0
    for i, (x, y, (_, ref)) in enumerate(zip(a, b, qs), 1):
        if x != y:
            changes.append(i)
        if x == ref and y == ref:
            both_correct += 1
        elif x == ref:
            regressions.append(i)
        elif y == ref:
            improvements.append(i)
        else:
            both_incorrect += 1
    gains, losses = len(improvements), len(regressions)
    delta, discordant = gains - losses, gains + losses
    # Under the null, each discordant pair is equally likely to favor either side.
    # Keep exact integer arithmetic until the optional display value.
    denominator = 2**discordant
    numerator = min(denominator, 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)))
    fraction = Fraction(numerator, denominator)
    warnings = ["Diagnostic comparison only; not release qualification or causal evidence about quantization."]
    if left["source_sha256"] != right["source_sha256"]:
        warnings.append("Source identities differ. This is a matched-workload comparison, not a same-source build comparison.")
    if synthetic:
        warnings.append("Both inputs are synthetic fixtures; the statistics do not measure model quality.")
    for label, side in (("Left", left), ("Right", right)):
        if side["provenance"]["runtime"] is None:
            warnings.append(f"{label}: the originating runtime, model files and tokenizer are unknown.")
        elif side["provenance"]["method"] != "halogen":
            warnings.append(f"{label}: exact model files and tokenizer are unknown; endpoint claims are not independent proof.")
    return {
        "schema": SCHEMA, "dataset": DATASET, "protocol_id": PROTOCOL, "synthetic": synthetic,
        "left": left, "right": right,
        "transitions": {"both_correct": both_correct, "both_incorrect": both_incorrect,
                        "improvements": gains, "regressions": losses},
        "improvement_ids": improvements, "regression_ids": regressions, "parsed_answer_change_ids": changes,
        "delta_correct": delta, "delta_percentage_points": 100 * delta / len(qs),
        "uncertainty": {
            "method": "exact-mcnemar-two-sided", "discordant_pairs": discordant,
            "p_value": float(fraction) or None,
            "p_value_exact": {"numerator": str(fraction.numerator), "denominator": str(fraction.denominator)},
            "assumption": ("Conditional on discordance, independent question pairs favor either side with equal probability "
                           "under the null. Correctness ties are excluded, p=1 if all pairs tie, and no CI is estimated. "
                           "A null numeric p-value means floating-point underflow; use the exact fraction."),
        },
        "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use unchanged GET /api/jobs/{id} exports and each job's corresponding prediction artifact. "
               "LEFT and RIGHT are positional; deltas are right minus left. Requires all 1,319 pinned test rows. "
               "Limits: 64 MiB per job export and 90 MiB per prediction artifact. "
               "Pairs of synthetic fixtures are labeled; mixed pairs are rejected. "
               "Output excludes prompts and responses and does not qualify a release.",
    )
    parser.add_argument("left_job", type=Path, metavar="LEFT_JOB")
    parser.add_argument("right_job", type=Path, metavar="RIGHT_JOB")
    parser.add_argument("--left-predictions", type=Path, required=True, metavar="FILE")
    parser.add_argument("--right-predictions", type=Path, required=True, metavar="FILE")
    args = parser.parse_args(argv)
    try:
        qs = questions()
        sides = []
        for label, job, pred in (("left", args.left_job, args.left_predictions),
                                 ("right", args.right_job, args.right_predictions)):
            try:
                sides.append(read_side(job, pred, qs))
            except (KeyError, TypeError, RecursionError) as e:
                raise Invalid(f"{label}: malformed job or prediction evidence") from e
            except (OSError, ValueError) as e:
                raise Invalid(f"{label}: {e}") from e
        report = analyze(*sides, qs)
    except (OSError, ValueError, RecursionError) as e:
        print(f"paired_quality: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
