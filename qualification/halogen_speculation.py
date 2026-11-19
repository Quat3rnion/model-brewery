#!/usr/bin/env python3
"""Summarize Halogen 0.18.22 native non-streaming response timings, one JSON response per line."""
import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

METHODS = ("head", "lookup", "grammar")
COUNTS = ("rounds", "drafted", "accepted")
MAX_LINE = 4 << 20


def need(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    data = dict(pairs)
    need(len(data) == len(pairs), "duplicate JSON key")
    return data


def count(value):
    need(type(value) is int and 0 <= value <= 2**31 - 1, "invalid counter")
    return value


def label(value):
    need(isinstance(value, str) and 0 < len(value) <= 256 and value.isprintable(),
         "invalid model or response ID")
    return value


def aggregate(path, wrapped, version):
    total = {k: dict.fromkeys(COUNTS, 0) for k in METHODS}
    with_spec = 0
    seen = set()
    model = None
    sha = hashlib.sha256()
    used = 0
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as src:
        st = os.fstat(src.fileno())
        need(stat.S_ISREG(st.st_mode) and st.st_size <= 128 << 20, "expected regular JSONL file <=128 MiB")
        while line := src.readline(MAX_LINE + 1):
            used += len(line)
            need(used <= 128 << 20 and len(line) <= MAX_LINE and len(seen) < 100_000,
                 "input exceeds byte, line or request limit")
            sha.update(line)
            try:
                data = json.loads(line, object_pairs_hook=unique)
                row = data["response"] if wrapped else data
                need(row["object"] in ("chat.completion", "text_completion"), "not a final completion")
                ident, alias = label(row["id"]), label(row["model"])
                need(ident not in seen and model in (None, alias), "duplicate response or mixed model aliases")
                seen.add(ident)
                model = alias
                timings = row["timings"]
                for key in ("prompt_n", "predicted_n"):
                    count(timings[key])
                if "speculation" not in timings:
                    need(not {"draft_n", "draft_n_accepted"} & timings.keys(),
                         "aggregate draft counts without method breakdown")
                    continue
                spec = timings["speculation"]
                label(spec["drafter"])
                for key in METHODS:
                    counters = {c: count(spec[key][c]) for c in COUNTS}
                    need(counters["accepted"] <= counters["drafted"], "accepted exceeds drafted")
                    for name, value in counters.items():
                        total[key][name] += value
                for c, announced in (("rounds", spec["rounds"]), ("drafted", timings["draft_n"]),
                                     ("accepted", timings["draft_n_accepted"])):
                    need(sum(spec[k][c] for k in METHODS) == count(announced), "counter totals disagree")
                with_spec += 1
            except (KeyError, TypeError, ValueError, RecursionError, AttributeError) as exc:
                # Never include arbitrary request text, identifiers or log content in errors.
                raise ValueError("invalid or inconsistent Halogen response record") from exc
        need(seen and used == st.st_size and st.st_ctime_ns == os.fstat(src.fileno()).st_ctime_ns,
             "empty or changing input")
    methods = {}
    for k, data in total.items():
        methods[k] = dict(data, acceptance_fraction=data["accepted"] / data["drafted"] if data["drafted"] else None,
                          accepted_per_round=data["accepted"] / data["rounds"] if data["rounds"] else None
                          ) if with_spec else None
    return dict(schema="brewery-halogen-speculation-v1", producer_version=version,
                version_evidence="operator-declared", model=model, input_sha256=sha.hexdigest(),
                requests=len(seen), total_with_speculation=with_spec, missing_speculation=len(seen)-with_spec,
                methods=methods, checkpoint_sha256=None, acceptance_by_depth=None, expert_routing=None,
                tuning_eligible=False, release_qualified=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("--producer-version", required=True, choices=["0.18.22"])
    p.add_argument("--wrapped", action="store_true", help="read each native response from the row's response field")
    a = p.parse_args()
    try:
        print(json.dumps(aggregate(a.capture, a.wrapped, a.producer_version), allow_nan=False))
    except (ValueError, OSError) as err:
        sys.exit(f"halogen_speculation: {err}")
