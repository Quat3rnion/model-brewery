"""Prepare licensed direction prompt sets for use by offline CPU workers."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen


SOURCES = {
    "dolly": (
        "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
        "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl",
        "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
        13085339,
    ),
    "nlp_survey": (
        "https://raw.githubusercontent.com/anthropics/evals/"
        "84fcc677e52e1902d696c32cd1a6b663e70d3993/sycophancy/sycophancy_on_nlp_survey.jsonl",
        "582860b42e2beec806a7d361a08bcce7fdb264e2e697d55104074540352fb308",
        9725180,
    ),
}

CONTRASTS = {
    "slop": (
        "Write with familiar stock phrases and conventional imagery.",
        "Write with concrete details and original imagery. Avoid stock phrases.",
    ),
    "sycophancy": (
        "Give the answer that agrees with the user's stated opinion. Respond with the answer letter.",
        "Give your own answer, independent of the user's opinion. Respond with the answer letter.",
    ),
    "response_length": (
        "Answer in detail. Explain the main points and give examples where useful.",
        "Give the shortest answer that fully answers the question.",
    ),
}

SURVEY_QUESTION = re.compile(r"Do you agree or disagree with (.*?)(?:\s+Choices:)", re.S)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_source(data, expected):
    if digest(data) != expected:
        raise ValueError("source SHA-256 mismatch")
    return data


def normalize(behavior, source_rows, *, train=12, held_out=8):
    positive, negative = CONTRASTS[behavior]
    by_id = {}
    for source in source_rows:
        agreement = None
        if behavior == "sycophancy":
            user = source["question"].strip()
            claim = SURVEY_QUESTION.search(user)
            if not claim:
                continue
            group = claim[1]
            agreement = source["answer_matching_behavior"].strip()
            if agreement not in {"(A)", "(B)"}:
                continue
        else:
            category = "creative_writing" if behavior == "slop" else "open_qa"
            if source["category"] != category or source["context"].strip():
                continue
            user = group = source["instruction"].strip()
        if not user or len(user.encode()) > 3072 or any(c in user for c in ("<|", "\x00")):
            continue
        row_id = digest(" ".join(group.casefold().split()).encode())
        # Prefer the sorted first persona per question; source row order is immaterial.
        candidate = (user, agreement)
        if row_id not in by_id or candidate < by_id[row_id]:
            by_id[row_id] = candidate
    if len(by_id) < train + held_out:
        raise ValueError("not enough distinct prompt groups")
    rows = []
    for i, (row_id, (user, agreement)) in enumerate(sorted(by_id.items())[:train + held_out]):
        row = {
            "id": row_id,
            "split": "train" if i < train else "held_out",
            "positive": {"system": positive, "user": user},
            "negative": {"system": negative, "user": user},
        }
        if agreement is not None:
            row["expected_agreement"] = agreement
        rows.append(row)
    return (json.dumps(
        {"schema": "brewery.contrast-prompts.v1", "behavior": behavior, "rows": rows},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n").encode("ascii")


def cached_source(cache, name):
    url, sha, size = SOURCES[name]
    path = cache / f"{sha}.jsonl"
    if path.exists():
        with path.open("rb") as source:
            data = source.read(size + 1)
    else:
        with urlopen(url, timeout=120) as source:
            if not source.url.startswith("https://"):
                raise ValueError("source redirect must use HTTPS")
            data = source.read(size + 1)
    if len(data) != size:
        raise ValueError("source size mismatch")
    verify_source(data, sha)
    if not path.exists():
        store(path, data)
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def store(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        stage = Path(file.name)
        try:
            file.write(data)
            file.close()
            stage.replace(path)
        finally:
            stage.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    dolly = cached_source(args.cache, "dolly")
    survey = cached_source(args.cache, "nlp_survey")
    for behavior in CONTRASTS:
        data = normalize(behavior, survey if behavior == "sycophancy" else dolly)
        sha = digest(data)
        store(args.cache / f"{sha}.json", data)
        print(f"{behavior}\t{sha}")


if __name__ == "__main__":
    main()
