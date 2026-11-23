"""Measure parent/child GGUF behavior on the fixed edit-check protocol."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.request import urlopen


DATA = Path(__file__).resolve().parents[1] / "crates/eval/data/edit-checks"
MAX_SOURCE_BYTES = 48 * 1024 * 1024
MAX_TOKEN_LOSS_BYTES = 256 * 1024


def file_sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_source(cache, source, fetch):
    path = cache / source["cache"]
    try:
        with path.open("rb") as f:
            raw = f.read(MAX_SOURCE_BYTES + 1)
    except FileNotFoundError:
        if not fetch:
            raise ValueError(f"{path} is missing; supply the pinned source or use --fetch") from None
        with urlopen(source["url"], timeout=30) as response:
            if not response.url.startswith("https://"):
                raise ValueError("refusing a non-HTTPS source redirect")
            raw = response.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES or hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"incorrect upstream hash or size: {source['url']}")
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
            staged = Path(tmp.name)
            try:
                tmp.write(raw)
                tmp.flush()
                staged.replace(path)
            finally:
                staged.unlink(missing_ok=True)
    if len(raw) > MAX_SOURCE_BYTES or hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError(f"incorrect upstream hash or size: {path}")
    return raw


def project(name, raw, item_ids):
    def pair(prompt, answer):
        return [" ".join(prompt.split()), " " + " ".join(answer.split())]

    source = json.loads(raw)
    if name == "counterfact.jsonl":
        source = {row["case_id"]: row for row in source}
    result = []
    for i in item_ids:
        row = source[i]
        if name == "counterfact.jsonl":
            edit = row["requested_rewrite"]
            prompt = edit["prompt"].format(edit["subject"])
            alt, truth = edit["target_new"]["str"], edit["target_true"]["str"]
            result.append(dict(id=i, target=pair(prompt, alt),
                               paraphrases=[pair(p, alt) for p in row["paraphrase_prompts"][:2]],
                               locality=[pair(p, truth) for p in row["neighborhood_prompts"][:2]]))
        elif name == "zsre.jsonl":
            alt = row["alt"]
            result.append(dict(id=i, target=pair(row["src"], alt),
                               paraphrases=[pair(row["rephrase"], alt)],
                               locality=[pair(row["loc"].removeprefix("nq question: "),
                                              row["loc_ans"])]))
        else:
            raise ValueError(f"unknown fixture: {name}")
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for r in result).encode("utf-8")


def load_fixtures(cache, fetch=False, data=DATA):
    pins = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    sources = json.loads((data / "sources.json").read_text(encoding="utf-8"))
    rows = []
    for name, pin in pins.items():
        if name == "heldout.txt":
            raw = (data / name).read_bytes()
        else:
            source = sources[name]
            raw = project(name, read_source(cache, source, fetch), source["item_ids"])
        lines = raw.decode("utf-8").splitlines()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError(f"incorrect fixture hash: {name}")
        if name == "heldout.txt":
            for line in lines:
                prompt, suffix = line.split(" ", 1)
                rows.append(("perplexity", prompt, " " + suffix))
        else:
            for line in lines:
                item = json.loads(line)
                dataset = name.split(".")[0]
                rows.append((f"{dataset}_target", *item["target"]))
                rows.extend((f"{dataset}_paraphrase", *p) for p in item["paraphrases"])
                rows.extend((f"{dataset}_locality", *p) for p in item["locality"])
    return pins, rows


@dataclass(frozen=True)
class RowScore:
    tokens: tuple[int, ...]
    prompt_tokens: int
    logprob: float
    probability: float
    rank: int
    top1: bool
    details: tuple[tuple[float, str], ...] | None = None


def aggregate(groups):
    results = {}
    for group, scores in groups.items():
        count = sum(len(s.tokens) - s.prompt_tokens for s in scores)
        if group == "perplexity":
            results[group] = (math.exp(-math.fsum(s.logprob for s in scores) / count), "lower")
            continue
        locality = group.endswith("_locality")
        results[f"{group}_probability"] = (
            math.fsum(s.probability for s in scores) / count, "neither" if locality else "higher",
        )
        results[f"{group}_rank"] = (
            sum(s.rank for s in scores) / count, "neither" if locality else "lower",
        )
        if not locality:
            name = group.removesuffix("_target") + "_efficacy" if group.endswith("_target") else group
            results[name] = (sum(s.top1 for s in scores) / len(scores), "higher")
    return results


def score(runner, path, threads, rows):
    text = "".join(f"{prompt}\t{answer}\n" for _, prompt, answer in rows)
    proc = subprocess.run(
        [runner, path, str(threads)], input=text, text=True, encoding="utf-8",
        stdout=subprocess.PIPE, check=True,
    )
    generation, *lines = proc.stdout.splitlines()
    header, generation = generation.split("\t")
    if header != "generation" or len(lines) != len(rows):
        raise ValueError("incorrect native runner result")
    groups = defaultdict(list)
    for (group, _, _), line in zip(rows, lines, strict=True):
        tokens, prefix, logp, prob, ranks, top1, *extra = line.split("\t")
        tokens = tuple(map(int, tokens.split(",")))
        prefix = int(prefix)
        if not 0 < prefix < len(tokens) <= 512 or len(extra) > 1:
            raise ValueError("incorrect native runner token boundary")
        details = None
        if group == "perplexity" and extra:
            details = tuple((float(lp), piece) for lp, piece in
                            (entry.split(":") for entry in extra[0].split(",")))
            if len(details) != len(tokens) - prefix or any(
                not math.isfinite(lp) or lp > 0 or len(piece) > 1024
                or bytes.fromhex(piece).hex() != piece for lp, piece in details
            ) or not math.isclose(math.fsum(lp for lp, _ in details), float(logp),
                                  rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("invalid native runner token log probabilities or pieces")
        groups[group].append(RowScore(
            tokens, prefix, float(logp), float(prob), int(ranks), top1 == "1", details,
        ))
    return bytes.fromhex(generation).decode("utf-8", errors="replace"), groups


def token_loss(rows, parent, child):
    heldout = [(p, c) for p, c in zip(parent["perplexity"], child["perplexity"], strict=True)]
    if len(heldout) != sum(group == "perplexity" for group, *_ in rows):
        raise ValueError("missing heldout sentences")
    if all(p.details is None and c.details is None for p, c in heldout):
        return None  # The old pinned runner still supports aggregate checks.
    result = []
    for p, c in heldout:
        if (p.tokens, p.prompt_tokens) != (c.tokens, c.prompt_tokens):
            raise ValueError("models do not tokenize the scored inputs identically")
        if p.details is None or c.details is None:
            raise ValueError("missing heldout token scores")
        sentence = []
        for id_, (before, a), (after, b) in zip(
            p.tokens[p.prompt_tokens:], p.details, c.details, strict=True,
        ):
            if a != b:
                raise ValueError("models do not decode the same token pieces")
            sentence.append(dict(id=id_, piece_hex=a, parent_logprob=before, child_logprob=after))
        result.append(sentence)
    if len(json.dumps(result, allow_nan=False)) > MAX_TOKEN_LOSS_BYTES:
        raise ValueError("heldout token scores exceed the size limit")
    return result


def progress(completed):
    dest = os.environ.get("BREWERY_NATIVE_PROGRESS")
    if not dest:
        return
    path = Path(dest)
    staged = path.with_suffix(".partial")
    try:
        staged.write_text(json.dumps(dict(
            phase="scratch_checks", unit="items", completed=completed, total=4,
            updated_at_ms=time.time_ns() // 1_000_000,
        )), encoding="utf-8")
        staged.replace(path)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "parent-gguf", "child-gguf", "parent-safetensors", "child-safetensors"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--cache", type=Path, required=True, help="local dataset cache")
    parser.add_argument("--fetch", action="store_true", help="fetch missing pinned sources")
    args = parser.parse_args()
    files = {
        "runner_sha256": args.runner,
        "parent_sha256": args.parent_safetensors,
        "revision_sha256": args.child_safetensors,
        "parent_gguf_sha256": args.parent_gguf,
        "revision_gguf_sha256": args.child_gguf,
    }
    hashes = {name: file_sha(path) for name, path in files.items()}
    pins, rows = load_fixtures(args.cache, args.fetch)
    progress(1)
    parent_text, parent = score(args.runner, args.parent_gguf, args.threads, rows)
    progress(2)
    child_text, child = score(args.runner, args.child_gguf, args.threads, rows)
    progress(3)
    for group in parent:
        if any((p.tokens, p.prompt_tokens) != (c.tokens, c.prompt_tokens)
               for p, c in zip(parent[group], child[group], strict=True)):
            raise ValueError("models do not tokenize the scored inputs identically")
    after = aggregate(child)
    metrics = [
        dict(name=name, parent=value, child=after[name][0], better=better)
        for name, (value, better) in aggregate(parent).items()
    ]
    if any(file_sha(files[name]) != sha for name, sha in hashes.items()):
        raise ValueError("a model or runner file changed during scoring")
    report = dict(
        schema="brewery.scratch-behavior.v1", qualification="not release-qualified",
        **hashes, fixture_sha256=pins, metrics=metrics,
        parent_generation=parent_text, child_generation=child_text,
    )
    if (tokens := token_loss(rows, parent, child)) is not None:
        report["token_loss"] = tokens
    print(json.dumps(report, allow_nan=False))
    progress(4)


if __name__ == "__main__":
    main()
