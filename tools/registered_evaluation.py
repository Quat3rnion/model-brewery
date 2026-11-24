"""Build pinned evaluation caches from reviewed local source files."""

import argparse
import csv
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit


SCHEMA = "brewery.registered-evaluation-prompts.v1"
SIDES = ("positive", "negative")
PARTITIONS = ("tune", "final_gate")
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


def serial(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def source_rows(source, directory):
    from direction_prompt_cache import group_id, verify_source

    url = urlsplit(source["url"])
    if (url.scheme != "https" or not url.hostname or url.username is not None
            or url.query or url.fragment or not url.path.rsplit("/", 1)[-1]
            or not any(re.fullmatch("[a-f0-9]{40}", part) for part in url.path.split("/")[1:-1])):
        raise ValueError("source URL must be HTTPS and contain a fixed commit path segment")
    sha, size, fmt = (source[k] for k in ("sha256", "size", "format"))
    if (not re.fullmatch("[a-f0-9]{64}", sha) or not 1 <= size <= 48 * 1024 * 1024
            or fmt not in {"csv", "json", "jsonl"} or not source["license"].strip()):
        raise ValueError("invalid evaluation source pin")
    with (directory / f"{sha}.{fmt}").open("rb") as file:
        raw = file.read(size + 1)
    if len(raw) != size:
        raise ValueError("source size mismatch")
    data = verify_source(raw, sha).decode("utf-8-sig")
    if fmt == "csv":
        rows = csv.DictReader(io.StringIO(data, newline=""))
    elif fmt == "json":
        rows = json.loads(data)
        if not isinstance(rows, list):
            raise ValueError("JSON source must be an array of objects")
    else:
        rows = (json.loads(line) for line in data.split("\n") if line.strip())
    groups = {}
    rule = source.get("filter")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("evaluation source rows must be objects")
        if rule is not None:
            field, value = rule["field"], rule["equals"]
            if field not in row or type(row[field]) is not type(value) or row[field] != value:
                continue
        user = row.get(source["field"])
        if not isinstance(user, str):
            raise ValueError("selected evaluation field must be a string")
        user = user.strip()
        if valid_message(user) and user:
            group = group_id(user)
            groups[group] = min(user, groups.get(group, user))
    return groups


def valid_message(text):
    return isinstance(text, str) and len(text.encode()) <= 4096 and "\0" not in text and "<|" not in text


def build(spec, sources, output):
    """Read only SHA-named local sources; return distinct tune/final_gate pins."""
    from direction_prompt_cache import store
    from hashlib import sha256

    sizes = spec["sizes"]
    if any(type(sizes[p]) is not int or not 1 <= sizes[p] <= 32 for p in PARTITIONS):
        raise ValueError("partition sizes must be 1 to 32 per side")
    groups = {side: source_rows(spec[side], sources) for side in SIDES}
    overlap = groups["positive"].keys() & groups["negative"].keys()
    rows = {p: [] for p in PARTITIONS}
    for side in SIDES:
        system = spec[side]["system"]
        if not valid_message(system):
            raise ValueError("invalid evaluation system message")
        selected = sorted(groups[side].keys() - overlap)
        if len(selected) < sum(sizes[p] for p in PARTITIONS):
            raise ValueError("not enough non-overlapping evaluation groups")
        start = 0
        for partition in PARTITIONS:
            end = start + sizes[partition]
            rows[partition].extend(dict(side=side, system=system, user=groups[side][group])
                                   for group in selected[start:end])
            start = end
    output.mkdir(mode=0o700, exist_ok=True, parents=True)
    pins = {}
    for partition in PARTITIONS:
        data = serial(dict(schema=SCHEMA, name=spec["name"], partition=partition,
                           rows=rows[partition]))
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("evaluation partition exceeds the cache size limit")
        pin = sha256(data).hexdigest()
        store(output / (pin + ".json"), data)
        pins[partition] = pin
    return pins


def hit(classes, text):
    clean = text.casefold().lstrip()
    result = {"counts": False}
    for cls in classes:
        if (any(clean.startswith(p.casefold()) for p in cls.get("prefixes", []))
                or any(p.casefold() in clean for p in cls.get("substrings", []))):
            result = {"class": cls["name"], "counts": cls["kind"] == "counts"}
            break
    # Short outputs have no prefix: a prefix must never contain the entire reply.
    if len(text) > 120:
        result["prefix"] = text[:120]
    return result


def marker_report(config, outputs):
    classes = config["classifier"]["classes"]
    report = {key: config[key] for key in ("name", "partition", "sha256")}
    report["schema"] = "brewery.registered-marker-rate.v1"
    report["max_tokens"] = config["classifier"]["max_tokens"]
    for side in SIDES:
        hits = [{"parent": hit(classes, parent), "child": hit(classes, child)}
                for kind, parent, child in outputs if kind == side]
        stats = {"hits": hits}
        for role in ("parent", "child"):
            stats[f"{role}_rate"] = sum(pair[role]["counts"] for pair in hits) / len(hits)
            stats[f"{role}_class_counts"] = {
                cls["name"]: sum(pair[role].get("class") == cls["name"] for pair in hits)
                for cls in classes
            }
        report[side] = stats
    return report


def replay(runner, parent, child, threads, cache, config):
    from hashlib import sha256

    pin = config["sha256"]
    if not re.fullmatch("[a-f0-9]{64}", pin):
        raise ValueError("invalid evaluation partition pin")
    with (cache / (pin + ".json")).open("rb") as file:
        raw = file.read(MAX_ARTIFACT_BYTES + 1)
    if len(raw) > MAX_ARTIFACT_BYTES or sha256(raw).hexdigest() != pin:
        raise ValueError("evaluation partition hash or size mismatch")
    data = json.loads(raw)
    if (data["schema"], data["name"], data["partition"]) != (
            SCHEMA, config["name"], config["partition"]):
        raise ValueError("incorrect evaluation set or partition")
    rows = data["rows"]
    if ([row["side"] for row in rows]
            != [side for side in SIDES for _ in range(config["size_per_side"])]
            or any(not valid_message(r["system"]) or not valid_message(r["user"])
                   or not r["user"].strip() for r in rows)):
        raise ValueError("invalid evaluation partition rows")
    cap = config["classifier"]["max_tokens"]
    if type(cap) is not int or not 1 <= cap <= 128:
        raise ValueError("invalid marker token cap")
    lines = "".join(f"marker:{cap}\t{r['system'].encode().hex()}\t{r['user'].encode().hex()}\n"
                    for r in rows)
    proc = subprocess.run([runner, parent, str(threads), "--behavior-v1", child],
                          input=lines, text=True, encoding="ascii", stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    if proc.returncode:
        raise ValueError("native marker generation failed (requires a marker-capable runner)")
    outputs = []
    try:
        for row, line in zip(rows, proc.stdout.splitlines(), strict=True):
            mode, a, na, b, nb = line.split("\t")
            if mode != f"marker:{cap}" or not (0 <= int(na) <= cap and 0 <= int(nb) <= cap):
                raise ValueError
            outputs.append((row["side"], bytes.fromhex(a).decode(errors="replace"),
                            bytes.fromhex(b).decode(errors="replace")))
    except ValueError:
        raise ValueError("invalid native marker result") from None
    return marker_report(config, outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    pins = build(spec, args.sources, args.output)
    print(serial(dict(**spec, pins=pins)).decode(), end="")


if __name__ == "__main__":
    main()
