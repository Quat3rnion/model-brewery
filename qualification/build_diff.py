#!/usr/bin/env python3
"""Compare two saved GET /api/jobs/{id} responses for successful convert jobs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


MAX_BYTES = 16 << 20
MAX_FILES = 16_384
U64 = 2**64 - 1
DIGEST = re.compile(r"[a-f0-9]{64}")
ID = re.compile(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}")
SIDES = ("assets", "ngram", "vision", "mtp")
ROLES = ("checkpoint", "sidecar", "assets", "manifest")
BACKENDS = ("ht_backend", "i4r_backend", "q4cp_backend")
NUMERIC = {"seed": (0, U64), "threads": (1, 64), "ht_beam": (1, 256)}
CHECKPOINT = "native/model.hgn"
MANIFEST = "native/conversion-manifest.json"
LIMITS = [
    "Only saved receipts were read. File-tree digests were recomputed from recorded paths, "
    "sizes, and hashes. Artifact bytes, input bytes, and receipt authenticity were not verified.",
    "Source identity compares role-qualified physical input bundles, not decoded tensor "
    "values. Aliases are display labels. Branch and revision are reported only if explicitly recorded.",
    "Convert receipts do not attest a loaded runtime or a controlled comparison. A generated "
    "inference manifest is not runtime evidence. Quantization-only attribution is not established.",
    "Sizes are reported publication file bytes, not disk allocation, runtime memory, or quality. "
    "This report does not qualify a release.",
]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, low=0, high=U64):
    return type(value) is int and low <= value <= high


def matches(value, pattern):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def text(value, limit=4096):
    return (isinstance(value, str) and bool(value) and len(value.encode("utf-8")) <= limit
            and value.isprintable())


def obj(value):
    need(isinstance(value, dict), "expected JSON object")
    return value


def arr(value):
    need(isinstance(value, list), "expected JSON array")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"unexpected JSON constant: {value}")


def read_export(path):
    # Nonblocking + fstat avoids opening a FIFO as a blocking stream, and
    # O_NOFOLLOW prevents treating symlinks as direct input files.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        need(stat.S_ISREG(info.st_mode), "expected a regular JSON file")
        need(info.st_size <= MAX_BYTES, f"JSON exceeds {MAX_BYTES} bytes")
        data = stream.read(MAX_BYTES + 1)
    need(len(data) <= MAX_BYTES, f"JSON exceeds {MAX_BYTES} bytes")
    return obj(json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant))


def location(value):
    need(text(value) and value.startswith("/") and "\\" not in value, "invalid recorded worker path")
    path = PurePosixPath(value)
    need(path.root == "/" and path.name and str(path) == value and ".." not in path.parts,
         "invalid recorded worker path")
    return path


def bundle(record, entry_allowed):
    root = location(record["path"])
    members = arr(record["files"])
    need(0 < len(members) <= MAX_FILES and integer(record["bytes"]), "invalid bundle size or count")
    files = {}
    for member in members:
        name, size, digest = (obj(member)[key] for key in ("path", "bytes", "sha256"))
        need(text(name) and "\\" not in name and name not in files, "invalid or duplicate member path")
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and rel.name and str(rel) == name and ".." not in rel.parts,
             "invalid bundle member path")
        need(integer(size) and matches(digest, DIGEST), "invalid bundle member identity")
        files[name] = {"bytes": size, "sha256": digest}
    tree = sha256()
    for name, member in sorted(files.items()):
        tree.update(f"{name}\0{member['bytes']}\0{member['sha256']}\n".encode("utf-8"))
        need(not any(str(p) in files for p in PurePosixPath(name).parents),
             "bundle member is also a parent path")
    digest = tree.hexdigest()
    entry = files.get(root.name)
    need(record["bundle_sha256"] == digest
         and record["bytes"] == sum(member["bytes"] for member in files.values()),
         "bundle tree digest or total size does not match the recorded inventory")
    need(record["sha256"] == digest or (
        entry_allowed and entry and record["sha256"] == entry["sha256"]),
        "bundle entry SHA-256 does not match")
    return root, digest, files


def validate_job(job):
    need(matches(job["id"], ID) and text(job["host"], 256), "invalid controller job identity")
    need(job["kind"] == "convert" and job["state"] == "succeeded"
         and job["has_result"] is True and job["cancel_requested"] is False and "error" not in job,
         "expected a successful terminal convert job export")
    events = arr(job["events"])
    states = [obj(e)["state"] for e in events]
    times = [e["at"] for e in events]
    need(len(events) >= 3 and states[0] == "queued" and states[-1] == "succeeded"
         and "running" in states[1:-1] and all(s in ("queued", "running") for s in states[:-1])
         and "queued" not in states[states.index("running"):]
         and all(map(integer, times)) and times == sorted(times)
         and times[0] == job["created_at"] and times[-1] == job["updated_at"]
         and all(text(e["progress"]) and "error" not in e for e in events)
         and job["progress"] == events[-1]["progress"], "invalid controller job lifecycle")
    result = obj(job["result"])
    need(result["kind"] == "convert" and result["host"] == job["host"],
         "controller and worker conversion receipt do not match")
    return result


def validate_source(job, result, options):
    inputs, ids = arr(result["inputs"]), arr(job["input_ids"])
    need(1 <= len(inputs) <= 1 + len(SIDES) and len(inputs) == len(ids)
         and all(matches(id_, ID) for id_ in ids) and len(set(ids)) == len(ids),
         "invalid source input registrations")
    primary = obj(inputs[0])
    kind = primary["kind"]
    need(kind in ("hf", "hgn", "gguf") and (kind != "hf" or len(inputs) == 1),
         "invalid source kind or auxiliary inputs")
    indices = {"source": 0}
    for role in SIDES:
        if role in options:
            i = options[role]
            need(integer(i, 1, len(inputs) - 1) and i not in indices.values()
                 and obj(inputs[i])["kind"] == role, "invalid auxiliary input selection")
            indices[role] = i
    need(len(indices) == len(inputs), "unreferenced auxiliary input")
    digest = sha256(b"brewery-canonical-source-v1\n")
    for role, i in indices.items():
        item = obj(inputs[i])
        need(item["id"] == ids[i] and item["host"] == job["host"],
             "source input does not match controller registration")
        _, fingerprint, _ = bundle(item, entry_allowed=item["kind"] not in ("hf", "assets"))
        digest.update(f"{role}\0{item['kind']}\0{fingerprint}\n".encode("utf-8"))
    need(result["source_digest_scope"] == "role-qualified physical source bundles"
         and result["source_sha256"] == digest.hexdigest(), "invalid role-qualified source digest")
    # Standard exports carry only the alias. A producer may supply explicit
    # metadata on the primary registration; it never changes the content ID.
    source = {"kind": kind, "sha256": digest.hexdigest(), "alias": primary["name"]}
    need(text(source["alias"]), "invalid source alias")
    for key in ("model", "revision", "branch"):
        value = primary.get(key)
        need(value is None or text(value), f"invalid recorded source {key}")
        source[key] = value
    return source


def validate_process(result, options):
    receipt = obj(result["backend_receipt"])
    need(receipt["tool"] == "bf16-to-hgn"
         and obj(receipt["requested_options"]) == options, "invalid converter backend receipt")
    procs = arr(receipt["processes"])
    need(len(procs) == 1, "expected one converter process")
    proc = obj(procs[0])
    need(integer(proc["exit_code"], 0, 0) and integer(proc["pid"], 1, 2**32 - 1)
         and matches(proc["executable_sha256"], DIGEST) and matches(proc["stderr_sha256"], DIGEST),
         "invalid successful converter process receipt")
    location(proc["executable"])
    location(proc["stderr_log"])
    return {"name": receipt["tool"], "sha256": proc["executable_sha256"]}


def artifact_role(name):
    if name == CHECKPOINT:
        return "checkpoint"
    if name in (MANIFEST, "manifest.json"):
        return "manifest"
    path = PurePosixPath(name)
    need(path.parts[0] == "native" and len(path.parts) > 1, f"unexpected publication path: {name!r}")
    return "sidecar" if len(path.parts) == 2 and path.suffix == ".hgn" else "assets"


def validate_artifact(result, source, options, tool):
    artifact, report = obj(result["artifact"]), obj(result["report"])
    need(artifact["kind"] == "hgn", "expected a published HGN directory bundle")
    root, fingerprint, files = bundle(artifact, entry_allowed=False)
    need(CHECKPOINT in files and files[CHECKPOINT]["bytes"] > 104
         and MANIFEST in files and files[MANIFEST]["bytes"] > 0,
         "publication lacks the HGN checkpoint or native conversion manifest")
    for member in files:
        artifact_role(member)
    profile = report["profile"]
    need(profile in ("v2", "ht43") and profile == options.get("profile", "v2")
         and report["policy"] == f"bf16-hgn-{profile}-rtn-nf4-signed-rms-trellis-v1"
         and report["published"] is True and report["manifest"] == MANIFEST
         and location(report["native_directory"]) == root / "native"
         and report["source_format"] == source["kind"]
         and integer(report["source_tensors"], 1) and integer(report["output_tensors"], 1)
         and matches(report["encoder_source_sha256"], DIGEST),
         "invalid or inconsistent native conversion publication report")
    tool["encoder_sha256"] = report["encoder_source_sha256"]
    parameters = {}
    for key in BACKENDS:
        value = report[key]
        need(value in ("cpu", "cuda", "hip") and value == options.get(key, value),
             f"inconsistent native {key}")
        parameters[key] = value
    for key, bounds in NUMERIC.items():
        requested = options.get(key)
        need(key not in options or integer(requested, *bounds), f"invalid requested {key}")
        parameters[key] = requested
    runtime = obj(result["inference_bundle"])
    if runtime["status"] == "manifest_created":
        fmt = "hgn-v2" if profile == "v2" else "ht43"
        need(runtime["format"] == fmt and location(runtime["manifest"]) == root / "manifest.json"
             and {"manifest.json", "native/qwen38-flash-next-ngram.hgn",
                  "native/qwen38-flash-next-vision.hgn", "native/tokenizer/tokenizer.json"} <= files.keys()
             and all(files[key]["bytes"] > 0 for key in (
                 "manifest.json", "native/tokenizer/tokenizer.json",
                 "native/qwen38-flash-next-ngram.hgn", "native/qwen38-flash-next-vision.hgn")),
             "inference manifest does not match publication records")
        runtime = {"status": "manifest_created", "format": fmt, "loaded": None}
    else:
        need(runtime["status"] == "not_created" and text(runtime["reason"]) and "manifest.json" not in files,
             "invalid inference manifest status")
        runtime = None
    return {"profile": profile, "policy": report["policy"], "options": parameters,
            "artifact": {"bundle_sha256": fingerprint, "bytes": artifact["bytes"]},
            "runtime": runtime}, files


def load(path):
    try:
        job = read_export(path)
        result = validate_job(job)
        options = obj(job["options"])
        need(options.keys() <= {"profile", *SIDES, *BACKENDS, *NUMERIC}, "unsupported conversion options")
        source = validate_source(job, result, options)
        tool = validate_process(result, options)
        build, files = validate_artifact(result, source, options, tool)
        return {"job_id": job["id"], "source": source, "tool": tool, **build}, files
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def relation(left, right):
    return "same" if left == right else "changed"


def compare_options(left, right):
    pairs = [(left[key], right[key]) for key in left]
    if any(a is not None and b is not None and a != b for a, b in pairs):
        return "changed"
    if any(a is None or b is None for a, b in pairs):
        return "unknown"
    return "same"


def diff(left, right, a, b):
    need(left["job_id"] != right["job_id"], "expected two distinct convert jobs")
    comparison = {
        "source": relation(left["source"]["sha256"], right["source"]["sha256"]),
        "checkpoint": relation(a[CHECKPOINT], b[CHECKPOINT]),
        "profile": relation(left["profile"], right["profile"]),
        "options": compare_options(left["options"], right["options"]),
        "tool": relation(left["tool"]["sha256"], right["tool"]["sha256"]),
        "encoder": relation(left["tool"]["encoder_sha256"], right["tool"]["encoder_sha256"]),
        # Publication receipts do not attest runtime execution or comparable quality runs.
        "runtime": "unknown", "quantization_only": False,
    }
    totals = {role: {"left_bytes": 0, "right_bytes": 0, "delta_bytes": 0} for role in ROLES}
    rows = []
    for name in sorted(a.keys() | b.keys()):
        x, y, role = a.get(name), b.get(name), artifact_role(name)
        status = ("added" if x is None else "removed" if y is None else
                  "unchanged" if x == y else "changed")
        size_x, size_y = x["bytes"] if x is not None else 0, y["bytes"] if y is not None else 0
        delta = size_y - size_x
        rows.append({"path": name, "role": role, "status": status,
                     "left": x, "right": y, "delta_bytes": delta})
        totals[role]["left_bytes"] += size_x
        totals[role]["right_bytes"] += size_y
        totals[role]["delta_bytes"] += delta
    return {"schema": "brewery-build-diff-v1", "left": left, "right": right,
            "comparison": comparison, "files": rows, "bytes_by_role": totals, "limits": LIMITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Reads two regular JSON files (16 MiB each); "
                                            "does not open artifact paths or contact the controller.")
    parser.add_argument("left", type=Path, metavar="LEFT_CONVERT_JOB.json")
    parser.add_argument("right", type=Path, metavar="RIGHT_CONVERT_JOB.json")
    args = parser.parse_args()
    try:
        left, a = load(args.left)
        right, b = load(args.right)
        report = diff(left, right, a, b)
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
