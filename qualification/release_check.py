#!/usr/bin/env python3
"""Offline cross-check of untouched workbench job details and their evidence."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from release_source import METADATA

TARGETS = ("gfx1151", "sm_89", "sm_120")
PROFILES = ("v2", "ht43")
ROWS = tuple((t, p) for t in TARGETS for p in PROFILES)
CHECKPOINT = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
SHARDS_SHA = "ea5d44e7207f28dc1525d17769d1f429d53acfd37f58b3ee1634ddcccf6b95f3"
SHARDS = {f"model-{i:05}-of-00131.safetensors" for i in range(1, 132)}
HEX = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
JOB_KINDS = {"convert", "verify", "transfer", "generate", "evaluate", "baseline"}
MAX_JOB = 64 << 20
MAX_PREDICTIONS = 90 << 20
ASSETS = frozenset("config.json tokenizer.json tokenizer_config.json chat_template.jinja "
                   "merges.txt generation_config.json vocab.json special_tokens_map.json "
                   "added_tokens.json preprocessor_config.json processor_config.json "
                   "video_preprocessor_config.json".split())
INDEX = "model.safetensors.index.json"
ROLES = {"checkpoint": "model.hgn", "lookup": "qwen38-flash-next-ngram.hgn",
         "vision": "qwen38-flash-next-vision.hgn"}
DATASET = {"name": "openai/grade-school-math", "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
           "split": "test", "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
           "rows": 1319, "full_split": True}
PROTOCOL = "gsm8k-zero-shot-raw-v1"
REQUIRED_ASSETS = {"config.json", "tokenizer.json", "tokenizer_config.json"}
NATIVE = "native/"
CUDA_CAPABILITY = {"sm_89": [8, 9], "sm_120": [12, 0]}
CUDA_SCHEMA = "brewery-cupti-encoder-v1"
CUDA_CODECS = {"Ht3": ("ht3", "ht_backend"), "Ht": ("ht4", "ht_backend"),
               "I4r": ("i4r", "i4r_backend"), "Q4Cp": ("q4cp", "q4cp_backend")}


class Invalid(ValueError):
    pass


class Missing(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise Invalid(message)


def hash_ok(value):
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def integer(value, low=0, high=2**64 - 1):
    return type(value) is int and low <= value <= high


def text(value):
    return isinstance(value, str) and bool(value.strip())


def digest(data):
    return sha256(data).hexdigest()


def pairs(items):
    out = {}
    for key, value in items:
        need(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def bad_constant(name):
    raise Invalid(f"not a JSON number: {name}")


def decode(data):
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def read_file(path, limit):
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_size <= limit,
         f"not a regular file or above {limit} bytes: {path}")
    with path.open("rb") as f:
        data = f.read(limit + 1)
    need(len(data) <= limit, f"above {limit} bytes: {path}")
    return data


def object_(value):
    need(isinstance(value, dict), "expected JSON object")
    return value


def array(value):
    need(isinstance(value, list), "expected JSON array")
    return value


def unique(seq, key):
    out = {}
    for item in array(seq):
        k = object_(item)[key]
        need(text(k) and k not in out, f"invalid or duplicate {key}: {k!r}")
        out[k] = item
    return out


def job_id(value):
    return isinstance(value, str) and ID.fullmatch(value) is not None


def location(value):
    need(text(value) and value.startswith("/") and ".." not in PurePosixPath(value).parts,
         "invalid worker path")
    return PurePosixPath(value)


def tree(entry, transfer=False):
    files = unique(entry["files"], "path")
    parts = []
    for name, member in sorted(files.items()):
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and str(rel) == name and name != "." and ".." not in rel.parts
             and not any(ord(c) < 32 for c in name)
             and integer(member["bytes"], 1) and hash_ok(member["sha256"]), "invalid file member")
        parts.append(f"{name}\0{member['bytes']}\0{member['sha256']}\n")
    fingerprint = digest("".join(parts).encode())
    need(bool(files) and entry["bundle_sha256"] == fingerprint
         and (transfer or entry["sha256"] == fingerprint)
         and entry["bytes"] == sum(m["bytes"] for m in files.values()), "invalid bundle tree digest or size")
    return files


def copy_from(root, job, basename, limit, meta):
    need(PurePosixPath(basename).name == basename and basename not in (".", ".."),
         "invalid artifact basename")
    need(integer(meta["bytes"], 1, limit) and hash_ok(meta["sha256"]), "invalid artifact identity")
    path = root / job["id"] / basename
    try:
        data = read_file(path, limit)
    except FileNotFoundError:
        raise Missing(f"copy {basename} to {path}") from None
    need(len(data) == meta["bytes"] and digest(data) == meta["sha256"],
         f"artifact bytes/digest mismatch: {basename}")
    return data


def json_copy(root, job, entry, kind):
    need(entry["kind"] == kind, f"expected {kind} artifact")
    return object_(decode(copy_from(root, job, location(entry["path"]).name, MAX_PREDICTIONS, entry)))


def identity_shards(path):
    try:
        data = read_file(path, 32768)
    except FileNotFoundError:
        raise Missing(f"copy the pinned raw shard manifest to {path} or use --shard-manifest") from None
    need(digest(data) == SHARDS_SHA, "source shard manifest is not the pinned manifest")
    files = {}
    for line in data.decode("ascii").splitlines():
        fingerprint, file = line.split("  ")
        need(file not in files and hash_ok(fingerprint), "invalid pinned shard manifest")
        files[file] = fingerprint
    need(files.keys() == SHARDS, "pinned manifest does not enumerate all 131 shard files")
    return files


def complete(job):
    need(job["state"] == "succeeded" and job["has_result"] is True
         and job["cancel_requested"] is False and "error" not in job, "job is not a successful terminal run")
    events = array(job["events"])
    need(events and events[0]["state"] == "queued" and events[-1]["state"] == "succeeded"
         and any(e["state"] == "running" for e in events), "missing controller job lifecycle")
    result = object_(job["result"])
    # Worker transfer replies contain resolved inputs and transfer statistics,
    # not a kind discriminator; the controller distinguishes transfer/preview.
    if job["kind"] != "transfer":
        need(result["kind"] == job["kind"], "worker and controller job kind differ")
    return result


def source_input(job, result, pin, count):
    inputs = array(result["inputs"])
    need(len(inputs) == count and len(job["input_ids"]) == count, "expected HF source without auxiliary inputs")
    for idx, item in enumerate(inputs):
        need(item["id"] == job["input_ids"][idx] and item["host"] == job["host"] == result["host"],
             "resolved input does not match controller input or worker host")
        location(item["path"])
    hf = inputs[0]
    need(hf["kind"] == "hf", "source is a derivative rather than an indexed HF snapshot")
    members = tree(hf)
    need({f for f in members if f.endswith(".safetensors")} == SHARDS,
         "source file tree does not cover exactly the 131 pinned shards")
    need(all(members[f]["sha256"] == pin[f] for f in SHARDS), "source raw shard SHA-256 mismatch")
    need({INDEX, *REQUIRED_ASSETS} <= members.keys(), "source lacks index or required tokenizer assets")
    source = digest(f"brewery-canonical-source-v1\nsource\0hf\0{hf['bundle_sha256']}\n".encode())
    need(result["source_sha256"] == source, "role-qualified HF source digest mismatch")
    return source, members


def process(job, result):
    receipt = result["backend_receipt"]
    p, = array(receipt["processes"])
    need(receipt["tool"] == "bf16-to-hgn" and receipt["requested_options"] == job["options"]
         and integer(p["pid"], 1, 2**32 - 1) and integer(p["exit_code"], 0, 0)
         and hash_ok(p["executable_sha256"]) and hash_ok(p["stderr_sha256"]), "invalid converter process receipt")
    location(p["executable"])
    location(p["stderr_log"])
    return p


def upstream_source(job):
    # Called after conversion() checks the shard digests and the native manifest
    # against the input tree. Ignore non-input repository files such as README.
    files = unique(job["result"]["inputs"][0]["files"], "path")
    for name, identity in METADATA.items():
        need(name in files and file_meta(files[name]) == identity, f"upstream metadata mismatch: {name}")
    need(not ((ASSETS & files.keys()) - METADATA.keys()), "unrecognized converter asset in pinned source")
    need(sum(files[n]["bytes"] for n in SHARDS) == 360_000_192_888, "upstream shard size mismatch")
    return section("record_validated", revision=REVISION, shards=len(SHARDS), metadata_files=len(METADATA))


def output_ids(profile):
    return {"checkpoint": f"qwen3.8-flash-next-{profile}", "lookup": "qwen3.8-flash-next-ngram",
            "vision": "qwen3.8-flash-next-vision"}


def file_meta(item):
    return item["bytes"], item["sha256"]


def conversion(job, row, pin, root):
    target, profile = row
    result = complete(job)
    source, inputs = source_input(job, result, pin, 1)
    p = process(job, result)
    options, report = job["options"], result["report"]
    backend = "hip" if target == "gfx1151" else "cuda"
    need(options.get("profile", "v2") == profile and report["profile"] == profile,
         "requested row and converter profile differ")
    need(report["published"] is True and report["manifest"] == NATIVE + "conversion-manifest.json"
         and report["source_format"] == "hf", "missing full native conversion publication record")
    hgn = result["artifact"]
    need(hgn["kind"] == "hgn", "wrong conversion artifact kind")
    need(location(report["native_directory"]) == location(hgn["path"]) / "native",
         "incorrect published native directory")
    outputs = tree(hgn)
    raw = copy_from(root, job, "conversion-manifest.json", MAX_JOB, outputs[NATIVE + "conversion-manifest.json"])
    m = object_(decode(raw))
    opts = m["options"]
    need(opts["profile"] == profile and all(opts[k] == v for k, v in options.items()),
         "job and manifest conversion options differ")
    # The workbench names backend families, not device architecture. These are
    # encoder selections only; a separately correlated profiler is still needed.
    need(all(opts[k] == report[k] for k in ("ht_backend", "i4r_backend", "q4cp_backend"))
         and all(opts[k] in (backend, "cpu") for k in ("ht_backend", "i4r_backend", "q4cp_backend"))
         and opts["ht_backend"] == backend, "manifest lacks the expected GPU encoder configuration")
    need(integer(opts["seed"]) and integer(opts["threads"], 1, 64) and integer(opts["ht_beam"], 1, 256),
         "invalid encoding settings")
    need(m["schema"] == 2 and m["source_kind"] == "hf" and m["tool"] == "bf16-to-hgn"
         and hash_ok(m["encoder_sha256"]) and m["encoder_sha256"] == report["encoder_source_sha256"]
         and m["policy"] == report["policy"]
         and m["policy"] == f"bf16-hgn-{profile}-rtn-nf4-signed-rms-trellis-v1"
         and text(m["spec_version"]) and text(m["spec_commit"]), "invalid native schema-2 manifest")
    dep = m["source_provenance"]
    deps = unique(dep["dependencies"], "file")
    assets = ASSETS & inputs.keys()
    need(dep["format"] == "huggingface-safetensors" and deps.keys() == SHARDS | {INDEX} | assets,
         "native source provenance does not cover all shards, index and tracked assets")
    for name, d in deps.items():
        need(d["size"] == inputs[name]["bytes"], f"source size mismatch in manifest: {name}")
        if name in SHARDS:
            need(integer(d["header_bytes"], 9, d["size"] - 1) and hash_ok(d["header_sha256"]),
                 f"missing native shard header identity: {name}")
        else:
            need(d["sha256"] == inputs[name]["sha256"], f"index/asset digest mismatch: {name}")
    need(m["config_sha256"] == inputs["config.json"]["sha256"] and m["index_sha256"] == inputs[INDEX]["sha256"],
         "manifest config/index identity mismatch")
    index = object_(decode(copy_from(root, job, INDEX, MAX_JOB, inputs[INDEX])))
    weight_map = object_(index["weight_map"])
    tensors = {}
    ranges = {shard: [] for shard in SHARDS}
    widths = {"Bf16": 2, "F32": 4, "I64": 8}
    for s in array(m["sources"]):
        t = s["tensor"]
        name, shard = t["name"], t["shard"]
        shape = array(t["shape"])
        need(text(name) and name not in tensors and shard in SHARDS and weight_map.get(name) == shard
             and integer(t["offset"], deps[shard]["header_bytes"])
             and integer(t["size"], 1) and t["offset"] + t["size"] <= inputs[shard]["bytes"]
             and t["dtype"] in widths and 1 <= len(shape) <= 5
             and all(integer(n, 1) for n in shape) and t["size"] == math.prod(shape) * widths[t["dtype"]]
             and hash_ok(s["sha256"]),
             "malformed or duplicate canonical source tensor")
        tensors[name] = s
        ranges[shard].append((t["offset"], t["size"]))
    need({s["tensor"]["shard"] for s in tensors.values()} == SHARDS
         and tensors.keys() == weight_map.keys()
         and len(tensors) == report["source_tensors"], "incomplete canonical tensor coverage")
    for shard, covered in ranges.items():
        end = deps[shard]["header_bytes"]
        for start, nbytes in sorted(covered):
            need(start == end, f"overlapping or missing tensor bytes in {shard}")
            end += nbytes
        need(end == inputs[shard]["bytes"], f"incomplete physical tensor coverage in {shard}")
    plans = unique(m["plan"]["files"], "filename")
    containers = unique(m["containers"], "filename")
    need(m["plan"]["architecture"] == "qwen4_exp" and plans.keys() == containers.keys() == set(ROLES.values())
         and set(m["plan"]["assets"]) == assets, "invalid full checkpoint/lookup/vision plan")
    output_count = 0
    cuda_codecs = set()
    ids = output_ids(profile)
    for role, name in ROLES.items():
        c, plan = containers[name], plans[name]
        output_records = array(c["tensors"])
        planned = unique(plan["tensors"], "name")
        written = unique([record["entry"] for record in output_records], "name")
        need(plan["identity"] == c["layout"]["identity"] == ids[role]
             and c["layout"]["entries"] == [record["entry"] for record in output_records]
             and planned and planned.keys() == written.keys()
             and outputs[NATIVE + name]["bytes"] == c["layout"]["file_size"]
             and all(hash_ok(x["fingerprint"]["sha256"]) for x in output_records),
             f"container records do not match plan/bundle: {name}")
        if target in CUDA_CAPABILITY:
            for tensor, expected_tensor in planned.items():
                encoding = expected_tensor["encoding"]
                need(encoding == written[tensor]["encoding"], f"tensor encoding mismatch: {name}/{tensor}")
                if encoding in CUDA_CODECS:
                    codec, flag = CUDA_CODECS[encoding]
                    if opts[flag] == "cuda":
                        cuda_codecs.add(codec)
        output_count += len(written)
    need(output_count == report["output_tensors"], "output tensor count does not match publication")
    exported_assets = unique(m["assets"], "destination")
    expected = {f"tokenizer/{n}": n for n in assets if not n.endswith("processor_config.json")}
    expected.update({n: n for n in assets if n == "config.json" or n.endswith("processor_config.json")})
    need(exported_assets.keys() == expected.keys(), "missing or unexpected output assets")
    for dest, name in expected.items():
        a = exported_assets[dest]
        need(a["source"] == name and (a["size"], a["sha256"]) == file_meta(inputs[name]) == file_meta(outputs[NATIVE + dest]),
             f"output asset mismatch: {dest}")
    need(outputs.keys() == {"manifest.json"} | {NATIVE + f for f in (*expected, *ROLES.values(), "conversion-manifest.json")},
         "missing or unexpected bundle constituents")
    fmt = "hgn-v2" if profile == "v2" else "ht43"
    inference = result["inference_bundle"]
    need(inference["status"] == "manifest_created" and inference["format"] == fmt
         and location(inference["manifest"]) == location(hgn["path"]) / "manifest.json",
         "missing generated inference manifest")
    generation_manifest = object_(decode(copy_from(root, job, "manifest.json", 64 << 10, outputs["manifest.json"])))
    tokenizer = {NATIVE + dest: outputs[NATIVE + dest]["sha256"] for dest in expected if dest.startswith("tokenizer/")}
    need(generation_manifest == {"format": fmt, "source_sha256": source,
                                 "files": [{"role": role, "path": NATIVE + name, "sha256": outputs[NATIVE + name]["sha256"]}
                                           for role, name in ROLES.items()],
                                 "tokenizer": [{"path": name, "sha256": hash_}
                                               for name, hash_ in sorted(tokenizer.items())]},
         "generated runtime manifest does not match all conversion members")
    # Ignore conversion-manifest.json in runtime reuse: it records encoder
    # provenance, not bytes loaded by the engine. Include every other member.
    runtime_tree = {name: file_meta(member) for name, member in outputs.items() if name != NATIVE + "conversion-manifest.json"}
    return {"source": source, "outputs": outputs, "process": p, "report": report, "cuda_codecs": cuda_codecs,
            "artifact": hgn, "format": fmt, "ids": ids, "tokenizer": tokenizer,
            "runtime_key": digest(json.dumps(runtime_tree, sort_keys=True).encode()),
            "common": (source, digest(json.dumps(tensors, sort_keys=True).encode()),
                       opts["seed"], opts["ht_beam"], m["encoder_sha256"], m["spec_version"], m["spec_commit"])}


def cuda_evidence_bytes(root, job, worker_path, expected_sha, limit):
    need(hash_ok(expected_sha), "invalid CUDA evidence SHA-256")
    name = worker_path.name
    need(name not in ("", ".", ".."), "invalid CUDA evidence filename")
    path = root / job["id"] / name
    try:
        data = read_file(path, limit)
    except FileNotFoundError:
        raise Missing(f"copy {name} to {path}") from None
    need(data and digest(data) == expected_sha, f"CUDA evidence SHA-256 mismatch: {name}")
    return data


def cuda_activity(job, target, bundle, root):
    if "encoder_activity" not in job["result"]:
        return section("external_review", "converter process completed; no encoder activity receipt exported")
    receipt = object_(job["result"]["encoder_activity"])
    status = receipt["status"]
    need(target in CUDA_CAPABILITY, "CUDA activity cannot establish this non-CUDA target")
    if status in ("unavailable", "not_observed"):
        need(text(receipt["reason"]), "missing CUDA capture failure reason")
        return section(status, receipt["reason"])
    p = bundle["process"]
    need(status == "observed" and receipt["schema"] == CUDA_SCHEMA and receipt["backend"] == "cuda"
         and integer(receipt["pid"], 1, 2**32 - 1) and receipt["pid"] == p["pid"]
         and receipt["executable_sha256"] == p["executable_sha256"]
         and receipt["output_bundle_sha256"] == bundle["artifact"]["bundle_sha256"],
         "CUDA receipt does not identify the successful converter and published output")
    cap = array(receipt["compute_capability"])
    need(len(cap) == 2 and integer(cap[0], 1) and integer(cap[1], 0, 9)
         and cap == CUDA_CAPABILITY[target], f"CUDA compute capability does not match {target}")
    uuid = receipt["gpu_uuid"]
    need(isinstance(uuid, str) and uuid.startswith("GPU-") and job_id(uuid[4:]), "invalid physical GPU UUID")
    capture = location(receipt["capture"])
    collector = location(receipt["collector"])
    executable = location(p["executable"])
    need(capture == location(p["stderr_log"]).with_name("encoder-activity.json")
         and collector.suffix == ".so" and ":" not in str(collector)
         and len({capture.name, collector.name, executable.name}) == 3,
         "CUDA capture, collector or executable evidence path is inconsistent")
    raw = object_(decode(cuda_evidence_bytes(root, job, capture, receipt["capture_sha256"], 1024)))
    need(raw.keys() == {"schema", "pid", "exe_dev", "exe_ino", "gpu_uuid", "compute_major", "compute_minor", "counts"}
         and raw["schema"] == CUDA_SCHEMA and integer(raw["pid"], 1, 2**32 - 1)
         and raw["pid"] == receipt["pid"] and integer(raw["exe_dev"]) and integer(raw["exe_ino"], 1)
         and raw["gpu_uuid"] == uuid and integer(raw["compute_major"], 1)
         and integer(raw["compute_minor"], 0, 9) and [raw["compute_major"], raw["compute_minor"]] == cap,
         "raw CUPTI capture disagrees with converter PID, GPU UUID, or capability")
    counts, reported_counts = object_(raw["counts"]), object_(receipt["completed_encoder_activity"])
    need(counts.keys() == {codec for codec, _ in CUDA_CODECS.values()}
         and all(integer(n) for n in counts.values())
         and counts == reported_counts and all(integer(n) for n in reported_counts.values()),
         "raw CUPTI counts disagree with the CUDA receipt")
    # Counts are for completed encoder kernels, not decoded tensors, and are
    # not a claim about every tensor having run on the GPU.
    expected = bundle["cuda_codecs"]
    report = bundle["report"]
    profile_codecs = {"ht4"}
    if report["profile"] == "ht43":
        profile_codecs.add("ht3")
    if report["profile"] == "v2" and report["i4r_backend"] == "cuda":
        profile_codecs.add("i4r")
    if report["q4cp_backend"] == "cuda":
        profile_codecs.add("q4cp")
    observed = {codec for codec, count in counts.items() if count > 0}
    need(expected == profile_codecs, "CUDA codec plan does not match the full conversion profile")
    need(observed == expected,
         f"unexpected CUDA codec coverage: wanted {', '.join(sorted(expected))}; observed {', '.join(sorted(observed))}")
    cuda_evidence_bytes(root, job, collector, receipt["collector_sha256"], MAX_JOB)
    cuda_evidence_bytes(root, job, executable, p["executable_sha256"], MAX_JOB)
    return section("record_validated", "CUPTI kernel activity correlated; independent host review still required",
                   schema=CUDA_SCHEMA, gpu_uuid=uuid, compute_capability=cap,
                   completed_encoder_activity=counts, capture_sha256=receipt["capture_sha256"],
                   collector_sha256=receipt["collector_sha256"],
                   output_bundle_sha256=bundle["artifact"]["bundle_sha256"])


def verify(job, pin, root):
    result = complete(job)
    source, _ = source_input(job, result, pin, 2)
    process(job, result)
    checked = result["inputs"][1]
    need(checked["kind"] in ("hgn", "hgn-v2", "ht43"), "verify input is not HGN")
    files = tree(checked)
    report = result["report"]
    need(result["file_verification"]["status"] == "passed" and report["manifest_schema"] == 2
         and report["provenance_attested"] is True and integer(report["payload_bytes"], 1)
         and integer(report["source_tensors"], 1) and integer(report["output_tensors"], 1)
         and isinstance(report["errors"], dict), "no schema-2 native source/output verification")
    artifact_report = json_copy(root, job, result["artifact"], "verify")
    need(all(r.get("passed", True) is True for r in (report, artifact_report)),
         "native verification report has conflicting or invalid passed value")
    need(artifact_report == report, "verification artifact and job report differ")
    return source, checked["bundle_sha256"], files, report


def transfer(job):
    r = complete(job)
    inputs = array(r["inputs"])
    need(len(inputs) == len(job["input_ids"]) and isinstance(r["bytes"], int),
         "transfer input inventory differs from the controller")
    entries = []
    for i, entry in zip(job["input_ids"], inputs):
        need(entry["host"] == job["host"] and entry["id"] == i,
             "transfer input does not match controller host or registration")
        if entry["kind"] not in ("hgn", "hgn-v2", "ht43", "brewery-bundle"):
            continue
        path = location(entry["path"])
        need(entry["cached"] is True and text(entry["source_host"]),
             "transfer did not resolve to a ready bundle")
        files = tree(entry, transfer=True)
        entries.append((str(path), entry["bundle_sha256"], files, entry["source_host"]))
    return entries


def runtime(identity, bundle):
    proof = identity["load_proof"]
    qualifier = proof["qualification"]
    files = unique(qualifier["files"], "role")
    obs = proof["observation"]
    mapped = unique(obs["mapped"], "path")
    image_id = proof["image_id"]
    need(identity["source_sha256"] == bundle["source"]
         and identity["manifest_sha256"] == bundle["outputs"]["manifest.json"]["sha256"]
         and {k: v["sha256"] for k, v in unique(identity["tokenizer"], "path").items()} == bundle["tokenizer"],
         "generation provenance does not match the exact bundle manifest and assets")
    need(proof["schema"] == "halogen-linux-engine-load-v1" and isinstance(image_id, str)
         and image_id.startswith("sha256:") and hash_ok(image_id[7:]) and hash_ok(proof["container_id"])
         and qualifier["schema"] == "halogen-hgn-preflight-v1"
         and qualifier["format"] == bundle["format"] and files.keys() == ROLES.keys()
         and all(integer(obs[k], 1) for k in ("engine_pid", "start_ticks", "listener_inode")),
         "invalid isolated runtime load evidence")
    need(mapped.keys() == {f"/brewery-{role}.hgn" for role in ROLES}, "incomplete mapped sidecar coverage")
    roots = set()
    for role, name in ROLES.items():
        f, member = files[role], bundle["outputs"][NATIVE + name]
        ident = f["identity"]
        path = location(f["path"])
        need(path.name == name and path.parent.name == "native" and f["sha256"] == member["sha256"]
             and f["hgn_model_id"] == bundle["ids"][role]
             and ident["bytes"] == member["bytes"] and integer(ident["inode"], 1)
             and integer(ident["dev_major"]) and integer(ident["dev_minor"]),
             f"runtime loaded unexpected {role} content")
        roots.add(str(path.parent.parent))
        key = f"/brewery-{role}.hgn"
        need(mapped[key] == {"path": key, "major": ident["dev_major"], "minor": ident["dev_minor"],
                             "inode": ident["inode"], "bytes": ident["bytes"]},
             f"runtime mapping does not match qualified {role} file")
    need(len(roots) == 1, "runtime sidecars were not loaded from one publication")
    return image_id, roots.pop()


def generation(job, bundle):
    r = complete(job)
    proto = r["protocol"]
    need(len(job["input_ids"]) == 1 and text(r["text"]) and len(r["text"].encode()) <= 65536
         and proto["id"] == "raw-greedy-v1" and integer(proto["temperature"], 0, 0)
         and integer(proto["max_tokens"], 1, 512)
         and proto["max_tokens"] == job["options"].get("max_tokens", 128)
         and hash_ok(proto["prompt_sha256"]), "empty generation or invalid generation protocol")
    return runtime(r["identity"], bundle)


# Match the evaluator's final-line-only decimal parser, without floating point.
NUMBER = re.compile(r"[+-]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]*)?|\.[0-9]+)\Z")


def answer(s):
    line = s.rstrip().rsplit("\n", 1)[-1].strip()
    raw = line[4:].strip()
    if not line.startswith("####") or not NUMBER.fullmatch(raw):
        return None
    negative = raw.startswith("-")
    whole, _, frac = raw.lstrip("+-").replace(",", "").partition(".")
    whole, frac = whole.lstrip("0"), frac.rstrip("0")
    return ("-" if negative and (whole or frac) else "") + (whole or "0") + (f".{frac}" if frac else "")


def questions():
    p = Path(__file__).resolve().parent.parent / "crates/eval/data/gsm8k-test.jsonl"
    data = read_file(p, 2 << 20)
    need(digest(data) == DATASET["sha256"], "local dataset does not match pinned GSM8K test split")
    rows = [decode(line) for line in data.splitlines()]
    out = [(digest(row["question"].encode()), answer(row["answer"])) for row in rows]
    need(len(out) == DATASET["rows"] and all(a is not None for _, a in out), "invalid pinned GSM8K rows")
    return out


def score(job, root, result, artifact_kind, qs):
    need(result["dataset"] == DATASET and result["protocol_id"] == PROTOCOL,
         "not the pinned full GSM8K test split and protocol")
    r = result["identity"]
    run = json_copy(root, job, result["artifact"], artifact_kind)
    need(r["synthetic"] is False and run["synthetic"] is False
         and run["schema"] == "brewery-gsm8k-predictions-v1" and run["protocol_id"] == PROTOCOL
         and run["dataset_sha256"] == DATASET["sha256"] and hash_ok(r["source_sha256"])
         and run["source_sha256"] == r["source_sha256"]
         and object_(run["provenance"]) == object_(r["provenance"]),
         "prediction artifact does not match a genuine evaluator result")
    predictions = array(run["predictions"])
    need(len(predictions) == len(qs), "need all 1319 predictions")
    parsed = []
    for i, (p, (q, _)) in enumerate(zip(predictions, qs), 1):
        need(integer(p["id"], i, i) and p["question_sha256"] == q
             and isinstance(p["text"], str) and len(p["text"].encode()) <= 65536,
             f"invalid pinned prediction row {i}")
        parsed.append(answer(p["text"]))
    total = len(qs)
    correct = sum(a == ref for a, (_, ref) in zip(parsed, qs))
    computed = {"correct": correct, "total": total, "unparsed": parsed.count(None),
                "accuracy_basis_points": (correct * 10_000 + total // 2) // total}
    need(result["score"] == computed and all(type(v) is int for v in result["score"].values()),
         "reported score does not match exported predictions")
    return computed, parsed


def baseline(job, root, qs):
    r = complete(job)
    stats, parsed = score(job, root, r, "gsm8k-baseline", qs)
    prov = r["identity"]["provenance"]
    method = prov["method"]
    need((method == "imported-predictions" and hash_ok(prov["import_sha256"])
          and len(job["input_ids"]) == 1 and "baseline_source" not in job["options"])
         or (method == "operator-bf16-endpoint" and prov["source_evidence"] == "matched-response-headers"
             and all(text(prov[k]) for k in ("runtime_id", "model", "baseline_source"))
             and job["options"].get("baseline_source") == prov["baseline_source"]
             and len(job["input_ids"]) == 0), "unsupported or unlinked baseline provenance")
    return r, stats, parsed


def evaluation(job, bundle, base, root, qs):
    r = complete(job)
    stats, parsed = score(job, root, r, "gsm8k-evaluation", qs)
    need(r["identity"]["source_sha256"] == bundle["source"], "evaluation source and bundle differ")
    runtime_id = runtime(r["identity"]["provenance"], bundle)
    if base is None:
        raise Missing("selected BF16 baseline evidence is unavailable")
    b, baseline_stats, baseline_parsed = base
    need(b["identity"]["source_sha256"] == bundle["source"]
         and r["baseline_identity"]["source_sha256"] == bundle["source"]
         and r["baseline_identity"]["synthetic"] is False and r["baseline_score"] == baseline_stats,
         "evaluation does not reference the selected baseline source and score")
    inline = r["baseline_artifact"]
    if inline is not None:
        need(len(job["input_ids"]) == 1 and
             job["options"].get("baseline_source") == b["identity"]["provenance"].get("baseline_source")
             and b["identity"]["provenance"]["method"] == "operator-bf16-endpoint"
             and r["baseline_identity"] == b["identity"] and file_meta(inline) == file_meta(b["artifact"]),
             "inline baseline is not the exact selected endpoint artifact")
        json_copy(root, job, inline, "gsm8k-baseline")
    else:
        p = r["baseline_identity"]["provenance"]
        selected = b["identity"]["provenance"]
        allowed = {b["artifact"]["sha256"]}
        if selected["method"] == "imported-predictions":
            allowed.add(selected["import_sha256"])
        need(len(job["input_ids"]) == 2 and "baseline_source" not in job["options"]
             and p["method"] == "imported-predictions" and p["import_sha256"] in allowed,
             "evaluation imported a different baseline artifact")
    delta = stats["correct"] - baseline_stats["correct"]
    total = DATASET["rows"]
    changed_ids = [i for i, (a, b) in enumerate(zip(parsed, baseline_parsed), 1) if a != b]
    needed = {"changed_answers": len(changed_ids), "changed_answer_ids": changed_ids,
              "delta_correct": delta, "within_two_percentage_points": delta * 100 >= -2 * total}
    need(r["comparison"] == needed and type(r["comparison"]["within_two_percentage_points"]) is bool
         and type(r["comparison"]["delta_correct"]) is int
         and type(r["comparison"]["changed_answers"]) is int
         and all(type(i) is int for i in r["comparison"]["changed_answer_ids"]),
         "comparison mismatch versus exact baseline predictions")
    return runtime_id, stats["correct"], delta


def section(status, reason=None, **data):
    if reason is not None:
        data["reason"] = reason
    return {"status": status, **data}


def pending():
    return {
        "schema": "brewery-release-check-v1",
        "checkpoint": {"repo": CHECKPOINT, "revision": REVISION,
                       "shard_count": len(SHARDS), "shard_manifest_sha256": SHARDS_SHA},
        "release_qualified": False,
        "paths": [{"target": target, "profile": profile,
                   "conversion": section("missing", "no selected convert job"),
                   "upstream_source": section("missing", "no structurally validated conversion"),
                   "verification": section("missing", "no matching full conversion"),
                   "native_execution": section("missing", "no matching full conversion")}
                  for target, profile in ROWS],
        "bundles": [],
        "baseline": section("missing", "no selected BF16 baseline job"),
        "external_requirements": [
            "Review trusted device inventory and native encoder kernel activity correlated to each "
            "full conversion PID, executable, interval, device ID, and output. "
            "A backend flag, diagnostic log, or trace of a subset is not proof.",
            "Verify that the exact baseline artifact came from this full checkpoint in BF16, "
            "with the pinned protocol; endpoint response headers alone are self-reported.",
            "The upstream_source field checks recorded input identities; it does not authenticate the worker.",
            "Native runtime parity is an independent gate; job exports do not establish it.",
        ],
        "errors": [],
    }


INPUT_ERRORS = (Invalid, KeyError, TypeError, ValueError, OSError, RecursionError, OverflowError)


def attempt(fn, *args):
    try:
        return fn(*args), None
    except Missing as e:
        return None, section("missing", str(e))
    except INPUT_ERRORS as e:
        return None, section("invalid", str(e))


def exported_job(path):
    job = object_(decode(read_file(path, MAX_JOB)))
    need(job_id(job["id"]) and job["kind"] in JOB_KINDS and text(job["host"])
         and isinstance(job["options"], dict) and isinstance(job["input_ids"], list)
         and all(job_id(i) for i in job["input_ids"])
         and integer(job["created_at"], 1) and integer(job["updated_at"], job["created_at"])
         and job["state"] in ("queued", "running", "succeeded", "failed", "cancelled"),
         "not a workbench job detail")
    array(job["events"])
    return job


def terminal_reason(job):
    if job["state"] == "succeeded":
        return None
    return section(job["state"], job.get("error") or "selected job did not succeed")


def collect(paths, row_ids, baseline_id, root, shard_manifest):
    out = pending()
    jobs = {}
    duplicate_ids = set()
    for path in paths:
        job, err = attempt(exported_job, path)
        if err:
            out["errors"].append({"export": str(path), **err})
        elif job["id"] in jobs:
            duplicate_ids.add(job["id"])
        else:
            jobs[job["id"]] = job
    for ident in duplicate_ids:
        del jobs[ident]
        out["errors"].append({"job_id": ident, **section("invalid", "duplicate job export")})

    groups = {kind: [j for j in jobs.values() if j["kind"] == kind] for kind in JOB_KINDS}
    for j in groups["convert"]:
        if j["id"] not in row_ids.values():
            out["errors"].append({"job_id": j["id"], **section("invalid", "convert job needs explicit --row assignment")})
    pin, pin_error = attempt(identity_shards, shard_manifest)
    qs, qs_error = attempt(questions)
    base = None
    if baseline_id:
        if baseline_id not in jobs:
            out["baseline"] = section("missing", "selected baseline export is absent", job_id=baseline_id)
        elif jobs[baseline_id]["kind"] != "baseline":
            out["baseline"] = section("invalid", "selected ID is not a baseline job", job_id=baseline_id)
        else:
            unavailable = terminal_reason(jobs[baseline_id]) or qs_error
            base, err = (None, unavailable) if unavailable else attempt(baseline, jobs[baseline_id], root, qs)
            if err:
                out["baseline"] = {**err, "job_id": baseline_id}
            else:
                out["baseline"] = section(
                    "external_review", "full split and score checked; confirm exact BF16 provenance independently",
                    job_id=baseline_id, source_sha256=base[0]["identity"]["source_sha256"],
                    artifact_sha256=base[0]["artifact"]["sha256"], score=base[1],
                    provenance=base[0]["identity"]["provenance"],
                )

    validated = []
    for row in out["paths"]:
        key = row["target"], row["profile"]
        ident = row_ids.get(key)
        if ident is None:
            continue
        job = jobs.get(ident)
        if job is None:
            err = section("missing", "assigned convert export is absent")
        elif job["kind"] != "convert":
            err = section("invalid", "assigned ID is not a convert job")
        else:
            err = terminal_reason(job) or pin_error
            if not err:
                bundle, err = attempt(conversion, job, key, pin, root)
        if err:
            row["conversion"] = {**err, "job_id": ident}
            continue
        validated.append((row, job, bundle))
        identity, error = attempt(upstream_source, job)
        row["upstream_source"] = error or identity
        row["conversion"] = section("record_validated", job_id=ident, source_sha256=bundle["source"],
                                    bundle_sha256=bundle["artifact"]["bundle_sha256"])
        p = bundle["process"]
        activity, activity_error = attempt(cuda_activity, job, key[0], bundle, root)
        row["native_execution"] = {
            **(activity_error or activity),
            "job_id": ident, "host": job["host"], "pid": p["pid"],
            "executable_sha256": p["executable_sha256"],
            "stderr_sha256": p["stderr_sha256"], "stderr_log": p["stderr_log"],
            "window_ms": [job["created_at"], job["updated_at"]], "output_path": bundle["artifact"]["path"],
        }

    sources = {b["common"] for _, _, b in validated}
    if len(sources) > 1:
        for row, _, _ in validated:
            row["conversion"].update(status="invalid", reason="full source, conversion recipe or encoder source differs across rows")
    if base is not None and validated and any(b["source"] != base[0]["identity"]["source_sha256"] for _, _, b in validated):
        out["baseline"]["status"] = "invalid"
        out["baseline"]["reason"] = "selected baseline does not match every validated conversion source"
        base = None

    # Hashing exact native bytes is the engine's job. Here we require its
    # independently run verify result to bind the whole output artifact.
    verifies = []
    if pin is not None and validated:
        for j in groups["verify"]:
            v, err = attempt(verify, j, pin, root)
            if err:
                out["errors"].append({"job_id": j["id"], **err})
            else:
                verifies.append((j, v))
    bundles = {}
    processes = set()
    destinations = set()
    for row, j, b in validated:
        output = b["artifact"]
        row["verification"] = section("missing", "no successful native verification for exact output bundle")
        for vjob, (vsource, vtree, vfiles, stats) in verifies:
            if (vsource, vtree, vfiles) == (b["source"], output["bundle_sha256"], b["outputs"]):
                if all(stats[k] == b["report"][k] for k in ("source_tensors", "output_tensors")):
                    row["verification"] = section("record_validated", job_id=vjob["id"])
                    break
        process_key = (j["host"], b["process"]["pid"], j["created_at"])
        dest_key = (j["host"], output["path"])
        if process_key in processes or dest_key in destinations:
            row["conversion"].update(status="invalid", reason="another row reused the converter process or publication")
        processes.add(process_key)
        destinations.add(dest_key)
        if row["conversion"]["status"] != "record_validated":
            continue
        key = b["runtime_key"]
        row["runtime_bundle_sha256"] = key
        group = bundles.setdefault(key, (b, [], set(), []))
        group[1].append(f"{row['target']}/{row['profile']}")
        group[2].add(dest_key)
        group[3].append((j["host"], output["bundle_sha256"], b["outputs"]))

    # A single qualified inference worker may receive complete bundles from
    # other workers. Bind the transferred path to the verified full publication.
    for j in groups["transfer"]:
        entries, err = attempt(transfer, j)
        if err:
            out["errors"].append({"job_id": j["id"], **err})
            continue
        for path, tree_sha, files, source_host in entries:
            for _, _, published, originals in bundles.values():
                if (source_host, tree_sha, files) in originals:
                    published.add((j["host"], path))

    def evaluate_identity(job):
        try:
            r = job["result"]
            return r["identity"] if job["kind"] == "generate" else r["identity"]["provenance"]
        except (KeyError, TypeError):
            return {}

    for key, (bundle, rows, published, _) in bundles.items():
        item = {"runtime_bundle_sha256": key, "paths": rows,
                "generation": section("missing", "no generation on the exact bundle"),
                "evaluation": section("missing", "no full GSM8K evaluation on the exact bundle"),
                "quality": section("missing", "no generation/evaluation pair with matched runtime and exact baseline"),
                "evidence_errors": []}
        generations, evaluations = [], []
        for kind, validated_jobs in (("generate", generations), ("evaluate", evaluations)):
            for j in groups[kind]:
                ident = evaluate_identity(j)
                if not isinstance(ident, dict) or ident.get("manifest_sha256") != bundle["outputs"]["manifest.json"]["sha256"]:
                    continue
                if kind == "generate":
                    value, err = attempt(generation, j, bundle)
                else:
                    value, err = (None, qs_error) if qs_error else attempt(evaluation, j, bundle, base, root, qs)
                if not err:
                    _, runtime_root = value if kind == "generate" else value[0]
                    if (j["host"], runtime_root) not in published:
                        err = section("invalid", "runtime did not use a matching validated publication on this host")
                if err:
                    item["evidence_errors"].append({"job_id": j["id"], **err})
                else:
                    validated_jobs.append((j, value))
        if generations:
            item["generation"] = section("record_validated", job_ids=[j["id"] for j, _ in generations])
        if evaluations:
            item["evaluation"] = section("record_validated", job_ids=[j["id"] for j, _ in evaluations])
        matched = [(g, e, image, correct, delta) for g, (gimage, gpath) in generations
                   for e, ((image, epath), correct, delta) in evaluations
                   if (g["host"], gimage, gpath) == (e["host"], image, epath)
                   and g["input_ids"][0] == e["input_ids"][0]
                   and g["result"]["identity"]["load_proof"]["container_id"]
                   != e["result"]["identity"]["provenance"]["load_proof"]["container_id"]]
        if matched:
            # Report conflicting or repeated observations, not just the best run.
            quality = [{
                "generation_job_id": g["id"], "evaluation_job_id": e["id"], "host": g["host"],
                "image_id": image, "correct": correct, "baseline_correct": base[1]["correct"],
                "total": DATASET["rows"], "delta_correct": delta, "meets_margin": delta >= -26,
            } for g, e, image, correct, delta in matched]
            all_meet = all(q["meets_margin"] for q in quality)
            item["quality"] = section("within_margin_pending_baseline_review" if all_meet else "below_margin",
                                      runs=quality)
        elif generations and evaluations:
            item["quality"] = section("invalid", "generation and evaluation used different runtime, input registration or reused a container")
        out["bundles"].append(item)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=("Export selected convert, verify, generate, evaluate and baseline GET /api/jobs/{id} "
                "responses unchanged. Include a successful transfer export when inference uses a remote bundle. "
                "Under ROOT/<job-id>/ copy conversion-manifest.json, manifest.json and the source index for "
                "each conversion, verify.json for each verify job, and prediction JSON for each baseline/evaluation. "
                "For observed CUDA activity, also copy encoder-activity.json, the converter executable, "
                "and the collector library referenced by the export. "
                "Use original basenames. No remote calls. JSON report on stdout; "
                "exit 1 while release or external review remains outstanding."),
    )
    parser.add_argument("--baseline-job-id", help="select one exact baseline export by its controller ID")
    parser.add_argument("--row", action="append", default=[], metavar="TARGET/PROFILE=ID",
                        help="assign one separate convert job to a row; this does not prove its GPU architecture")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"),
                        help="job-keyed root of locally copied evidence files (default: artifacts)")
    parser.add_argument("--shard-manifest", type=Path, default=Path(__file__).with_name("release_check_shards.sha256"),
                        help="raw shard SHA-256 list; must have the pinned digest")
    parser.add_argument("exports", nargs="*", type=Path, metavar="JOB_EXPORT.json")
    args = parser.parse_args(argv)
    rows = {}
    for entry in args.row:
        label, _, ident = entry.partition("=")
        key = tuple(label.split("/"))
        if key not in ROWS or key in rows or not job_id(ident) or ident in rows.values():
            parser.error(f"invalid or duplicate row assignment: {entry}")
        rows[key] = ident
    if args.baseline_job_id is not None and not job_id(args.baseline_job_id):
        parser.error("--baseline-job-id requires a controller UUID")
    report = collect(args.exports, rows, args.baseline_job_id, args.artifacts, args.shard_manifest)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
