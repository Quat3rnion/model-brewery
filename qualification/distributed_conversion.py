#!/usr/bin/python3
"""Resumable stage-2 conversion through the allowlisted Brewery dispatcher."""
import argparse
import concurrent.futures
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid

SPLIT = "conversion-partition"
PACKAGE = "conversion-package"
UNIT = "conversion-unit"
COLLECTION = "conversion-units"
ARTIFACT = "hgn"
FAN_IN = 126
PACK_BATCH = 16
ENCODING = {
    "profile": ("v2", "ht43"),
    "seed": (0, 2**64 - 1),
    "threads": (1, 64),
    "ht-beam": (1, 256),
    "ht-backend": ("cpu", "hip", "cuda"),
    "i4r-backend": ("cpu", "hip", "cuda"),
    "q4cp-backend": ("cpu", "cuda"),
}


class Failure(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise Failure(message)


def token(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", value),
            "invalid worker identifier")
    return value


def path(value):
    require(isinstance(value, str) and value.startswith("/") and
            all(part not in ("", ".", "..") for part in value[1:].split("/")) and
            not any(ord(c) < 32 or ord(c) == 127 for c in value), "expected a safe absolute path")
    return Path(value)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid digest")
    return value


def sha(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def number(n, low=0, high=2**64 - 1):
    require(type(n) is int and low <= n <= high, "integer outside protocol bounds")
    return n


def cpu_threads():
    value = os.environ.get("BREWERY_CPU_THREADS", "4")
    require(re.fullmatch(r"[0-9]{1,2}", value), "invalid BREWERY_CPU_THREADS")
    return min(number(int(value), 1, 64), len(os.sched_getaffinity(0)))


def tree(root):
    require(stat.S_ISDIR(root.lstat().st_mode), f"not a real directory: {root}")
    entries = []
    pending = [root]
    while pending:
        for p in pending.pop().iterdir():
            info = p.lstat()
            if stat.S_ISDIR(info.st_mode):
                pending.append(p)
            else:
                require(stat.S_ISREG(info.st_mode), f"not a regular artifact file: {p}")
                name = p.relative_to(root).as_posix()
                require(not any(ord(c) < 32 or ord(c) == 127 or c == "\\" for c in name),
                        "unsafe artifact name")
                entries.append((name, info.st_size, p))
                require(len(entries) <= 16384, "oversized artifact tree")
    require(0 < len(entries) <= 16384, "empty or oversized artifact tree")
    h = hashlib.sha256()
    entries.sort()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(cpu_threads(), 16, len(entries))) as pool:
        for (name, size, _), checksum in zip(entries, pool.map(sha, (p for _, _, p in entries)), strict=True):
            h.update(f"{name}\0{size}\0{checksum}\n".encode("utf-8"))
    return h.hexdigest(), sum(size for _, size, _ in entries)


def artifact(root, kind):
    checksum, size = tree(root)
    return {"path": str(root), "kind": kind, "sha256": checksum, "bytes": size}


def result_artifact(result, kind):
    result = result["artifact"]
    require(result["kind"] == kind, f"wrong artifact kind: expected {kind}")
    path(result["path"])
    digest(result["sha256"])
    number(result["bytes"])
    return result


def native_receipt(receipt, partition=None, unit=None):
    require(type(receipt["schema"]) is int and receipt["schema"] == 1,
            "wrong native shard protocol")
    checksum = digest(receipt["partition_sha256"])
    if partition is not None:
        require(partition == checksum, "mixed partition receipt")
    if unit is not None:
        require(number(receipt["unit"]) == unit, "mixed unit receipt")
    return checksum


def unit_ids(units):
    require(isinstance(units, list) and 0 < len(units) <= 8192, "invalid partition size")
    for i, unit in enumerate(units):
        require(number(unit["id"]) == i, "partition units must be consecutive and unique")
        number(unit["input_bytes"])
        number(unit["output_bytes"])
    return list(range(len(units)))


def check_merge(receipt, partition):
    native_receipt(receipt, partition)
    report = receipt["verification"]
    require(receipt["published"] is True and report["provenance_attested"] is True and
            type(report["manifest_schema"]) is int and report["manifest_schema"] == 2,
            "native merge did not independently verify its result")


def encoding_options(value, default_threads=1):
    options = {"seed": 0, "threads": default_threads, "ht-beam": 128,
               "ht-backend": "cpu", "i4r-backend": "cpu", "q4cp-backend": "cpu"}
    require(isinstance(value, dict) and set(value) <= set(ENCODING), "unknown encoding option")
    options.update(value)
    for k, bounds in ENCODING.items():
        if type(bounds[0]) is int:
            number(options[k], *bounds)
        else:
            require(options.get(k) in bounds, f"invalid {k}")
    return options


def admitted(backends, expected):
    if any(b != "cpu" for b in backends):
        cfg = json.loads(path(os.environ["BREWERY_WORKER_CONFIG"]).read_text())
        gate = path(cfg.get("admission"))
        require(sha(gate) == digest(expected),
                "GPU stage requires the pinned shared worker admission gate")


def run_native(native, *args):
    completed = subprocess.run([native, *map(str, args)], stdout=subprocess.PIPE,
                               text=True, check=False)
    require(completed.returncode == 0, f"native {args[0]} failed with exit {completed.returncode}")
    return json.loads(completed.stdout)


def check_input(item, kind, expected=None, rehash=False):
    require(item["kind"] == kind, f"expected input kind {kind}")
    persisted = digest(item["bundle_sha256"])
    if expected is not None:
        require(persisted == expected, f"input digest changed for {kind}")
    p = path(item["path"])
    if rehash:
        require(tree(p)[0] == persisted, f"input digest changed on disk for {kind}")
    return p


def groups(inputs, bindings):
    """Yield individually verified unit directories, with no extras."""
    require(len(inputs) == len(bindings), "incomplete collection inputs")
    seen = set()
    for item, b in zip(inputs, bindings, strict=True):
        ids = b["ids"]
        require(isinstance(ids, list) and ids and
                all(type(i) is int and i >= 0 for i in ids) and
                sorted(set(ids)) == ids and not seen.intersection(ids), "duplicate or invalid unit IDs")
        seen.update(ids)
        kind = b["kind"]
        require(kind in (UNIT, COLLECTION), "invalid unit collection kind")
        p = check_input(item, kind, digest(b["sha256"]), rehash=True)
        if kind == UNIT:
            require(len(ids) == 1, "individual result must name one unit")
            yield ids[0], p
        else:
            require({c.name for c in p.iterdir()} == {str(i) for i in ids},
                    "collection contains missing or extra unit directories")
            for i in ids:
                child = p / str(i)
                require(stat.S_ISDIR(child.lstat().st_mode), "non-directory unit in collection")
                yield i, child


def stage_collection(root, entries):
    root.mkdir()
    for i, source in entries:
        # A content-addressed worker cache and job work may share a filesystem.
        # Hardlink when possible; otherwise copy; never symlink.
        def link_or_copy(src, dst):
            try:
                return os.link(src, dst)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                return shutil.copy2(src, dst)
        shutil.copytree(source, root / str(i), copy_function=link_or_copy)


def engine(job):
    kind, inputs, options = (job[k] for k in ("kind", "inputs", "options"))
    work = Path.cwd()
    if kind == "collect_units":
        entries = list(groups(inputs, options["bindings"]))
        root = work / "collection"
        stage_collection(root, entries)
        return {"partition_sha256": digest(options["partition_sha256"]),
                "ids": sorted(i for i, _ in entries), "artifact": artifact(root, COLLECTION)}
    native = path(os.environ["BREWERY_STAGE2_NATIVE"])
    require(sha(native) == digest(options["native_sha256"]), "native binary digest changed")
    if kind == "stage2_preflight":
        require(not inputs, "preflight takes no inputs")
        admitted(options["backends"], options["admission_sha256"])
        return {"native_sha256": options["native_sha256"],
                "admission_sha256": options["admission_sha256"]}
    part = options.get("partition_sha256")
    unit_id = options.get("unit")
    if kind == "partition":
        require(len(inputs) == 1, "partition requires the source bundle")
        source = check_input(inputs[0], "hf", options.get("source_sha256"))
        opts = encoding_options(options["encoding"])
        # `partition` can use a CPU-only worker: this records a tuple, not admission for encoding.
        flags = [item for k, v in sorted(opts.items()) for item in ("--" + k, v)]
        root = work / "partition"
        receipt = run_native(native, "partition", "--source", source, "--output", root, *flags)
        part = native_receipt(receipt)
        require(sha(root / "partition.json") == part, "native partition digest differs")
        unit_ids(receipt["units"])
        return {"source_sha256": inputs[0]["bundle_sha256"], "receipt": receipt,
                "artifact": artifact(root, SPLIT)}
    if kind == "encode_unit":
        require(len(inputs) == 2, "encode_unit requires partition and package")
        partition = check_input(inputs[0], SPLIT, options["partition_tree"]) / "partition.json"
        package = check_input(inputs[1], PACKAGE, options["package_tree"], rehash=True)
        require(sha(partition) == digest(part), "partition digest changed")
        partition_options = json.loads(partition.read_text())["options"]
        actual_backends = [partition_options[k] for k in ("ht_backend", "i4r_backend", "q4cp_backend")]
        require(actual_backends == options["backends"], "worker backend differs from native partition")
        admitted(actual_backends, options["admission_sha256"])
        root = work / "result"
        receipt = run_native(native, "encode-unit", "--partition", partition,
                             "--package", package, "--output", root)
        native_receipt(receipt, part, number(unit_id))
        return {"receipt": receipt, "artifact": artifact(root, UNIT)}
    require(kind in ("pack_unit", "merge_units") and len(inputs) >= 2, "unknown native stage")
    source = check_input(inputs[0], "hf", digest(options["source_sha256"]))
    partition = check_input(inputs[1], SPLIT, options["partition_tree"]) / "partition.json"
    require(sha(partition) == digest(part), "partition digest changed")
    if kind == "pack_unit":
        require(len(inputs) == 2, "pack_unit requires source and partition")
        ids = options["units"]
        require(isinstance(ids, list) and 0 < len(ids) <= PACK_BATCH and
                sorted({number(i) for i in ids}) == ids, "invalid package batch")
        root = work / "packages"
        root.mkdir()
        packages = []
        for i in ids:
            target = root / str(i)
            receipt = run_native(native, "pack-unit", "--source", source, "--partition", partition,
                                 "--unit", i, "--output", target)
            native_receipt(receipt, part, i)
            packages.append({"receipt": receipt, "artifact": artifact(target, PACKAGE)})
        return {"packages": packages, "artifact": artifact(root, PACKAGE)}
    entries = list(groups(inputs[2:], options["bindings"]))
    require(sorted(i for i, _ in entries) == list(range(number(options["unit_count"], 1, 8192))),
            "missing or mixed units at merge")
    gathered = work / "results"
    stage_collection(gathered, entries)
    backend = options["decode_backend"]
    require(backend in ("cpu", "hip", "cuda"), "invalid decode backend")
    admitted([backend], options["admission_sha256"])
    root = work / "converted"
    receipt = run_native(native, "merge", "--source", source, "--partition", partition,
                         "--results", gathered, "--output", root, "--decode-backend", backend)
    check_merge(receipt, part)
    return {"receipt": receipt, "artifact": artifact(root, ARTIFACT)}


def encoded_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def snapshot(file, value):
    with tempfile.NamedTemporaryFile(dir=file.parent, prefix=".journal-", delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(encoded_json(value).encode())
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(pending, file)
            fd = os.open(file.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            pending.unlink(missing_ok=True)


class Run:
    def __init__(self, directory, cfg, deadline, poll):
        self.directory, self.cfg, self.deadline, self.poll = directory, cfg, deadline, poll
        self.journal = directory / "journal.json"
        self.dispatch = path(cfg["dispatcher"])
        self.env = dict(os.environ, BREWERY_DISPATCH_CONFIG=str(path(cfg["dispatch_config"])))
        self.dispatch_digest = sha(self.dispatch)
        self.config_digest = sha(path(cfg["dispatch_config"]))
        identity = hashlib.sha256(encoded_json(
            [cfg, self.dispatch_digest, self.config_digest]).encode()).hexdigest()
        if self.journal.exists():
            self.record = json.loads(self.journal.read_text())
            require(self.record["config_sha256"] == identity, "configuration changed since run creation")
        else:
            self.record = {"schema": 1, "run_id": uuid.uuid4().hex, "default_threads": cpu_threads(),
                           "config_sha256": identity, "jobs": {}}
            snapshot(self.journal, self.record)

    def call(self, action, j):
        remain = self.deadline - time.monotonic()
        require(remain > 0, "run deadline reached; resume the same run directory")
        try:
            completed = subprocess.run(
                [self.dispatch, action, j["host"], j["id"]],
                input=encoded_json(j["request"]) if action == "start" else "",
                capture_output=True, text=True, env=self.env, timeout=remain,
            )
        except subprocess.TimeoutExpired as err:
            raise Failure("run deadline reached during dispatch; resume the same run directory") from err
        reply = json.loads(completed.stdout)
        require(completed.returncode == 0, f"{action} transport failed for {j['id']}: {reply.get('error')}")
        return reply

    def job(self, label, host, kind, inputs, options):
        ident = f"s2-{self.record['run_id']}-{label}"
        request = {"kind": kind, "inputs": inputs, "options": options}
        proposed = {"id": ident, "host": token(host), "request": request}
        j = self.record["jobs"].get(label)
        if j is not None:
            require(all(j[k] == v for k, v in proposed.items()), "recorded job identity changed")
        else:
            j = self.record["jobs"][label] = proposed
            snapshot(self.journal, self.record)
        return j

    def wait(self, j):
        reply = self.call("status", j)
        state = reply["state"]
        if "result" in j:
            require(state == "succeeded" and reply.get("result") == j["result"],
                    f"worker lost or changed recorded result for {j['id']}")
        elif state == "missing" or (state == "queued" and reply["progress"] != "launch submitted"):
            reply = self.call("start", j)
        while reply["state"] in ("queued", "running"):
            require(time.monotonic() < self.deadline, "run deadline reached; resume the same run directory")
            time.sleep(self.poll)
            reply = self.call("status", j)
        require(reply["state"] == "succeeded",
                f"{j['id']} {reply['state']}: {reply.get('error', reply.get('progress'))}")
        return reply["result"]

    def finish(self, j, result):
        if "result" not in j:
            j["result"] = result
            snapshot(self.journal, self.record)
        return result

    def submit(self, label, host, kind, inputs, options):
        j = self.job(label, host, kind, inputs, options)
        return self.finish(j, self.wait(j))


def input_from(host, artifact):
    return {"host": host, "kind": artifact["kind"], "path": artifact["path"]}


def collect(run, coordinator, groups_, partition):
    layer = 0
    while len(groups_) > FAN_IN:
        next_layer = []
        for offset in range(0, len(groups_), FAN_IN):
            chunk = groups_[offset:offset + FAN_IN]
            ids = sorted(i for _, binding in chunk for i in binding["ids"])
            result = run.submit(f"collect-{layer}-{offset}", coordinator, "collect_units",
                                [i for i, _ in chunk],
                                {"bindings": [b for _, b in chunk], "partition_sha256": partition})
            a = result_artifact(result, COLLECTION)
            require(result["ids"] == ids and result["partition_sha256"] == partition,
                    "collector returned mixed units")
            next_layer.append((input_from(coordinator, a),
                               {"kind": COLLECTION, "ids": ids, "sha256": a["sha256"]}))
        layer += 1
        groups_ = next_layer
    return groups_


def workflow(run):
    cfg = run.cfg
    require(set(cfg) <= {"dispatcher", "dispatch_config", "source", "coordinator", "workers",
                         "encoding", "decode_backend"}, "unknown workflow config field")
    coordinator = token(cfg["coordinator"])
    workers = cfg["workers"]
    # Preserve the encoding identity of runs created before CPU budgets were recorded.
    opts = encoding_options(cfg["encoding"], run.record.get("default_threads", 1))
    backend = [opts[k] for k in ("ht-backend", "i4r-backend", "q4cp-backend")]
    decode_backend = cfg.get("decode_backend", "cpu")
    require(decode_backend in ("cpu", "hip", "cuda"), "invalid decode_backend")
    source = {"kind": "hf", "host": token(cfg["source"]["host"]),
              "path": str(path(cfg["source"]["path"]))}
    pin = cfg["source"].get("sha256")
    if pin is not None:
        digest(pin)
    require(isinstance(workers, dict) and 2 <= len(workers) <= 32, "need 2..32 configured workers")
    for host, details in workers.items():
        token(host)
        require(set(details) <= {"native_sha256", "backend", "admission_sha256"},
                "unknown worker metadata")
        digest(details["native_sha256"])
    eligible = sorted(host for host, details in workers.items() if details["backend"] == backend)
    require(coordinator in workers and len(eligible) >= 2,
            "need a coordinator and at least two distinct eligible worker IDs")
    checks = {host: backend for host in eligible}
    checks[coordinator] = [*checks.get(coordinator, []), decode_backend]
    for host, used_backends in sorted(checks.items()):
        details = workers[host]
        gate = details.get("admission_sha256")
        if any(b != "cpu" for b in used_backends):
            digest(gate)
        run.submit("preflight-" + host, host, "stage2_preflight", [],
                   {"native_sha256": details["native_sha256"], "admission_sha256": gate,
                    "backends": used_backends})
    native = workers[coordinator]["native_sha256"]
    part = run.submit("partition", coordinator, "partition", [source],
                      {"encoding": opts, "native_sha256": native, "source_sha256": pin})
    partition = native_receipt(part["receipt"])
    ids = unit_ids(part["receipt"]["units"])
    source_digest = digest(part["source_sha256"])
    require(pin is None or pin == source_digest, "source did not match configured digest")
    a = result_artifact(part, SPLIT)
    part_input = input_from(coordinator, a)
    common = {"partition_sha256": partition, "partition_tree": a["sha256"],
              "source_sha256": source_digest}
    # Keep one running unit per worker. Prepare packages in sequence to avoid
    # competing reads of the source on the coordinator.
    outputs, packages = [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(eligible)) as pool:
        for start in range(0, len(ids), len(eligible)):
            batch = []
            for unit, host in zip(ids[start:start + len(eligible)], eligible):
                if unit not in packages:
                    pack_ids = ids[unit:unit + PACK_BATCH]
                    packed = run.submit(f"pack-{unit}", coordinator, "pack_unit", [source, part_input],
                                        dict(common, units=pack_ids, native_sha256=native))
                    parent = path(result_artifact(packed, PACKAGE)["path"])
                    for i, pack in zip(pack_ids, packed["packages"], strict=True):
                        native_receipt(pack["receipt"], partition, i)
                        a = result_artifact(pack, PACKAGE)
                        require(path(a["path"]) == parent / str(i), "package outside its batch")
                        packages[i] = a
                a = packages.pop(unit)
                j = run.job(f"encode-{unit}", host, "encode_unit",
                            [part_input, input_from(coordinator, a)],
                            dict(common, unit=unit, package_tree=a["sha256"],
                                 native_sha256=workers[host]["native_sha256"], backends=backend,
                                 admission_sha256=workers[host].get("admission_sha256")))
                batch.append((unit, host, j, pool.submit(run.wait, j)))
            for unit, host, job, future in batch:
                result = run.finish(job, future.result())
                native_receipt(result["receipt"], partition, unit)
                a = result_artifact(result, UNIT)
                outputs.append((input_from(host, a),
                                {"kind": UNIT, "ids": [unit], "sha256": a["sha256"]}))
    collected = collect(run, coordinator, outputs, partition)
    merged = run.submit("merge", coordinator, "merge_units",
                        [source, part_input, *(i for i, _ in collected)],
                        dict(common, bindings=[b for _, b in collected], native_sha256=native,
                             decode_backend=decode_backend, unit_count=len(ids),
                             admission_sha256=workers[coordinator].get("admission_sha256")))
    check_merge(merged["receipt"], partition)
    return {"schema": 1, "run_id": run.record["run_id"], "source_sha256": source_digest,
            "partition_sha256": partition, "units": len(ids), "artifact_host": coordinator,
            "artifact": result_artifact(merged, ARTIFACT), "receipt": merged["receipt"],
            "release_qualified": False}


def main():
    # No arguments is the explicitly configured worker engine protocol.
    if len(sys.argv) == 1:
        return engine(json.load(sys.stdin))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--timeout", default=3600, type=float)
    parser.add_argument("--poll", default=.1, type=float)
    parser.add_argument("--reserved-workers", action="store_true",
                        help="confirm operator reservation of the configured worker instances")
    args = parser.parse_args()
    require(args.reserved_workers, "reserve worker instances before using stage 2")
    require(0 < args.timeout <= 86400 and .02 <= args.poll <= 10, "invalid timeout or polling interval")
    args.run_dir.mkdir(mode=0o700, exist_ok=True)
    st = args.run_dir.lstat()
    require(stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and not st.st_mode & 0o077,
            "run directory must be private and owned by the operator")
    cfg = json.loads(args.config.read_text())
    with (args.run_dir / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Failure("another coordinator already owns this run") from error
        run = Run(args.run_dir, cfg, time.monotonic() + args.timeout, args.poll)
        result = workflow(run)
        snapshot(args.run_dir / "result.json", result)
        return result


if __name__ == "__main__":
    os.umask(0o077)
    try:
        print(encoded_json(main()))
    except (Failure, OSError, ValueError, KeyError, TypeError) as error:
        print(f"distributed conversion: {error}", file=sys.stderr)
        sys.exit(2)
