#!/usr/bin/env python3
"""Report read-only Linux observations of the legacy local-sm120 input pass."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import tempfile
import threading
import time
from uuid import UUID


ROOT = Path("/mnt/games/model-brewery")
HOST = "local-sm120"
UID = os.getuid()
PROC = Path("/proc")
DIST = Path(__file__).resolve().parents[2] / "web/dist"
QUIET = (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, RuntimeError)


@dataclass(frozen=True)
class Layout:
    run: Path
    source: Path
    receipt: Path
    receipt_sha: str
    worker_sha: str = ""

    def exe(self, program):
        return self.run / "bin" / program


LOCAL = Layout(
    ROOT / HOST,
    ROOT / "Qwen3.8-Flash-Next-de4b8e4d43b917e7706784d8bb445c9af86a3540",
    ROOT / ".qwen3.8-flash-next-de4b8e4d/published.json",
    "85679c93d41466b415e3adde0021ff8db70148f0e37b5589972f5184be2ee7a5",
    "0b5291596a75611cb902f436b14ed819fc291758ee1c830793d9e74732547ac0",
)


def require(condition):
    if not condition:
        raise ValueError("observation not verified")


def safe(meta):
    return meta.st_uid == UID and not meta.st_mode & 0o022


def canonical(path, kind):
    meta = path.lstat()
    require(safe(meta) and kind(meta.st_mode) and path.resolve(strict=True) == path)
    return meta


def small(path, limit=262144, *, private=False):
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if private:
        canonical(path, stat.S_ISREG)
        flags |= os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags), "rb") as file:
        if private:
            meta = os.fstat(file.fileno())
            require(safe(meta) and stat.S_ISREG(meta.st_mode) and not meta.st_mode & 0o077)
        data = file.read(limit + 1)
    require(len(data) <= limit)
    return data


def private_json(path):
    return json.loads(small(path, private=True))


def manifest(layout):
    data = small(layout.receipt, private=True)
    # The receipt itself is pinned to the verified publication; never hash the corpus here.
    require(hashlib.sha256(data).hexdigest() == layout.receipt_sha)
    obj = json.loads(data)
    require(obj["schema"] == "model-brewery.source-publication.v1"
            and obj["model_dir"] == str(layout.source))
    sizes = {name: entry["size"] for name, entry in obj["files"].items()}
    require(sizes and all(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name)
                          and name not in (".", "..") and type(size) is int and size > 0
                          for name, size in sizes.items()))
    canonical(layout.source, stat.S_ISDIR)
    return sizes


def kv(data, sep):
    return dict(line.split(sep, 1) for line in data.splitlines() if sep in line)


def matches_env(proc, expected):
    keys = {os.fsencode(key): os.fsencode(value) for key, value in expected.items()}
    env = small(proc / "environ").split(b"\0")
    return all([line for line in env if line.startswith(key + b"=")] == [key + b"=" + value]
               for key, value in keys.items())


def process(proc):
    require(proc.stat().st_uid == UID)
    # comm is parenthesized and can contain spaces or closing parentheses.
    fields = small(proc / "stat", 4096).rsplit(b")", 1)[1].split()
    require(fields[0] != b"Z")
    return int(fields[1]), int(fields[19])  # PPID and process start ticks


@dataclass(frozen=True)
class Process:
    proc: Path
    exe: Path
    parent: int
    start: int
    args: tuple

    def unchanged(self):
        return (process(self.proc) == (self.parent, self.start)
                and os.readlink(self.proc / "exe") == str(self.exe)
                and self.exe.samefile(self.proc / "exe")
                and argv(self.proc) == (str(self.exe), *self.args))


def argv(proc):
    return tuple(os.fsdecode(arg) for arg in small(proc / "cmdline").split(b"\0")[:-1])


def processes(layout):
    exes = {str(layout.exe(name)): name for name in
            ("brewery", "brewery-dispatch", "brewery-worker", "brewery-engine")}
    result = {name: [] for name in exes.values()}
    for proc in PROC.iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != UID:
                continue
            link = os.readlink(proc / "exe")
            if link not in exes:
                continue
            exe = Path(link)
            canonical(exe, stat.S_ISREG)
            if not exe.samefile(proc / "exe"):
                continue
            parent, start = process(proc)
            args = argv(proc)
            if args and args[0] == link:
                result[exes[link]].append(Process(proc, exe, parent, start, args[1:]))
        except QUIET:
            continue
    return result


def observation(process, source, sizes):
    hits = []
    for fd in (process.proc / "fd").iterdir():
        try:
            target = Path(os.readlink(fd))
            if target.parent != source or target.name not in sizes:
                continue
            info = kv(small(process.proc / "fdinfo" / fd.name, 4096), b":")
            length = sizes[target.name]
            pos = int(info[b"pos"])
            flags = int(info[b"flags"], 8)
            meta = canonical(target, stat.S_ISREG)
            opened = fd.stat()
            if (0 < pos <= length and not flags & (os.O_ACCMODE | os.O_PATH)
                    and meta.st_size == length == opened.st_size
                    and meta.st_ino == int(info[b"ino"])
                    and os.path.samestat(meta, opened) and os.readlink(fd) == str(target)):
                hits.append((target.name, pos, length))
        except QUIET:
            continue
    return hits[0] if len(hits) == 1 else None


def job_state(layout, ident, state):
    require(str(UUID(ident)) == ident)
    jobdir = layout.run / "worker/jobs" / ident
    obj = private_json(jobdir / "state.json")
    job = obj["job"]
    require(obj["action"] == "start" and obj["cancel"] is False
            and obj["reply"]["state"] == state
            and job["kind"] in ("inspect", "plan", "convert", "verify")
            and has_source(layout, job))
    return jobdir, job


def has_source(layout, job):
    return any(i["path"] == str(layout.source) and i["host"] == HOST and i["kind"] == "hf"
               for i in job["inputs"])


def candidates(layout, procs, controller, inherited):
    workers = procs["brewery-worker"]
    for dispatch in procs["brewery-dispatch"]:
        if (dispatch.parent != int(controller.proc.name) or len(dispatch.args) != 3
                or dispatch.args[:2] != ("start", HOST)):
            continue
        ident = dispatch.args[2]
        for worker in workers:
            if worker.parent == int(dispatch.proc.name) and worker.args == ("rpc",):
                try:
                    _, job = job_state(layout, ident, "queued")
                    if matches_env(dispatch.proc, inherited):
                        yield ident, (controller, dispatch, worker), "Resolving input (observed file)", job
                except QUIET:
                    continue
    for worker in workers:
        if len(worker.args) != 2 or worker.args[0] != "supervise":
            continue
        ident = worker.args[1]
        for engine in procs["brewery-engine"]:
            if engine.parent != int(worker.proc.name) or engine.args:
                continue
            try:
                jobdir, job = job_state(layout, ident, "running")
                jobfile = jobdir / "engine-input.json"
                require(os.readlink(engine.proc / "cwd") == str(jobdir / "work")
                        and os.readlink(engine.proc / "fd/0") == str(jobfile)
                        and jobfile.samefile(engine.proc / "fd/0")
                        and has_source(layout, private_json(jobfile)))
                yield ident, (controller, worker, engine), "Engine input read (observed file)", job
            except QUIET:
                continue


@lru_cache(maxsize=4)
def file_sha(path, stamp):
    # Only the pinned legacy worker binary; no model files are opened by this observer.
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def rchar(proc):
    return int(kv(small(proc / "io", 4096), b":")[b"rchar"])


def pass_prefix(layout, sizes, worker):
    require(layout.worker_sha)
    meta = canonical(worker.exe, stat.S_ISREG)
    require(meta.st_size < 64 * 1024 * 1024)
    require(file_sha(worker.exe, (meta.st_ino, meta.st_mtime_ns, meta.st_ctime_ns))
            == layout.worker_sha)
    receipt = layout.receipt.stat()
    source = layout.source.stat()
    require(source.st_ctime_ns <= receipt.st_mtime_ns)
    prefixes = {}
    total = 0
    # The pinned worker hashes while enumerating readdir order, then sorts its
    # manifest afterwards. Read only metadata and reject any inventory change.
    with os.scandir(layout.source) as entries:
        for entry in entries:
            meta = entry.stat(follow_symlinks=False)
            require(entry.name in sizes and stat.S_ISREG(meta.st_mode) and safe(meta)
                    and not meta.st_mode & 0o222 and meta.st_size == sizes[entry.name]
                    and meta.st_ctime_ns <= receipt.st_mtime_ns)
            prefixes[entry.name] = total
            total += meta.st_size
    require(prefixes.keys() == sizes.keys() and layout.source.stat().st_ctime_ns == source.st_ctime_ns)
    return prefixes, total


def sample(layout, sizes, chain, phase, job, inherited):
    worker = chain[-1] if len(chain) == 3 and chain[-1].args == ("rpc",) else chain[-2]
    require(matches_env(worker.proc, {**inherited, "BREWERY_WORKER_CONFIG": layout.run / "worker.json"}))
    reader = chain[-1]
    estimate = None
    if reader == worker and len(job["inputs"]) == 1 and layout.worker_sha:
        try:
            estimate = pass_prefix(layout, sizes, worker)
            before = rchar(reader.proc)
        except QUIET:
            estimate = None
    hit = observation(reader, layout.source, sizes)
    if hit is None:
        return None
    name, pos, length = hit
    detail = {"phase": phase, "unit": "bytes", "completed": pos, "total": length,
              "current_file": name}
    if estimate is not None:
        try:
            after = rchar(reader.proc)
            prefixes, total = estimate
            scanned = prefixes[name] + pos
            # rchar includes the initial RPC/config/state reads. Bracket the FD
            # sample and accept only a small bounded overhead for this one direct
            # input. Disagreement (including a seek or second read) falls back.
            if before <= scanned + 65536 and after >= scanned and 0 <= after - before <= 1048576:
                detail.update(phase="Resolving input (estimated pass)", completed=scanned,
                              total=total, estimated=True)
        except QUIET:
            pass
    require(all(p.unchanged() for p in chain))
    return detail


def observation_sample(layout):
    try:
        procs = processes(layout)
        inherited = {"BREWERY_DATA": layout.run / "controller",
                     "BREWERY_DISPATCH": layout.exe("brewery-dispatch"),
                     "BREWERY_DISPATCH_CONFIG": layout.run / "dispatch.json"}
        controllers = [p for p in procs["brewery"] if not p.args
                       and matches_env(p.proc, inherited)]
        if len(controllers) != 1:
            return None
        dispatch = private_json(layout.run / "dispatch.json")["hosts"][HOST]
        config = private_json(layout.run / "worker.json")
        require(dispatch["local"] is True and dispatch["worker"] == str(layout.exe("brewery-worker"))
                and dispatch["config"] == str(layout.run / "worker.json")
                and config["root"] == str(layout.run / "worker")
                and config["source_roots"] == [str(layout.source)]
                and config["engine"] == str(layout.exe("brewery-engine")))
        canonical(layout.run / "controller", stat.S_ISDIR)
        canonical(layout.run / "worker", stat.S_ISDIR)
        sizes = manifest(layout)
        matches = []
        for ident, chain, phase, job in candidates(layout, procs, controllers[0], inherited):
            try:
                detail = sample(layout, sizes, chain, phase, job, inherited)
                if detail:
                    reader = chain[-1]
                    matches.append((ident, detail, (reader.proc.name, reader.start)))
            except QUIET:
                continue
        if len(matches) != 1:
            return None
        ident, detail, identity = matches[0]
        now = time.time_ns() // 1_000_000
        detail["updated_at_ms"] = now
        return ({"schema": "brewery-live-progress-v1", "job_id": ident,
                 "observed_at_ms": now, "source": "observe_progress", "progress_detail": detail},
                identity)
    except QUIET:
        return None


def observe(layout=LOCAL):
    hit = observation_sample(layout)
    return hit[0] if hit else None


def publish(output, value):
    data = (json.dumps(value, separators=(",", ":")) + "\n").encode("ascii")
    require(len(data) <= 4096)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".current-", delete=False) as file:
        tmp = Path(file.name)
        try:
            file.write(data)
            file.close()
            tmp.replace(output)
        finally:
            tmp.unlink(missing_ok=True)


def watch(output, stop, *, layout=LOCAL, interval=2, stale=20):
    canonical(output.parent.parent, stat.S_ISDIR)
    output.parent.mkdir(mode=0o700, exist_ok=True)
    canonical(output.parent, stat.S_ISDIR)
    previous = None
    last_change = 0
    try:
        while not stop.is_set():
            hit = observation_sample(layout)
            if hit:
                value, identity = hit
                current = (identity, value["job_id"],
                           {k: v for k, v in value["progress_detail"].items()
                            if k != "updated_at_ms"})
                if current != previous:
                    previous, last_change = current, time.monotonic()
                    publish(output, value)
                elif time.monotonic() - last_change >= stale:
                    output.unlink(missing_ok=True)
            else:
                output.unlink(missing_ok=True)
                previous = None
            stop.wait(interval)
    finally:
        output.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true",
                        help="Publish atomically to web/dist/live-progress/current.json.")
    args = parser.parse_args()
    if args.watch:
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        watch(DIST / "live-progress/current.json", stop)
        return 0
    value = observe()
    if value is None:
        return 1
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
