#!/usr/bin/env python3
"""Run one local sm_120 workbench qualification stage against persistent state."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, ProxyHandler, HTTPRedirectHandler, build_opener
from uuid import UUID

from release_check import CHECKPOINT, REVISION, identity_shards
from release_source import METADATA

ROOT = Path("/mnt/games/model-brewery")
RUN = ROOT / "local-sm120"
SOURCE = ROOT / f"Qwen3.8-Flash-Next-{REVISION}"
PUBLICATION = ROOT / ".qwen3.8-flash-next-de4b8e4d/published.json"
SOURCE_SHA = "8858fcd3bad25db0f243a0f72d30ad0674f4a99bf04d102ab61f01897df72308"
BUNDLE_SHA = "0eb8fb9e3b0f2422297174c880aa2ad8304e77a34d37b15c592ded1c3f9b4cd1"
CONVERTER_SHA = "cf629fc31a285d6af2004ef98cf66bca839e6e4e3ce207d5fd981843001a6eb5"
COLLECTOR_SHA = "90592e64c1b239e55661a0dab4eafe7d81fe8ef3a7e3866149602c43897f128e"
GPU = "GPU-dce9e8fa-faa3-af60-909f-004f99054cd1"
HOST = "local-sm120"
GIB = 1 << 30
TERMINAL = ("succeeded", "failed", "cancelled")
HOLD = 12 * 3600
RESOURCE_QUEUE = "/home/quaternion/.local/bin/resource-queue"
PAYLOAD = {"v2": 118_785_840_932, "ht43": 109_716_416_292}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def need(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, data):
    if path.exists() and path.read_bytes() == data:
        return
    pending = path.with_suffix(path.suffix + ".next")
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    pending.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path, obj):
    write(path, json.dumps(obj, sort_keys=True, indent=2).encode() + b"\n")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def private(path, directory=False):
    meta = path.lstat()
    need(path.resolve() == path and meta.st_uid == os.getuid() and not meta.st_mode & 0o077
         and (stat.S_ISDIR(meta.st_mode) if directory else stat.S_ISREG(meta.st_mode)),
         f"expected a private canonical {'directory' if directory else 'file'}: {path}")


def publication():
    private(ROOT, True)
    need(SOURCE.is_dir() and PUBLICATION.is_file(), "waiting for atomic upstream source publication and receipt")
    private(PUBLICATION.parent, True)
    private(PUBLICATION)
    need(SOURCE.resolve() == SOURCE, "published source is aliased")
    need(PUBLICATION.stat().st_size <= 128 * 1024, "source receipt is too large")
    receipt = json.loads(PUBLICATION.read_bytes())
    need(receipt["schema"] == "model-brewery.source-publication.v1"
         and receipt["repo_id"] == CHECKPOINT and receipt["revision"] == REVISION
         and receipt["model_dir"] == str(SOURCE) and receipt["shard_count"] == 131
         and receipt["weight_bytes"] == 360_000_192_888, "source receipt is not for the pinned revision")
    pins = identity_shards(Path(__file__).with_name("release_check_shards.sha256"))
    files = receipt["files"]
    need(set(files) == set(pins) | METADATA.keys() == {p.name for p in SOURCE.iterdir()},
         "published source inventory differs from pins")
    tree = hashlib.sha256()
    for name, meta in sorted(files.items()):
        child = SOURCE / name
        st = child.lstat()
        need(stat.S_ISREG(st.st_mode) and st.st_size == meta["size"], f"invalid published file: {name}")
        size, digest = meta["size"], meta["sha256"]
        need((size, digest) == METADATA[name] if name in METADATA else digest == pins[name],
             f"upstream digest mismatch: {name}")
        tree.update(f"{name}\0{size}\0{digest}\n".encode())
    need(tree.hexdigest() == BUNDLE_SHA, "wrong published source tree digest")
    # The engine hashes the real source on each job and checks SOURCE_SHA
    # before launching the native converter. This gate does not hash it twice.


def available(required, *, gpu):
    need(shutil.disk_usage(RUN).free >= required, f"need {required} free bytes on {RUN}")
    if not gpu:
        return
    ram = next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))
    need(int(ram) * 1024 >= 16 * GIB, f"need >=16 GiB MemAvailable; have {ram} KiB")
    query = ["/usr/bin/nvidia-smi", "-i", GPU]
    line = subprocess.check_output(query + ["--query-gpu=uuid,compute_cap,memory.free",
                                           "--format=csv,noheader,nounits"], text=True, timeout=10)
    uuid, capability, mib = (field.strip() for field in line.strip().split(","))
    need(uuid == GPU and capability == "12.0" and int(mib) >= 12 * 1024,
         f"need pinned sm_120 and >=12 GiB VRAM free; got {line.strip()}")
    active = subprocess.check_output(query + ["--query-compute-apps=pid", "--format=csv,noheader"],
                                     text=True, timeout=10)
    need(not active.strip(), "selected GPU already has a compute process")


def admitted(argv):
    """Private engine launcher. Exec the converter with its own real PID."""
    pid_file, exe, kind, *args = argv
    need(kind == "convert" and Path(pid_file).name == "admitted-pid"
         and Path(pid_file).resolve().is_relative_to(RUN / "worker/jobs")
         and exe == str(RUN / "bin/bf16-to-hgn"), "invalid native admission request")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
         and os.environ.get("CUDA_INJECTION64_PATH") == str(RUN / "bin/encoder_activity.so")
         and os.environ.get("BREWERY_CUPTI_CAPTURE") == str(Path(pid_file).with_name("encoder-activity.json")),
         "workbench did not arm the pinned CUPTI capture and exact GPU selection")
    publication()
    available(int(os.environ["BREWERY_SM120_REQUIRED_DISK"]), gpu=True)
    need(sha(Path(exe)) == CONVERTER_SHA, "pinned converter changed")
    need(sha(RUN / "bin/encoder_activity.so") == COLLECTOR_SHA, "pinned CUPTI collector changed")
    with open(pid_file, "x", encoding="ascii") as file:
        file.write(f"{os.getpid()}\n")
    os.execv(exe, [exe, kind, *args])


def install(run, bins, converter, collector):
    need(sha(converter) == CONVERTER_SHA and sha(collector) == COLLECTOR_SHA,
         "conversion tools do not match the pinned CUDA build")
    need(os.access(RESOURCE_QUEUE, os.X_OK), "shared resource-queue executable is unavailable")
    for dest in (run, run / "bin", run / "exports", run / "artifacts", run / "worker"):
        dest.mkdir(mode=0o700, exist_ok=True)
        private(dest, True)
    tools = {name: bins / name for name in ("brewery", "brewery-worker", "brewery-dispatch", "brewery-engine")}
    for name, path in {**tools, "bf16-to-hgn": converter, "encoder_activity.so": collector}.items():
        dest = run / "bin" / name
        if dest.exists():
            need(sha(dest) == sha(path), f"previously installed binary differs: {dest}")
        else:
            write(dest, path.read_bytes())
        dest.chmod(0o700)
    launcher = run / "bin/admit-converter"
    script = Path(__file__).resolve()
    import shlex
    word = shlex.quote
    write(launcher, (
        "#!/bin/sh\n"
        f'exec {word(RESOURCE_QUEUE)} run --resource gpu --flock gpu --holder brewery-local-sm120 '
        f'--wait 300 --max-hold {HOLD} -- '
        f'/usr/bin/python3 -B {word(str(script))} _admitted "$@"\n'
    ).encode())
    launcher.chmod(0o700)
    worker = run / "worker.json"
    save(worker, {
        "root": str(run / "worker"), "source_roots": [str(SOURCE)],
        "pinned_publication": {"source": str(SOURCE), "receipt": str(PUBLICATION),
                               "repo_id": CHECKPOINT, "revision": REVISION,
                               "bundle_sha256": BUNDLE_SHA},
        "engine": str(run / "bin/brewery-engine"),
        "engine_env": {"BREWERY_CONVERTER": str(run / "bin/bf16-to-hgn"),
                       "BREWERY_CUPTI_COLLECTOR": str(run / "bin/encoder_activity.so"),
                       "BREWERY_CUDA_GPU_UUID": GPU, "CUDA_DISABLE_PTX_JIT": "1",
                       "BREWERY_EXPECTED_SOURCE_SHA256": SOURCE_SHA,
                       "BREWERY_CONVERTER_ADMISSION": str(launcher)},
    })
    save(run / "dispatch.json", {"hosts": {HOST: {
        "local": True, "worker": str(run / "bin/brewery-worker"), "config": str(worker),
        "capabilities": {kind: False for kind in ("compare", "generate", "evaluate", "baseline",
                                                  "benchmark", "transfer", "transfer_preview", "cache_cleanup")},
    }}})
    print(f"Installed private local worker under {run}")


def request(url, path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = Request(url + "/api/" + path, payload, {"Content-Type": "application/json"})
    with build_opener(ProxyHandler({}), NoRedirect()).open(req, timeout=15) as reply:
        raw = reply.read((64 << 20) + 1)
    need(len(raw) <= 64 << 20, "oversized controller reply")
    return json.loads(raw), raw


def reconcile(intent, records):
    matches = [r["id"] for r in records if r["id"] not in intent["before"]
               and all(r.get(k) == v for k, v in intent["body"].items())]
    need(len(matches) == 1, "unresolved POST outcome; inspect controller state; not resubmitted")
    return matches[0]


def submit(url, run, label, resource, body):
    journal = run / f"{label}.json"
    if not journal.exists():
        state, _ = request(url, "state")
        intent = {"body": body, "before": [r["id"] for r in state[resource]]}
        save(journal, intent)
        try:
            row, _ = request(url, resource, body)
        except (OSError, ValueError) as error:
            try:
                state, _ = request(url, "state")
                ident = reconcile(intent, state[resource])
            except (OSError, ValueError) as recovery:
                raise ValueError(f"{label}: POST outcome unknown; not resubmitted ({recovery})") from error
        else:
            ident = reconcile(intent, [row])
    else:
        intent = json.loads(journal.read_bytes())
        need(intent["body"] == body, f"job spec changed for {label}")
        ident = intent.get("id")
        if ident is None:
            state, _ = request(url, "state")
            ident = reconcile(intent, state[resource])
    need(str(UUID(ident)) == ident, "controller did not return a job/model UUID")
    save(journal, {**intent, "id": ident})
    return ident


@contextmanager
def controller(run, required):
    bin_dir = run / "bin"
    cfg = run / "dispatch.json"
    log = run / "controller.log"
    env = {**os.environ, "BREWERY_DATA": str(run / "controller"), "BREWERY_LISTEN": "127.0.0.1:0",
           "BREWERY_DISPATCH": str(bin_dir / "brewery-dispatch"), "BREWERY_DISPATCH_CONFIG": str(cfg),
           "BREWERY_DISPATCH_TIMEOUT_SECS": "21600", "BREWERY_POLL_MS": "500",
           "BREWERY_SM120_REQUIRED_DISK": str(required)}
    with log.open("a+") as stream:
        start = stream.tell()
        proc = subprocess.Popen([bin_dir / "brewery"], env=env, stdout=stream, stderr=stream)
        try:
            until = time.monotonic() + 10
            while time.monotonic() < until and proc.poll() is None:
                with log.open() as reader:
                    reader.seek(start)
                    lines = [s.rsplit(" ", 1)[1] for s in reader.read().splitlines()
                             if s.startswith("brewery listening on http://127.0.0.1:")]
                if lines:
                    yield lines[-1]
                    return
                time.sleep(0.1)
            raise ValueError(f"controller did not start; see {log}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def record_job(url, run, label, kind, inputs, options, *, seconds=HOLD + 8 * 3600):
    ident = submit(url, run, label, "jobs", {
        "kind": kind, "host": HOST, "input_ids": inputs, "options": options,
    })
    print(f"{label}: {ident}", flush=True)
    until = time.monotonic() + seconds
    detail = f"jobs/{ident}"
    previous = None
    try:
        while time.monotonic() < until:
            try:
                job, raw = request(url, detail)
            except (OSError, URLError):
                time.sleep(2)
                continue
            msg = (job["state"], job["progress"])
            if msg != previous:
                print(*msg, flush=True)
                previous = msg
            if job["state"] in TERMINAL:
                write(run / "exports" / f"{ident}.json", raw)
                need(job["state"] == "succeeded" and job["has_result"], f"{label} ended {job['state']}")
                return job
            time.sleep(2)
        raise ValueError(f"wait deadline reached for {ident}")
    except (KeyboardInterrupt, ValueError):
        # Cancellation is idempotent; an aborted run must not leave a conversion
        # behind. If contact was lost, cancel that same ID on reconciliation.
        try:
            job, _ = request(url, detail)
            if job["state"] not in TERMINAL:
                request(url, detail + "/cancel", {})
                for _ in range(30):
                    job, raw = request(url, detail)
                    if job["state"] in TERMINAL:
                        write(run / "exports" / f"{ident}.json", raw)
                        break
                    time.sleep(1)
                else:
                    print(f"Cancel pending for {ident}; reopen the same run with cancel", file=sys.stderr)
        except (OSError, ValueError):
            print(f"Contact lost; restart same run and reconcile {ident} (do not reissue POST)", file=sys.stderr)
        raise


def options(profile, kind):
    result = {"profile": profile, "ht_backend": "cuda", "q4cp_backend": "cuda"}
    if profile == "v2":
        result["i4r_backend"] = "cuda"
    if kind != "inspect":
        result.update({"threads": 4, "ht_beam": 128})
    return result


def required_space(run):
    total = 16 * GIB
    for candidate in ("v2", "ht43"):
        plan = saved_job(run, f"{candidate}-plan")
        disk = plan["result"]["disk_preflight"]
        need(plan["kind"] == "plan" and plan["options"] == options(candidate, "plan")
             and plan["result"]["source_sha256"] == SOURCE_SHA
             and disk["fits"] is True and disk["payload_bytes"] == PAYLOAD[candidate]
             and disk["required_bytes"] >= PAYLOAD[candidate],
             f"{candidate} plan did not establish disk capacity")
        completed = (run / f"{candidate}-convert.json").exists()
        if completed:
            journal = json.loads((run / f"{candidate}-convert.json").read_bytes())
            completed = bool(journal.get("id") and (run / "exports" / f"{journal['id']}.json").exists())
        if not completed:
            total += disk["required_bytes"]
    return total


def saved_job(run, label):
    ident = json.loads((run / f"{label}.json").read_bytes())["id"]
    job = json.loads((run / "exports" / f"{ident}.json").read_bytes())
    need(job["id"] == ident and job["state"] == "succeeded", f"{label} is not a successful job")
    return job


def evidence(run, job, source):
    result = job["result"]
    native = result["backend_receipt"]["processes"][0]
    dest = run / "artifacts" / job["id"]
    dest.mkdir(exist_ok=True, mode=0o700)
    root = run / "worker/jobs" / job["id"] / "work"
    artifact = Path(result["artifact"]["path"])
    need(artifact.resolve() == artifact and artifact.is_relative_to(root), "unexpected artifact path")
    paths = [Path(native["stderr_log"])]
    if job["kind"] == "convert":
        paths.extend([artifact / "native/conversion-manifest.json", artifact / "manifest.json"])
        capture = result.get("encoder_activity", {}).get("capture")
        if capture:
            paths.append(Path(capture))
            for path, digest in [
                (Path(native["executable"]), CONVERTER_SHA),
                (Path(result["encoder_activity"]["collector"]), COLLECTOR_SHA),
            ]:
                need(path.parent == run / "bin" and sha(path) == digest, "CUDA evidence executable pin mismatch")
                write(dest / path.name, path.read_bytes())
        index = source / "model.safetensors.index.json"
        write(dest / index.name, index.read_bytes())
    else:
        paths.append(artifact)
    for path in paths:
        need(path.resolve() == path and path.is_relative_to(root), "evidence escaped worker directory")
        if path.is_file():
            need(path.stat().st_size <= 64 << 20, f"oversized evidence file: {path}")
            write(dest / path.name, path.read_bytes())


def stage(url, run, source, action, *, seconds=HOLD + 8 * 3600):
    src = submit(url, run, "source", "models", {
        "name": "Pinned BF16 source", "kind": "hf", "host": HOST, "path": str(source),
    })
    if action == "preflight":
        for profile in ("v2", "ht43"):
            for kind in ("inspect", "plan"):
                job = record_job(url, run, f"{profile}-{kind}", kind, [src], options(profile, kind),
                                 seconds=seconds)
                print(profile, kind, job["result"]["disk_preflight"], flush=True)
        return
    profile, kind = action.split("-")
    if kind == "convert":
        job = record_job(url, run, action, kind, [src], options(profile, kind), seconds=seconds)
    else:
        converted = saved_job(run, f"{profile}-convert")
        artifact = converted["result"]["artifact"]
        need(Path(artifact["path"]) == run / "worker/jobs" / converted["id"] / "work/converted",
             "converted bundle is not an output of this worker job")
        dest = submit(url, run, f"{profile}-artifact", "models", {
            "name": f"Local {profile} output", "kind": "hgn", "host": HOST, "path": artifact["path"],
        })
        job = record_job(url, run, action, kind, [src, dest], {"decode_backend": "cpu"}, seconds=seconds)
        need(job["result"]["file_verification"]["status"] == "passed", "file verification did not pass")
    evidence(run, job, source)
    if kind == "convert":
        need(job["result"].get("encoder_activity", {}).get("status") == "observed",
             "output preserved, but no matching GPU encoder activity was observed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_admitted":
        admitted(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("install", "check", "preflight", "v2-convert", "v2-verify",
                                         "ht43-convert", "ht43-verify", "cancel", "all"))
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--collector", type=Path)
    parser.add_argument("--wait-publication", type=int, default=0, metavar="SECONDS")
    args = parser.parse_args()
    need(os.path.ismount(ROOT.parent), "approved /mnt/games disk is not mounted")
    private(ROOT, True)
    if args.stage == "install":
        need(all((args.bin_dir, args.converter, args.collector)), "install needs --bin-dir, --converter and --collector")
        install(RUN, args.bin_dir, args.converter, args.collector)
        return
    need(0 <= args.wait_publication <= 86400, "publication wait must be between 0 and 86400 seconds")
    private(RUN, True)
    with (RUN / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage != "cancel":
            end = time.monotonic() + args.wait_publication
            if args.wait_publication and not PUBLICATION.is_file():
                print(f"Waiting up to {args.wait_publication}s for {PUBLICATION}", flush=True)
            while time.monotonic() < end and not PUBLICATION.is_file():
                time.sleep(min(30, max(0, end - time.monotonic())))
            publication()
            need(sha(RUN / "bin/bf16-to-hgn") == CONVERTER_SHA
                 and sha(RUN / "bin/encoder_activity.so") == COLLECTOR_SHA, "tool pins do not match")
        if args.stage == "check":
            available(16 * GIB, gpu=True)
            print("Source publication and provisional resource checks passed; run preflight next.")
            return
        if args.stage == "cancel":
            # Even if an old queued conversion races cancellation, do not admit it.
            with controller(RUN, 1 << 63) as url:
                state, _ = request(url, "state")
                for job in state["jobs"]:
                    if job["state"] not in TERMINAL:
                        request(url, f"jobs/{job['id']}/cancel", {})
                until = time.monotonic() + 90
                while time.monotonic() < until:
                    state, _ = request(url, "state")
                    if all(j["state"] in TERMINAL for j in state["jobs"]):
                        return
                    time.sleep(1)
                raise ValueError("cancellation not yet confirmed; repeat cancel against the same state")
        actions = ("preflight", "v2-convert", "v2-verify", "ht43-convert", "ht43-verify")
        for action in actions if args.stage == "all" else (args.stage,):
            required = 16 * GIB
            launching = action.endswith("-convert")
            if launching:
                required = required_space(RUN)
                if not (RUN / f"{action}.json").exists():
                    available(required, gpu=True)
            else:
                available(required, gpu=False)
            with controller(RUN, required) as url:
                stage(url, RUN, SOURCE, action)


if __name__ == "__main__":
    # Ensure a normal termination signal also unwinds the controller context.
    def interrupt(_sig, _frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, interrupt)
        main()
    except (OSError, ValueError, KeyError, KeyboardInterrupt, subprocess.SubprocessError) as err:
        sys.exit(f"Blocked: {err}")
