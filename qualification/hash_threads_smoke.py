"""Linux worker RPC smoke: independently verify hashes and count active hash threads."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def run(worker, root, limit):
    source = root / "shards"
    cfg = root / "worker.json"
    env = dict(os.environ, BREWERY_WORKER_CONFIG=str(cfg))
    env.pop("BREWERY_HASH_THREADS", None)
    if limit:
        env["BREWERY_HASH_THREADS"] = str(limit)
    with subprocess.Popen([worker, "rpc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env) as p:
        request = {"op": "describe", "input": {"path": str(source), "host": "probe", "kind": "hf"}}
        p.stdin.write(json.dumps(request).encode())
        p.stdin.close()
        active = set()
        start = time.monotonic()
        while p.poll() is None:
            for t in Path(f"/proc/{p.pid}/task").glob("*/stat"):
                try:
                    fields = t.read_text().rsplit(")", 1)[1].split()
                    if fields and int(fields[11]) > 0:  # utime since thread start, in ticks
                        active.add(t.parent.name)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if time.monotonic() - start > 30:
                p.kill()
                raise RuntimeError("worker did not finish")
            time.sleep(.005)
        reply = json.loads(p.stdout.read())
        if p.returncode or "ok" not in reply:
            raise RuntimeError(f"worker failed: {reply}")
    return reply["ok"]["bundle"], active


def main():
    if len(os.sched_getaffinity(0)) < 2:
        sys.exit("need at least two logical CPUs to verify concurrency")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "shards"
        source.mkdir()
        cfg = {"root": str(root / "state"), "engine": "/usr/bin/false", "source_roots": [str(source)]}
        (root / "worker.json").write_text(json.dumps(cfg))
        (root / "state").mkdir(mode=0o700)
        block = bytes(1 << 20)
        expected = {}
        for i in range(8):
            path = source / f"shard-{i}"
            with path.open("wb") as f:
                f.write(bytes([i]))
                f.truncate((64 << 20) + 1)
            h = hashlib.sha256(bytes([i]))
            for _ in range(64):
                h.update(block)
            expected[path.name] = h.hexdigest()
        serial, one = run(sys.argv[1], root, 1)
        parallel, many = run(sys.argv[1], root, None)
        if serial != parallel or {f["path"]: f["sha256"] for f in serial["files"]} != expected:
            raise RuntimeError("serial/parallel manifest or reference file hashes differ")
        print(f"CPU-active threads: serial={len(one)}, default={len(many)}; all 8 SHA-256 checksums match")
        if len(one) != 1 or len(many) < 2:
            raise RuntimeError("expected a single serial reader and multiple CPU-active default readers")


if __name__ == "__main__":
    main()
