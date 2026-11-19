"""Exercise the observer against real procfs with small local processes."""

import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unittest

from observe_progress import Layout, observe, watch


JOB = "64a9c1d5-6269-4ec7-b553-8a6696407642"
FIXTURE = r"""
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    const char *worker = getenv("FIXTURE_WORKER"), *engine = getenv("FIXTURE_ENGINE");
    if (argc == 1 && getenv("FIXTURE_ENGINE_RUN") == NULL) {
        const char *dispatch = getenv("BREWERY_DISPATCH");
        if (!fork()) execl(dispatch, dispatch, "start", "local-sm120", getenv("FIXTURE_JOB"), NULL);
        for (;;) pause();
    }
    if (argc == 4) {
        if (!fork()) execl(worker, worker, "rpc", NULL);
    } else if (argc == 3) {
        if (!fork()) {
            int fd = open(getenv("FIXTURE_ENGINE_INPUT"), O_RDONLY);
            if (fd < 0 || dup2(fd, 0) < 0 || chdir(getenv("FIXTURE_WORK"))) return 2;
            close(fd);
            setenv("FIXTURE_ENGINE_RUN", "1", 1);
            execl(engine, engine, NULL);
        }
    } else {
        int fd = open(getenv("FIXTURE_INPUT"), O_RDONLY);
        char bytes[17];
        const char *prefix = getenv("FIXTURE_PREFIX");
        if (prefix) {
            char buf[256];
            int previous = open(prefix, O_RDONLY);
            if (previous < 0) return 4;
            while (read(previous, buf, sizeof(buf)) > 0) {}
            close(previous);
        }
        if (fd < 0 || read(fd, bytes, sizeof(bytes)) != sizeof(bytes)) return 3;
        const char *seek = getenv("FIXTURE_SEEK");
        if (seek) lseek(fd, atol(seek), SEEK_SET);
        puts("ready");
        fflush(stdout);
    }
    for (;;) pause();
}
"""


class TestObserve(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        run = self.root / "local-sm120"
        self.source = self.root / "published"
        self.receipt = self.root / "receipt.json"
        self.job = run / "worker" / "jobs" / JOB
        for path in (self.source, run / "controller", run / "bin", self.job / "work"):
            path.mkdir(mode=0o700, parents=True)
        self.path = self.source / "z-first.safetensors"
        self.path.write_bytes(b"x" * 128)
        (self.source / "a-last.safetensors").write_bytes(b"y" * 128)
        self.write(self.receipt, {"schema": "model-brewery.source-publication.v1",
                                 "model_dir": str(self.source),
                                 "files": {p.name: {"size": 128} for p in self.source.iterdir()}})
        self.layout = Layout(run, self.source, self.receipt,
                             hashlib.sha256(self.receipt.read_bytes()).hexdigest())
        binary = run / "bin/brewery"
        subprocess.run(["cc", "-x", "c", "-", "-o", binary], input=FIXTURE,
                       text=True, check=True)
        for name in ("brewery-dispatch", "brewery-worker", "brewery-engine"):
            shutil.copyfile(binary, binary.with_name(name))
            binary.with_name(name).chmod(0o700)
        self.write(run / "dispatch.json", {"hosts": {"local-sm120": {
            "local": True, "config": str(run / "worker.json"),
            "worker": str(run / "bin/brewery-worker")}}})
        self.write(run / "worker.json", {"root": str(run / "worker"),
                                        "source_roots": [str(self.source)],
                                        "engine": str(run / "bin/brewery-engine")})
        self.state("queued")
        self.env = {**os.environ, "BREWERY_DATA": str(run / "controller"),
                    "BREWERY_DISPATCH": str(run / "bin/brewery-dispatch"),
                    "BREWERY_DISPATCH_CONFIG": str(run / "dispatch.json"),
                    "BREWERY_WORKER_CONFIG": str(run / "worker.json"),
                    "FIXTURE_WORKER": str(run / "bin/brewery-worker"),
                    "FIXTURE_ENGINE": str(run / "bin/brewery-engine"),
                    "FIXTURE_INPUT": str(self.path), "FIXTURE_JOB": JOB,
                    "FIXTURE_WORK": str(self.job / "work"),
                    "FIXTURE_ENGINE_INPUT": str(self.job / "engine-input.json")}

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))
        path.chmod(0o600)

    def state(self, state):
        job = {"kind": "inspect", "inputs": [{"host": "local-sm120", "kind": "hf",
                                             "path": str(self.source)}]}
        self.write(self.job / "state.json", {"action": "start", "cancel": False,
                                           "job": job, "reply": {"state": state}})
        self.write(self.job / "engine-input.json", job)

    def start(self, binary="brewery", args=(), ready=True, **env):
        proc = self.enterContext(subprocess.Popen(
            [self.layout.run / "bin" / binary, *args], env={**self.env, **env},
            start_new_session=True, stdout=subprocess.PIPE, text=True,
        ))
        self.addCleanup(self.stop, proc)
        if ready:
            self.assertTrue(select.select([proc.stdout], [], [], 3)[0], "fixture did not start")
            self.assertEqual(proc.stdout.readline(), "ready\n")
        return proc

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)

    def test_reports_offset_for_exact_worker_chain_not_a_pass_estimate(self):
        self.start()
        snapshot = observe(self.layout)
        self.assertEqual(snapshot["job_id"], JOB)
        self.assertEqual(snapshot["schema"], "brewery-live-progress-v1")
        self.assertEqual(snapshot["progress_detail"], {
            "phase": "Resolving input (observed file)", "unit": "bytes",
            "completed": 17, "total": 128, "current_file": "z-first.safetensors",
            "updated_at_ms": snapshot["observed_at_ms"],
        })
        self.path.write_bytes(b"x" * 127)
        self.assertIsNone(observe(self.layout))

    def test_pass_estimate_requires_pinned_binary_and_matches_fd_offset_to_rchar(self):
        first, second = self.source.iterdir()
        with first.open("wb") as file:
            file.truncate(2 * 1024 * 1024)
        for path in (first, second):
            path.chmod(0o400)
        self.write(self.receipt, {"schema": "model-brewery.source-publication.v1",
                                 "model_dir": str(self.source),
                                 "files": {p.name: {"size": p.stat().st_size}
                                           for p in (first, second)}})
        binary = self.layout.exe("brewery-worker")
        self.layout = Layout(self.layout.run, self.source, self.receipt,
                             hashlib.sha256(self.receipt.read_bytes()).hexdigest(),
                             hashlib.sha256(binary.read_bytes()).hexdigest())
        proc = self.start(FIXTURE_PREFIX=str(first), FIXTURE_INPUT=str(second))
        detail = observe(self.layout)["progress_detail"]
        self.assertTrue(detail["estimated"])
        self.assertEqual(detail["completed"], 2 * 1024 * 1024 + 17)
        self.assertEqual(detail["total"], 2 * 1024 * 1024 + 128)
        self.stop(proc)

        self.start(FIXTURE_INPUT=str(second))
        detail = observe(self.layout)["progress_detail"]
        self.assertNotIn("estimated", detail)
        self.assertEqual(detail["completed"], 17)
        self.assertEqual(detail["total"], 128)

    def test_detached_engine_is_tied_to_job_workdir_and_stdin(self):
        self.start(ready=False)
        self.state("running")
        self.start("brewery-worker", ("supervise", JOB), FIXTURE_WORK=str(self.job))
        self.assertIsNone(observe(self.layout))
        self.start("brewery-worker", ("supervise", JOB))
        detail = observe(self.layout)["progress_detail"]
        self.assertEqual(detail["phase"], "Engine input read (observed file)")
        self.assertEqual(detail["completed"], 17)

    def test_watch_withdraws_unchanged_or_inactive_observation(self):
        proc = self.start()
        dist = self.root / "dist"
        dist.mkdir()
        output = dist / "live-progress/current.json"
        stop = threading.Event()
        thread = threading.Thread(target=watch, args=(output, stop), kwargs={
            "layout": self.layout, "interval": 0.02, "stale": 0.5})
        thread.start()
        try:
            self.wait_until(output.exists)
            self.assertEqual(json.loads(output.read_text())["job_id"], JOB)
            self.wait_until(lambda: not output.exists())
            self.stop(proc)
            self.start()
            self.wait_until(output.exists)
        finally:
            stop.set()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(output.exists())

    def wait_until(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            self.assertLess(time.monotonic(), deadline, "observer timed out")
            time.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
