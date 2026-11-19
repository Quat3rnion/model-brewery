"""CLI regression tests. The loopback server provides test-only API records."""
import contextlib
import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


MATRIX = Path(__file__).with_name("release_matrix.py")
JOB = "43240cef-879b-4b71-9fa7-8ced52ec3e92"
UNSELECTED = "0048eba6-371a-4fe5-86f1-399edbf650f0"
MISSING = "795b7fc3-64c4-430f-9049-54d429bb1298"


@contextlib.contextmanager
def controller(routes):
    requests = []

    class Server(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            status, body = routes.get(self.path, (404, b'{"error":"job not found"}'))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with http.server.HTTPServer(("127.0.0.1", 0), Server) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            thread.join()


class ReleaseMatrix(unittest.TestCase):
    def run_matrix(self, url, exports, *args):
        return subprocess.run(
            [sys.executable, str(MATRIX), "--controller", url, "--exports", str(exports), *args],
            capture_output=True, text=True, timeout=15,
        )

    def test_empty_collection_queries_live_controller_but_does_not_select_jobs(self):
        state = json.dumps({"jobs": [{"id": UNSELECTED}]}).encode()
        with tempfile.TemporaryDirectory() as exports:
            with controller({"/api/state": (200, state)}) as (url, requests):
                result = self.run_matrix(url, exports)
            self.assertEqual(requests, ["/api/state"])
            self.assertFalse(list(Path(exports).iterdir()))
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["release_qualified"])
        self.assertEqual(
            {(item["target"], item["profile"], item["conversion"]["status"]) for item in report["paths"]},
            {
                ("gfx1151", "v2", "missing"), ("gfx1151", "ht43", "missing"),
                ("sm_89", "v2", "missing"), ("sm_89", "ht43", "missing"),
                ("sm_120", "v2", "missing"), ("sm_120", "ht43", "missing"),
            },
        )

    def test_running_and_failed_exports_remain_distinct_from_missing_on_resume(self):
        path = f"/api/jobs/{JOB}"
        missing = f"/api/jobs/{MISSING}"
        running = (
            json.dumps({
                "id": JOB, "kind": "convert", "host": "frame", "state": "running",
                "created_at": 1, "updated_at": 1, "has_result": False,
                "input_ids": [UNSELECTED], "options": {"profile": "v2"},
                "events": [{"state": "queued"}, {"state": "running"}],
            }, indent=2) + "\n"
        ).encode()
        failed = running.replace(b'"running"', b'"failed"')
        routes = {path: (200, running)}
        args = ("--row", f"sm_89/v2={JOB}", "--row", f"sm_120/v2={MISSING}")
        with tempfile.TemporaryDirectory() as exports:
            with controller(routes) as (url, requests):
                first = self.run_matrix(url, exports, *args)
                routes[path] = (200, failed)
                second = self.run_matrix(url, exports, *args)
                del routes[path]
                third = self.run_matrix(url, exports, *args)
            self.assertEqual(requests, [path, missing] * 3)
            self.assertCountEqual(
                [p.read_bytes() for p in Path(exports).glob(f"snapshot-*/{JOB}.json")],
                [running, failed],
            )
            self.assertFalse(list(Path(exports).rglob(f"{MISSING}.json")))
        for result, status in zip((first, second, third), ("running", "failed", "missing")):
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(f"{JOB}: {status}", result.stderr)
            self.assertIn(f"{MISSING}: missing (HTTP 404)", result.stderr)
            report = json.loads(result.stdout)
            assigned = {
                r["conversion"]["job_id"]: r["conversion"]["status"]
                for r in report["paths"] if "job_id" in r["conversion"]
            }
            self.assertEqual(assigned[MISSING], "missing")
            if status == "missing":
                self.assertEqual(assigned[JOB], "missing")
            else:
                self.assertIn(assigned[JOB], {status, "invalid", "pending"})

    def test_wrong_job_id_is_not_exported_or_used_as_baseline(self):
        body = json.dumps({"id": UNSELECTED, "state": "succeeded", "events": []}).encode()
        with tempfile.TemporaryDirectory() as exports:
            with controller({f"/api/jobs/{JOB}": (200, body)}) as (url, requests):
                result = self.run_matrix(url, exports, "--baseline-job-id", JOB)
            self.assertEqual(requests, [f"/api/jobs/{JOB}"])
            self.assertFalse(list(Path(exports).rglob("*.json")))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(f"{JOB}: unavailable", result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["baseline"]["status"], "missing")
        self.assertEqual(report["baseline"]["job_id"], JOB)


if __name__ == "__main__":
    unittest.main()
