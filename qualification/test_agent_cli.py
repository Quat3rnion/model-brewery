"""Public agent-client behavior with a disposable loopback HTTP server."""
import contextlib
import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "tools/brewery-agent"


class AgentCLI(unittest.TestCase):
    @contextlib.contextmanager
    def server(self, content=b"{}", mime="application/json", status=200):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def reply(self):
                data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                calls.append((self.command, self.path, dict(self.headers), data))
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.end_headers()
                self.wfile.write(content)
            do_GET = do_POST = reply

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
            thread = threading.Thread(target=http.serve_forever)
            thread.start()
            try:
                yield f"http://127.0.0.1:{http.server_port}", calls
            finally:
                http.shutdown()
                thread.join()

    def invoke(self, url, *args, data=""):
        return subprocess.run([sys.executable, CLI, "--url", url, *args], input=data, text=True,
                              capture_output=True, timeout=7)

    def test_create_sends_one_key_and_json(self):
        with self.server(b'{"id":"receipt"}') as (url, calls):
            out = self.invoke(url, "create", "prompts", "--key", "test.v1", data='{"name":"x","text":"y"}')
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(json.loads(out.stdout), {"id": "receipt"})
            [(method, path, headers, body)] = calls
            self.assertEqual((method, path), ("POST", "/api/prompts"))
            self.assertEqual({k.lower(): v for k, v in headers.items()}["idempotency-key"], "test.v1")
            self.assertEqual(json.loads(body), {"name": "x", "text": "y"})

    def test_workflow_commands_do_not_chain_child_jobs(self):
        ident = "6f36d330-f5ed-48fe-86b3-4b7df5c7296d"
        with self.server() as (url, calls):
            for args in (("create", "conversions", "--key", "example"), ("conversions",),
                         ("conversion", ident), ("cancel-conversion", ident)):
                out = self.invoke(url, *args, data='{"input_ids":["a"]}')
                self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual([(method, path) for method, path, _, _ in calls], [
            ("POST", "/api/conversions"), ("GET", "/api/conversions"),
            ("GET", f"/api/conversions/{ident}"), ("POST", f"/api/conversions/{ident}/cancel")])

    def test_resume_delivers_only_complete_frame_and_marks_resync(self):
        for kind, payload, rc in [("change", '{"id":9,"kind":"job.updated","entity_id":"j","state":"running"}', 0),
                                  ("resync", '{"reason":"ahead","event_cursor":3}', 3)]:
            marker = "id: 9\r\n" if kind == "change" else ""
            body = f": heartbeat\r\n\r\n{marker}event: {kind}\r\ndata: {payload}\r\n\r\n".encode()
            with self.server(body, "text/event-stream") as (url, _):
                out = self.invoke(url, "events", "--after", "8", "--limit", "1")
                self.assertEqual(out.returncode, rc, out.stderr)
                self.assertEqual(json.loads(out.stdout)["data"], json.loads(payload))
        with self.server(b"event: change\nid: 9\ndata: {}", "text/event-stream") as (url, _):
            out = self.invoke(url, "events", "--after", "8", "--limit", "1")
            self.assertNotEqual(out.returncode, 0)
            self.assertEqual(out.stdout, "")

    def test_no_external_destination_or_redirect(self):
        out = self.invoke("https://example.org", "state")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("loopback", out.stderr)

    def test_progress_reports_only_the_active_phase_not_a_job_percentage(self):
        ident = "6f36d330-f5ed-48fe-86b3-4b7df5c7296d"
        sample = {"phase": "hashing_input", "completed": 51, "total": 100, "unit": "bytes",
                  "updated_at_ms": 1}
        for state in ("running", "succeeded"):
            with self.server(json.dumps({"id": ident, "state": state, "progress": "Measured",
                                         "progress_detail": sample}).encode()) as (url, calls):
                out = self.invoke(url, "progress", ident)
                self.assertEqual(out.returncode, 0, out.stderr)
                expected = sample if state == "running" else None
                self.assertEqual(json.loads(out.stdout)["progress_detail"], expected)
                self.assertEqual(len(calls), 1)

    def test_wait_returns_terminal_detail_without_submitting_or_cancelling(self):
        ident = "6f36d330-f5ed-48fe-86b3-4b7df5c7296d"
        for state, code in [("succeeded", 0), ("failed", 4), ("running", 5)]:
            detail = {"id": ident, "state": state, "result": {"test": True}}
            with self.server(json.dumps(detail).encode()) as (url, calls):
                out = self.invoke(url, "wait", ident, "--timeout", "1")
                self.assertEqual(out.returncode, code, out.stderr)
                self.assertTrue(all((m, path) == ("GET", f"/api/jobs/{ident}") for m, path, _, _ in calls))
                if state != "running":
                    self.assertEqual(json.loads(out.stdout), detail)
