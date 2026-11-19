"""Executable interface tests against pinned upstream CLIs, not model benchmarks."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

HERE = Path(__file__).resolve().parent
RECIPES = {
    "tool-eval-bench": ("3.0.0", "tool-eval-safe-v1"),
    "betterbench": ("0.6.1", "betterbench-decode-chat-v1"),
}


class MockEndpoint(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body))
        if self.headers.get("X-Config-Probe"):
            self.send_error(400, "Operator configuration escaped into the benchmark")
            return
        if self.server.mode == "unauthorized":
            self.send_error(401)
            return
        if self.path != "/v1/chat/completions" or self.server.mode == "failed":
            self.send_error(503)
            return
        if self.server.mode == "redirect":
            self.send_response(307)
            self.send_header("Location", "/other")
            self.end_headers()
            return
        if self.server.mode in ("tool", "tool_mock"):
            # This model declines to use any tools. That is a measured quality
            # failure, not an upstream execution failure.
            message = {"role": "assistant", "content": "I cannot answer."}
            finish_reason = "stop"
            if self.server.mode == "tool_mock" and body["messages"][-1]["role"] != "tool":
                message = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "call-1", "type": "function",
                        "function": {"name": "run_code",
                                     "arguments": json.dumps({"code": "print(1)"})},
                    }],
                }
                finish_reason = "tool_calls"
            data = {"choices": [{"message": message, "finish_reason": finish_reason}]}
            if not body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
                return
            if "tool_calls" in message:
                message["tool_calls"][0]["index"] = 0
                del message["content"]
            events = [{"choices": [{"delta": message, "finish_reason": finish_reason}]}]
        else:
            events = [
                {"choices": [{"delta": {"content": text}}]}
                for text in ("One", " two", " three")
            ] + [{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        if self.server.mode != "no_usage":
            events.append({"choices": [], "usage": {
                "completion_tokens": 3, "prompt_tokens": 10, "total_tokens": 13,
            }})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class Adapters(unittest.TestCase):
    def invoke(self, benchmark, mode, *, argv=(), **overrides):
        version, protocol = RECIPES[benchmark]
        with ThreadingHTTPServer(("127.0.0.1", 0), MockEndpoint) as server:
            server.requests, server.mode = [], mode
            listener = Thread(target=server.serve_forever, daemon=True)
            listener.start()
            try:
                request = dict(benchmark=benchmark, target="local-smoke",
                               endpoint=f"http://127.0.0.1:{server.server_port}",
                               model="mock-model", version=version, protocol=protocol)
                request.update(overrides)
                with TemporaryDirectory(dir=os.environ.get("TMPDIR", "/arboretum")) as temp:
                    # The tool CLI normally walks CWD ancestors for dotenv.
                    (Path(temp) / ".env").write_text("TOOL_EVAL_HEADERS=X-Config-Probe=dotenv\n")
                    cwd = Path(temp) / "run"
                    cwd.mkdir(mode=0o700)
                    result = subprocess.run(
                        [sys.executable, "-EBs", str(HERE / benchmark), *argv],
                        input=json.dumps(request), capture_output=True, text=True,
                        cwd=cwd, timeout=30,
                    )
                    raw = Path(cwd) / "raw.json"
                    raw = json.loads(raw.read_text()) if raw.exists() else None
                    return result, raw, server.requests
            finally:
                server.shutdown()
                listener.join()

    def test_tool_failures_to_call_are_counted_as_quality_not_infrastructure(self):
        result, raw, requests = self.invoke("tool-eval-bench", "tool")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            "metrics": [{"name": "mock_tool_subset_score", "value": 0,
                         "unit": "%", "direction": "higher"}], "samples": 3,
        })
        self.assertEqual({r["scenario_id"] for r in raw["scores"]["scenario_results"]},
                         {"TC-01", "TC-02", "TC-04"})
        self.assertEqual([path for path, _ in requests], ["/v1/chat/completions"] * 3)
        self.assertTrue(all(body["tools"] for _, body in requests))

    def test_betterbench_emits_measured_streaming_medians(self):
        result, raw, requests = self.invoke("betterbench", "stream")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["samples"], 4)
        self.assertEqual(raw["betterbench_version"], "0.6.1")
        self.assertEqual(set(raw["single_stream"]), {"chat"})
        self.assertNotIn("prefill", raw)
        self.assertNotIn("concurrency", raw)
        self.assertEqual([path for path, _ in requests], ["/v1/chat/completions"] * 5)
        self.assertTrue(all(body["stream"] and body["stream_options"]["include_usage"]
                            and "tools" not in body for _, body in requests))
        self.assertEqual({(m["name"], m["unit"], m["direction"]) for m in report["metrics"]},
                         {("ttft_p50", "ms", "lower"), ("decode_tps_p50", "tokens/s", "higher")})
        self.assertTrue(all(m["value"] > 0 for m in report["metrics"]))

    def test_tool_code_requests_get_an_in_memory_disabled_result(self):
        result, _, requests = self.invoke("tool-eval-bench", "tool_mock")
        self.assertEqual(result.returncode, 0, result.stderr)
        mock_results = [m["content"] for _, body in requests for m in body["messages"]
                        if m["role"] == "tool"]
        self.assertEqual(len(mock_results), 3)
        for mock_result in mock_results:
            self.assertIn("Code execution is disabled in benchmark mocks.",
                          json.loads(mock_result)["error"])

    def test_missing_usage_is_not_reported_as_tokens_per_second(self):
        result, raw, _ = self.invoke("betterbench", "no_usage")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("reported usage", result.stderr)
        self.assertIsNotNone(raw)

    def test_redirects_fail_without_following_location(self):
        for benchmark in RECIPES:
            with self.subTest(benchmark=benchmark):
                result, _, requests = self.invoke(benchmark, "redirect")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertGreater(len(requests), 0)
                self.assertEqual({path for path, _ in requests}, {"/v1/chat/completions"})

    def test_endpoint_failure_never_emits_a_score(self):
        for benchmark in RECIPES:
            with self.subTest(benchmark=benchmark):
                result, raw, requests = self.invoke(benchmark, "unauthorized")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIsNotNone(raw)
                self.assertGreater(len(requests), 0)

    def test_rejects_unpinned_recipe_and_user_supplied_arguments(self):
        for benchmark in RECIPES:
            for overrides in ({"version": "5.0"}, {"protocol": "openai"},
                              {"endpoint": "http://localhost:1234"},
                              {"argv": ["--help"]}, {"extra": "value"}):
                with self.subTest(benchmark=benchmark, overrides=overrides):
                    result, raw, requests = self.invoke(benchmark, "failed", **overrides)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIsNone(raw)
                    self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
