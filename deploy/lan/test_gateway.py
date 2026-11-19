"""Isolated HTTPS smoke and regression tests; run with Python and CADDY set."""

import base64
import contextlib
import http.client
import http.server
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import ssl
import subprocess
import tempfile
import threading
import time
import unittest


class Backend(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: first\n\n")
            self.wfile.flush()
            if self.server.continue_stream.wait(timeout=5):
                self.wfile.write(b"data: second\n\n")
            return
        self.reply(200, json.dumps(dict(self.headers)).encode())

    def do_POST(self):
        if self.path == "/api/admin/actions":
            self.do_GET()
            return
        # Model the controller's no-body POST guard; the fake backend needs no
        # running controller/worker or data directory.
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{self.headers['Host']}":
            self.reply(403)
        elif (
            origin is None
            and self.headers.get("Sec-Fetch-Site") != "same-origin"
            and self.headers.get_content_type() != "application/json"
        ):
            self.reply(415)
        else:
            self.reply(204)

    def reply(self, status, body=b""):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


class UnixBackend(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class GatewayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resources = contextlib.ExitStack()
        cls.addClassCleanup(resources.close)
        state = Path(resources.enter_context(tempfile.TemporaryDirectory(
            prefix="brewery-gateway-", dir="/arboretum",
        )))
        (state / "tls").mkdir(mode=0o700)
        proxy_socket = state / "upstream.sock"
        backend = resources.enter_context(UnixBackend(str(proxy_socket), Backend))
        cls.continue_stream = backend.continue_stream = threading.Event()
        thread = threading.Thread(target=backend.serve_forever, daemon=True)
        thread.start()
        resources.callback(backend.shutdown)
        cls.port_file = Path(f"{proxy_socket}.port")
        cls.port_file.write_text("38175\n")

        with socket.socket(socket.AF_INET) as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            cls.port = port_socket.getsockname()[1]
        cls.host = f"127.0.0.1:{cls.port}"
        template = Path(__file__).with_name("Caddyfile").read_text()
        (state / "Caddyfile").write_text(template.replace(":8443", f":{cls.port}"))
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt",
             "ec_paramgen_curve:prime256v1", "-nodes", "-days", "1",
             "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
             "-keyout", "tls/server.key", "-out", "tls/server.crt"],
            cwd=state, check=True, capture_output=True,
        )
        cls.tls = ssl.create_default_context(cafile=state / "tls/server.crt")
        caddy = os.environ.get("CADDY", "caddy")
        password = secrets.token_urlsafe(24)
        digest = subprocess.run(
            [caddy, "hash-password", "--algorithm", "bcrypt"],
            input=f"{password}\n", text=True, check=True, capture_output=True,
        ).stdout.strip()
        (state / "basic-auth.caddy").touch(mode=0o600)
        (state / "basic-auth.caddy").write_text(f"brewery {digest}\n")
        cls.authorization = "Basic " + base64.b64encode(
            f"brewery:{password}".encode(),
        ).decode()
        log = resources.enter_context((state / "gateway.log").open("w+"))
        # Use the shipped endpoint and TLS defaults, not the operator's environment.
        env = {key: value for key, value in os.environ.items() if not key.startswith("BREWERY_")}
        proc = subprocess.Popen(
            [caddy, "run", "--config", "Caddyfile", "--adapter", "caddyfile"],
            cwd=state, stdout=log, stderr=log,
            env=dict(env, BREWERY_UPSTREAM_SOCKET=str(proxy_socket),
                     XDG_CONFIG_HOME=str(state), XDG_DATA_HOME=str(state)),
        )

        def stop():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        resources.callback(stop)
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", cls.port), timeout=0.2):
                    break
            except ConnectionRefusedError:
                if proc.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(f"Caddy startup failed:\n{log.read()}")
                time.sleep(0.05)
        else:
            raise RuntimeError("Caddy did not start")

    @contextlib.contextmanager
    def request(self, method="GET", path="/", headers=(), body=b"", *,
                auth=True, host=None, source_ip="127.0.0.1"):
        with contextlib.closing(http.client.HTTPSConnection(
            "127.0.0.1", self.port, context=self.tls, timeout=3,
            source_address=(source_ip, 0),
        )) as conn:
            conn.putrequest(method, path, skip_host=True)
            conn.putheader("Host", self.host if host is None else host)
            if auth:
                conn.putheader("Authorization", self.authorization)
            for name, value in headers:
                conn.putheader(name, value)
            if body:
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body)
            with conn.getresponse() as response:
                yield response

    def test_login(self):
        with self.request(auth=False) as response:
            self.assertEqual(response.status, 401)
            self.assertTrue(response.headers["WWW-Authenticate"].startswith("Basic "))
        with self.request(auth=False, headers=[
            ("Authorization", "Basic " + base64.b64encode(b"brewery:wrong").decode()),
        ]) as response:
            self.assertEqual(response.status, 401)
        with self.request() as response:
            self.assertEqual(response.status, 200)

    def test_direct_peer_and_exact_host(self):
        with self.request(source_ip="127.0.0.2", headers=[
            ("Forwarded", "for=127.0.0.1"),
            ("X-Forwarded-For", "127.0.0.1"),
        ]) as response:
            self.assertEqual(response.status, 403)
        for host in ("127.0.0.1", "localhost", f"127.0.0.1:0{self.port}"):
            with self.subTest(host=host), self.request(host=host) as response:
                self.assertEqual(response.status, 403)
        with self.request(path=f"https://{self.host}/", host="localhost") as response:
            self.assertEqual(response.status, 403)

    def test_disallowed_browser_signals(self):
        same = f"https://{self.host}"
        for name, values in [
            ("Origin", ("https://elsewhere.invalid",)),
            ("Origin", (same.replace("https:", "http:"),)),
            ("Origin", (same + "/",)),
            ("Origin", ("null",)),
            ("Origin", ("",)),
            ("Origin", (same, same)),
            ("Origin", (same, "https://elsewhere.invalid")),
            ("Sec-Fetch-Site", ("cross-site",)),
            ("Sec-Fetch-Site", ("same-site",)),
            ("Sec-Fetch-Site", ("",)),
            ("Sec-Fetch-Site", ("same-origin", "same-origin")),
            ("Sec-Fetch-Site", ("none, same-origin",)),
            ("Connection", ("keep-alive, oRiGiN",)),
            ("Connection", ("sec-fetch-site",)),
            ("Connection", ("content-type",)),
        ]:
            with self.subTest(name=name, values=values), self.request(
                path="/api/events", headers=[(name, value) for value in values],
            ) as response:
                self.assertEqual(response.status, 403)

    def test_browser_and_cli_post(self):
        for headers, status in [
            ([], 415),
            ([("Content-Type", "application/x-www-form-urlencoded")], 415),
            ([("Sec-Fetch-Site", "none")], 415),
            ([("Content-Type", "application/json")], 204),
            ([("Origin", f"https://{self.host}")], 204),
            ([("Sec-Fetch-Site", "same-origin")], 204),
            ([("Origin", "https://elsewhere.invalid"),
              ("Content-Type", "application/json")], 403),
            ([("Sec-Fetch-Site", "cross-site"),
              ("Content-Type", "application/json")], 403),
        ]:
            with self.subTest(headers=headers), self.request(
                "POST", "/api/jobs/test/cancel", headers=headers,
            ) as response:
                self.assertEqual(response.status, status)

    def test_upstream_headers(self):
        with self.request(headers=[
            ("Origin", f"https://{self.host}"),
            ("Sec-Fetch-Site", "same-origin"),
            ("Content-Type", "application/json; charset=utf-8"),
            ("Proxy-Authorization", "Basic x"),
            ("Cookie", "credential=x"),
            ("Forwarded", "for=192.0.2.1"),
            ("X-Forwarded-For", "192.0.2.1"),
            ("X-Forwarded-Extra", "client-supplied"),
            ("X-Real-IP", "192.0.2.1"),
        ]) as response:
            self.assertEqual(response.status, 200)
            headers = {key.lower(): value for key, value in json.loads(response.read()).items()}
            self.assertEqual(headers["host"], "127.0.0.1:38175")
            self.assertEqual(headers["origin"], "http://127.0.0.1:38175")

            self.assertEqual(headers["sec-fetch-site"], "same-origin")
            self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
            for name in ("authorization", "proxy-authorization", "cookie", "forwarded",
                         "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
                         "x-forwarded-extra", "x-real-ip"):
                self.assertNotIn(name, headers)
        with self.request() as response:
            self.assertEqual(response.status, 200)
            names = [name.lower() for name in json.loads(response.read())]
            for name in ("origin", "sec-fetch-site", "content-type"):
                self.assertNotIn(name, names)

    def test_admin_header_forwarded_only_on_exact_admin_operations(self):
        name = "X-Brewery-Admin-Key"
        for method, path, forwarded in (
            ("GET", "/api/state", False),
            ("GET", "/api/settings", False),
            ("GET", "/api/admin/state", True),
            ("GET", "/api/admin/state/extra", False),
            ("GET", "/api/admin/actions", False),
            ("POST", "/api/admin/actions", True),
        ):
            with self.subTest(method=method, path=path), self.request(
                method, path, headers=[(name, "stub"), ("Content-Type", "application/json")],
            ) as reply:
                self.assertEqual(reply.status, 200)
                headers = {k.lower(): v for k, v in json.loads(reply.read()).items()}
                self.assertEqual(headers.get(name.lower()), "stub" if forwarded else None)
                self.assertNotIn("authorization", headers)
        with self.request(path="/api/admin/state", auth=False, headers=[(name, "stub")]) as reply:
            self.assertEqual(reply.status, 401)

    def test_controller_port_update(self):
        original = self.port_file.read_text()
        try:
            for invalid in ("0", "65536", "tcp://listener", ""):
                with self.subTest(invalid=invalid):
                    self.port_file.write_text(invalid)
                    with self.request() as response:
                        self.assertEqual(response.status, 503)
            self.port_file.write_text("57139")
            with self.request(headers=[("Origin", f"https://{self.host}")]) as response:
                self.assertEqual(response.status, 200)
                headers = json.loads(response.read())
                self.assertEqual(headers["Host"], "127.0.0.1:57139")
                self.assertEqual(headers["Origin"], "http://127.0.0.1:57139")
        finally:
            self.port_file.write_text(original)

    def test_sse_delivers_each_event_without_waiting_for_close(self):
        try:
            with self.request(path="/api/events") as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Type"], "text/event-stream")
                self.assertEqual(response.read(len(b"data: first\n\n")), b"data: first\n\n")
                self.continue_stream.set()
                self.assertEqual(response.read(), b"data: second\n\n")
        finally:
            self.continue_stream.set()


if __name__ == "__main__":
    unittest.main()
