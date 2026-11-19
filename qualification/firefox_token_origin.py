#!/usr/bin/env python3
"""Check Firefox's token PUT against a synthetic loopback Origin guard.

From the repository root, run in a loopback-only network namespace:
    unshare -Urn sh -c 'ip link set lo up && python3 -B qualification/firefox_token_origin.py'
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
import shutil
import signal
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.parse import parse_qs, urlsplit


TOKEN_PATH = "/api/settings/backend-hf/hf-token"
JS = b"""
(async () => {
  const policy = {cache: 'no-store', credentials: 'same-origin',
                  redirect: 'error', mode: 'same-origin'};
  for (const [label, options] of [
    ['inherited', {}], ['same-origin', {referrerPolicy: 'same-origin'}],
  ]) {
    let status = 'fetch-error';
    try {
      const response = await fetch('/api/settings/backend-hf/hf-token?' + label, {
        ...policy, ...options, method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({token: 'not-a-real-token'}),
      });
      status = response.status;
    } finally {
      await fetch('/result?' + new URLSearchParams({label, status}), policy);
    }
  }
})();
"""


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, status, content_type="text/plain", body=b""):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlsplit(self.path)
        if route.path == "/":
            self.reply(200, "text/html", b"<!doctype html><script src='/check.js' defer></script>")
        elif route.path == "/check.js":
            self.reply(200, "text/javascript", JS)
        elif route.path == "/result":
            query = parse_qs(route.query)
            self.reply(204)
            self.server.results.put((query["label"][0], query["status"][0]))
        else:
            self.reply(404)

    def do_PUT(self):
        route = urlsplit(self.path)
        if route.path != TOKEN_PATH or route.query not in ("inherited", "same-origin"):
            self.reply(404)
            return
        self.rfile.read(int(self.headers["Content-Length"]))
        origin = self.headers.get("Origin")
        site = self.headers.get("Sec-Fetch-Site")
        # Match the controller's Origin/Fetch Metadata checks; never store a token.
        accepted = origin in (None, self.server.origin) and site in (None, "same-origin", "none")
        self.server.headers.put((route.query, {"Origin": origin, "Sec-Fetch-Site": site}))
        self.reply(204 if accepted else 403)


def main():
    firefox = shutil.which("firefox")
    if not firefox:
        raise SystemExit("Firefox is required on PATH")
    with TemporaryDirectory(prefix="firefox-token-origin.", dir="/arboretum") as name:
        root = Path(name)
        profile = root / "profile"
        profile.mkdir(mode=0o700)
        env = {"PATH": os.environ["PATH"], "LANG": "C.UTF-8", "HOME": name, "TMPDIR": name,
               "XDG_RUNTIME_DIR": name, "MOZ_CRASHREPORTER_DISABLE": "1"}
        with ThreadingHTTPServer(("127.0.0.1", 0), Fixture) as server:
            server.origin = f"http://127.0.0.1:{server.server_port}"
            server.results = Queue()
            server.headers = Queue()
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with (root / "firefox.log").open("wb") as log, subprocess.Popen(
                    [firefox, "--headless", "--no-remote", "--profile", str(profile), server.origin],
                    env=env, stdout=log, stderr=log, start_new_session=True,
                ) as browser:
                    try:
                        statuses = dict(server.results.get(timeout=15) for _ in range(2))
                    except Empty:
                        raise SystemExit("Firefox did not finish the probe:\n"
                                         + (root / "firefox.log").read_text(errors="replace")) from None
                    finally:
                        try:
                            os.killpg(browser.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                observed = dict(server.headers.get_nowait() for _ in range(server.headers.qsize()))
                print(json.dumps({"headers": observed, "status": statuses}, indent=2))
                expected = {
                    "inherited": {"Origin": "null", "Sec-Fetch-Site": "same-origin"},
                    "same-origin": {"Origin": server.origin, "Sec-Fetch-Site": "same-origin"},
                }
                if observed != expected or statuses != {"inherited": "403", "same-origin": "204"}:
                    raise SystemExit("Expected a rejected null Origin and an accepted same-origin override")
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    main()
