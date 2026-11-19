#!/usr/bin/env python3
"""Check fake-token downloads and redirect denial with real pinned hfr."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import mkdtemp
from threading import Thread
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from managed_hfr_smoke import (
    COMMIT, FIRST, FILES, HOST, REPO, Hub, Smoke, controller, need, save, tree, worker_config,
)

FAKE = "hf_fixture_NotARealCredential"
UNTRUSTED = "brewery-smoke/redirect"
PAGINATE = "brewery-smoke/paginate"


class TokenHub(Hub):
    def handle(self):
        try:
            super().handle()
        except ConnectionResetError:
            pass

    def do_GET(self):
        url = urlsplit(self.path)
        repo = next((repo for repo in (REPO, UNTRUSTED, PAGINATE)
                     if url.path.startswith((f"/api/models/{repo}/", f"/{repo}/resolve/",
                                             f"/api/resolve-cache/models/{repo}/"))), None)
        auth = self.headers.get("Authorization") == f"Bearer {FAKE}"
        self.server.requests.append((self.command, url.path, auth))
        if not auth or not repo:
            return self.respond(401, b"authorization required", "text/plain")
        api = f"/api/models/{repo}"
        if url.path == f"{api}/revision/main":
            body = {"id": repo, "sha": COMMIT, "private": False, "gated": "auto"}
        elif url.path == f"{api}/tree/{COMMIT}":
            if repo == PAGINATE:
                return self.respond(200, b"[]", "application/json",
                                    Link=f"<{self.server.other}/tree>; rel=\"next\"")
            body = tree()
        elif url.path == f"/{repo}/resolve/{COMMIT}/{FIRST}":
            if repo == UNTRUSTED:
                return self.respond(307, b"", "text/plain", Location=f"{self.server.other}/payload")
            return self.respond(307, b"", "text/plain",
                                Location=f"/api/resolve-cache/models/{repo}/{COMMIT}/{FIRST}")
        elif url.path == f"/api/resolve-cache/models/{repo}/{COMMIT}/{FIRST}":
            data = FILES[FIRST]
            need(self.headers.get("Range") == f"bytes=0-{len(data)-1}", "range is not bounded")
            return self.respond(206, data, "application/octet-stream",
                                Content_Range=f"bytes 0-{len(data)-1}/{len(data)}")
        else:
            return self.respond(404, b"unexpected", "text/plain")
        return self.respond(200, json.dumps(body).encode(), "application/json")


class OtherOrigin(Hub):
    def do_GET(self):
        self.server.requests.append(self.command)
        return self.respond(403, b"must not be contacted", "text/plain")


def run(bin_dir, hfr):
    need(os.getpid() == 1 and socket.if_nameindex() == [(1, "lo")], "isolated loopback required")
    subprocess.run(["/usr/bin/ip", "link", "set", "lo", "up"], check=True, timeout=3)
    os.umask(0o077)
    root = Path(mkdtemp(prefix="managed-hfr-token.", dir="/arboretum"))
    for name in ("home", "tmp", "inputs", "credentials"):
        (root / name).mkdir()
    token_file = root / "credentials" / "hub"
    token_file.write_text(FAKE)
    with ThreadingHTTPServer(("127.0.0.1", 0), TokenHub) as hub, \
         ThreadingHTTPServer(("127.0.0.1", 0), OtherOrigin) as other:
        hub.other = f"http://127.0.0.1:{other.server_port}"
        for server in (hub, other):
            server.requests = []
            Thread(target=server.serve_forever, daemon=True).start()
        try:
            worker_config(root, bin_dir, hfr, hub.server_port)
            config = root / HOST / "worker.json"
            policy = json.loads(config.read_text())
            policy["source_roots"] = []
            policy["hf_download"]["token_file"] = str(token_file)
            save(config, policy)
            env = {"PATH": "/usr/bin:/bin", "HOME": str(root / "home"),
                   "TMPDIR": str(root / "tmp"), "HF_TOKEN": "hf_must_not_be_inherited"}
            with controller(root, bin_dir, env) as url:
                smoke = Smoke(root, url, hub)
                for label, repo, expected in (("gated", REPO, "succeeded"),
                                              ("redirect", UNTRUSTED, "failed"),
                                              ("pagination", PAGINATE, "failed")):
                    job = smoke.submit(label, {"repo_id": repo, "filename": FIRST})
                    result = smoke.wait(label, job, expected)
                    if expected == "succeeded":
                        smoke.verified(result, {}, FIRST)
                    need(FAKE not in json.dumps(result), f"credential in response for {label}")
                before = len(hub.requests)
                token_file.chmod(0o644)
                job = smoke.submit("unsafe-permissions", {"repo_id": REPO, "filename": FIRST})
                smoke.wait("unsafe-permissions", job, "failed")
                need(len(hub.requests) == before, "unsafe token permissions did not fail closed")
                token_file.unlink()
                job = smoke.submit("removed", {"repo_id": REPO, "filename": FIRST})
                smoke.wait("removed", job, "failed")
                need(any(not auth for _, _, auth in hub.requests), "empty credential store did not fall back to anonymous")
            need(not other.requests, "authenticated Hub attempted to contact a different origin")
            need(FAKE not in "\n".join(p.read_text(errors="ignore") for p in (root / HOST / "jobs").rglob("*") if p.is_file()),
                 "credential persisted to worker job files or logs")
        finally:
            hub.shutdown()
            other.shutdown()
    print(f"PASS: gated model, redirect and pagination denied, token removal and mode checks; {root}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--isolated":
        run(*(Path(p).resolve(strict=True) for p in sys.argv[2:]))
    elif len(sys.argv) == 3:
        raise SystemExit(subprocess.run([
            "/usr/bin/unshare", "--user", "--map-root-user", "--net", "--pid", "--fork", "--kill-child",
            "--mount-proc", sys.executable, "-E", "-s", "-B", str(Path(__file__).resolve()),
            "--isolated", *sys.argv[1:],
        ], env={"PATH": "/usr/bin:/bin"}).returncode)
    else:
        raise SystemExit(f"usage: {sys.argv[0]} BREWERY_BIN_DIR PINNED_HFR_ELF")
