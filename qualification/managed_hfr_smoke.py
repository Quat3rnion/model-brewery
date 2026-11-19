#!/usr/bin/env python3
"""Exercise managed Hugging Face downloads on real services with a loopback Hub.

Requires Linux user/PID/network namespaces, the four built Brewery binaries,
and a trusted pinned Rust hfr build. No public Hub, credentials, or model load.
Writes controller, job, Hub-request and result evidence to a fresh /arboretum root.
"""

import argparse
from contextlib import contextmanager
from hashlib import file_digest, sha1, sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
from tempfile import mkdtemp
from threading import Event, Thread
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener
from uuid import UUID

from local_sm120 import NoRedirect, need, save

HOST = "fixture"
OFFLINE = "conversion"
COMMIT = "514255873be215c799bca92d4086291349fb12b1"
REPO = "brewery-smoke/indexed"
BAD = "brewery-smoke/corrupt"
CANCEL = "brewery-smoke/cancel"
FIRST = "model-00001-of-00002.safetensors"
SECOND = "model-00002-of-00002.safetensors"
TERMINAL = ("succeeded", "failed", "cancelled")
BINARIES = ("brewery", "brewery-dispatch", "brewery-worker", "brewery-engine")
CONVERSION = ("inspect", "plan", "convert", "verify")


def st_tensor(name, value):
    header = json.dumps({name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", value)


FILES = {
    "config.json": b'{"model_type":"fixture"}\n',
    "model.safetensors.index.json": json.dumps({
        "metadata": {"total_size": 8}, "weight_map": {"x.weight": FIRST, "y.weight": SECOND},
    }).encode(),
    FIRST: st_tensor("x.weight", 1.25),
    SECOND: st_tensor("y.weight", -0.5),
}


def git_oid(data):
    return sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def tree():
    result = []
    for name, data in FILES.items():
        entry = {"type": "file", "path": name, "size": len(data), "oid": git_oid(data)}
        if name.endswith(".safetensors"):
            pointer = (b"version https://git-lfs.github.com/spec/v1\n"
                       + f"oid sha256:{sha256(data).hexdigest()}\nsize {len(data)}\n".encode())
            entry.update(oid=git_oid(pointer), lfs={"oid": sha256(data).hexdigest(), "size": len(data)})
        result.append(entry)
    return result


class Hub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def respond(self, status, body, kind, **headers):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected if a held transfer is cancelled.

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        server = self.server
        url = urlsplit(self.path)
        range_header = self.headers.get("Range")
        auth = "Authorization" in self.headers or "Cookie" in self.headers
        server.requests.append({"method": self.command, "path": url.path,
                                "query": url.query, "range": range_header, "credential": auth})
        if auth or len(server.requests) > 128:
            return self.respond(403, b"unexpected request", "text/plain")
        for repo in (REPO, BAD, CANCEL):
            prefix = f"/api/models/{repo}"
            if url.path in (f"{prefix}/revision/main", f"{prefix}/revision/{COMMIT}"):
                body = {"id": repo, "sha": COMMIT, "private": False, "gated": False}
            elif url.path == f"{prefix}/tree/{COMMIT}":
                body = tree()
            else:
                path = f"/{repo}/resolve/{COMMIT}/"
                if not url.path.startswith(path) or url.path[len(path):] not in FILES:
                    continue
                name = url.path[len(path):]
                data = FILES[name]
                if self.command == "HEAD":
                    return self.respond(200, data, "application/octet-stream", Accept_Ranges="bytes")
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header or "")
                if not match:
                    return self.respond(416, b"expected one bounded range", "text/plain")
                start, end = map(int, match.groups())
                if not 0 <= start <= end < len(data):
                    return self.respond(416, b"range outside fixture", "text/plain")
                if repo == BAD and name == FIRST:
                    data = data[:-1] + bytes([data[-1] ^ 1])  # Same length, wrong LFS SHA-256.
                if repo in server.holds:
                    entered, release = server.holds[repo]
                    entered.set()
                    if not release.wait(20):
                        return self.respond(503, b"fixture hold timed out", "text/plain")
                return self.respond(206, data[start:end + 1], "application/octet-stream",
                                    Content_Range=f"bytes {start}-{end}/{len(data)}")
            return self.respond(200, json.dumps(body).encode(), "application/json")
        return self.respond(404, b"not found", "text/plain")


@contextmanager
def hub(root):
    with ThreadingHTTPServer(("127.0.0.1", 0), Hub) as server:
        server.requests = []
        server.holds = {repo: (Event(), Event()) for repo in (REPO, CANCEL)}
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            for _, release in server.holds.values():
                release.set()
            server.shutdown()
            thread.join(timeout=3)
            save(root / "hub-requests.json", server.requests)


def worker_config(root, bins, hfr, fixture_port):
    hosts = {}
    for name in (HOST, OFFLINE):
        work = root / name
        work.mkdir()
        config = work / "worker.json"
        worker = {"root": str(work), "source_roots": [str(root / "inputs")],
                  "engine": str(bins / "brewery-engine")}
        host = {"local": True, "worker": str(bins / "brewery-worker"), "config": str(config)}
        if name == HOST:
            worker["hf_download"] = {"hfr": str(hfr), "max_bytes": 1 << 20, "fixture_port": fixture_port}
            host["capabilities"] = {"hf_download": True, **dict.fromkeys(CONVERSION, False)}
        save(config, worker)
        hosts[name] = host
    save(root / "dispatch.json", {"hosts": hosts})


@contextmanager
def controller(root, bins, env):
    config = {**env, "BREWERY_DATA": str(root / "controller"), "BREWERY_LISTEN": "127.0.0.1:0",
              "BREWERY_DISPATCH": str(bins / "brewery-dispatch"),
              "BREWERY_DISPATCH_CONFIG": str(root / "dispatch.json"),
              "BREWERY_DISPATCH_TIMEOUT_SECS": "12", "BREWERY_POLL_MS": "100"}
    log = root / "controller.log"
    with log.open("wb") as stream, subprocess.Popen(
        [bins / "brewery"], cwd=root, env=config, stdout=stream, stderr=stream,
    ) as proc:
        try:
            end = time.monotonic() + 8
            while time.monotonic() < end and proc.poll() is None:
                match = re.search(r"^brewery listening on (http://127\.0\.0\.1:\d+)$",
                                  log.read_text(), re.M)
                if match:
                    yield match[1]
                    return
                time.sleep(0.05)
            raise ValueError(f"controller did not bind; see {log}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()


class Smoke:
    def __init__(self, root, url, server):
        self.root = root
        self.url = url
        self.server = server
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def call(self, route, body=None, status=200, key=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Idempotency-Key"] = key
        req = Request(f"{self.url}/api/{route}", None if body is None else json.dumps(body).encode(),
                      headers)
        try:
            reply = self.opener.open(req, timeout=4)
        except HTTPError as error:
            reply = error
        with reply:
            raw = reply.read((1 << 20) + 1)
            need(len(raw) <= 1 << 20 and reply.code == status,
                 f"{route}: expected HTTP {status}, got {reply.code}: {raw[:512]!r}")
        return json.loads(raw)

    def models(self):
        return {m["id"]: m for m in self.call("state")["models"]}

    def submit(self, label, options, host=HOST, status=201):
        body = {"kind": "hf_download", "host": host, "input_ids": [], "options": options}
        save(self.root / f"{label}-request.json", body)
        result = self.call("jobs", body, status=status, key=f"managed-hfr.{label}")
        if status in (201, 200):
            ident = result["id"]
            need(str(UUID(ident)) == ident, "invalid job ID")
            return ident
        save(self.root / f"{label}-response.json", result)

    def wait(self, label, ident, expected):
        until = time.monotonic() + 25
        while time.monotonic() < until:
            detail = self.call(f"jobs/{ident}")
            if detail["state"] in TERMINAL:
                save(self.root / f"{label}-job.json", detail)
                need(detail["state"] == expected, f"{label}: expected {expected}, got {detail}")
                return detail
            time.sleep(0.08)
        raise ValueError(f"{label} {ident}: job did not reach {expected}")

    def no_publication(self, detail, models_before):
        ident = detail["id"]
        result = detail.get("result") or {}
        need(not any(result.get(k) for k in ("artifact", "library_model_id", "verified")),
             f"{ident}: unverified result claims an artifact")
        path = self.root / HOST / "jobs" / ident / "work/download"
        need(not path.exists() and not path.is_symlink(), f"{ident}: unexpected published output")
        need(self.models() == models_before, f"{ident}: library changed without a verified result")

    def verified(self, detail, models_before, filename=None):
        ident = detail["id"]
        result = detail["result"]
        need(result["kind"] == "hf_download" and result["verified"] is True
             and result["repo_id"] == REPO and result["commit"] == COMMIT
             and result.get("filename") == filename, f"{ident}: wrong verified identity")
        kind, expected_files = ("hf", FILES) if filename is None else ("safetensors", {filename: FILES[filename]})
        download = self.root / HOST / "jobs" / ident / "work/download"
        expected = download if filename is None else download / filename
        artifact = result["artifact"]
        need(artifact["path"] == str(expected) and artifact["kind"] == kind,
             f"{ident}: unexpected artifact location or kind")
        need(artifact["bytes"] == sum(map(len, expected_files.values())), f"{ident}: wrong artifact size")
        need(download.resolve() == download and download.is_dir(), f"{ident}: noncanonical output directory")
        observed = {}
        for p in download.iterdir():
            need(p.name in expected_files and not p.is_symlink() and p.is_file()
                 and p.stat().st_size == len(expected_files[p.name]), f"{ident}: unexpected output {p}")
            observed[p.name] = p.read_bytes()
        need(observed == expected_files, f"{ident}: wrong file inventory or content")
        expected_digest = (
            sha256(expected_files[filename]).hexdigest() if filename is not None else
            sha256("".join(f"{name}\0{len(data)}\0{sha256(data).hexdigest()}\n"
                           for name, data in sorted(expected_files.items())).encode()).hexdigest()
        )
        need(artifact["sha256"] == expected_digest, f"{ident}: wrong artifact digest")
        models = self.models()
        model_id = result["library_model_id"]
        need(set(models) == set(models_before) | {model_id} and model_id not in models_before
             and all(models[k] == v for k, v in models_before.items()),
             f"{ident}: expected one new Library artifact")
        model = models[model_id]
        need((model["host"], model["kind"], model["path"]) == (HOST, kind, str(expected)),
             f"{ident}: Library entry does not identify the verified artifact")
        return {"job_id": ident, "model_id": model_id, "artifact": artifact}

    def transfers(self, since, repo):
        rows = self.server.requests[since:]
        prefix = f"/{repo}/resolve/{COMMIT}/"
        paths, metadata = set(), set()
        for req in rows:
            if "/resolve/" in req["path"]:
                need(req["path"].startswith(prefix), f"non-immutable or wrong repository transfer: {req}")
                if req["method"] == "GET":
                    paths.add(req["path"][len(prefix):])
                else:
                    metadata.add(req["path"][len(prefix):])
        need(metadata <= paths, f"downloader probed an unselected file: {metadata - paths}")
        return paths

    def run(self):
        state = self.call("state")
        save(self.root / "initial-state.json", state)
        need(state["jobs"] == state["models"] == [], "expected a fresh empty controller")
        caps = {h["id"]: h["capabilities"] for h in state["hosts"]}
        need(caps[HOST].get("hf_download") is True and caps[OFFLINE].get("hf_download") is False,
             "requires integrated download-capable controller and opt-out worker")
        need(all(caps[OFFLINE][k] is True and caps[HOST][k] is False for k in CONVERSION),
             "download-only or conversion-worker capabilities changed")
        self.submit("disabled", {"repo_id": REPO}, host=OFFLINE, status=409)
        need(not self.call("state")["jobs"] and not self.server.requests,
             "disabled worker admitted a job or contacted the Hub")

        start = len(self.server.requests)
        ident = self.submit("full", {"repo_id": REPO, "revision": "main"})
        need(self.submit("full", {"repo_id": REPO, "revision": "main"}, status=200) == ident,
             "idempotent replay queued another job")
        entered, release = self.server.holds[REPO]
        need(entered.wait(10), "full job did not reach pinned downloader; see isolated worker/Hub logs")
        try:
            running = self.call(f"jobs/{ident}")
            need(running["state"] == "running", "job finished before fixture bytes were available")
            self.no_publication(running, {})
        finally:
            release.set()
        full_detail = self.wait("full", ident, "succeeded")
        need(full_detail["result"]["revision"] == "main", "full job lost requested revision")
        full = self.verified(full_detail, {})
        need(self.transfers(start, REPO) == set(FILES), "full job did not fetch the exact fixture inventory")
        # The downloader resolves the already verified commit; it must not
        # resolve a moving branch a second time.
        need(any(r["path"] == f"/api/models/{REPO}/revision/{COMMIT}" and r["query"] == "expand=sha"
                 for r in self.server.requests[start:]), "no pinned hfr commit resolution in fixture trace")

        start = len(self.server.requests)
        models = self.models()
        ident = self.submit("exact", {"repo_id": REPO, "revision": COMMIT, "filename": SECOND})
        detail = self.wait("exact", ident, "succeeded")
        exact = self.verified(detail, models, SECOND)
        need(detail["result"]["revision"] == COMMIT and self.transfers(start, REPO) == {SECOND},
             "exact job broadened selection or changed its immutable revision")

        models = self.models()
        rejected = {}
        for label, repo, filename in [("corrupt", BAD, FIRST), ("missing", REPO, "absent.safetensors")]:
            start = len(self.server.requests)
            ident = self.submit(label, {"repo_id": repo, "filename": filename})
            detail = self.wait(label, ident, "failed")
            self.no_publication(detail, models)
            need(self.transfers(start, repo) == ({FIRST} if label == "corrupt" else set()),
                 f"{label}: failure did not exercise the expected fixture boundary")
            rejected[label] = {"job_id": ident, "error": detail.get("error")}

        ident = self.submit("cancel", {"repo_id": CANCEL, "filename": FIRST})
        entered, release = self.server.holds[CANCEL]
        need(entered.wait(10), "cancel job did not reach held fixture transfer")
        try:
            self.call(f"jobs/{ident}/cancel", {})
            cancelled = self.wait("cancel", ident, "cancelled")
            self.no_publication(cancelled, models)
        finally:
            release.set()
        final = self.call("state")
        save(self.root / "final-state.json", final)
        need(len(final["jobs"]) == 5 and {h["id"]: h["capabilities"] for h in final["hosts"]} == caps,
             "unexpected jobs or changed worker capabilities")
        need(not any(r["credential"] for r in self.server.requests), "fixture received credentials")
        return {"full": full, "exact": exact, "rejected": rejected, "cancelled_job_id": ident,
                "conversion_capabilities": {k: caps[OFFLINE][k] for k in CONVERSION}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path, help="integrated Brewery build artifacts")
    parser.add_argument("--hfr", required=True, type=Path, help="trusted pinned Rust hfr ELF")
    parser.add_argument("--hfr-sha256", required=True, help="binary SHA-256 from the trusted pinned build receipt")
    parser.add_argument("--isolated", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.isolated:
        command = ["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
                   "--kill-child", "--mount-proc", "--", sys.executable, "-E", "-s", "-B",
                   str(Path(__file__).resolve()),
                   *sys.argv[1:], "--isolated"]
        raise SystemExit(subprocess.run(command, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}).returncode)
    need(os.getpid() == 1 and socket.if_nameindex() == [(1, "lo")],
         "requires a fresh private PID/network namespace with only loopback")
    subprocess.run(["/usr/bin/ip", "link", "set", "lo", "up"], check=True, timeout=3)
    bins, hfr = args.bin_dir.resolve(strict=True), args.hfr.resolve(strict=True)
    need(all(os.access(bins / bin_name, os.X_OK) for bin_name in BINARIES), "missing integrated Brewery binaries")
    need(hfr.is_file() and os.access(hfr, os.X_OK), "missing pinned hfr binary")
    with hfr.open("rb") as source:
        need(source.read(4) == b"\x7fELF", "hfr must be the real pinned ELF, not a wrapper")
        source.seek(0)
        digest = file_digest(source, "sha256").hexdigest()
    need(digest == args.hfr_sha256, "hfr SHA-256 differs from the build receipt")
    os.umask(0o077)
    root = Path(mkdtemp(prefix="managed-hfr-smoke.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    for child in ("home", "inputs", "tmp"):
        (root / child).mkdir()
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(root / "home"),
           "TMPDIR": str(root / "tmp"), "XDG_CONFIG_HOME": str(root / "home"),
           "HF_HOME": str(root / "home" / "hf")}
    with hub(root) as server:
        worker_config(root, bins, hfr, server.server_port)
        with controller(root, bins, env) as url:
            result = Smoke(root, url, server).run()
    report = {"scratch": str(root), "commit": COMMIT, "hfr_sha256": digest, **result}
    save(root / "summary.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
