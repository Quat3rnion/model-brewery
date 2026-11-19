#!/usr/bin/env python3
"""Exercise native hfr Xet downloads with the official hf-xet 1.7.0 LocalServer."""

import argparse
from contextlib import ExitStack, contextmanager
from hashlib import sha1, sha256, shake_256
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

REPO = "fixture/native-xet"
COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
WEIGHTS = "model.safetensors"
CONFIG = "config.json"
CAS_TOKEN = "fixture-only-cas-token"
HUB_TOKEN = "fixture-only-hub-token"
LOOPBACK = "127.0.0.1"


def tensor():
    # Repeating a long pseudorandom block lets us prove dedup independently of compression.
    data = shake_256(b"model-brewery native xet fixture v1").digest(512 << 10) * 4
    header = (
        b'{"fixture":{"dtype":"U8","shape":[2097152],"data_offsets":[0,2097152]}}'
    )
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + data


FILES = {CONFIG: b'{"model_type":"fixture"}\n', WEIGHTS: tensor()}
WEIGHTS_SHA256 = "a663b61a4a7a16994e21bc92c84e3651ccf5f8635beeba3570aec9500c2d25c1"
CORRUPT_SHA256 = "4196bfb4b7f838742e97d46405b80fd59ec541560414fe7d5a81f9a48417e437"


def git_oid(data):
    return sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def hfr_env(root):
    """Do not inherit a real token, token file, proxy, or shared Xet cache."""
    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": os.defpath, "HOME": str(home), "HF_HOME": str(home),
        "XDG_CACHE_HOME": str(home), "NO_PROXY": LOOPBACK,
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
    }


class WireHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_HEAD(self):
        self.dispatch()

    def do_GET(self):
        self.dispatch()

    def dispatch(self):
        fx = self.server.fx
        origin = self.server.origin
        url = urlsplit(self.path)
        bearer = self.headers.get("Authorization")
        if self.headers.get("Cookie"):
            auth = "unexpected"
        elif bearer is None:
            auth = "none"
        elif fx.hf_token is not None and bearer == f"Bearer {fx.hf_token}":
            auth = "hub"
        elif bearer == f"Bearer {CAS_TOKEN}":
            auth = "cas"
        else:
            auth = "unexpected"
        row = {
            "hop": origin, "method": self.command, "path": url.path,
            "query": url.query if origin == "hub" else "",
            "range": self.headers.get("Range"), "authorization": auth,
        }
        fx.requests.append(row)

        def reply(status, body=b"", *, length=None, **headers):
            row["status"] = status
            row["bytes"] = len(body) if self.command != "HEAD" else 0
            if fx.trace:
                fx.trace(dict(row))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body) if length is None else length))
            for key, value in headers.items():
                self.send_header(key.replace("_", "-"), value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        try:
            if origin == "hub":
                self.hub(fx, url.path, auth, reply)
            else:
                self.cas(fx, origin, url.path, auth, row, reply)
        except (BrokenPipeError, ConnectionResetError):
            pass  # The managed cancellation case may close a held response.

    def hub(self, fx, path, auth, reply):
        if auth != ("hub" if fx.hf_token is not None else "none"):
            return reply(403)
        prefix = f"/api/models/{REPO}"
        kind = {"Content_Type": "application/json"}
        if path in (f"{prefix}/revision/main", f"{prefix}/revision/{COMMIT}"):
            data = {"id": REPO, "sha": COMMIT, "private": False,
                    "gated": "auto" if fx.hf_token is not None else False}
        elif path == f"{prefix}/tree/{COMMIT}":
            data = fx.tree
        elif path == f"{prefix}/xet-read-token/{COMMIT}":
            data = {
                "casUrl": f"http://{LOOPBACK}:{fx.cas_port}",
                "accessToken": CAS_TOKEN, "exp": int(time.time()) + 600,
            }
        elif path == f"/{REPO}/resolve/{COMMIT}/{CONFIG}":
            content = FILES[CONFIG]
            if self.command == "HEAD":
                return reply(200, length=len(content), Accept_Ranges="bytes")
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            if not match:
                return reply(416)
            start, end = map(int, match.groups())
            if not 0 <= start <= end < len(content):
                return reply(416)
            return reply(206, content[start:end + 1], Content_Range=f"bytes {start}-{end}/{len(content)}")
        else:
            # In particular, the Xet weights have no /resolve fallback.
            return reply(404)
        return reply(200, json.dumps(data).encode(), **kind)

    def cas(self, fx, origin, path, auth, row, reply):
        if origin == "cas":
            if path not in (f"/v2/reconstructions/{fx.xet_hash}", f"/v1/reconstructions/{fx.xet_hash}"):
                return reply(404)
            if auth != "cas":
                return reply(403)
        else:
            if path != "/v1/fetch_term":
                return reply(404)
            if auth != "none":
                return reply(403)
            fx.blob_entered.set()
            if not fx.blob_release.wait(timeout=30):
                return reply(504)
        headers = {"Host": f"{LOOPBACK}:{fx.blob_port}"}
        if row["range"] is not None:
            headers["Range"] = row["range"]
        connection = HTTPConnection(LOOPBACK, fx.local_port, timeout=5)
        try:
            # Never forward either bearer to the LocalServer backend.
            connection.request(self.command, self.path, headers=headers)
            response = connection.getresponse()
            data = response.read(4 << 20)
            if len(data) >= 4 << 20:
                return reply(502)
            if origin == "cas" and response.status == 200:
                reconstruction = json.loads(data)
                row["terms"] = len(reconstruction["terms"])
                row["xorbs"] = list(reconstruction.get("xorbs", reconstruction.get("fetch_info", {})))
            return reply(response.status, data, **{
                name.replace("-", "_"): value for name, value in response.getheaders()
                if name.lower() in {"content-type", "content-range"}
            })
        finally:
            connection.close()


@contextmanager
def fixture(root, *, helper=None, corrupt=False, hf_token=None, hold_blob=False, trace=None):
    """Provide loopback Hub, CAS and Xorb routes; only synthetic tokens are allowed."""
    helper = helper or os.environ.get("HFR_XET_FIXTURE_HELPER")
    if not helper:
        raise ValueError("set HFR_XET_FIXTURE_HELPER or pass helper=")
    # Do not substitute a real Hub token; keep the two fixture roles distinct.
    if hf_token is not None and (not hf_token or hf_token == CAS_TOKEN):
        raise ValueError("pass a distinct synthetic Hub token")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    seed = root / WEIGHTS
    data = FILES[WEIGHTS]
    seed_bytes = data if not corrupt else data[:-1] + bytes([data[-1] ^ 1])
    seed.write_bytes(seed_bytes)
    assert sha256(data).hexdigest() == WEIGHTS_SHA256
    fx = SimpleNamespace(
        repo_id=REPO, commit=COMMIT, files=FILES, xet_files={WEIGHTS}, hf_token=hf_token,
        requests=[], trace=trace,
        blob_entered=threading.Event(), blob_release=threading.Event(),
    )
    if not hold_blob:
        fx.blob_release.set()
    with socket.socket() as port_socket:
        port_socket.bind((LOOPBACK, 0))
        fx.local_port = port_socket.getsockname()[1]
    with ExitStack() as stack:
        log = stack.enter_context((root / "local-server.log").open("wb"))
        proc = stack.enter_context(subprocess.Popen(
            [str(Path(helper).resolve(strict=True)), str(fx.local_port), str(root / "cas"), str(seed)],
            cwd=root, env=hfr_env(root), stdout=subprocess.PIPE, stderr=log,
        ))

        def terminate():
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)

        stack.callback(terminate)
        if not select.select([proc.stdout], [], [], 20)[0]:
            raise TimeoutError("LocalServer did not report seeded metadata")
        metadata = json.loads(proc.stdout.readline())
        assert metadata["file_size"] == len(data) and metadata["sha256"] == sha256(seed_bytes).hexdigest()
        fx.xet_hash = metadata["hash"]
        assert re.fullmatch(r"[0-9a-f]{64}", fx.xet_hash)
        deadline = time.monotonic() + 5
        while True:
            conn = HTTPConnection(LOOPBACK, fx.local_port, timeout=1)
            try:
                conn.request("GET", "/health")
                if conn.getresponse().status == 200:
                    break
            except OSError:
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise
            finally:
                conn.close()
            if time.monotonic() >= deadline:
                raise TimeoutError("LocalServer was not healthy")
            time.sleep(0.025)
        pointer = (f"version https://git-lfs.github.com/spec/v1\n"
                   f"oid sha256:{WEIGHTS_SHA256}\nsize {len(data)}\n").encode()
        fx.tree = [
            {"type": "file", "path": CONFIG, "size": len(FILES[CONFIG]), "oid": git_oid(FILES[CONFIG])},
            {"type": "file", "path": WEIGHTS, "size": len(data), "oid": git_oid(pointer),
             "xetHash": fx.xet_hash,
             "lfs": {"oid": WEIGHTS_SHA256, "size": len(data), "pointerSize": len(pointer)}},
        ]
        for origin in ("blob", "cas", "hub"):
            server = stack.enter_context(ThreadingHTTPServer((LOOPBACK, 0), WireHandler))
            server.fx, server.origin = fx, origin
            setattr(fx, f"{origin}_port", server.server_port)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            stack.callback(worker.join)
            stack.callback(server.shutdown)
        stack.callback(fx.blob_release.set)
        fx.endpoint = f"http://{LOOPBACK}:{fx.hub_port}"
        yield fx


def smoke(args):
    for corrupt in (False, True):
        root = args.root.resolve() / ("corrupt" if corrupt else "good")
        root.mkdir(parents=True, exist_ok=False)
        with fixture(root, helper=args.helper, corrupt=corrupt, hf_token=HUB_TOKEN) as fx:
            dest = root / "download"
            env = hfr_env(root / "client")
            env["HF_TOKEN"] = HUB_TOKEN
            command = [
                str(args.hfr.resolve(strict=True)), "download", REPO, "--endpoint", fx.endpoint,
                "--repo-type", "model", "--revision", COMMIT, "--local-dir", str(dest),
                "--transfer", "xet", "--verify", "--connections", "1", "--xet-buffer-mib", "64",
            ]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=25)
            (root / "stderr").write_bytes(result.stderr)
            (root / "stdout").write_bytes(result.stdout)
            (root / "requests.json").write_text(json.dumps(fx.requests, indent=2) + "\n")
            hub, recon, blobs = ([row for row in fx.requests if row["hop"] == hop]
                                 for hop in ("hub", "cas", "blob"))
            assert any(row["path"].endswith(f"/xet-read-token/{COMMIT}") and row["status"] == 200 for row in hub), hub
            assert all(row["authorization"] == "hub" for row in hub), hub
            assert not any("/resolve/" in row["path"] and row["path"].endswith(WEIGHTS) for row in hub), hub
            # hf-xet prefetches file ranges past EOF; LocalServer returns 416 for those.
            assert all(row["authorization"] == "cas" and row["status"] in (200, 416) for row in recon), recon
            assert any(row["status"] == 200 and row["terms"] > 1 and row["xorbs"] for row in recon), recon
            assert blobs and all(row["authorization"] == "none" and row["status"] in (200, 206) for row in blobs), blobs
            assert sum(row["bytes"] for row in blobs) < len(FILES[WEIGHTS]), blobs
            # The pinned native path verifies reconstruction, not the advertised
            # LFS digest. A managed caller must reject this mismatch separately.
            assert result.returncode == 0, result.stderr
            digest = sha256((dest / WEIGHTS).read_bytes()).hexdigest()
            assert digest == (CORRUPT_SHA256 if corrupt else WEIGHTS_SHA256), digest
            assert (dest / CONFIG).read_bytes() == FILES[CONFIG]
            assert not list(dest.rglob("*.incomplete*"))
            print(f"{root.name}: exit={result.returncode}, Xet hash {fx.xet_hash}, "
                  f"{sum(row['bytes'] for row in blobs)} Xorb bytes for {len(FILES[WEIGHTS])} file bytes, "
                  f"SHA-256 {digest}" + (" (published LFS SHA-256 differs)" if corrupt else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("serve", "smoke"):
        mode = modes.add_parser(name)
        mode.add_argument("--root", required=True, type=Path, help="scratch directory under /arboretum")
        mode.add_argument("--helper", type=Path)
    modes.choices["smoke"].add_argument("--hfr", required=True, type=Path)
    modes.choices["serve"].add_argument("--corrupt", action="store_true")
    modes.choices["serve"].add_argument("--require-hub-token", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke":
        return smoke(args)
    with fixture(
        args.root, helper=args.helper, corrupt=args.corrupt,
        hf_token=HUB_TOKEN if args.require_hub_token else None,
        trace=lambda row: print(json.dumps(row), flush=True, file=sys.stderr),
    ) as fx:
        print(json.dumps({
            "endpoint": fx.endpoint, "cas_port": fx.cas_port, "blob_port": fx.blob_port,
            "repo": REPO, "revision": COMMIT, "file": WEIGHTS, "xet_hash": fx.xet_hash,
            "size": len(FILES[WEIGHTS]), "sha256": sha256(FILES[WEIGHTS]).hexdigest(),
            "tree": fx.tree,
        }), flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
