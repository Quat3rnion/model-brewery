#!/usr/bin/env python3
"""Verify mixed Xet/HTTP imports through the real Brewery jobs API.

Requires integrated Brewery binaries, pinned Rust hfr, and the native Xet
fixture helper. Uses only a loopback-only namespace and synthetic bearers.
"""

import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from tempfile import mkdtemp

from managed_hfr_smoke import BINARIES, HOST, Smoke, controller, need, save, worker_config

FAKE_HF = "hf_fixture_xet_not_a_real_credential"


def verify_wire(fx, corrupt):
    rows = list(fx.requests)
    hub, cas, blobs = ([row for row in rows if row["hop"] == hop]
                       for hop in ("hub", "cas", "blob"))
    need(len(rows) == len(hub) + len(cas) + len(blobs), "unrecognized wire hop")
    for group, credential in ((hub, "hub"), (cas, "cas"), (blobs, "none")):
        need(group and all(row["authorization"] == credential for row in group),
             f"wrong bearer confinement at {credential!r} hop")

    api = f"/api/models/{fx.repo_id}"
    resolve = f"/{fx.repo_id}/resolve/{fx.commit}/"
    revisions = [row for row in hub if "/revision/" in row["path"]]
    need(len(revisions) == 1 and revisions[0]["path"] == f"{api}/revision/main"
         and revisions[0]["status"] == 200
         and any(r["path"] == f"{api}/tree/{fx.commit}" and r["status"] == 200 for r in hub),
         "branch must be resolved once before using immutable metadata")
    tickets = [r for r in hub if "/xet-read-token/" in r["path"]]
    need(tickets and all(r["path"] == f"{api}/xet-read-token/{fx.commit}"
                         and r["status"] == 200 for r in tickets), "CAS tickets were not scoped to the commit")
    transfers = [r for r in hub if "/resolve/" in r["path"] or "/resolve-cache/" in r["path"]]
    http_files = fx.files.keys() - fx.xet_files
    need(all(r["path"].startswith(resolve) and r["path"][len(resolve):] in http_files for r in transfers),
         "Xet file accessed through an ordinary HTTP fallback or a mutable revision")
    fetched = {r["path"][len(resolve):] for r in transfers
               if r["method"] == "GET" and r["status"] in (200, 206)}
    if not corrupt:
        need(fetched == http_files, "ordinary HTTP files were not transferred")
    need(any(r["path"].endswith(f"/reconstructions/{fx.xet_hash}") and r["status"] == 200
             and r["terms"] > 1 and r["xorbs"] for r in cas),
         "no successful native CAS reconstruction request")
    xorb_bytes = sum(r["bytes"] for r in blobs if r["method"] == "GET" and r["status"] in (200, 206))
    need(xorb_bytes > 0, "no native Xorb bytes served")
    return {"hub_requests": len(hub), "cas_requests": len(cas), "xorb_requests": len(blobs),
            "xorb_bytes": xorb_bytes, "http_files": sorted(fetched), "xet_files": sorted(fx.xet_files)}


def verified(smoke, detail, fx):
    ident = detail["id"]
    result = detail["result"]
    path = smoke.root / HOST / "jobs" / ident / "work/download"
    need(result["kind"] == "hf_download" and result["verified"] is True
         and result["repo_id"] == fx.repo_id and result["revision"] == "main"
         and result["commit"] == fx.commit and "filename" not in result,
         "verified receipt did not identify the immutable whole model")
    need(path.resolve() == path and path.is_dir(), "not a canonical download directory")
    actual = {}
    for file in path.iterdir():
        need(file.name in fx.files and file.is_file() and not file.is_symlink()
             and file.stat().st_size == len(fx.files[file.name]), f"unexpected output file {file.name}")
        actual[file.name] = file.read_bytes()
    need(actual == fx.files, "published model bytes differ from fixture source bytes")
    tree = b"".join(f"{name}\0{len(data)}\0{sha256(data).hexdigest()}\n".encode()
                    for name, data in sorted(fx.files.items()))
    artifact = {"kind": "hf", "path": str(path), "bytes": sum(map(len, fx.files.values())),
                "sha256": sha256(tree).hexdigest()}
    need(result["artifact"] == artifact, "wrong verified artifact path, byte count, or digest")
    model_id = result["library_model_id"]
    models = smoke.models()
    need(set(models) == {model_id}, "did not register exactly one Library model")
    model = models[model_id]
    need((model["host"], model["kind"], model["name"], model["path"])
         == (HOST, "hf", fx.repo_id, str(path)), "Library row does not identify the verified model")
    return {"job_id": ident, "library_model_id": model_id, "artifact": artifact}


def configure(root, bins, hfr, fx, token_file):
    worker_config(root, bins, hfr, fx.hub_port)
    path = root / HOST / "worker.json"
    policy = json.loads(path.read_text())
    policy["source_roots"] = []
    policy["hf_download"].update(
        fixture_xet_cas_port=fx.cas_port, token_file=str(token_file), max_bytes=8 << 20,
    )
    save(path, policy)


def run_case(root, args, fixture, *, corrupt):
    for name in ("home", "tmp", "inputs", "wire", "credentials"):
        (root / name).mkdir(parents=True)
    token_file = root / "credentials" / "hub"
    token_file.write_text(FAKE_HF)
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(root / "home"),
           "TMPDIR": str(root / "tmp"), "HF_HOME": str(root / "home" / "hf")}
    with (root / "wire.jsonl").open("w", buffering=1) as trace, fixture(
        root / "wire", helper=args.fixture_helper, corrupt=corrupt, hold_blob=not corrupt,
        hf_token=FAKE_HF, trace=lambda row: trace.write(json.dumps(row) + "\n"),
    ) as fx:
        need(re.fullmatch(r"[0-9a-f]{40}", fx.commit), "fixture has no immutable commit")
        http_files = fx.files.keys() - fx.xet_files
        need(fx.xet_files and fx.xet_files <= fx.files.keys() and "config.json" in http_files,
             "fixture must contain both Xet content and a small non-Xet file")
        configure(root, args.bin_dir, args.hfr, fx, token_file)
        with controller(root, args.bin_dir, env) as url:
            smoke = Smoke(root, url, None)
            state = smoke.call("state")
            need(state["jobs"] == state["models"] == [], "controller is not fresh")
            need(any(host["id"] == HOST and host["capabilities"].get("hf_download") is True
                     for host in state["hosts"]), "download worker not enabled")
            ident = smoke.submit("mixed", {"repo_id": fx.repo_id, "revision": "main"})
            if not corrupt:
                try:
                    need(fx.blob_entered.wait(10), "job did not reach native Xet blob transfer")
                    running = smoke.call(f"jobs/{ident}")
                    need(running["state"] == "running", "job finished before native bytes arrived")
                    smoke.no_publication(running, {})
                finally:
                    fx.blob_release.set()
                detail = smoke.wait("mixed", ident, "succeeded")
                result = verified(smoke, detail, fx)
            else:
                detail = smoke.wait("corrupt", ident, "failed")
                smoke.no_publication(detail, {})
                # A parseable native reconstruction can still disagree with
                # the pinned LFS digest. Require the independent verifier's
                # rejection, not an unrelated CAS, credential, or job error.
                stderr = root / HOST / "jobs" / ident / "engine-stderr.log"
                diagnostic = "download checksum or size did not match the pinned Hub inventory"
                need(diagnostic in stderr.read_text(), "download did not fail content verification")
                result = {"job_id": ident, "error": detail.get("error"), "verification_error": diagnostic}
            save(root / "final-state.json", smoke.call("state"))
        result["wire"] = verify_wire(fx, corrupt)
        result["commit"] = fx.commit
        return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--bin-dir", type=Path, required=True)
    cli.add_argument("--hfr", type=Path, required=True, help="trusted pinned Rust hfr ELF")
    cli.add_argument("--hfr-sha256", required=True, help="SHA-256 from the trusted hfr build receipt")
    cli.add_argument("--fixture-helper", type=Path, required=True, help="native Xet fixture helper")
    cli.add_argument("--isolated", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if not args.isolated:
        raise SystemExit(subprocess.run([
            "/usr/bin/unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
            "--kill-child", "--mount-proc", sys.executable, "-E", "-s", "-B",
            str(Path(__file__).resolve()), *sys.argv[1:], "--isolated",
        ], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}).returncode)
    need(os.getpid() == 1 and socket.if_nameindex() == [(1, "lo")],
         "requires a fresh private PID/network namespace with only loopback")
    subprocess.run(["/usr/bin/ip", "link", "set", "lo", "up"], check=True, timeout=3)
    for name in ("bin_dir", "hfr", "fixture_helper"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    need(all(os.access(args.bin_dir / name, os.X_OK) for name in BINARIES), "missing integrated binaries")
    need(os.access(args.hfr, os.X_OK) and os.access(args.fixture_helper, os.X_OK),
         "hfr and native fixture helper must be executable")
    with args.hfr.open("rb") as binary:
        need(binary.read(4) == b"\x7fELF", "expected real pinned Rust hfr ELF, not a wrapper")
        binary.seek(0)
        digest = file_digest(binary, "sha256").hexdigest()
    need(digest == args.hfr_sha256, "hfr digest differs from the trusted build receipt")
    from hfr_xet_fixture import fixture

    os.umask(0o077)
    root = Path(mkdtemp(prefix="managed-hfr-xet.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    # Neither the fixture helper nor hfr may inherit a real credential or
    # write to an operator's home, cache, or system temporary directory.
    for name in ("home", "tmp"):
        (root / name).mkdir()
    os.environ.clear()
    os.environ.update(PATH="/usr/bin:/bin", LANG="C.UTF-8", HOME=str(root / "home"),
                      TMPDIR=str(root / "tmp"), HF_HOME=str(root / "home" / "hf"))
    results = {"scratch": str(root), "hfr_sha256": digest,
               "mixed": run_case(root / "good", args, fixture, corrupt=False),
               "corrupt": run_case(root / "corrupt", args, fixture, corrupt=True)}
    save(root / "summary.json", results)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
