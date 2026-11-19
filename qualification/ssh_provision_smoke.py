#!/usr/bin/python3
"""Pinned real SSH provisioning on a loopback-only test host with a scratch home."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def keygen(path):
    subprocess.run(["/usr/bin/ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", path], check=True)


def smoke(root, fixture, project):
    assert os.geteuid() != 0 and socket.if_nameindex() == [(1, "lo")]
    os.umask(0o077)
    account = pwd.getpwuid(os.geteuid())
    home = Path(account.pw_dir)
    assert list(home.iterdir()) == []
    target = home / ".local/share/model-brewery/workers/test-worker"
    for name in ("client", "server"):
        keygen(root / name)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = root / "sshd.conf"
    server.write_text(f"""
HostKey {root}/server
AuthorizedKeysFile {root}/client.pub
# The outer root UID is intentionally unmapped in this scratch namespace.
# The only authorized key is the fresh synthetic key in this fixture.
StrictModes no
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
AllowUsers {account.pw_name}
DisableForwarding yes
ListenAddress 127.0.0.1
Port {port}
PidFile {root}/sshd.pid
""")
    public_key = (root / "server.pub").read_text().split()[:2]
    fingerprint = base64.b64encode(hashlib.sha256(base64.b64decode(public_key[1])).digest())
    fingerprint = "SHA256:" + fingerprint.decode().rstrip("=")
    host = {"id": "test-worker", "host": "127.0.0.1", "user": account.pw_name,
            "port": port, "fingerprint": fingerprint}
    env = {"PATH": "/usr/bin", "TMPDIR": str(root), "BREWERY_PROVISION_IDENTITY": str(root / "client"),
           "BREWERY_PROVISION_BUNDLE": str(root / "bundle"), "BREWERY_PROVISION_MANIFEST_SHA256": fixture}

    def request(op, fields=None):
        run = subprocess.run([project / "tools/brewery-provision"],
                             input=json.dumps(dict(host, op=op, **(fields or {}))), text=True,
                             env=env, capture_output=True, timeout=20, check=True)
        return json.loads(run.stdout)

    with (root / "sshd.log").open("w") as log, subprocess.Popen(
        ["/usr/bin/sshd", "-D", "-e", "-f", server], stderr=log, stdout=log,
    ) as daemon:
        try:
            end = time.monotonic() + 3
            while True:
                assert daemon.poll() is None and time.monotonic() < end, "sshd did not start"
                try:
                    with socket.create_connection(("127.0.0.1", port), 0.2):
                        break
                except OSError:
                    time.sleep(.03)
            bad = dict(host, fingerprint="SHA256:" + "A" * 43, op="test")
            reply = subprocess.run([project / "tools/brewery-provision"], input=json.dumps(bad),
                                   env=env, text=True, capture_output=True, check=True, timeout=12)
            assert json.loads(reply.stdout)["state"] == "failed"
            assert "Accepted publickey" not in (root / "sshd.log").read_text()
            assert request("test") == {"state": "verified", "host_key": " ".join(public_key)}
            assert not (home / ".local/share/model-brewery").exists()
            plan = request("plan")
            assert plan["state"] == "planned" and plan["bundle_id"] == fixture, plan
            assert not (home / ".local/share/model-brewery").exists()
            assert request("install", {"confirmation": "a" * 64})["state"] == "failed"
            assert not (home / ".local/share/model-brewery").exists()
            # Keep the manifest and file size unchanged but replace one byte.
            # Confirmation must never authorize a different payload.
            with (root / "bundle/brewery-engine").open("r+b") as binary:
                binary.seek(-1, os.SEEK_END)
                last = binary.read(1)
                binary.seek(-1, os.SEEK_END)
                binary.write(bytes([last[0] ^ 1]))
                binary.flush()
                denied = request("install", {"confirmation": plan["confirmation"]})
                assert denied == {"state": "failed", "error": "Worker binary does not match its pinned hash"}, denied
                binary.seek(-1, os.SEEK_END)
                binary.write(last)
            assert not (home / ".local/share/model-brewery").exists()
            result = request("install", {"confirmation": plan["confirmation"]})
            assert result == {k: v for k, v in plan.items() if k != "confirmation"} | {"state": "installed"}, result
            worker = Path(result["worker"])
            config = Path(result["config"])
            assert worker == target / fixture / "brewery-worker"
            assert config == target / fixture / "worker.json"
            assert worker.read_bytes() == (root / "bundle/brewery-worker").read_bytes()
            assert worker.with_name("brewery-engine").read_bytes() == (root / "bundle/brewery-engine").read_bytes()
            assert config.stat().st_mode & 0o777 == 0o600
            assert request("install", {"confirmation": plan["confirmation"]}) == result
            (root / "known").write_text(f"brewery-fixture {' '.join(public_key)}\n")
            rpc = subprocess.run(["/usr/bin/ssh", "-F", "/dev/null", "-T", "-i", root / "client", "-p", str(port),
                                  "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                                  "-o", "GlobalKnownHostsFile=/dev/null",
                                  "-o", f"UserKnownHostsFile={root}/known",
                                  "-o", "HostKeyAlias=brewery-fixture", "-o", "IdentityAgent=none",
                                  "-l", account.pw_name, "127.0.0.1",
                                  f"BREWERY_WORKER_CONFIG={config} {worker} rpc"],
                                 input='{"op":"status","id":"fixture-job"}', capture_output=True, text=True,
                                 check=True, timeout=15)
            assert json.loads(rpc.stdout)["ok"]["state"] == "missing"
        finally:
            daemon.terminate()
            daemon.wait(timeout=5)
    print(f"PASS: wrong SSH/binary pins rejected, dry plan, confirmed install and worker RPC; {root}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("bin_dir", type=Path)
    cli.add_argument("--isolated", nargs=2, metavar=("ROOT", "BUNDLE_ID"))
    args = cli.parse_args()
    project = Path(__file__).resolve().parent.parent
    if args.isolated:
        root, fixture = args.isolated
        smoke(Path(root), fixture, project)
        return
    os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix="brewery-provision-", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    (root / "home").mkdir()
    bundle = root / "bundle"
    bundle.mkdir()
    files = {}
    for name in ("brewery-worker", "brewery-engine"):
        path = bundle / name
        shutil.copyfile(args.bin_dir / name, path)
        with path.open("rb") as f:
            files[name] = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = json.dumps({"version": 1, "arch": os.uname().machine, "files": files}).encode()
    (bundle / "manifest.json").write_bytes(manifest)
    pin = hashlib.sha256(manifest).hexdigest()
    # The outer namespace enables loopback. Inner bubblewrap runs as the
    # unprivileged login with a scratch-only home and a read-only host FS.
    uid, gid, home = os.getuid(), os.getgid(), Path(pwd.getpwuid(os.getuid()).pw_dir)
    subtree = root / "project"
    (subtree / "tools").mkdir(parents=True)
    (subtree / "qualification").mkdir()
    for src in ("tools/brewery-provision", "qualification/ssh_provision_smoke.py"):
        shutil.copy2(project / src, subtree / src)
    child = ["/usr/bin/bwrap", "--unshare-user", "--uid", str(uid), "--gid", str(gid),
             "--unshare-pid", "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev",
             "--proc", "/proc", "--bind", str(root), str(root), "--bind", str(root / "home"), str(home),
             "--chdir", str(root), "--", "/usr/bin/python3", "-I", "-B",
             str(subtree / "qualification/ssh_provision_smoke.py"),
             str(args.bin_dir), "--isolated", str(root), pin]
    raise SystemExit(subprocess.run(
        ["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--", "/usr/bin/sh", "-c",
         '/usr/bin/ip link set lo up && exec "$@"', "ssh-provision-test", *child],
        env={"PATH": "/usr/bin", "HOME": str(home), "TMPDIR": str(root)},
    ).returncode)


if __name__ == "__main__":
    main()
