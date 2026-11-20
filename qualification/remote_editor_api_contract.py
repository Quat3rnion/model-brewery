#!/usr/bin/python3
"""Provision a weight editor on a real, isolated, forced-command SSH planner."""
import argparse
import base64
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
from urllib.request import Request

from managed_hfr_smoke import Smoke, controller, save, st_tensor
from ssh_provision_smoke import keygen

HOST = "frame-hy4-plan"
PROJECT = Path(__file__).resolve().parents[1]


def verify(smoke, host, models, missing, credential="fixture", extra=()):
    def row():
        return next(x for x in smoke.call("admin/state")["hosts"] if x["id"] == host)

    def install(revision, path, selected=credential, expect=200):
        return smoke.call("admin/actions", {
            "action": "setup_weight_editor", "worker": host, "revision": revision,
            "model_directory": path, "credential": selected,
        }, status=expect)

    before = row()
    assert before["weight_editor"]["setup"]["status"] == "needs_setup"
    choices = before["weight_editor"]["setup"]["credentials"]
    assert {"id": "fixture", "label": "Fixture key"} in choices
    assert {choice["id"] for choice in choices} == {"fixture", *extra}
    rev = before["revision"]
    install(rev, models, "not-configured", 409)
    assert row()["revision"] == rev
    install(rev, models)
    ready = row()
    assert ready["weight_editor"]["setup"]["status"] == "verified"
    assert ready["weight_editor"]["setup"]["model_directory"] == models
    assert set(ready["allowed_jobs"]) - set(before["allowed_jobs"]) == {
        "weight_inspect", "weight_propose", "weight_apply", "weight_tune"
    }
    install(rev, models, expect=409)
    install(ready["revision"], missing, expect=409)
    assert row()["weight_editor"]["setup"]["status"] == "verified"
    assert row()["weight_editor"]["setup"]["model_directory"] == models
    return ready


def saved_logins(api, root, user, port, other_port, fingerprint, editors, server_log):
    base = {
        "host": "127.0.0.1", "user": user, "port": port,
        "fingerprint": fingerprint, "identity_file": str(root / "provision-key"),
    }

    def save_login(id, fields):
        definition = api.call("admin/ssh-hosts", {"id": id, "name": id, **base, **fields}, status=201)
        return api.call(f"admin/ssh-hosts/{id}/action", {"revision": definition["revision"], "action": "test"})

    def setup(credential, *, status=409):
        return api.call("admin/actions", {
            "action": "setup_weight_editor", "worker": HOST, "revision": 0,
            "model_directory": str(root / "models"), "credential": credential,
        }, status=status)

    same = save_login("separate-login", {})
    other = save_login("wrong-port", {"port": other_port})
    wrong_host = save_login("wrong-host", {"host": "127.0.0.2"})
    stale = f"saved-ssh:separate-login:{same['revision']}"
    setup_view = next(x for x in api.call("admin/state")["hosts"] if x["id"] == HOST)["weight_editor"]["setup"]
    assert {choice["id"] for choice in setup_view["credentials"]} == {"fixture", stale}, setup_view

    # The two endpoints intentionally share the same Ed25519 host key. No SSH
    # login (not even a pre-auth keyscan) may reach the actual worker.
    log = server_log.read_bytes()
    setup(f"saved-ssh:wrong-port:{other['revision']}")
    setup(f"saved-ssh:wrong-host:{wrong_host['revision']}")
    assert server_log.read_bytes() == log and not editors.exists()
    with api.opener.open(Request(f"{api.url}/api/admin/ssh-hosts/separate-login",
                         json.dumps({**base, "name": "Renamed login", "revision": same["revision"]}).encode(),
                         {"Content-Type": "application/json"}, method="PUT"), timeout=4) as reply:
        same = json.load(reply)
    setup(stale)
    assert server_log.read_bytes() == log and not editors.exists()
    current = f"saved-ssh:separate-login:{same['revision']}"
    identity = root / "provision-key"
    identity.chmod(0o644)
    try:
        setup(current)
        assert not editors.exists()
    finally:
        identity.chmod(0o600)
    return current, (current,)


def fixture(root, bins, editor_runtime, old_controller, editor_archive):
    assert os.geteuid() != 0 and socket.if_nameindex() == [(1, "lo")]
    home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    assert list(home.iterdir()) == []
    os.umask(0o077)
    for name in ("server", "rpc-key", "provision-key"):
        keygen(root / name)
    key = (root / "server.pub").read_text().split()[:2]
    digest = base64.b64encode(sha256(base64.b64decode(key[1])).digest()).decode().rstrip("=")
    user = pwd.getpwuid(os.geteuid()).pw_name
    planner = root / "planner.json"
    models = root / "models"
    models.mkdir()
    (models / "tensor.safetensors").write_bytes(st_tensor("weight", 1.0))
    save(planner, {
        "root": str(root / "planner"), "engine": str(bins / "brewery-engine"),
        "hy4_plan": {"host": HOST, "source": str(root / "planner-source"), "halo_tools": str(root / "halo-tools")},
    })
    planner_digest = sha256(planner.read_bytes()).hexdigest()
    authorized = root / "authorized"
    authorized.write_text(
        f'command="BREWERY_WORKER_CONFIG={planner} {bins}/brewery-worker rpc",restrict '
        + (root / "rpc-key.pub").read_text()
        + "restrict " + (root / "provision-key.pub").read_text()
    )
    with socket.socket() as listener, socket.socket() as other_listener:
        listener.bind(("127.0.0.1", 0))
        other_listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        other_port = other_listener.getsockname()[1]
    server = root / "sshd.conf"
    server.write_text(f"""HostKey {root}/server
AuthorizedKeysFile {authorized}
StrictModes no
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
AllowUsers {user}
DisableForwarding yes
ListenAddress 127.0.0.1
ListenAddress 127.0.0.2
Port {port}
PidFile {root}/sshd.pid
""")
    other_server = root / "other_sshd.conf"
    other_server.write_text(server.read_text().replace(f"Port {port}\n", f"Port {other_port}\n")
                            .replace("sshd.pid", "other_sshd.pid"))
    known = root / "known_hosts"
    known.write_text(f"{HOST} {' '.join(key)}\n")
    alias = root / "ssh"
    alias.write_text(f"""Host {HOST}
HostName 127.0.0.1
Port {port}
User {user}
HostKeyAlias {HOST}
UserKnownHostsFile {known}
GlobalKnownHostsFile /dev/null
StrictHostKeyChecking yes
IdentityAgent none
IdentitiesOnly yes
IdentityFile {root / 'rpc-key'}
""")
    save(root / "dispatch.json", {"hosts": {HOST: {
        "ssh": HOST, "ssh_config": str(alias), "capabilities": {"hy4_plan": True, "plan": True},
    }}})
    credentials = root / "credentials.json"
    row = {"worker": HOST, "label": "Fixture key", "user": user, "fingerprint": f"SHA256:{digest}",
           "identity": str(root / "provision-key")}
    original_key_row = {**row, "label": "Original RPC key", "identity": str(root / "rpc-key")}
    save(credentials, {"fixture": row})
    package = root / "editor.tar"
    if editor_archive:
        shutil.copyfile(editor_archive, package)
    else:
        with tarfile.open(package, "w", format=tarfile.USTAR_FORMAT) as tar:
            for name in ("brewery-worker", "brewery-engine"):
                tar.add(bins / name, arcname=name)
            for name in ("brewery-weight-edit", "brewery-weight-tune", "brewery-weight-runtime"):
                tar.add(editor_runtime / name, arcname=name)
    with package.open("rb") as f:
        pin = file_digest(f, "sha256").hexdigest()
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(home), "TMPDIR": str(root),
        "BREWERY_WEB_DIST": str(root / "unused"),
        "BREWERY_REMOTE_EDITOR_PROVISION": str(PROJECT / "tools/brewery-editor-provision"),
        "BREWERY_REMOTE_EDITOR_CREDENTIALS": str(credentials),
        "BREWERY_REMOTE_EDITOR_PACKAGE": str(package),
        "BREWERY_REMOTE_EDITOR_PACKAGE_SHA256": pin,
        "BREWERY_SSH_PROVISION": str(PROJECT / "tools/brewery-provision"),
    }
    server_log = root / "sshd.log"
    with server_log.open("wb") as log, (root / "other_sshd.log").open("wb") as other_log, subprocess.Popen(
        ["/usr/bin/sshd", "-D", "-e", "-f", server], stdout=log, stderr=log
    ) as daemon, subprocess.Popen(
        ["/usr/bin/sshd", "-D", "-e", "-f", other_server], stdout=other_log, stderr=other_log
    ) as other_daemon:
        try:
            end = time.monotonic() + 3
            while True:
                assert daemon.poll() is None and other_daemon.poll() is None and time.monotonic() < end, "sshd did not start"
                try:
                    with socket.create_connection(("127.0.0.1", port), 0.2), \
                         socket.create_connection(("127.0.0.1", other_port), 0.2):
                        break
                except OSError:
                    time.sleep(.03)
            if old_controller:
                # Assert the new contract is absent on the pre-change binary.
                with controller(root, old_controller, env) as url:
                    rows = Smoke(root, url, None).call("admin/state")["hosts"]
                    assert next(x for x in rows if x["id"] == HOST)["weight_editor"].get("setup") is None
            save(credentials, {"fixture": row, "original": original_key_row})
            with controller(root, bins, env) as url:
                api = Smoke(root, url, None)
                api.call("admin/actions", {
                    "action": "setup_weight_editor", "worker": HOST, "revision": 0,
                    "model_directory": str(models), "credential": "original",
                }, status=409)
                assert not (home / ".local/share/model-brewery/editors").exists()
            save(credentials, {"fixture": row})
            with controller(root, bins, env) as url:
                api = Smoke(root, url, None)
                current, choices = saved_logins(api, root, user, port, other_port, f"SHA256:{digest}",
                                               home / ".local/share/model-brewery/editors", server_log)
                ready = verify(api, HOST, str(models), str(root / "missing"), current, choices)
                assert "plan" in ready["allowed_jobs"]
                model = api.call("models", {
                    "host": HOST, "kind": "safetensors", "name": "fixture",
                    "path": str(models / "tensor.safetensors"),
                }, status=201)
                job = api.call("jobs", {
                    "kind": "weight_inspect", "host": HOST, "input_ids": [model["id"]],
                    "options": {"mode": "inventory", "offset": 0, "limit": 10},
                }, status=201)
                api.wait("remote-inventory", job["id"], "succeeded")
            with controller(root, bins, env) as url:
                api = Smoke(root, url, None)
                def state():
                    return next(x for x in api.call("admin/state")["hosts"] if x["id"] == HOST)
                assert state() == ready
                api.call("admin/actions", {"action": "set_weight_editor", "worker": HOST,
                                           "revision": ready["revision"], "enabled": False})
                assert state()["allowed_jobs"] == ["plan"]
                assert state()["weight_editor"]["setup"]["status"] == "verified"
                api.call("admin/actions", {"action": "set_weight_editor", "worker": HOST,
                                           "revision": ready["revision"] + 1, "enabled": True})
                assert set(state()["allowed_jobs"]) == set(ready["allowed_jobs"])
            assert sha256(planner.read_bytes()).hexdigest() == planner_digest
            proc = subprocess.run(["/usr/bin/ssh", "-F", alias, "-T", HOST, "brewery-worker rpc"],
                                  input='{"op":"status","id":"fixture"}', capture_output=True, text=True,
                                  check=True, timeout=5)
            assert json.loads(proc.stdout)["ok"]["state"] == "missing"
        finally:
            other_daemon.terminate()
            other_daemon.wait(timeout=5)
            daemon.terminate()
            daemon.wait(timeout=5)
    print(f"PASS: forced-command planner unchanged; verified separate SSH runtime, inventory, restart, invalid key/path, stale revision and rollback; {root}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("bin_dir", type=Path)
    cli.add_argument("editor_runtime", type=Path)
    cli.add_argument("--old-controller", type=Path)
    cli.add_argument("--editor-archive", type=Path)
    cli.add_argument("--isolated", type=Path)
    args = cli.parse_args()
    if args.isolated:
        fixture(args.isolated, args.bin_dir, args.editor_runtime, args.old_controller, args.editor_archive)
        return
    root = Path(tempfile.mkdtemp(prefix="brewery-editor-ssh-", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    (root / "home").mkdir(mode=0o700)
    (root / "tools").mkdir()
    (root / "qualification").mkdir()
    for src in ("tools/brewery-provision", "tools/brewery-editor-provision",
                "qualification/remote_editor_api_contract.py", "qualification/managed_hfr_smoke.py",
                "qualification/local_sm120.py", "qualification/release_check.py", "qualification/release_source.py",
                "qualification/ssh_provision_smoke.py"):
        shutil.copy2(PROJECT / src, root / src)
    home = pwd.getpwuid(os.getuid()).pw_dir
    # Match the normal SSH-provision fixture: external networking and real
    # worker homes are unavailable to every test process.
    child = ["/usr/bin/bwrap", "--unshare-user", "--uid", str(os.getuid()), "--gid", str(os.getgid()),
             "--unshare-pid", "--die-with-parent", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
             "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
             "--symlink", "usr/lib", "/lib64", "--dev", "/dev", "--proc", "/proc",
             "--dir", "/tmp",
             "--bind", str(root), str(root), "--bind", str(root / "home"), home]
    for folder in {path.resolve() for path in (args.bin_dir, args.editor_runtime, args.old_controller) if path}:
        child += ["--ro-bind", str(folder), str(folder)]
    if args.editor_archive:
        archive = str(args.editor_archive.resolve())
        child += ["--ro-bind", archive, archive]
    child += ["--chdir", str(root), "--", "/usr/bin/python3", "-B", str(root / "qualification" / Path(__file__).name),
             str(args.bin_dir.resolve()), str(args.editor_runtime.resolve()), "--isolated", str(root)]
    if args.old_controller:
        child += ["--old-controller", str(args.old_controller.resolve())]
    if args.editor_archive:
        child += ["--editor-archive", archive]
    raise SystemExit(subprocess.run(
        ["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--", "/usr/bin/sh", "-c",
         '/usr/bin/ip link set lo up && exec "$@"', "editor-provision-test", *child],
        env={"PATH": "/usr/bin", "HOME": home, "TMPDIR": str(root)},
    ).returncode)


if __name__ == "__main__":
    main()
