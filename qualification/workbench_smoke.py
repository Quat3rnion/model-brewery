"""Exercise the installed workbench through loopback HTTP, not internal Rust APIs.

Point --bin-dir at a workspace build. Pass --gguf twice to additionally run
the real CPU tracing engine. Set BREWERY_LANTERNFISH to its pinned binary.
All state is private and temporary; registered original models are read-only.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


@contextlib.contextmanager
def transport_fixture(enabled, root, bin_dir):
    """Private loopback sshd with a forced worker; never read or change the user's SSH config."""
    if not enabled:
        yield None
        return
    for prog in ("ssh", "sshd", "ssh-keygen"):
        require(shutil.which(prog), f"{prog} needed for --ssh")
    fixture = root / "ssh"
    fixture.mkdir(mode=0o700)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    for key in ("host", "client"):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", fixture / key], check=True)
    forced = fixture / "worker-rpc"
    forced.write_text("#!/bin/sh\nset -eu\n"
                      "[ \"$SSH_ORIGINAL_COMMAND\" = 'brewery-worker rpc' ] || exit 126\n"
                      f"export BREWERY_WORKER_CONFIG={shlex.quote(str(root / 'compute.json'))}\n"
                      f"exec {shlex.quote(str(bin_dir / 'brewery-worker'))} rpc\n")
    forced.chmod(0o700)
    (fixture / "authorized").write_text((fixture / "client.pub").read_text())
    (fixture / "known").write_text(f"[127.0.0.1]:{port} " + (fixture / "host.pub").read_text())
    daemon = fixture / "server.conf"
    daemon.write_text(f"""ListenAddress 127.0.0.1
Port {port}
HostKey {fixture}/host
AuthorizedKeysFile {fixture}/authorized
PidFile {fixture}/sshd.pid
PasswordAuthentication no
KbdInteractiveAuthentication no
DisableForwarding yes
UsePAM no
ForceCommand {forced}
""")
    client = fixture / "client.conf"
    client.write_text(f"""Host compute
    HostName 127.0.0.1
    Port {port}
    IdentityFile {fixture}/client
    IdentitiesOnly yes
    UserKnownHostsFile {fixture}/known
    GlobalKnownHostsFile /dev/null
    BatchMode yes
    StrictHostKeyChecking yes
""")
    with (fixture / "sshd.log").open("wb") as log:
        proc = subprocess.Popen([shutil.which("sshd"), "-D", "-e", "-f", daemon],
                                stdout=log, stderr=log)
        try:
            for _ in range(100):
                if proc.poll() is not None:
                    raise RuntimeError((fixture / "sshd.log").read_text()[-2048:])
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.1) as check:
                        require(check.recv(80).startswith(b"SSH-"), "wrong SSH listener")
                    break
                except OSError:
                    time.sleep(.05)
            else:
                raise RuntimeError("sshd not ready")
            yield fixture
        finally:
            proc.terminate()
            proc.wait(timeout=8)


def smoke(bin_dir, web_dist, models, root, ssh_fixture, suite=None, report_file=None):
    for exe in ("brewery", "brewery-worker", "brewery-dispatch", "brewery-engine", "brewery-eval"):
        require(os.access(bin_dir / exe, os.X_OK), f"missing workspace build: {bin_dir / exe}")
    require((web_dist / "index.html").is_file(), "missing built Svelte UI")
    source = root / "original-files"
    source.mkdir()
    original = source / "source with spaces.txt"
    payload = b"original source: never delete on unregister\n" * 50000
    original.write_bytes(payload)
    allowed = [str(source), *(str(p.parent) for p in models)]
    # Only the compute worker advertises this root, even in loopback SSH mode.
    compute_models = root / "compute models"
    compute_models.mkdir()
    remote_file = compute_models / "Granite β model.gguf"
    remote_file.touch()
    (compute_models / ".private").touch()
    (compute_models / "linked").symlink_to(source, target_is_directory=True)
    hosts = {}
    for host in ("source", "compute"):
        config = root / f"{host}.json"
        worker_root = root / host
        worker_root.mkdir(mode=0o700)
        worker = {"root": str(worker_root),
                  "source_roots": allowed + ([str(compute_models)] if host == "compute" else []),
                  "engine": str(bin_dir / "brewery-engine")}
        if host == "compute" and suite:
            worker["engine_env"] = {"BREWERY_BENCHMARK_CONFIG": str(suite[0])}
        config.write_text(json.dumps(worker))
        hosts[host] = ({"ssh": "compute", "ssh_config": str(ssh_fixture / "client.conf")}
                       if ssh_fixture and host == "compute" else
                       {"local": True, "worker": str(bin_dir / "brewery-worker"), "config": str(config)})
        if host == "compute" and suite:
            hosts[host].update(capabilities={"benchmark": True}, benchmark_targets=[{"id": suite[1]}],
                               benchmarks=[{"id": name} for name in suite[2]])
    config = root / "dispatch.json"
    config.write_text(json.dumps({"hosts": hosts}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    address = f"127.0.0.1:{port}"
    base = f"http://{address}"
    env = dict(os.environ, BREWERY_DATA=str(root / "state"), BREWERY_DISPATCH_CONFIG=str(config),
               BREWERY_DISPATCH=str(bin_dir / "brewery-dispatch"), BREWERY_WEB_DIST=str(web_dist),
               BREWERY_LISTEN=address, PATH=f"{bin_dir}:{os.environ['PATH']}")

    def call(path, body=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", "Origin": base})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{method or ('POST' if body is not None else 'GET')} {path}: "
                               f"{error.code} {error.read(2048).decode(errors='replace')}") from error

    @contextlib.contextmanager
    def server():
        with (root / "controller.log").open("ab") as log:
            proc = subprocess.Popen([bin_dir / "brewery"], env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    if proc.poll() is not None:
                        raise RuntimeError((root / "controller.log").read_text()[-2048:])
                    try:
                        call("/api/state")
                        break
                    except urllib.error.URLError:
                        time.sleep(.05)
                else:
                    raise RuntimeError("controller did not bind loopback")
                yield
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(8)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

    def job(kind, host, ids, options=None, **extra):
        j = call("/api/jobs", dict(kind=kind, host=host, input_ids=ids,
                                   options=options or {}, **extra))
        route = "/api/jobs/" + j["id"]
        deadline = time.monotonic() + (1800 if kind == "benchmark" else 600 if kind == "compare" else 90)
        while time.monotonic() < deadline:
            j = call(route)
            if j["state"] == "succeeded":
                print(f"{kind}: {j['state']}", flush=True)
                return j
            if j["state"] in ("failed", "cancelled", "interrupted"):
                raise RuntimeError(f"{kind}: {j.get('error') or j.get('progress')}")
            time.sleep(.25)
        call(route + "/cancel", {})
        raise RuntimeError(f"{kind} timed out; cancellation requested")

    def agent(*args, data=None, code=0):
        cli = Path(__file__).resolve().parents[1] / "tools/brewery-agent"
        proc = subprocess.run([cli, "--url", base, *args], text=True, capture_output=True,
                              input=json.dumps(data) if data is not None else None, timeout=40)
        require(proc.returncode == code, f"agent {args[0]} exited {proc.returncode}: {proc.stderr}")
        return json.loads(proc.stdout or proc.stderr)

    with server():
        with urllib.request.urlopen(base, timeout=5) as r:
            require("text/html" in r.headers.get("Content-Type", ""), "no application shell")
            require(b"<script" in r.read(), "UI entry point missing")
        def suggestions(host, prefix):
            return call("/api/suggestions/paths", {"host": host, "prefix": prefix})["items"]

        target_root = {"path": str(compute_models), "kind": "directory"}
        require(target_root in suggestions("compute", ""), "compute suggestions did not expose its configured root")
        require(target_root not in suggestions("source", ""), "a source host suggested another worker's root")
        require(suggestions("compute", str(compute_models) + "/") ==
                [{"path": str(remote_file), "kind": "file"}], "unexpected worker directory entries")
        require(suggestions("compute", str(compute_models / "Granite β")) ==
                [{"path": str(remote_file), "kind": "file"}], "worker prefix completion changed UTF-8 or spaces")
        require(not call("/api/state")["jobs"], "suggestions must not enqueue jobs")
        print("path suggestions: selected worker, UTF-8, host isolation, no jobs", flush=True)
        schema = agent("describe")
        require(schema["openapi"].startswith("3.1") and "/api/events" in schema["paths"], "agent discovery missing")
        snapshot = agent("state")
        cursor = str(snapshot["event_cursor"])
        source_req = {"name": "agent source", "host": "source", "path": str(original), "kind": "gsm8k-predictions"}
        item = agent("create", "models", "--key", "smoke.source", data=source_req)
        change = agent("events", "--after", cursor, "--limit", "1")
        require(change["event"] == "change" and change["data"]["entity_id"] == item["id"]
                and change["data"]["kind"] == "model.created", "agent did not see committed registration")
        require("path" not in change["data"], "event contains model payload")
        submit = {"kind": "transfer_preview", "host": "compute", "input_ids": [item["id"]], "options": {}}
        queued = agent("create", "jobs", "--key", "smoke.preview", data=submit)
        same = agent("create", "jobs", "--key", "smoke.preview", data=submit)
        require(queued["id"] == same["id"], "retry enqueued a duplicate")
        detail = agent("wait", same["id"], "--timeout", "30")
        require(detail["state"] == "succeeded" and detail["result"]["missing_bytes"] == len(payload),
                "agent did not retrieve verified preview")
        # Capture a bookmark before removing the registration; replay it after controller restart.
        before_remove = str(agent("state")["event_cursor"])
        agent("forget-model", item["id"])
        first = call("/api/models", {"name": "preserved source", "host": "source",
                                     "path": str(original), "kind": "gsm8k-predictions"})
        prompt = call("/api/prompts", {"name": "short test", "text": "Name a prime number."})
        j1 = job("transfer_preview", "compute", [first["id"]])
        j2 = job("transfer", "compute", [first["id"]])
        j3 = job("transfer_preview", "compute", [first["id"]])
        for j in (j1, j2, j3):
            require(isinstance(j.get("result"), dict), f"missing transfer report for {j['id']}")
        require(j1["result"]["missing_bytes"] == len(payload), "preview did not report missing bytes")
        require(j3["result"]["missing_bytes"] == 0, "verified transfer not reused")
        staged = j2["result"]["inputs"][0]
        require(Path(staged["path"]).read_bytes() == payload, "staged bytes differ from original")
        cleanup = job("cache_cleanup", "compute", [], {"bundle_sha256": staged["bundle_sha256"]})
        require(staged["bundle_sha256"] in cleanup["result"]["removed"]
                and not Path(staged["path"]).exists(), "worker cache entry still present")
        if suite:
            _, target, adapters = suite
            measured = job("benchmark", "compute", [], {"benchmark_target": target, "benchmarks": adapters})
            report = measured["result"]
            require(report["schema"] == "brewery-benchmark-report-v1" and report["target"] == target
                    and report["status"] == "complete" and report["release_qualified"] is False,
                    f"suite did not produce a complete report: {report.get('runs')}")
            require([r["benchmark"] for r in report["runs"]] == adapters and
                    all(r["status"] == "succeeded" and r["metrics"] for r in report["runs"]),
                    "missing results for one or more selected benchmarks")
            print("benchmark metrics per runner:", {r["benchmark"]: len(r["metrics"]) for r in report["runs"]})
        # For the real-trace test both models are already available on source.
        if models:
            ids = [call("/api/models", {"name": p.name, "kind": "gguf", "host": "source",
                                        "path": str(p)})["id"] for p in models]
            comparison = job("compare", "source", ids, {"raw": True, "n": 1, "labels": 0,
                                                       "threads": 2}, prompt_id=prompt["id"])
            traces = comparison["result"]["traces"]
            require(len(traces) == 2 and all(t.get("tokens") for t in traces),
                    "compare did not publish two real token traces")
        call("/api/models/" + first["id"], method="DELETE")
        require(original.read_bytes() == payload, "unregister modified the original")
    with server():
        resumed = agent("events", "--after", before_remove, "--limit", "1")
        require(resumed["data"]["kind"] == "model.deleted" and resumed["data"]["entity_id"] == item["id"],
                "agent could not resume a committed change after restart")
        require(agent("create", "jobs", "--key", "smoke.preview", data=submit)["id"] == queued["id"],
                "job submission receipt lost across restart")
        require(agent("events", "--after", str(2**62), "--limit", "1", code=3)["event"] == "resync",
                "invalidated cursor must require resync")
        print("agent API: discovery, keyed job, result, SSE restart/resync", flush=True)
        state = call("/api/state")
        require(any(p["id"] == prompt["id"] for p in state["prompts"]), "prompt lost on restart")
        require(not any(m["id"] == first["id"] for m in state["models"]), "entry reappeared")
        require(call("/api/jobs/" + j2["id"])["state"] == "succeeded", "job lost on restart")
        if suite:
            require(call("/api/jobs/" + measured["id"])["result"] == report, "benchmark results lost on restart")
            if report_file:
                # Keep the actual controller export; the fixture's worker paths are temporary.
                with report_file.open("x") as out:
                    json.dump(measured, out, indent=2)
    print("source unchanged:", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin-dir", type=Path, required=True)
    p.add_argument("--web-dist", type=Path, default=Path("web/dist"))
    p.add_argument("--gguf", type=Path, action="append", default=[])
    p.add_argument("--ssh", action="store_true", help="use an isolated loopback SSH worker for transfers")
    p.add_argument("--benchmark-config", type=Path, help="trusted worker config for optional managed suite smoke")
    p.add_argument("--benchmark-target", help="ID of an admitted endpoint in the trusted config")
    p.add_argument("--benchmark", action="append", help="adapter ID to execute; repeat to run several")
    p.add_argument("--benchmark-export", type=Path, help="save the validated benchmark job to a new file")
    args = p.parse_args()
    if len(args.gguf) not in (0, 2):
        p.error("supply zero or two --gguf files")
    if any((args.benchmark_config, args.benchmark_target, args.benchmark)) and not all(
            (args.benchmark_config, args.benchmark_target, args.benchmark)):
        p.error("use --benchmark-config, --benchmark-target and --benchmark together")
    if args.benchmark_export and not args.benchmark:
        p.error("--benchmark-export requires a benchmark run")
    suite = ((args.benchmark_config.resolve(strict=True), args.benchmark_target, args.benchmark)
             if args.benchmark else None)
    with tempfile.TemporaryDirectory(prefix="brewery-workbench-") as tmp:
        with transport_fixture(args.ssh, Path(tmp), args.bin_dir.resolve()) as fixture:
            smoke(args.bin_dir.resolve(), args.web_dist.resolve(),
                  [g.resolve(strict=True) for g in args.gguf], Path(tmp), fixture, suite, args.benchmark_export)
