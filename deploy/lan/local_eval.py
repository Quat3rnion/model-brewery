#!/usr/bin/python3
"""Stage a catalog-only local-eval worker at a new private path; do not publish it."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess

REPO = Path(__file__).resolve().parents[2]
SUITES = {
    "betterbench": ("0.6.1", "betterbench-decode-chat-v1", "betterbench"),
    "tool-eval-bench": ("3.0.0", "tool-eval-safe-v1", "tool_eval_bench"),
}
HOST = "local-eval"
DISABLED = ("baseline", "benchmark", "cache_cleanup", "compare", "convert", "evaluate",
            "generate", "inspect", "plan", "transfer", "transfer_preview", "verify")


def verify_venv(root):
    requirements = {name: (version, module) for name, (version, _, module) in SUITES.items()}
    requirements.update({"httpx": ("0.28.1", "httpx"), "python-dotenv": ("1.2.4", "dotenv")})
    python = root / "bin/python"
    code = """import sys, json, importlib
from importlib.metadata import version
for name, (pin, module) in json.loads(sys.argv[1]).items():
    if version(name) != pin:
        raise SystemExit('wrong version of ' + name)
    importlib.import_module(module)
"""
    subprocess.run([python, "-EBs", "-c", code, json.dumps(requirements)], check=True, timeout=10,
                   env={"PATH": "/usr/bin", "LC_ALL": "C.UTF-8"})
    for suite in SUITES:
        # Verify CLI startup only: no endpoint options and no requests.
        subprocess.run([python, "-EBs", root / "bin" / suite, "--help"], check=True, timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env={"PATH": "/usr/bin", "LC_ALL": "C.UTF-8"})


def write_json(path, value):
    with path.open("x") as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stage(root, release, venv, base_dispatch, port):
    root = root.absolute()
    if root.parent.resolve(strict=True) != root.parent:
        raise ValueError("enrollment root parent must be canonical")
    dispatch = json.loads(base_dispatch.read_bytes())
    if HOST in dispatch["hosts"]:
        raise ValueError(f"{HOST} already exists")
    venv = venv.resolve(strict=True)
    for path in (venv, *venv.rglob("*")):
        if path == venv / ".lock":
            continue
        resolved = path.resolve(strict=True)
        meta = resolved.stat()
        if meta.st_uid not in (0, os.getuid()) or meta.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(f"untrusted Python runtime path: {path}")
    verify_venv(venv)
    os.umask(0o077)
    root.mkdir(mode=0o700)
    try:
        binary = root / "bin"
        binary.mkdir()
        (root / "worker").mkdir()
        for name in ("brewery-worker", "brewery-engine"):
            dest = binary / name
            shutil.copyfile(release / name, dest)
            dest.chmod(0o700)
        runtime = root / "runtime"
        shutil.copytree(venv, runtime, symlinks=True, ignore=shutil.ignore_patterns(".lock"))
        verify_venv(runtime)
        adapters = {}
        for name, (version, protocol, _) in SUITES.items():
            script = root / name
            shutil.copyfile(REPO / "benchmarks" / name, script)
            wrapper = binary / name
            wrapper.write_text("#!/bin/sh\nexec " + " ".join(map(shlex.quote,
                [str(runtime / "bin/python"), "-EBs", str(script)])) + "\n")
            wrapper.chmod(0o700)
            adapters[name] = {"executable": str(wrapper), "version": version, "protocol": protocol}
        shutil.copyfile(REPO / "benchmarks/_adapter.py", root / "_adapter.py")
        write_json(root / "catalog.json", {"adapters": adapters})
        write_json(root / "worker.json", {
            "root": str(root / "worker"),
            "source_roots": [],
            "engine": str(binary / "brewery-engine"),
            "managed_settings": {
                "catalog": str(root / "catalog.json"),
                "allowed_ports": [port],
                "baseline_allowed": False,
                "evaluation_allowed": False,
            },
            "engine_env": {
                "BREWERY_CPU_THREADS": "4",
                "BREWERY_ALLOW_GPU_LOAD": "0",
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
            },
        })
        dispatch["hosts"][HOST] = {
            "local": True, "worker": str(binary / "brewery-worker"),
            "config": str(root / "worker.json"), "managed_settings": True,
            "capabilities": dict.fromkeys(DISABLED, False),
        }
        write_json(root / "dispatch.proposed.json", dispatch)
    except BaseException:
        shutil.rmtree(root)
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "release", "venv", "dispatch"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--approved-port", type=int, required=True, metavar="PORT")
    args = cli.parse_args()
    if not 1 <= args.approved_port <= 65535:
        cli.error("--approved-port must be 1 through 65535")
    stage(args.root, args.release, args.venv, args.dispatch, args.approved_port)
