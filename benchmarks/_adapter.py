"""Private support for the two fixed benchmark executables (Linux worker)."""

import importlib.metadata
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import sysconfig
from urllib.parse import urlsplit

MAX_REPORT = 4 * 1024 * 1024
RAW = Path("raw.json")


class AdapterError(Exception):
    pass


def read_request(benchmark, version, protocol):
    if len(sys.argv) != 1:
        raise AdapterError("this adapter accepts JSON stdin only (no arguments)")
    data = sys.stdin.buffer.read(8193)
    if len(data) > 8192:
        raise AdapterError("request exceeds 8 KiB")
    request = json.loads(data)
    if not isinstance(request, dict) or set(request) != {
        "benchmark", "target", "endpoint", "model", "version", "protocol",
    }:
        raise AdapterError("unexpected request fields")
    if any(not isinstance(v, str) or not v or not v.isprintable()
           for v in request.values()):
        raise AdapterError("request fields must be nonempty printable strings")
    if (request["benchmark"], request["version"], request["protocol"]) != (
        benchmark, version, protocol,
    ):
        raise AdapterError(f"expected benchmark/version/protocol {benchmark}/{version}/{protocol}")
    return endpoint(request["endpoint"]), request["model"]


def endpoint(value):
    # Do not use a hostname, scoped address, implicit path, URL credentials,
    # query/fragment, or anything that needs DNS to establish the destination.
    if not re.fullmatch(r"https?://(?:[0-9.]+|\[::1\])(?::[0-9]+)?(?:/v1)?/?", value):
        raise AdapterError("expected a numeric loopback HTTP(S) endpoint at / or /v1")
    parsed = urlsplit(value)
    address = ipaddress.ip_address(parsed.hostname)
    if not address.is_loopback or parsed.port == 0:
        raise AdapterError("expected a numeric loopback HTTP(S) endpoint")
    host = str(address) if address.version == 4 else "[::1]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{host}{port}/v1"


def _limit_files():
    # Invoked in the child of a single-threaded adapter. Covers the upstream
    # log, report, and any auxiliary files without buffering logs in memory.
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_REPORT, MAX_REPORT))


def run_cli(name, version, args):
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        raise AdapterError(f"install {name} {version} in the adapter Python environment") from None
    if installed != version:
        raise AdapterError(f"expected {name} {version}, installed {installed}")
    cli = Path(sysconfig.get_path("scripts")) / name
    cwd = Path.cwd()
    with open("upstream.log", "xb") as log:
        result = subprocess.run(
            [sys.executable, "-EBs", str(cli), *args],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            # No proxy/credentials/dotenv. No PATH commands (including
            # BetterBench's best-effort host GPU fingerprint subprocesses).
            env={"PATH": "/dev/null", "HOME": str(cwd), "TMPDIR": str(cwd),
                 "XDG_CACHE_HOME": str(cwd), "XDG_CONFIG_HOME": str(cwd),
                 "XDG_DATA_HOME": str(cwd), "PYTHON_DOTENV_DISABLED": "1"},
            timeout=285, preexec_fn=_limit_files, check=False,
        )
    if result.returncode:
        raise AdapterError(f"{name} exited {result.returncode}; see private upstream.log")
    with RAW.open("rb") as report:
        raw = report.read(MAX_REPORT + 1)
    if len(raw) >= MAX_REPORT:
        raise AdapterError("raw report reached the 4 MiB limit")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("error"):
        raise AdapterError("upstream returned an invalid or error report")
    return payload


def metric(name, value, unit, direction):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise AdapterError(f"invalid metric {name}")
    return dict(name=name, value=value, unit=unit, direction=direction)


def main(benchmark, version, protocol, run):
    try:
        base, model = read_request(benchmark, version, protocol)
        os.umask(0o077)
        result = run(base, model)
        print(json.dumps(result, allow_nan=False, separators=(",", ":")))
    except AdapterError as exc:
        print(f"{benchmark}: {exc}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, KeyError, TypeError, OSError, OverflowError, subprocess.SubprocessError) as exc:
        # Do not echo model-controlled text, endpoint errors, or stack traces.
        print(f"{benchmark}: invalid input, incomplete report, or CLI failure ({type(exc).__name__})",
              file=sys.stderr)
        sys.exit(1)
