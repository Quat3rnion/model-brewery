#!/usr/bin/env python3
"""Export selected live controller jobs and run the release evidence checker."""
import argparse
from http.client import HTTPException
import ipaddress
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener
from uuid import UUID


MAX_RESPONSE = 32 * 1024 * 1024
STATES = ("queued", "running", "succeeded", "failed", "cancelled")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def controller_url(value):
    try:
        url = urlsplit(value)
        if (
            url.scheme != "http" or url.path not in ("", "/") or url.query
            or url.fragment or url.username is not None
            or not ipaddress.ip_address(url.hostname).is_loopback or not url.port
        ):
            raise ValueError
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            "use an explicit loopback HTTP address and port, for example http://127.0.0.1:4141"
        ) from None
    return f"http://{url.netloc}"


def job_id(value):
    try:
        ident = UUID(value)
        if ident.version == 4 and str(ident) == value:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use the job UUID returned by the controller")


def row(value):
    label, equals, ident = value.partition("=")
    if not equals or "/" not in label:
        raise argparse.ArgumentTypeError("use TARGET/PROFILE=JOB_ID")
    return label, job_id(ident)


def get(opener, url):
    with opener.open(url, timeout=10) as response:
        if response.status != 200:
            raise ValueError(f"expected HTTP 200, got {response.status}")
        raw = response.read(MAX_RESPONSE + 1)
    if len(raw) > MAX_RESPONSE:
        raise ValueError("controller response too large")
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError("expected a controller JSON object")
    return raw, body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=controller_url, default="http://127.0.0.1:4141")
    parser.add_argument("--exports", type=Path, required=True, help="directory for fresh API snapshots")
    parser.add_argument("--artifacts", type=Path, help="checker artifact directory, organized by job ID")
    parser.add_argument("--baseline-job-id", type=job_id, help="selected baseline job (also fetched)")
    parser.add_argument("--row", type=row, action="append", default=[], metavar="TARGET/PROFILE=JOB_ID",
                        help="requested matrix row and selected conversion job (repeatable)")
    parser.add_argument("job_ids", metavar="JOB_ID", type=job_id, nargs="*",
                        help="selected verify, transfer, generate and evaluate jobs")
    args = parser.parse_args(argv)
    ids = list(dict.fromkeys(
        ([args.baseline_job_id] if args.baseline_job_id else []) + [ident for _, ident in args.row] + args.job_ids
    ))
    opener = build_opener(ProxyHandler({}), NoRedirect())
    checker = [sys.executable, str(Path(__file__).with_name("release_check.py"))]
    if args.baseline_job_id:
        checker += ["--baseline-job-id", args.baseline_job_id]
    if args.artifacts is not None:
        checker += ["--artifacts", str(args.artifacts.resolve())]
    for label, ident in args.row:
        checker += ["--row", f"{label}={ident}"]

    collection_complete = bool(ids)
    try:
        args.exports.mkdir(parents=True, exist_ok=True)
        if not ids:
            _, state = get(opener, f"{args.controller}/api/state")
            if not isinstance(state.get("jobs"), list):
                raise ValueError("controller state has no jobs list")
            print(f"No job IDs selected ({len(state['jobs'])} jobs in controller).", file=sys.stderr)
        else:
            snapshot = Path(tempfile.mkdtemp(prefix="snapshot-", dir=args.exports.resolve()))
            print(f"Controller exports from {args.controller}: {snapshot}", file=sys.stderr)
            for ident in ids:
                try:
                    raw, detail = get(opener, f"{args.controller}/api/jobs/{ident}")
                    state = detail.get("state")
                    if detail.get("id") != ident or state not in STATES or not isinstance(detail.get("events"), list):
                        raise ValueError("response is not the requested controller job detail")
                    path = snapshot / f"{ident}.json"
                    temporary = path.with_suffix(".partial")
                    temporary.write_bytes(raw)
                    temporary.replace(path)
                    checker.append(str(path))
                    collection_complete &= state == "succeeded"
                    print(f"{ident}: {state}", file=sys.stderr)
                except HTTPError as exc:
                    collection_complete = False
                    status = "missing" if exc.code == 404 else "unavailable"
                    print(f"{ident}: {status} (HTTP {exc.code})", file=sys.stderr)
                except (ValueError, OSError, HTTPException) as exc:
                    collection_complete = False
                    print(f"{ident}: unavailable ({exc})", file=sys.stderr)
    except (ValueError, OSError, HTTPException) as exc:
        collection_complete = False
        print(f"Collection unavailable: {exc}", file=sys.stderr)

    # Only pass this invocation's exports. Prior snapshots are audit records,
    # not a substitute for unavailable jobs on the selected controller.
    result = subprocess.run(checker, check=False)
    return result.returncode if result.returncode > 0 else int(result.returncode != 0 or not collection_complete)


if __name__ == "__main__":
    sys.exit(main())
