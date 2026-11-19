#!/usr/bin/python3
"""Exercise Settings against an isolated controller and staged local-eval worker."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from bootstrap_admin import bootstrap


def verify(ctrl, scratch, staged):
    scratch = scratch.resolve(strict=True)
    staged = staged.resolve(strict=True)
    if not scratch.is_relative_to("/arboretum") or not staged.is_relative_to(scratch):
        raise ValueError("use a staged worker under a private /arboretum scratch directory")
    with tempfile.TemporaryDirectory(prefix="settings-smoke-", dir=scratch) as name:
        path = Path(name)
        bootstrap(path / "admin")
        data = path / "controller"
        dispatch = path / "dispatch.json"
        cfg = json.loads((staged / "dispatch.proposed.json").read_bytes())
        isolated = {}
        for key, val in cfg["hosts"].items():
            # Leave the other hosts visible but do not point to production
            # worker code/config/state, even if somebody tests dispatch later.
            isolated[key] = val if key == "local-eval" else {"local": True, "worker": "/usr/bin/false",
                                                               "config": str(path / "unused.json")}
        dispatch.write_text(json.dumps({"hosts": isolated}))
        catalog = path / "admin/catalog.json"
        catalog.touch(mode=0o600)
        catalog.write_text('{"workers":{"local-eval":{"local":true,"initially_active":true}}}\n')
        token = (path / "admin/operator.key").read_text().strip()
        env = {"PATH": "/usr/bin", "HOME": name, "BREWERY_ADMIN_KEY_FILE": str(path / "admin/verify.sha256"),
               "BREWERY_ADMIN_CATALOG": str(catalog),
               "BREWERY_DATA": str(data), "BREWERY_LISTEN": "127.0.0.1:0",
               "BREWERY_DISPATCH": str(ctrl.resolve(strict=True).with_name("brewery-dispatch")),
               "BREWERY_DISPATCH_CONFIG": str(dispatch)}
        opener = build_opener(ProxyHandler({}))
        with subprocess.Popen([ctrl.resolve(strict=True)], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as proc:
            try:
                import select
                assert select.select([proc.stdout], [], [], 5)[0], "controller startup timed out"
                address = re.fullmatch(r"brewery listening on (http://127\.0\.0\.1:\d+)", proc.stdout.readline().strip())
                assert address, "controller did not start"

                def request(route, status=200, method=None, value=None, key=None):
                    headers = {"Content-Type": "application/json"}
                    if key:
                        headers["X-Brewery-Admin-Key"] = key
                    req = Request(address[1] + "/api/" + route, method=method, headers=headers,
                                  data=json.dumps(value).encode() if value is not None else None)
                    try:
                        with opener.open(req, timeout=3) as response:
                            assert response.status == status, (route, response.status, status)
                            return json.load(response)
                    except HTTPError as error:
                        assert error.code == status, (route, error.code, status)
                        return None

                def config():
                    return next(row for row in request("settings")["hosts"] if row["id"] == "local-eval")

                def caps():
                    return next(row for row in request("state")["hosts"] if row["id"] == "local-eval")["capabilities"]

                assert request("admin/state", 401) is None
                def admin():
                    rows = request("admin/state", key=token)["workers"]
                    assert [row["id"] for row in rows] == ["local-eval"]
                    return rows[0]["status"]

                assert admin() == "active"
                initial = config()
                assert initial["ready"] and initial["managed"] and initial["approved_ports"] == [8080]
                assert {s["id"] for s in initial["approved_suites"]} == {"betterbench", "tool-eval-bench"}
                assert not initial["evaluation_allowed"] and not initial["baseline_allowed"]
                assert not caps()["benchmark"]
                values = {k: initial[k] for k in ("revision", "benchmark_targets", "enabled_suites",
                                                  "baseline_endpoints", "evaluation_enabled")}
                values["benchmark_targets"] = [{"id": "fixture-only", "endpoint": "http://127.0.0.1:8080/v1",
                                                "model": "fixture-only"}]
                values["enabled_suites"] = ["betterbench", "tool-eval-bench"]
                route = "settings/local-eval"
                rejected = json.loads(json.dumps(values))
                rejected["benchmark_targets"][0]["endpoint"] = "http://127.0.0.1:8081/v1"
                request(route, status=400, method="PUT", value=rejected)
                request(route, method="PUT", value=values)
                observed = config()
                assert observed["revision"] == values["revision"] + 1 and caps()["benchmark"]
                request(route, status=409, method="PUT", value=values)
                values["revision"] = observed["revision"]
                values["benchmark_targets"] = []
                values["enabled_suites"] = []
                request(route, method="PUT", value=values)
                assert not caps()["benchmark"] and request("state")["jobs"] == []
                request("admin/actions", method="POST", key=token,
                        value={"action": "deactivate", "worker": "local-eval"})
                assert admin() == "inactive"
                request("admin/actions", method="POST", key=token,
                        value={"action": "approve", "worker": "local-eval"})
                assert admin() == "active" and request("state")["jobs"] == []
            finally:
                proc.terminate()
                proc.wait(timeout=5)
                (staged / "worker/operational-settings.json").unlink(missing_ok=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--controller", type=Path, required=True)
    cli.add_argument("--scratch", type=Path, required=True)
    cli.add_argument("--staged", type=Path, required=True)
    args = cli.parse_args()
    verify(args.controller, args.scratch, args.staged)
    print("Settings round trip and separate admin key passed; no job or inference sent.")
