#!/usr/bin/python3
"""Check actual profile envelopes against local admission in an isolated controller."""
import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
from urllib.request import ProxyHandler, Request, build_opener


def fake_dispatch(root, action, host, ident):
    import local_components as admission

    admission.HOME = root
    admission.STATE = root / "worker"
    admission.COMPONENTS = root / "components"
    admission.CATALOG = root / "catalog.json"
    admission.source_path = lambda: root / "base"
    # Approve scratch budgets only. This fixture never starts a real worker.
    admission.ram_available = lambda: 20 << 30
    admission.disk_available = lambda _: 300_000_000_000
    record = root / f"{ident}.json"
    if action == "start":
        job = json.load(sys.stdin)
        admission.precheck({"op": "prepare", "action": "start", "job": job})
        assert admission.validate_job(job) == job["kind"]  # same check as engine admission
        record.write_text(json.dumps(job))
    else:
        job = json.loads(record.read_text())
    kind = job["kind"]
    result = {"kind": kind, "host": host, "disk_preflight": {"fits": True}, "source_sha256": "b" * 64,
              "custom_profile": {"revision": job["options"]["profile_recipe"]["revision"], "release_qualified": False}}
    if kind == "convert":
        output = admission.STATE / "jobs" / ident / "work/converted"
        output.mkdir(parents=True, exist_ok=True)
        result.update(report={"published": True}, artifact={"path": str(output), "kind": "hgn",
                      "sha256": "a" * 64, "bytes": 1})
    if kind == "verify":
        result.update(report={"passed": True, "manifest_schema": 2, "provenance_attested": True,
                              "source_tensors": 1, "output_tensors": 1, "payload_bytes": 1, "errors": {}},
                      inputs=[{}, {**job["inputs"][1], "sha256": "a" * 64, "bytes": 1}],
                      file_verification={"status": "passed"})
    print(json.dumps({"state": "succeeded", "progress": "fixture", "result": result}))


def run(controller):
    with tempfile.TemporaryDirectory(prefix="brewery-profile-smoke-", dir="/arboretum") as scratch:
        root = Path(scratch)
        (root / "base").mkdir()
        (root / "components").mkdir()
        (root / "components/vision.safetensors").touch()
        dispatch = root / "dispatch.json"
        dispatch.write_text(json.dumps({"hosts": {"local-components": {"local": True,
                        "component_cpu_only": True, "capabilities": {
                            "custom_profiles": True, "convert_components": True}}}}))
        fake = root / "dispatch"
        fake.write_text(f'#!/bin/sh\nexec /usr/bin/python3 -B "{Path(__file__).resolve()}" --fake "{root}" "$@"\n')
        fake.chmod(0o700)
        env = {"PATH": "/usr/bin", "BREWERY_DATA": str(root / "db"), "BREWERY_DISPATCH": str(fake),
               "BREWERY_DISPATCH_CONFIG": str(dispatch), "BREWERY_LISTEN": "127.0.0.1:0", "BREWERY_POLL_MS": "50"}
        with subprocess.Popen([controller], env=env, stdout=subprocess.PIPE, text=True) as proc:
            try:
                assert select.select([proc.stdout], [], [], 5)[0]
                url = proc.stdout.readline().strip().rsplit(" ", 1)[-1]
                assert url.startswith("http://127.0.0.1:") and url != "http://127.0.0.1:4141"
                opener = build_opener(ProxyHandler({}))

                def api(path, data=None):
                    with opener.open(Request(url + "/api/" + path,
                       data=json.dumps(data).encode() if data is not None else None,
                       headers={"Content-Type": "application/json"}), timeout=4) as response:
                        return json.load(response)

                catalog = api("profiles/catalog")
                (root / "catalog.json").write_text(json.dumps(catalog))
                tensor = next(row["name"] for row in catalog["tensors"] if row["editable"])
                saved = api("profiles", {"name": "Scratch é", "base": "v2",
                        "catalog_sha256": catalog["catalog_sha256"], "overrides": {tensor: "ht43"}})
                ids = [api("models", {"name": name, "kind": kind, "host": "local-components",
                                    "path": str(root / path)})["id"]
                       for name, kind, path in (
                           ("Pinned fixture", "hf", "base"),
                           ("Overlay fixture", "vision", "components/vision.safetensors"))]
                for count in (1, 2):
                    workflow = api("conversions", {"host": "local-components", "input_ids": ids[:count],
                                   "profile_revision": saved["revision"],
                                   "options": {"vision": 1} if count == 2 else {}})
                    while workflow["state"] == "running":
                        for _ in range(100):
                            time.sleep(0.1)
                            workflow = api(f"conversions/{workflow['id']}")
                            if workflow["state"] != "running":
                                break
                        else:
                            raise AssertionError("workflow did not finish")
                    assert workflow["state"] == "succeeded", workflow
                    for kind in ("inspect", "plan", "convert", "verify"):
                        envelope = json.loads((root / f"{workflow['job_ids'][kind]}.json").read_bytes())
                        assert envelope["options"]["profile_recipe"]["revision"] == saved["revision"]
                        if count == 2:
                            assert envelope["options"]["vision"] == (2 if kind == "verify" else 1)
            finally:
                proc.terminate()
                proc.wait(timeout=5)


if __name__ == "__main__":
    if sys.argv[1] == "--fake":
        fake_dispatch(Path(sys.argv[2]), *sys.argv[3:])
    else:
        run(Path(sys.argv[1]).resolve(strict=True))
        print("Actual controller recipe-only and combined envelopes passed admission for all four stages.")
