"""Exercise a settings revision through real local worker dispatch; no model or GPU."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


def check(bins, root):
    def save(p, value):
        p.write_text(json.dumps(value))
        p.chmod(0o600)

    work = root / "worker"
    work.mkdir(mode=0o700)
    runner = root / "check-target"
    runner.write_text("""#!/usr/bin/python3
import json,sys
x=json.load(sys.stdin)
assert x['endpoint']=='http://127.0.0.1:8080/v1' and x['model']=='selected-model'
print(json.dumps({'samples':1,'metrics':[{'name':'ok','unit':'count','direction':'higher','value':1}]}))
""")
    runner.chmod(0o700)
    catalog = root / "catalog.json"
    save(catalog, {"adapters":{"check":{"executable":str(runner),"version":"1","protocol":"settings-smoke"}}})
    cfg = root / "worker.json"
    save(cfg, {"root":str(work),"source_roots":[],"engine":str(bins / "brewery-engine"),
               "managed_settings":{"catalog":str(catalog),"allowed_ports":[8080],"evaluation_allowed":False}})
    dc = root / "dispatch.json"
    save(dc, {"hosts":{"probe":{"local":True,"config":str(cfg),"worker":str(bins / "brewery-worker"),
                                "managed_settings":True}}})
    with socket.socket() as s:
        s.bind(("127.0.0.1",0))
        port = s.getsockname()[1]
    addr = f"127.0.0.1:{port}"
    base = "http://" + addr
    env = dict(os.environ,BREWERY_LISTEN=addr,BREWERY_DATA=str(root/"db"),BREWERY_DISPATCH_CONFIG=str(dc),
               BREWERY_DISPATCH=str(bins/"brewery-dispatch"),BREWERY_POLL_MS="50")
    opener = build_opener(ProxyHandler({}))

    def call(method, path, body=None):
        q = Request(base+"/api/"+path, method=method, data=json.dumps(body).encode() if body is not None else None,
                    headers={"Content-Type":"application/json","Origin":base})
        with opener.open(q,timeout=5) as r:
            return json.load(r)

    @contextlib.contextmanager
    def server():
        with (root/"controller.log").open("ab") as log:
            p = subprocess.Popen([bins/"brewery"],env=env,stdout=log,stderr=log)
            try:
                for _ in range(60):
                    if p.poll() is not None:
                        raise RuntimeError((root/"controller.log").read_text()[-1000:])
                    try:
                        call("GET","state")
                        break
                    except URLError:
                        time.sleep(.05)
                yield
            finally:
                p.terminate()
                p.wait(timeout=6)

    with server():
        current = call("GET","settings")["hosts"][0]
        assert current["managed"] and "check" in [s["id"] for s in current["approved_suites"]],current
        data = dict(revision=current["revision"],benchmark_targets=[
            dict(id="local-test",endpoint="http://127.0.0.1:8080/v1",model="selected-model")
        ],enabled_suites=["check"],baseline_endpoints=[],evaluation_enabled=False)
        updated = call("PUT","settings/probe",data)
        assert updated["revision"]>current["revision"]
        try:
            call("PUT","settings/probe",data)
        except HTTPError as e:
            assert e.code == 409
        else:
            raise AssertionError("stale save accepted")
        assert not call("GET","state")["jobs"],"settings save started a job"
    with server():
        state = call("GET","state")
        assert state["hosts"][0]["capabilities"]["benchmark"],state["hosts"][0]
        assert call("GET","settings")["hosts"][0]["revision"] == updated["revision"]
        job = call("POST","jobs",dict(kind="benchmark",host="probe",input_ids=[],
                                     options={"benchmark_target":"local-test","benchmarks":["check"]}))
        for _ in range(120):
            job = call("GET","jobs/"+job["id"])
            if job["state"] in ("succeeded","failed","cancelled"):
                break
            time.sleep(.1)
        assert job["state"]=="succeeded",job
        assert job["result"]["status"]=="complete" and job["result"]["runs"][0]["metrics"][0]["value"]==1,job["result"]
        print("settings applied to worker, stale save rejected, no auto job, survived restart; adapter got selected endpoint/model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    with tempfile.TemporaryDirectory(prefix="brewery-settings-test-") as folder:
        check(parser.parse_args().bin_dir.resolve(), Path(folder))
