"""CPU-only integration test with real Rust workers and a synthetic native CLI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

SCRIPT = Path(__file__).with_name("distributed_conversion.py")
NATIVE = r"""#!/usr/bin/python3
import argparse, hashlib, json, os, pathlib, time
p = argparse.ArgumentParser()
p.add_argument("action")
for k in ("source", "partition", "unit", "package", "results", "output",
          "profile", "seed", "threads", "ht-beam", "ht-backend",
          "i4r-backend", "q4cp-backend", "decode-backend"):
    p.add_argument("--" + k)
a = p.parse_args()
o = pathlib.Path(a.output)
o.mkdir()
if a.action == "partition":
    part = json.dumps({"source":(pathlib.Path(a.source)/"data").read_text(),"options":{
        k+"_backend":getattr(a,k+"_backend") for k in ("ht","i4r","q4cp")}})
    (o/"partition.json").write_text(part)
    out = {"schema":1,"partition_sha256":hashlib.sha256(part.encode()).hexdigest(),
           "units":[{"id":i,"input_bytes":1,"output_bytes":1} for i in range(3)]}
else:
    digest = hashlib.sha256(pathlib.Path(a.partition).read_bytes()).hexdigest()
    out = {"schema":1, "partition_sha256":digest}
    if a.action == "pack-unit":
        (o/"data").write_text(a.unit)
        out["unit"] = int(a.unit)
    elif a.action == "encode-unit":
        time.sleep(float(os.environ.get("FIXTURE_DELAY", "0")))
        unit = int((pathlib.Path(a.package)/"data").read_text())
        (o/"unit.json").write_text(json.dumps([digest,unit]))
        out["unit"] = unit
    elif a.action == "merge":
        for i in range(3):
            assert json.loads((pathlib.Path(a.results)/str(i)/"unit.json").read_text()) == [digest,i]
        (o/"fixture-not-hgn").write_text("synthetic")
        out.update(published=True,verification={"provenance_attested":True,"manifest_schema":2,"synthetic":True})
    else:
        raise AssertionError(a.action)
print(json.dumps(out))
"""


@unittest.skipUnless(os.getenv("BREWERY_TEST_BIN"), "build worker and set BREWERY_TEST_BIN")
class DistributedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/arboretum")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        binaries = Path(os.environ["BREWERY_TEST_BIN"]).resolve()
        native = self.root / "fixture"
        native.write_text(NATIVE)
        native.chmod(0o700)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "data").write_text("fixture")
        engine = self.root / "engine"
        engine.symlink_to(SCRIPT.resolve())
        hosts = {}
        for name in ("a", "b"):
            cfg = self.root / f"{name}.json"
            cfg.write_text(json.dumps({
                "root": str(self.root / name),
                "source_roots": [str(self.source)],
                "engine": str(engine), "disk_reserve_bytes": 0,
                "engine_env": {"BREWERY_STAGE2_NATIVE": str(native), "FIXTURE_DELAY": "1.5"},
            }))
            hosts[name] = {"local": True, "worker": str(binaries / "brewery-worker"),
                           "config": str(cfg)}
        dispatch = self.root / "dispatch.json"
        dispatch.write_text(json.dumps({"hosts": hosts}))
        digest = hashlib.sha256(native.read_bytes()).hexdigest()
        self.config = self.root / "workflow.json"
        self.config.write_text(json.dumps({
            "dispatcher": str(binaries / "brewery-dispatch"), "dispatch_config": str(dispatch),
            "source": {"host": "a", "path": str(self.source)}, "coordinator": "a",
            "workers": {k: {"native_sha256": digest, "backend": ["cpu", "cpu", "cpu"]}
                        for k in hosts},
            "encoding": {"profile": "v2"},
        }))

    def execute(self, seconds):
        return subprocess.run(
            ["/usr/bin/python3", SCRIPT, "run", "--config", self.config, "--run-dir",
             self.root / "run", "--timeout", str(seconds), "--reserved-workers"],
            capture_output=True, text=True, timeout=30,
        )

    def interrupt_during_encode(self):
        stopped = self.execute(1)
        self.assertNotEqual(stopped.returncode, 0, stopped)
        self.assertIn("deadline", stopped.stderr)
        journal = json.loads((self.root / "run/journal.json").read_text())
        self.assertIn("encode-0", journal["jobs"])
        return journal

    def test_resume_two_local_workers(self):
        before = self.interrupt_during_encode()
        completed = self.execute(15)
        self.assertEqual(completed.returncode, 0, completed)
        result = json.loads(completed.stdout)
        self.assertFalse(result["release_qualified"])
        artifact = Path(result["artifact"]["path"])
        self.assertEqual((artifact / "fixture-not-hgn").read_text(), "synthetic")
        later = json.loads((self.root / "run/journal.json").read_text())
        self.assertEqual(later["run_id"], before["run_id"])
        self.assertEqual(later["jobs"]["encode-0"]["id"], before["jobs"]["encode-0"]["id"])
        self.assertEqual({later["jobs"][f"encode-{i}"]["host"] for i in range(3)}, {"a", "b"})
        self.assertEqual(json.loads(self.execute(15).stdout), result)

    def test_mutated_unit_blocks_merge(self):
        journal = self.interrupt_during_encode()
        j = journal["jobs"]["encode-0"]
        deadline = time.monotonic() + 8
        while True:
            worker = self.root / j["host"]
            found = list(worker.glob(f"**/{j['id']}/work/result/unit.json"))
            status = subprocess.check_output(
                [Path(os.environ["BREWERY_TEST_BIN"]) / "brewery-dispatch", "status", j["host"], j["id"]],
                env=dict(os.environ, BREWERY_DISPATCH_CONFIG=str(self.root / "dispatch.json")),
            )
            if json.loads(status)["state"] == "succeeded":
                self.assertEqual(len(found), 1)
                found[0].write_text("corrupted")
                break
            self.assertLess(time.monotonic(), deadline, "fixture encode did not finish")
            time.sleep(0.05)
        result = self.execute(15)
        self.assertNotEqual(result.returncode, 0, result)
        logs = list(self.root.glob("**/engine-stderr.log"))
        diagnostics = result.stderr + "\n".join(log.read_text() for log in logs)
        self.assertRegex(diagnostics, "digest|SHA-256", diagnostics)
        self.assertFalse(list(self.root.glob("**/fixture-not-hgn")))

    @unittest.skipUnless(os.getenv("BREWERY_TEST_NATIVE_FIXTURE"), "set native scratch fixture harness")
    def test_native_tiny_v2_and_ht43_match_serial_reference(self):
        harness = Path(os.environ["BREWERY_TEST_NATIVE_FIXTURE"]).resolve()
        native_sha256 = hashlib.sha256(harness.read_bytes()).hexdigest()
        for profile in ("v2", "ht43"):
            with self.subTest(profile=profile):
                fixture = self.root / profile
                subprocess.run([harness, "fixture", "--root", fixture, "--profile", profile],
                               check=True, capture_output=True, timeout=20)
                binaries = Path(os.environ["BREWERY_TEST_BIN"]).resolve()
                hosts = {}
                for name in ("a", "b"):
                    cfg = self.root / f"{profile}-{name}.json"
                    cfg.write_text(json.dumps({
                        "root": str(self.root / f"{profile}-{name}"),
                        "source_roots": [str(fixture / "source")],
                        "engine": str(SCRIPT.resolve()), "disk_reserve_bytes": 0,
                        "engine_env": {"BREWERY_STAGE2_NATIVE": str(harness)},
                    }))
                    hosts[name] = {"local": True, "worker": str(binaries / "brewery-worker"),
                                   "config": str(cfg)}
                dispatch = self.root / f"{profile}-dispatch.json"
                dispatch.write_text(json.dumps({"hosts": hosts}))
                config = json.loads(self.config.read_text())
                config.update(dispatch_config=str(dispatch),
                              source={"host": "a", "path": str(fixture / "source")},
                              workers={k: {"backend": ["cpu"] * 3, "native_sha256": native_sha256}
                                       for k in hosts},
                              encoding={"profile": profile, "seed": 7, "ht-beam": 8})
                config_path = self.root / f"{profile}-workflow.json"
                config_path.write_text(json.dumps(config))
                output = subprocess.run(
                    ["/usr/bin/python3", SCRIPT, "run", "--config", config_path,
                     "--run-dir", self.root / f"{profile}-run", "--timeout", "20", "--reserved-workers"],
                    check=True, capture_output=True, timeout=25,
                )
                report = json.loads(output.stdout)
                self.assertFalse(report["release_qualified"])
                self.assertGreaterEqual(report["units"], 2)
                actual = Path(report["artifact"]["path"])
                reference = fixture / "reference"
                names = sorted(x.name for x in reference.glob("*.hgn"))
                self.assertEqual(len(names), 3)
                for name in names:
                    self.assertEqual((actual / name).read_bytes(), (reference / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
