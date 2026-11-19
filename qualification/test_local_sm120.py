import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import local_sm120
from local_sm120 import HOST, controller, reconcile, request, save, stage


class LocalInstall(unittest.TestCase):
    def test_pins_the_publication_in_new_private_worker_config(self):
        with tempfile.TemporaryDirectory(prefix="sm120-install-fixture-") as folder:
            root = Path(folder).resolve()
            bins = root / "fixtures"
            bins.mkdir()
            for name in ("brewery", "brewery-worker", "brewery-dispatch", "brewery-engine",
                         "bf16-to-hgn", "encoder_activity.so", "queue"):
                path = bins / name
                path.write_bytes(name.encode())
                path.chmod(0o700)
            run = root / "fresh-install"
            converter = bins / "bf16-to-hgn"
            collector = bins / "encoder_activity.so"
            with patch.multiple(local_sm120, CONVERTER_SHA=local_sm120.sha(converter),
                                COLLECTOR_SHA=local_sm120.sha(collector),
                                RESOURCE_QUEUE=str(bins / "queue")):
                local_sm120.install(run, bins, converter, collector)
            config = json.loads((run / "worker.json").read_text())
            self.assertEqual(config["pinned_publication"], {
                "source": str(local_sm120.SOURCE), "receipt": str(local_sm120.PUBLICATION),
                "repo_id": local_sm120.CHECKPOINT, "revision": local_sm120.REVISION,
                "bundle_sha256": local_sm120.BUNDLE_SHA,
            })
            self.assertEqual(config["engine_env"]["BREWERY_EXPECTED_SOURCE_SHA256"],
                             local_sm120.SOURCE_SHA)


class SubmissionRecovery(unittest.TestCase):
    def test_recovers_new_exact_job_but_never_guesses(self):
        spec = {"kind": "convert", "host": "local-sm120", "options": {"profile": "v2"}}
        intent = {"body": spec, "before": ["old"]}
        self.assertEqual(reconcile(intent, [
            {"id": "old", **spec}, {"id": "new", **spec},
            {"id": "other", **spec, "options": {"profile": "ht43"}},
        ]), "new")
        for records in ([], [{"id": "one", **spec}, {"id": "two", **spec}]):
            with self.assertRaisesRegex(ValueError, "not resubmitted"):
                reconcile(intent, records)

    @unittest.skipUnless(os.environ.get("BREWERY_TEST_BIN_DIR"), "requires compiled workbench binaries")
    def test_local_http_round_trip_reuses_jobs_across_controller_restart(self):
        # Stub native reports test the real controller/dispatcher/worker/engine
        # plumbing. They are not conversion evidence and never touch the GPU.
        with tempfile.TemporaryDirectory(prefix="sm120-smoke-") as folder:
            run = Path(folder)
            for name in ("source", "worker", "exports", "artifacts"):
                (run / name).mkdir(mode=0o700)
            source = run / "source"
            (source / "config.json").write_text("{}")
            (source / "model.safetensors.index.json").write_text("{}")
            (source / "weights").write_bytes(b"bounded test input only")
            (run / "bin").symlink_to(Path(os.environ["BREWERY_TEST_BIN_DIR"]).resolve())
            converter = run / "fixture-converter"
            converter.write_text("""#!/usr/bin/python3
import json, sys
from pathlib import Path
kind = sys.argv[1]
profile = sys.argv[sys.argv.index('--profile') + 1] if '--profile' in sys.argv else 'v2'
summary = {"source_format":"hf","tensor_payloads_read":False,"outputs":[],"output_hgn_bytes":1024}
if kind == "inspect":
    out = summary
elif kind == "plan":
    out = {"summary":summary,"plan":{},"layouts":[]}
elif kind == "convert":
    dst = Path(sys.argv[sys.argv.index('--output') + 1])
    dst.mkdir()
    (dst / 'model.hgn').write_bytes(b'fixture; not HGN evidence')
    (dst / 'conversion-manifest.json').write_text(json.dumps({
        "schema":2,"source_kind":"hf","source_provenance":{},"encoder_sha256":"1"*64,
        "options":{"profile":profile},"plan":{"architecture":"synthetic fixture"}
    }))
    out = {"published":True,"manifest":"conversion-manifest.json","source_format":"hf","profile":profile}
elif kind == "verify":
    assert (Path(sys.argv[sys.argv.index('--output') + 1]) / 'model.hgn').is_file()
    out = {"manifest_schema":2,"provenance_attested":True,"source_tensors":1,
           "output_tensors":1,"payload_bytes":64,"errors":{}}
else:
    raise SystemExit(3)
print(json.dumps(out))
""")
            converter.chmod(0o700)
            admission = run / "fake-admission"
            admission.write_text(
                '#!/bin/sh\nset -eu\n( set -C; echo "$$" > "$1" )\nshift\nexec "$@"\n')
            admission.chmod(0o700)
            cfg = run / "worker.json"
            save(cfg, {"root": str(run / "worker"), "source_roots": [str(source)],
                       "engine": str(run / "bin/brewery-engine"),
                       "engine_env": {"BREWERY_CONVERTER": str(converter),
                                      "BREWERY_CONVERTER_ADMISSION": str(admission)}})
            save(run / "dispatch.json", {"hosts": {HOST: {"local": True,
                "worker": str(run / "bin/brewery-worker"), "config": str(cfg)}}})
            with controller(run, 0) as url:
                stage(url, run, source, "preflight", seconds=30)
                before, _ = request(url, "state")
                self.assertEqual(len(before["jobs"]), 4)
                with self.assertRaisesRegex(ValueError, "no matching GPU encoder"):
                    stage(url, run, source, "v2-convert", seconds=30)
            with controller(run, 0) as url:
                stage(url, run, source, "preflight", seconds=30)
                stage(url, run, source, "v2-verify", seconds=30)
                after, _ = request(url, "state")
                self.assertEqual(len(after["jobs"]), 6)
                self.assertEqual([j["id"] for j in after["jobs"][:4]], [j["id"] for j in before["jobs"]])
                converted = json.loads((run / "exports" / f"{after['jobs'][4]['id']}.json").read_text())
                verify = json.loads((run / "exports" / f"{after['jobs'][5]['id']}.json").read_text())
                self.assertEqual(verify["result"]["file_verification"]["status"], "passed")
                root = run / "worker/jobs" / converted["id"] / "work/converter-evidence"
                self.assertEqual(int((root / "admitted-pid").read_text()),
                                 converted["result"]["backend_receipt"]["processes"][0]["pid"])
                # One HTTP submission; client never creates the four child jobs
                # or registers the generated bundle for the verification stage.
                source_id = json.loads((run / "source.json").read_bytes())["id"]
                flow, _ = request(url, "conversions", {"input_ids": [source_id],
                                                       "host": HOST, "options": {"profile": "ht43"}})
                deadline = time.monotonic() + 40
                while flow["state"] == "running" and time.monotonic() < deadline:
                    time.sleep(.2)
                    flow, _ = request(url, f"conversions/{flow['id']}")
                self.assertEqual(flow["state"], "succeeded", flow)
                self.assertEqual(set(flow["job_ids"]), {"inspect", "plan", "convert", "verify"})
                self.assertEqual(len(set(flow["job_ids"].values())), 4)
                last, _ = request(url, "jobs/" + flow["job_ids"]["verify"])
                self.assertEqual(last["input_ids"], [source_id, flow["artifact_model_id"]])
                self.assertEqual(last["result"]["file_verification"]["status"], "passed")
                self.assertEqual((source / "weights").read_bytes(), b"bounded test input only")
