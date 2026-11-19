"""Public CLI tests; miniature receipt data is not release evidence."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).with_name("build_diff.py")
NATIVE_EXPORTS = os.environ.get("BREWERY_BUILD_DIFF_TEST_EXPORTS")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree(root, kind, contents):
    files = [{"path": path, "bytes": len(data), "sha256": sha(data)}
             for path, data in sorted(contents.items())]
    digest = sha(b"".join(b"%s\0%d\0%s\n" % (
        f["path"].encode(), f["bytes"], f["sha256"].encode()) for f in files))
    return {"kind": kind, "path": root, "files": files, "bundle_sha256": digest,
            "sha256": digest, "bytes": sum(f["bytes"] for f in files)}


def job(n, profile="v2", weights=b"upstream", model=b"checkpoint"):
    source = tree("/work/source", "hf", {"model.safetensors": weights,
                  "model.safetensors.index.json": b"{}", "config.json": b"{}"})
    source.update(id="f3445168-8f10-4450-9a7c-b9d2e7c03878", host="host",
                  name="org/model@not-a-branch", source_host="host")
    root = f"/work/job{n}/converted"
    artifact = tree(root, "hgn", {"native/model.hgn": model * 300,
                    "native/qwen38-flash-next-ngram.hgn": b"sidecar" * 50,
                    "native/conversion-manifest.json": profile.encode(),
                    "native/config.json": b"{}", "native/tokenizer/tokenizer.json": b"{}"})
    opts = {"profile": profile}
    report = {"published": True, "profile": profile, "source_format": "hf",
              "manifest": "native/conversion-manifest.json", "native_directory": root + "/native",
              "source_tensors": 1, "output_tensors": 2, "encoder_source_sha256": sha(b"encoder"),
              "policy": f"bf16-hgn-{profile}-rtn-nf4-signed-rms-trellis-v1",
              "ht_backend": "cpu", "i4r_backend": "cpu", "q4cp_backend": "cpu"}
    source_sha = sha(f"brewery-canonical-source-v1\nsource\0hf\0{source['bundle_sha256']}\n".encode())
    return {
        "id": f"11223344-1122-4122-8122-{n:012}", "kind": "convert", "host": "host",
        "state": "succeeded", "progress": "succeeded", "has_result": True, "cancel_requested": False,
        "created_at": 100, "updated_at": 102, "input_ids": [source["id"]], "options": opts,
        "events": [{"state": state, "at": 100 + i, "progress": state}
                   for i, state in enumerate(("queued", "running", "succeeded"))],
        "result": {"kind": "convert", "host": "host", "inputs": [source],
                   "source_sha256": source_sha,
                   "source_digest_scope": "role-qualified physical source bundles",
                   "report": report, "artifact": artifact,
                   "inference_bundle": {"status": "not_created", "reason": "fixture"},
                   "backend_receipt": {"tool": "bf16-to-hgn", "requested_options": opts, "processes": [
                       {"pid": 4811, "exit_code": 0, "executable": "/bin/bf16-to-hgn",
                        "executable_sha256": sha(b"binary"),
                        "stderr_log": f"/work/job{n}/converter-evidence/stderr.log",
                        "stderr_sha256": sha(b"successful process")},
                   ]}},
    }


class BuildDiffCLI(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def run_jobs(self, left, right):
        for name, document in (("left", left), ("right", right)):
            (self.folder / name).write_text(json.dumps(document), encoding="utf-8")
        return self.run_paths(self.folder / "left", self.folder / "right")

    def run_paths(self, left, right):
        return subprocess.run([sys.executable, str(CLI), str(left), str(right)],
                              capture_output=True, text=True, timeout=5)

    def report(self, left, right):
        result = self.run_jobs(left, right)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def reject(self, result, error):
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn(error, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_same_reported_source_different_checkpoint_is_not_quantization_only(self):
        left, right = job(1), job(2, "ht43", model=b"compressed")
        report = self.report(left, right)
        self.assertEqual(report["schema"], "brewery-build-diff-v1")
        self.assertEqual(report["comparison"], {
            "source": "same", "profile": "changed", "checkpoint": "changed",
            "options": "unknown", "tool": "same", "encoder": "same", "runtime": "unknown",
            "quantization_only": False,
        })
        self.assertEqual(report["left"]["source"]["alias"], "org/model@not-a-branch")
        for field in ("model", "revision", "branch"):
            self.assertIsNone(report["left"]["source"][field])
        self.assertEqual(report["bytes_by_role"]["checkpoint"], {
            "left_bytes": 3000, "right_bytes": 3000, "delta_bytes": 0,
        })
        rows = {row["path"]: row for row in report["files"]}
        self.assertEqual(rows["native/model.hgn"]["status"], "changed")
        self.assertEqual(rows["native/qwen38-flash-next-ngram.hgn"]["status"], "unchanged")

    def test_checkpoint_identity_is_independent_of_source_bundle_and_paths(self):
        left, right = job(1), job(2, weights=b"other source weights")
        right["result"]["inputs"][0].update(model="org/other", revision="7a7a", branch="test")
        right["result"]["report"]["encoder_source_sha256"] = sha(b"another encoder")
        right["result"]["backend_receipt"]["processes"][0]["executable_sha256"] = sha(b"another binary")
        left["options"]["seed"] = 0
        right["options"]["seed"] = 4
        report = self.report(left, right)
        self.assertEqual(report["comparison"], {
            "source": "changed", "profile": "same", "checkpoint": "same",
            "options": "changed", "tool": "changed", "encoder": "changed", "runtime": "unknown",
            "quantization_only": False,
        })
        self.assertEqual(report["right"]["source"]["revision"], "7a7a")
        self.assertEqual(report["right"]["source"]["branch"], "test")
        self.assertEqual(report["right"]["source"]["model"], "org/other")

    def test_added_removed_and_changed_members_conserve_role_totals(self):
        left, right = job(1), job(2)
        prefix = "native/"
        for doc, contents in (
            (left, {"model.hgn": b"L" * 384, "ngram.hgn": b"0" * 448, "vision.hgn": b"v" * 320,
                    "conversion-manifest.json": b"{}", "tokenizer/left.json": b"left"}),
            (right, {"model.hgn": b"L" * 384, "ngram.hgn": b"1" * 384, "mtp.hgn": b"m" * 320,
                     "conversion-manifest.json": b"{} ", "tokenizer/right.json": b"right"}),
        ):
            res = doc["result"]
            res["artifact"] = tree(res["artifact"]["path"], "hgn",
                                   {prefix + k: v for k, v in contents.items()})
        report = self.report(left, right)
        rows = {row["path"]: row for row in report["files"]}
        for path, status, delta in (("model.hgn", "unchanged", 0),
                                    ("ngram.hgn", "changed", -64),
                                    ("vision.hgn", "removed", -320),
                                    ("mtp.hgn", "added", 320),
                                    ("conversion-manifest.json", "changed", 1)):
            row = rows[prefix + path]
            self.assertEqual((row["status"], row["delta_bytes"]), (status, delta))
        self.assertIsNone(rows["native/vision.hgn"]["right"])
        self.assertIsNone(rows["native/mtp.hgn"]["left"])
        self.assertEqual(report["bytes_by_role"]["sidecar"],
                         {"left_bytes": 768, "right_bytes": 704, "delta_bytes": -64})
        self.assertEqual(report["bytes_by_role"]["assets"],
                         {"left_bytes": 4, "right_bytes": 5, "delta_bytes": 1})
        for side in ("left", "right"):
            self.assertEqual(report[side]["artifact"]["bytes"],
                             sum(group[f"{side}_bytes"] for group in report["bytes_by_role"].values()))

    def test_reject_inconsistent_or_nonconvert_receipts(self):
        for mutate, error in (
            (lambda j: j.update(kind="plan"), "successful terminal convert job"),
            (lambda j: j.update(state="failed"), "successful terminal convert job"),
            (lambda j: j["events"].pop(1), "controller job lifecycle"),
            (lambda j: j["result"].update(source_sha256="0" * 64), "role-qualified source digest"),
            (lambda j: j["result"]["report"].update(published=False), "native conversion publication"),
            (lambda j: j["result"]["report"].update(profile="ht43"), "native conversion publication"),
            (lambda j: j["result"]["backend_receipt"]["processes"][0].update(exit_code=1),
             "successful converter process"),
            (lambda j: j["result"]["artifact"].update(bytes=99), "bundle tree digest or total size"),
            (lambda j: j["result"]["artifact"]["files"][0].update(sha256="1" * 64),
             "bundle tree digest or total size"),
            (lambda j: j["result"]["artifact"]["files"].append(j["result"]["artifact"]["files"][0]),
             "duplicate member path"),
            (lambda j: j["result"]["artifact"]["files"][0].update(path="../bad"),
             "invalid bundle member path"),
        ):
            with self.subTest(error=error):
                bad = job(2)
                mutate(bad)
                self.reject(self.run_jobs(job(1), bad), error)
        self.reject(self.run_jobs(job(1), job(1)), "two distinct convert jobs")

    def test_rejects_unbounded_and_nonregular_input_before_reading(self):
        left = self.folder / "left"
        left.write_text(json.dumps(job(1)), encoding="utf-8")
        fifo = self.folder / "fifo"
        os.mkfifo(fifo)
        self.reject(self.run_paths(left, fifo), "expected a regular JSON file")
        big = self.folder / "big"
        with big.open("wb") as stream:
            stream.truncate((16 << 20) + 1)
        self.reject(self.run_paths(left, big), "JSON exceeds 16777216 bytes")
        alias = self.folder / "alias"
        alias.symlink_to(left)
        self.assertEqual(self.run_paths(left, alias).returncode, 2)
        duplicate = self.folder / "duplicate"
        duplicate.write_text('{"kind":"convert","kind":"plan"}', encoding="utf-8")
        self.reject(self.run_paths(left, duplicate), "duplicate JSON key")

    @unittest.skipUnless(NATIVE_EXPORTS, "set BREWERY_BUILD_DIFF_TEST_EXPORTS to CPU fixture exports")
    def test_private_controller_native_cpu_fixture(self):
        folder = Path(NATIVE_EXPORTS)
        run = self.run_paths(folder / "v2-export.json", folder / "ht43-export.json")
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report["comparison"]["source"], "same")
        self.assertEqual(report["comparison"]["checkpoint"], "changed")
        self.assertEqual(report["comparison"]["runtime"], "unknown")
        self.assertEqual(report["bytes_by_role"]["checkpoint"],
                         {"left_bytes": 65488, "right_bytes": 65808, "delta_bytes": 320})


if __name__ == "__main__":
    unittest.main()
