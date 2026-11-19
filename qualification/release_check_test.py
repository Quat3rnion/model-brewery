"""Development fixtures only; nothing here is a release receipt."""
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECKER = Path(__file__).with_name("release_check.py")
DATA = CHECKER.parent.parent / "crates/eval/data/gsm8k-test.jsonl"
FINGERPRINT = "a" * 64
HOST = "development-only"
ROLES = {"checkpoint": "model.hgn", "lookup": "qwen38-flash-next-ngram.hgn",
         "vision": "qwen38-flash-next-vision.hgn"}


def h(data):
    return sha256(data).hexdigest()


def encode(data):
    return json.dumps(data, separators=(",", ":")).encode()


def ident(n):
    return f"11223344-1122-4122-8122-{n:012}"


def job(n, kind, inputs, opts, result):
    return {"id": ident(n), "kind": kind, "host": HOST, "state": "succeeded",
            "input_ids": inputs, "options": opts, "has_result": True, "cancel_requested": False,
            "created_at": n * 10, "updated_at": n * 10 + 1, "result": {"kind": kind, **result},
            "events": [{"state": state, "at": n * 10} for state in ("queued", "running", "succeeded")]}


def file(name, data):
    return {"path": name, "bytes": len(data), "sha256": h(data)}


def tree(files):
    sha = h("".join(f"{p}\0{v['bytes']}\0{v['sha256']}\n" for p, v in sorted(files.items())).encode())
    return {"files": list(files.values()), "bytes": sum(v["bytes"] for v in files.values()),
            "bundle_sha256": sha, "sha256": sha}


def fixture(root, baseline_correct=1319, deltas=(-26, -27)):
    def copy(n, name, data):
        directory = root / ident(n)
        directory.mkdir(exist_ok=True)
        (directory / name).write_bytes(data)

    pin = CHECKER.with_name("release_check_shards.sha256").read_text()
    shards = {line[66:]: line[:64] for line in pin.splitlines()}
    sharded = {f"development.{i}": p for i, p in enumerate(shards)}
    index = encode({"weight_map": sharded})
    assets = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    source_files = {a: file(a, a.encode()) for a in assets}
    source_files["model.safetensors.index.json"] = file("model.safetensors.index.json", index)
    deps = [{"file": p, "size": a["bytes"], "sha256": a["sha256"]} for p, a in source_files.items()]
    for p, digest in shards.items():
        source_files[p] = {"path": p, "bytes": 26, "sha256": digest}
        deps.append({"file": p, "size": 26, "header_bytes": 10, "header_sha256": FINGERPRINT})
    hf = {"path": "/development/hf", "kind": "hf", "host": HOST, "id": ident(1000),
          **tree(source_files)}
    source = h(f"brewery-canonical-source-v1\nsource\0hf\0{hf['bundle_sha256']}\n".encode())
    records, rows, generation_proofs = [], [], {}
    for n, (target, profile) in enumerate(
            [(t, p) for t in ("gfx1151", "sm_89", "sm_120") for p in ("v2", "ht43")], 1):
        options = {"profile": profile, "ht_backend": "hip" if target == "gfx1151" else "cuda",
                   "i4r_backend": "cpu", "q4cp_backend": "cpu"}
        path = f"/development/conv-{n}"
        roles = {r: f"qwen3.8-flash-next-{p}" for r, p in
                 zip(ROLES, (profile, "ngram", "vision"))}
        fmt = "hgn-v2" if profile == "v2" else "ht43"
        plan, containers, outputs, loaded, mapped, manifest_files = [], [], {}, [], [], []
        for i, (role, name) in enumerate(ROLES.items(), 200):
            runtime_name = f"native/{name}"
            blob = (role + profile).encode()
            member = outputs[runtime_name] = file(runtime_name, blob)
            checkpoint_name = roles[role]
            tensor = {"name": role,
                      "encoding": {"checkpoint": "Ht", "lookup": "Fp8G128", "vision": "Bf16"}[role]}
            plan.append({"filename": name, "identity": checkpoint_name, "tensors": [tensor]})
            containers.append({"filename": name,
                               "layout": {"identity": checkpoint_name, "file_size": len(blob), "entries": [tensor]},
                               "tensors": [{"entry": tensor, "fingerprint": {"sha256": h(blob)}}]})
            loaded.append({"role": role, "path": f"{path}/{runtime_name}", "sha256": member["sha256"],
                           "hgn_model_id": checkpoint_name, "identity": {"bytes": len(blob),
                           "dev_major": 3, "dev_minor": 1, "inode": i}})
            mapped.append({"path": f"/brewery-{role}.hgn", "bytes": len(blob), "major": 3, "minor": 1, "inode": i})
            manifest_files.append({"role": role, "path": runtime_name, "sha256": member["sha256"]})
        project_assets = []
        for name in assets:
            for dest in [f"tokenizer/{name}"] + ([name] if name == "config.json" else []):
                outputs[f"native/{dest}"] = file(f"native/{dest}", name.encode())
                project_assets.append({"source": name, "destination": dest, "size": len(name),
                                       "sha256": h(name.encode())})
        tokenizer = [{"path": f"native/tokenizer/{a}", "sha256": h(a.encode())} for a in sorted(assets)]
        runtime_manifest = encode({"format": fmt, "files": manifest_files, "tokenizer": tokenizer,
                                   "source_sha256": source})
        outputs["manifest.json"] = file("manifest.json", runtime_manifest)
        policy = f"bf16-hgn-{profile}-rtn-nf4-signed-rms-trellis-v1"
        native_manifest = encode({"schema": 2, "source_kind": "hf", "tool": "bf16-to-hgn",
                                  "encoder_sha256": FINGERPRINT, "policy": policy, "spec_version": "test",
                                  "spec_commit": "test",
                                  "source_provenance": {"format": "huggingface-safetensors", "dependencies": deps},
                                  "config_sha256": h(b"config.json"), "index_sha256": h(index),
                                  "options": dict(options, seed=0, threads=1, ht_beam=128),
                                  "plan": {"architecture": "qwen4_exp", "files": plan, "assets": assets},
                                  "containers": containers, "assets": project_assets,
                                  "sources": [{"tensor": {"name": k, "shard": v, "offset": 10, "size": 16,
                                                          "dtype": "Bf16", "shape": [8]},
                                               "sha256": FINGERPRINT} for k, v in sharded.items()]})
        outputs["native/conversion-manifest.json"] = file("native/conversion-manifest.json", native_manifest)
        for name, data in (("manifest.json", runtime_manifest), ("conversion-manifest.json", native_manifest),
                           ("model.safetensors.index.json", index)):
            copy(n, name, data)
        artifact = {"kind": "hgn", "path": path, **tree(outputs)}
        process = {"tool": "bf16-to-hgn", "requested_options": options,
                   "processes": [{"executable": "/development/executable", "executable_sha256": FINGERPRINT,
                                  "stderr_log": f"{path}/stderr.log", "stderr_sha256": FINGERPRINT,
                                  "exit_code": 0, "pid": 100 + n}]}
        report = {"published": True, "manifest": "native/conversion-manifest.json", "source_format": "hf",
                  "native_directory": f"{path}/native", "encoder_source_sha256": FINGERPRINT, "policy": policy,
                  "source_tensors": 131, "output_tensors": 3, **options}
        records.append(job(n, "convert", [hf["id"]], options, {"host": HOST, "inputs": [hf],
                       "source_sha256": source, "report": report, "artifact": artifact,
                       "backend_receipt": process, "hardware_proof": {"status": "passed"},
                       "inference_bundle": {"status": "manifest_created", "format": fmt, "manifest": f"{path}/manifest.json"}}))
        rows.extend(["--row", f"{target}/{profile}={ident(n)}"])
        vstats = {"manifest_schema": 2, "provenance_attested": True, "source_tensors": 131,
                  "output_tensors": 3, "payload_bytes": 10, "errors": {}, "passed": True}
        vbytes = encode(vstats)
        copy(200 + n, "verify.json", vbytes)
        verify_input = dict(artifact, id=ident(300 + n), host=HOST)
        records.append(job(200 + n, "verify", [hf["id"], verify_input["id"]], {},
                           {"host": HOST, "inputs": [hf, verify_input], "source_sha256": source,
                            "backend_receipt": dict(process, requested_options={}),
                            "file_verification": {"status": "passed"}, "report": vstats,
                            "artifact": {**file("/development/verify.json", vbytes), "kind": "verify"}}))
        generation_proofs.setdefault(profile, {"source_sha256": source, "manifest_sha256": h(runtime_manifest),
             "tokenizer": tokenizer, "load_proof": {"schema": "halogen-linux-engine-load-v1",
             "image_id": f"sha256:{FINGERPRINT}", "container_id": "b" * 64, "qualification": {
             "schema": "halogen-hgn-preflight-v1", "format": fmt, "files": loaded},
             "observation": {"engine_pid": 9, "start_ticks": 100, "listener_inode": 200, "mapped": mapped}}})
    ds_sha = h(DATA.read_bytes())
    ds = {"name": "openai/grade-school-math", "split": "test", "sha256": ds_sha,
          "revision": "3101c7d5072418e28b9008a6636bde82a006892c", "rows": 1319, "full_split": True}
    protocol = "gsm8k-zero-shot-raw-v1"
    question_rows = [json.loads(row) for row in DATA.read_bytes().splitlines()]
    reference = [row["answer"].splitlines()[-1] for row in question_rows]

    def predictions(n, kind, same, provenance):
        texts = [{"id": i, "question_sha256": h(row["question"].encode()),
                  "text": reference[i-1] if i <= same else "not parsed"}
                 for i, row in enumerate(question_rows, 1)]
        identity = {"source_sha256": source, "synthetic": False, "provenance": provenance}
        raw = encode(dict(identity, schema="brewery-gsm8k-predictions-v1", dataset_sha256=ds_sha,
                          protocol_id=protocol, predictions=texts))
        name = "predictions.json"
        copy(n, name, raw)
        return {"identity": identity, "dataset": ds, "protocol_id": protocol,
                "score": {"correct": same, "total": 1319, "unparsed": 1319-same,
                          "accuracy_basis_points": (same*10_000+659)//1319},
                "artifact": {**file(f"/development/{name}", raw), "kind": kind}}

    base = predictions(500, "gsm8k-baseline", baseline_correct,
                       {"method": "imported-predictions", "import_sha256": "e" * 64})
    records.append(job(500, "baseline", [ident(700)], {}, base))
    for n, profile in enumerate(("v2", "ht43"), 600):
        proof = generation_proofs[profile]
        inputs = [ident(n + 10)]
        records.append(job(n, "generate", inputs, {}, {"identity": proof, "text": "Development test response",
                       "protocol": {"id": "raw-greedy-v1", "temperature": 0, "max_tokens": 128,
                                    "prompt_sha256": FINGERPRINT}}))
        eval_proof = dict(proof, load_proof=dict(proof["load_proof"], container_id="c" * 64))
        delta = deltas[n - 600]
        candidate_correct = baseline_correct + delta
        eval_result = predictions(n+20, "gsm8k-evaluation", candidate_correct, eval_proof)
        records.append(job(n+20, "evaluate", [*inputs, ident(700)], {}, dict(eval_result,
                           baseline_identity=base["identity"], baseline_score=base["score"], baseline_artifact=None,
                           comparison={"changed_answers": abs(delta), "delta_correct": delta,
                                       "changed_answer_ids": list(range(min(candidate_correct, baseline_correct) + 1,
                                                                        max(candidate_correct, baseline_correct) + 1)),
                                       "within_two_percentage_points": delta >= -26})))
    return rows, records


class ReleaseCheck(unittest.TestCase):
    def run_check(self, *args):
        result = subprocess.run([sys.executable, str(CHECKER), *map(str, args)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        return json.loads(result.stdout)

    def test_empty_collection_reports_six_missing_rows_and_baseline(self):
        out = self.run_check()
        self.assertEqual({(r["target"], r["profile"]) for r in out["paths"]},
                         {(t, p) for t in ("gfx1151", "sm_89", "sm_120")
                          for p in ("v2", "ht43")})
        self.assertTrue(all(r["conversion"]["status"] == "missing" for r in out["paths"]))
        self.assertTrue(all(r["upstream_source"]["status"] == "missing" for r in out["paths"]))
        self.assertEqual(out["baseline"]["status"], "missing")
        self.assertFalse(out["release_qualified"])

    def check_fixture(self, root, rows, jobs):
        exports = []
        for j in jobs:
            path = root / f"{j['id']}.json"
            path.write_bytes(encode(j))
            exports.append(path)
        return self.run_check("--artifacts", root, "--baseline-job-id", ident(500), *rows, *exports)

    def test_agreeing_development_manifest_and_export_do_not_prove_upstream_identity(self):
        with tempfile.TemporaryDirectory(dir="/arboretum", prefix="release-source-") as d:
            root = Path(d)
            rows, jobs = fixture(root)
            row = self.check_fixture(root, rows[:2], [jobs[0]])["paths"][0]
            self.assertEqual(row["conversion"]["status"], "record_validated")
            self.assertEqual(row["upstream_source"]["status"], "invalid")

    def test_verify_rejects_contradictory_native_report_and_accepts_omission(self):
        with tempfile.TemporaryDirectory(dir="/arboretum", prefix="release-check-verify-") as d:
            root = Path(d)
            rows, jobs = fixture(root)
            convert, verify = jobs[:2]
            report = verify["result"]["report"]
            for passed, expected in [("absent", True), (True, True), (False, False),
                                     (None, False), ("true", False), (1, False)]:
                with self.subTest(passed=passed):
                    if passed == "absent":
                        report.pop("passed", None)
                    else:
                        report["passed"] = passed
                    raw = encode(report)
                    (root / verify["id"] / "verify.json").write_bytes(raw)
                    verify["result"]["artifact"].update(bytes=len(raw), sha256=h(raw))
                    result = self.check_fixture(root, rows[:2], [convert, verify])
                    phase = result["paths"][0]["verification"]
                    if expected:
                        self.assertEqual(phase["status"], "record_validated")
                    else:
                        self.assertEqual(phase["status"], "missing")
                        self.assertTrue(any(e.get("job_id") == verify["id"] and "passed" in e.get("reason", "")
                                            for e in result["errors"]), result["errors"])
            # Python considers True == 1; a numeric status in the artifact
            # must fail even when the worker report uses a real boolean.
            report["passed"] = True
            raw = encode(dict(report, passed=1))
            (root / verify["id"] / "verify.json").write_bytes(raw)
            verify["result"]["artifact"].update(bytes=len(raw), sha256=h(raw))
            self.assertEqual(self.check_fixture(root, rows[:2], [convert, verify])
                             ["paths"][0]["verification"]["status"], "missing")

    def test_cuda_receipt_cross_checks_bounded_raw_capture_and_binary_hashes(self):
        with tempfile.TemporaryDirectory(dir="/arboretum", prefix="release-check-cuda-") as d:
            root = Path(d)
            _, jobs = fixture(root)
            j = jobs[8]  # sm_120/v2
            rows = ["--row", f"sm_120/v2={j['id']}"]
            result = j["result"]
            native = result["backend_receipt"]["processes"][0]
            artifacts = root / j["id"]
            binaries = {"executable": b"development converter", "collector.so": b"development collector"}
            for name, content in binaries.items():
                (artifacts / name).write_bytes(content)
            native["executable_sha256"] = h(binaries["executable"])
            raw = {"schema": "brewery-cupti-encoder-v1", "pid": native["pid"],
                   "exe_dev": 1, "exe_ino": 12345,
                   "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                   "compute_major": 12, "compute_minor": 0,
                   "counts": {"ht3": 0, "ht4": 2, "i4r": 0, "q4cp": 0}}
            receipt = {"status": "observed", "schema": raw["schema"], "backend": "cuda",
                       "pid": native["pid"], "executable_sha256": native["executable_sha256"],
                       "output_bundle_sha256": result["artifact"]["bundle_sha256"],
                       "gpu_uuid": raw["gpu_uuid"], "compute_capability": [12, 0],
                       "completed_encoder_activity": deepcopy(raw["counts"]),
                       "capture": str(Path(native["stderr_log"]).with_name("encoder-activity.json")),
                       "collector": "/development/collector.so", "collector_sha256": h(binaries["collector.so"]),
                       "scope": "completed encoder kernels observed; not proof of full GPU coverage or release"}

            def check(change=None, raw_change=None, padding=0):
                evidence, record = deepcopy(receipt), deepcopy(raw)
                if change:
                    evidence.update(change)
                if raw_change:
                    record.update(raw_change)
                data = encode(record) + b" " * padding
                (artifacts / "encoder-activity.json").write_bytes(data)
                evidence.setdefault("capture_sha256", h(data))
                result["encoder_activity"] = evidence
                out = self.check_fixture(root, rows, [j])
                self.assertFalse(out["release_qualified"])
                return out["paths"][4]["native_execution"]

            valid = check()
            self.assertEqual(valid["status"], "record_validated", valid)
            self.assertEqual(valid["completed_encoder_activity"]["ht4"], 2)
            for label, change, raw_change in [
                ("executable hash", {"executable_sha256": "e" * 64}, None),
                ("output digest", {"output_bundle_sha256": "e" * 64}, None),
                ("PID", None, {"pid": native["pid"] + 1}),
                ("GPU UUID", None, {"gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdee"}),
                ("target", {"compute_capability": [8, 9]}, {"compute_major": 8, "compute_minor": 9}),
                ("wrong codec", {"completed_encoder_activity": dict(raw["counts"], ht4=0, q4cp=1)},
                 {"counts": dict(raw["counts"], ht4=0, q4cp=1)}),
                ("capture hash", {"capture_sha256": "e" * 64}, None),
                ("collector hash", {"collector_sha256": "e" * 64}, None),
            ]:
                with self.subTest(label=label):
                    self.assertEqual(check(change, raw_change)["status"], "invalid")
            for state in ("unavailable", "not_observed"):
                result["encoder_activity"] = {"status": state, "reason": f"development {state}"}
                self.assertEqual(self.check_fixture(root, rows, [j])["paths"][4]["native_execution"]["status"], state)
            self.assertEqual(check(padding=1024)["status"], "invalid")
            self.assertEqual(check()["status"], "record_validated")
            (artifacts / "encoder-activity.json").unlink()
            self.assertEqual(self.check_fixture(root, rows, [j])["paths"][4]["native_execution"]["status"], "missing")

    def test_six_paths_reuse_byte_identical_bundle_but_not_release_attestation(self):
        with tempfile.TemporaryDirectory(dir="/arboretum", prefix="release-check-test-") as d:
            root = Path(d)
            rows, jobs = fixture(root)
            out = self.check_fixture(root, rows, jobs)
            self.assertTrue(all(r["conversion"]["status"] == "record_validated"
                                and r["verification"]["status"] == "record_validated" for r in out["paths"]), out)
            self.assertTrue(all(r["native_execution"]["status"] == "external_review" for r in out["paths"]))
            self.assertEqual(len(out["bundles"]), 2)
            self.assertEqual([len(b["paths"]) for b in out["bundles"]], [3, 3])
            self.assertEqual(out["baseline"]["status"], "external_review")
            self.assertEqual([b["quality"]["status"] for b in out["bundles"]],
                             ["within_margin_pending_baseline_review", "below_margin"])
            self.assertFalse(out["release_qualified"])

    def test_missing_index_and_wrong_baseline_hash_are_not_promoted(self):
        with tempfile.TemporaryDirectory(dir="/arboretum", prefix="release-check-test-") as d:
            root = Path(d)
            rows, jobs = fixture(root)
            (root / ident(1) / "model.safetensors.index.json").unlink()
            changed = next(j for j in jobs if j["id"] == ident(620))
            changed["result"]["baseline_identity"] = {
                **changed["result"]["baseline_identity"],
                "provenance": {"method": "imported-predictions", "import_sha256": "0" * 64},
            }
            out = self.check_fixture(root, rows, jobs)
            self.assertEqual(out["paths"][0]["conversion"]["status"], "missing")
            v2 = next(b for b in out["bundles"] if "sm_89/v2" in b["paths"])
            self.assertEqual(v2["quality"]["status"], "missing")
            self.assertTrue(any("different baseline" in e.get("reason", "") for e in v2["evidence_errors"]), v2)

    def test_gain_above_two_points_and_changed_answer_ids(self):
        with tempfile.TemporaryDirectory(dir="/arboretum", prefix="release-check-test-") as d:
            root = Path(d)
            rows, jobs = fixture(root, baseline_correct=1250, deltas=(27, -27))
            out = self.check_fixture(root, rows, jobs)
            self.assertEqual(out["bundles"][0]["quality"]["status"],
                             "within_margin_pending_baseline_review", out["bundles"][0])
            self.assertEqual(out["bundles"][0]["quality"]["runs"][0]["delta_correct"], 27)
            changed = next(j for j in jobs if j["id"] == ident(620))
            changed["result"]["comparison"]["changed_answer_ids"][0] = 1
            out = self.check_fixture(root, rows, jobs)
            self.assertEqual(out["bundles"][0]["quality"]["status"], "missing")
            self.assertTrue(any("comparison mismatch" in e.get("reason", "")
                                for e in out["bundles"][0]["evidence_errors"]))

    def test_remote_generation_needs_a_real_transfer_of_the_exact_tree(self):
        with tempfile.TemporaryDirectory(dir="/arboretum", prefix="release-check-test-") as d:
            root = Path(d)
            rows, jobs = fixture(root)
            generate = next(j for j in jobs if j["id"] == ident(600))
            generate["host"] = "inference-worker"
            generate["result"]["identity"] = deepcopy(generate["result"]["identity"])
            files = generate["result"]["identity"]["load_proof"]["qualification"]["files"]
            for f in files:
                f["path"] = f["path"].replace("/development/conv-1/", "/development/cache/data/")
            out = self.check_fixture(root, rows, jobs)
            self.assertEqual(out["bundles"][0]["generation"]["status"], "missing")
            original = jobs[0]["result"]["artifact"]
            transferred = dict(original, id=ident(800), source_host=HOST,
                               host=generate["host"], cached=True, path="/development/cache/data")
            transferred.pop("sha256")
            transfer = job(800, "transfer", [ident(800)], {}, {"inputs": [transferred], "bytes": original["bytes"]})
            transfer["host"] = generate["host"]
            transfer["result"].pop("kind")
            out = self.check_fixture(root, rows, [*jobs, transfer])
            self.assertEqual(out["bundles"][0]["generation"]["status"], "record_validated")
            self.assertEqual(out["bundles"][0]["quality"]["status"], "invalid")  # evaluation was on a different host


if __name__ == "__main__":
    unittest.main()
