"""Development fixtures for the offline pairing CLI; no inference is run."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from release_check import DATASET, PROTOCOL, answer, digest

CLI = Path(__file__).with_name("paired_quality.py")
DATA = CLI.parent.parent / "crates/eval/data/gsm8k-test.jsonl"


def build_side(root, side, replies, source, *, synthetic=False, baseline=False):
    identity = {"source_sha256": source, "synthetic": synthetic}
    if baseline and not synthetic:
        identity["provenance"] = {"method": "operator-bf16-endpoint", "baseline_source": "endpoint",
                                  "runtime_id": "bf16-a", "model": "same-alias",
                                  "source_evidence": "matched-response-headers"}
    elif synthetic or baseline:
        identity["provenance"] = {"method": "imported-predictions", "import_sha256": "e" * 64}
    else:
        identity["provenance"] = {
            "source_sha256": source, "manifest_sha256": "b" * 64,
            "tokenizer": [{"path": "tokenizer/tokenizer.json", "sha256": "c" * 64}],
            "load_proof": {
                "schema": "halogen-linux-engine-load-v1", "image_id": "sha256:" + "d" * 64,
                "container_id": "f" * 64,
                "qualification": {"schema": "halogen-hgn-preflight-v1", "format": "ht43", "files": [
                    {"role": "checkpoint", "path": "/worker/bundle/model.hgn", "sha256": "a" * 64,
                     "hgn_model_id": "qwen3.8-flash-next-ht43", "identity": {
                         "bytes": 120, "dev_major": 8, "dev_minor": 0, "inode": 20,
                         "modified": [0, 0], "changed": [0, 0],
                     }},
                ]},
                "observation": {"engine_pid": 2, "start_ticks": 1, "listener_inode": 2,
                                "mapped": [{"path": "/brewery-checkpoint.hgn", "major": 8,
                                            "minor": 0, "inode": 20, "bytes": 120}]},
            },
        }
    qs = list(map(json.loads, DATA.read_bytes().splitlines()))
    text = [("private fixture reasoning\n" + q["answer"].splitlines()[-1]) if x is True else x
            for q, x in zip(qs, replies + ["unparsed"] * len(qs))]
    predictions = [{"id": i, "question_sha256": digest(q["question"].encode()), "text": completion}
                   for i, (q, completion) in enumerate(zip(qs, text), 1)]
    correct = sum(answer(t) == answer(q["answer"]) for q, t in zip(qs, text))
    score = {"total": len(qs), "correct": correct, "unparsed": sum(answer(t) is None for t in text),
             "accuracy_basis_points": (correct * 10_000 + len(qs) // 2) // len(qs)}
    kind = "baseline" if baseline else "evaluate"
    ident = f"12345678-1234-1234-1234-{1 if side == 'left' else 2:012}"
    envelope = {"schema": "brewery-gsm8k-predictions-v1", "dataset_sha256": DATASET["sha256"],
                "protocol_id": PROTOCOL, "predictions": predictions, **identity}
    job = {"id": ident, "kind": kind, "state": "succeeded", "has_result": True, "cancel_requested": False,
           "host": "worker", "created_at": 1, "updated_at": 2,
           "events": [{"state": s} for s in ("queued", "running", "succeeded")],
           "options": ({"baseline_source": "endpoint"} if baseline and not synthetic else {}),
           "input_ids": [] if baseline and not synthetic else ["12222222-1234-1234-1234-123456789012"],
           "result": {"kind": kind, "dataset": DATASET, "protocol_id": PROTOCOL,
                      "score": score, "identity": identity}}
    return root, side, job, envelope


def write_side(side_fixture):
    root, side, job, envelope = side_fixture
    pred_path = root / f"{side}.predictions.json"
    data = json.dumps(envelope).encode()
    pred_path.write_bytes(data)
    job["result"]["artifact"] = {"path": f"/worker/{pred_path.name}",
                                "kind": "gsm8k-baseline" if job["kind"] == "baseline" else "gsm8k-evaluation",
                                "sha256": digest(data), "bytes": len(data)}
    job_path = root / f"{side}.job.json"
    job_path.write_text(json.dumps(job))
    return job_path, pred_path


class PairedQuality(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/arboretum", prefix="paired-quality-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.left = build_side(self.root, "left", [True, True, "unparsed", "#### 999999", True],
                               "a" * 64, baseline=True)
        self.right = build_side(self.root, "right", [True, "unparsed", True, "#### 999998", "unparsed",
                                                    True, True, True], "9" * 64)

    def call(self, left=None, right=None):
        lj, lp = left or write_side(self.left)
        rj, rp = right or write_side(self.right)
        return subprocess.run([sys.executable, "-B", str(CLI), str(lj), str(rj),
                               "--left-predictions", str(lp), "--right-predictions", str(rp)],
                              capture_output=True, text=True, check=False)

    def test_exact_ordered_transitions_and_linkage(self):
        cp = self.call()
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertNotIn("private fixture reasoning", cp.stdout)
        self.assertNotIn("#### 999999", cp.stdout)
        r = json.loads(cp.stdout)
        self.assertEqual(r["schema"], "brewery-paired-quality-v1")
        self.assertEqual(r["left"]["job_id"], self.left[2]["id"])
        self.assertEqual(r["right"]["job_id"], self.right[2]["id"])
        self.assertEqual(r["transitions"], {"both_correct": 1, "both_incorrect": 1312,
                                           "improvements": 4, "regressions": 2})
        self.assertEqual(r["improvement_ids"], [3, 6, 7, 8])
        self.assertEqual(r["regression_ids"], [2, 5])
        self.assertEqual(r["parsed_answer_change_ids"], [2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(r["delta_correct"], 2)
        self.assertAlmostEqual(r["delta_percentage_points"], 200 / 1319)
        self.assertEqual(r["uncertainty"]["discordant_pairs"], 6)
        self.assertAlmostEqual(r["uncertainty"]["p_value"], 0.6875)
        self.assertEqual(r["uncertainty"]["p_value_exact"], {"numerator": "11", "denominator": "16"})
        self.assertTrue(any("source" in w.lower() and "differ" in w.lower() for w in r["warnings"]))
        self.assertIsNone(r["left"]["provenance"]["manifest_sha256"])
        self.assertIsNone(r["left"]["provenance"]["files"])
        self.assertIsNone(r["left"]["provenance"]["tokenizer"])
        self.assertEqual(r["right"]["provenance"]["manifest_sha256"], "b" * 64)
        self.assertEqual(r["right"]["provenance"]["files"][0]["sha256"], "a" * 64)

    def reject(self, message, left=None, right=None):
        cp = self.call(left, right)
        self.assertEqual(cp.returncode, 1, cp.stderr)
        self.assertEqual(cp.stdout, "")
        self.assertIn(message, cp.stderr)

    def test_ordinary_endpoint_can_be_a_quantized_reference(self):
        p = self.left[2]["result"]["identity"]["provenance"]
        p.update(method="operator-reference-endpoint", precision="q4_0", runtime_id="quant-a")
        cp = self.call()
        self.assertEqual(cp.returncode, 0, cp.stderr)
        provenance = json.loads(cp.stdout)["left"]["provenance"]
        self.assertEqual(provenance["method"], "operator-reference-endpoint")
        self.assertEqual(provenance["runtime"]["precision"], "q4_0")

    def test_rejects_mismatched_artifact_bytes_and_unsupported_protocol(self):
        rj, rp = write_side(self.right)
        data = rp.read_bytes()
        rp.write_bytes(data.replace(b"unparsed", b"noanswer", 1))
        self.reject("SHA-256 or size mismatch", right=(rj, rp))
        rp.write_bytes(data + b"\n")
        self.reject("SHA-256 or size mismatch", right=(rj, rp))
        _, _, job, env = self.right
        job["result"]["protocol_id"] = env["protocol_id"] = "some-other-protocol"
        self.reject("pinned GSM8K test split and protocol")

    def test_matches_same_source_builds_without_assigning_roles_by_name(self):
        left = build_side(self.root, "left", [True], "9" * 64)
        left[2]["result"]["identity"]["provenance"]["manifest_sha256"] = "0" * 64
        lhs = write_side(left)
        rhs = write_side(self.right)
        cp = self.call(rhs, lhs)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        result = json.loads(cp.stdout)
        self.assertEqual(result["left"]["job_id"], self.right[2]["id"])
        self.assertEqual(result["right"]["job_id"], left[2]["id"])
        self.assertEqual(result["right"]["provenance"]["manifest_sha256"], "0" * 64)
        self.assertEqual(result["delta_correct"], -4)
        self.assertFalse(any("Source identities differ" in x for x in result["warnings"]))
        self.reject("distinct job IDs", left=lhs, right=lhs)

    def test_rejects_wrong_payload_identity_question_order_and_score(self):
        _, _, job, env = self.right
        env["source_sha256"] = "1" * 64
        self.reject("payload identity")
        env["source_sha256"] = job["result"]["identity"]["source_sha256"]
        env["synthetic"] = True
        self.reject("synthetic flag differs from job")
        env["synthetic"] = False
        rows = env["predictions"]
        rows[0], rows[1] = rows[1], rows[0]
        self.reject("question hash mismatch at row 1")
        rows[0], rows[1] = rows[1], rows[0]
        rows[0]["question_sha256"], original = "0" * 64, rows[0]["question_sha256"]
        self.reject("question hash mismatch at row 1")
        rows[0]["question_sha256"] = original
        end = rows.pop()
        self.reject("cover all 1319")
        rows.append(end)
        job["result"]["score"]["correct"] += 1
        self.reject("recomputed score")

    def test_ties_and_synthetic_are_explicit(self):
        a = build_side(self.root, "left", ["#### 999", "unparsed"], "a" * 64, synthetic=True)
        b = build_side(self.root, "right", ["#### 00999.000", "#### 999"], "a" * 64, synthetic=True)
        self.reject("cannot be paired", left=write_side(a))
        cp = self.call(write_side(a), write_side(b))
        self.assertEqual(cp.returncode, 0, cp.stderr)
        report = json.loads(cp.stdout)
        self.assertIs(report["synthetic"], True)
        self.assertEqual(report["uncertainty"]["p_value"], 1)
        self.assertEqual(report["uncertainty"]["discordant_pairs"], 0)
        self.assertEqual(report["parsed_answer_change_ids"], [2])
        self.assertIsNone(report["right"]["provenance"]["runtime"])

    def test_underflow_emits_exact_fraction(self):
        a = build_side(self.root, "left", [], "a" * 64, synthetic=True)
        b = build_side(self.root, "right", [True] * 1319, "a" * 64, synthetic=True)
        cp = self.call(write_side(a), write_side(b))
        self.assertEqual(cp.returncode, 0, cp.stderr)
        u = json.loads(cp.stdout)["uncertainty"]
        self.assertIsNone(u["p_value"])
        self.assertEqual(u["p_value_exact"], {"numerator": "1", "denominator": str(2**1318)})
