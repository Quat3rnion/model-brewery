import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("halogen_speculation.py")


def response(idx, rounds, drafted, accepted):
    empty = {"rounds": 0, "drafted": 0, "accepted": 0}
    return {"object": "chat.completion", "model": "local", "id": f"x-{idx}",
            "choices": [{"message": {"content": "DO NOT EXPORT"}}],
            "timings": {"prompt_n": 3, "predicted_n": 8, "speculation": {
                "drafter": "mtp", "rounds": rounds,
                "head": dict(rounds=rounds, drafted=drafted, accepted=accepted),
                "lookup": empty, "grammar": empty}, "draft_n": drafted,
                "draft_n_accepted": accepted}}


class HalogenSpeculation(unittest.TestCase):
    def test_native_01822_observation(self):
        # Numeric fields of response #9 from the 112-request MTP benchmark.
        # Only the synthetic request ID and non-text fields remain.
        fixture = SCRIPT.parent / "fixtures/halogen-0.18.22-timing.jsonl"
        proc = subprocess.run([sys.executable, SCRIPT, "--producer-version", "0.18.22", fixture],
                              check=True, capture_output=True)
        report = json.loads(proc.stdout)
        self.assertEqual(report["methods"]["head"]["accepted_per_round"], 2.75)
        self.assertEqual(report["methods"]["lookup"]["acceptance_fraction"], 5/6)

    def invoke(self, rows, *opts):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return subprocess.run([sys.executable, SCRIPT, "--producer-version", "0.18.22",
                                   *opts, path], text=True, capture_output=True)

    def test_partial_coverage_weighted_totals_and_no_text(self):
        missing = response(3, 0, 0, 0)
        missing["timings"] = {"prompt_n": 2, "predicted_n": 1}
        proc = self.invoke([response(1, 1, 1, 1), response(2, 3, 6, 2), missing])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual((report["total_with_speculation"], report["missing_speculation"]), (2, 1))
        self.assertEqual(report["methods"]["head"]["acceptance_fraction"], 3 / 7)
        self.assertEqual(report["methods"]["head"]["accepted_per_round"], 3 / 4)
        self.assertIsNone(report["methods"]["lookup"]["acceptance_fraction"])
        self.assertIsNone(report["acceptance_by_depth"])
        self.assertNotIn("DO NOT EXPORT", proc.stdout)

    def test_invalid_reports_are_not_partially_published(self):
        item = response(1, 1, 2, 1)
        for rows in ([item, item], [item, dict(response(2, 0, 0, 0), model="other")],
                     [response(1, 1, 2, 3)], [dict(item, timings={"draft_n": 1})],
                     [dict(item, timings=dict(item["timings"], draft_n_accepted=0))]):
            with self.subTest(), self.assertRaises(subprocess.CalledProcessError):
                proc = self.invoke(rows)
                self.assertEqual(proc.stdout, "")
                proc.check_returncode()

    def test_distinguishes_absence_from_zero_and_explicit_wrapper(self):
        row = response(1, 0, 0, 0)
        row["timings"] = {"prompt_n": 3, "predicted_n": 2}
        proc = self.invoke([{"response": row, "metrics": {"head": 99}}], "--wrapped")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIsNone(json.loads(proc.stdout)["methods"]["head"])
