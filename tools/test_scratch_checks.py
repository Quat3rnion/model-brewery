import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scratch_checks import RowScore, aggregate, load_fixtures, project, score, token_loss


class ScratchChecks(unittest.TestCase):
    def test_heldout_scores_keep_only_aligned_predicted_tokens(self):
        rows = [("perplexity", "The", " box."), ("zsre_target", "A", " pen")]
        before = "generation\t61\n1,2,3\t1\t-5\t.3\t5\t0\t-3:20626f78,-2:2e\n1,4\t1\t-2\t.1\t1\t1\n"
        after = before.replace("-5\t.3", "-3\t.3").replace("-3:20626f78", "-1:20626f78")
        with patch("scratch_checks.subprocess.run", side_effect=[
            SimpleNamespace(stdout=before), SimpleNamespace(stdout=after),
            SimpleNamespace(stdout=after.replace("20626f78", "20626167")),
            SimpleNamespace(stdout=after.replace("-3\t.3", "-4\t.3")),
        ]):
            _, parent = score("runner", "parent", 1, rows)
            _, child = score("runner", "child", 1, rows)
            self.assertEqual(token_loss(rows, parent, child), [[
                dict(id=2, piece_hex="20626f78", parent_logprob=-3., child_logprob=-1.),
                dict(id=3, piece_hex="2e", parent_logprob=-2., child_logprob=-2.),
            ]])
            _, bad = score("runner", "child", 1, rows)
            with self.assertRaisesRegex(ValueError, "token pieces"):
                token_loss(rows, parent, bad)
            with self.assertRaisesRegex(ValueError, "log probabilities"):
                score("runner", "child", 1, rows)

    def test_legacy_runner_keeps_perplexity_but_has_no_token_evidence(self):
        rows = [("perplexity", "The", " box.")]
        output = f"generation\t61\n1,2,3\t1\t{math.log(.25)}\t1\t2\t1\n"
        with patch("scratch_checks.subprocess.run", return_value=SimpleNamespace(stdout=output)):
            _, measured = score("runner", "model", 1, rows)
        self.assertEqual(aggregate(measured)["perplexity"], (2., "lower"))
        self.assertIsNone(token_loss(rows, measured, measured))

    def test_fixture_counts_and_groups(self):
        row = dict(subject="Ada", src="Where did Ada go?", rephrase="Where was Ada going?",
                   alt="the park", loc="nq question: who wrote the note", loc_ans="Pat")
        data = json.dumps([row]).encode()
        expected = (b'{"id":0,"target":["Where did Ada go?"," the park"],'
                    b'"paraphrases":[["Where was Ada going?"," the park"]],'
                    b'"locality":[["who wrote the note"," Pat"]]}\n')
        self.assertEqual(project("zsre.jsonl", data, [0]), expected)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "cache").mkdir()
            corpus = b"The box is red.\n"
            (root / "data/heldout.txt").write_bytes(corpus)
            (root / "data/manifest.json").write_text(json.dumps({
                "zsre.jsonl": hashlib.sha256(expected).hexdigest(),
                "heldout.txt": hashlib.sha256(corpus).hexdigest(),
            }))
            src = {
                "url": "https://example.invalid/data.json", "cache": "data.json",
                "sha256": hashlib.sha256(data).hexdigest(), "item_ids": [0],
            }
            (root / "data/sources.json").write_text(json.dumps({"zsre.jsonl": src}))
            cached = root / "cache/data.json"
            cached.write_bytes(data)
            _, rows = load_fixtures(root / "cache", data=root / "data")
            self.assertEqual(rows[0], ("zsre_target", "Where did Ada go?", " the park"))
            self.assertEqual(rows[-1], ("perplexity", "The", " box is red."))
            cached.write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "upstream hash"):
                load_fixtures(root / "cache", data=root / "data")
            cached.write_bytes(data)
            (root / "data/heldout.txt").write_bytes(b"The box is blue.\n")
            with self.assertRaisesRegex(ValueError, "fixture hash"):
                load_fixtures(root / "cache", data=root / "data")

    def test_aggregation_weights_tokens_and_full_answers_separately(self):
        scores = {
            "counterfact_target": [
                RowScore((1, 4), 1, math.log(.5), .5, 1, True),
                RowScore((1, 2, 5, 6), 2, math.log(.24), 1, 4, False),
            ],
            "perplexity": [
                RowScore((1, 4), 1, math.log(.5), .5, 1, True),
                RowScore((1, 2, 5, 6), 2, math.log(.25), 1, 2, True),
            ],
        }
        self.assertEqual(aggregate(scores), {
            "counterfact_target_probability": (.5, "higher"),
            "counterfact_target_rank": (5 / 3, "lower"),
            "counterfact_efficacy": (.5, "higher"),
            "perplexity": (2., "lower"),
        })


if __name__ == "__main__":
    unittest.main()
