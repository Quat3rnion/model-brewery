import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scratch_checks import RowScore, aggregate, load_fixtures, project


class ScratchChecks(unittest.TestCase):
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
