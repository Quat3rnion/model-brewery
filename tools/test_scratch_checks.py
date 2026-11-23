import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scratch_checks import (RowScore, aggregate, load_behavior_rows, load_fixtures,
                            measure_behavior, project)


class ScratchChecks(unittest.TestCase):
    def test_behavior_cache_replays_only_held_out_rows_and_verifies_exact_bytes(self):
        behaviors = ("refusal", "slop", "sycophancy", "response_length")
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            pins = {}
            for behavior in behaviors:
                rows = [{
                    "id": split, "split": split,
                    "positive": {"system": "", "user": split + ": positive"},
                    "negative": {"system": "Be brief.", "user": split + ": negative"},
                    **({"expected_agreement": "Yes"} if behavior == "sycophancy" else {}),
                } for split in ("train", "held_out")]
                contents = json.dumps({
                    "schema": "brewery.contrast-prompts.v1", "behavior": behavior, "rows": rows,
                }).encode()
                pins[behavior] = hashlib.sha256(contents).hexdigest()
                (cache / (pins[behavior] + ".json")).write_bytes(contents)
            rows = load_behavior_rows(cache, pins)
            self.assertEqual(rows, [
                *((kind, "", "held_out: positive", "Yes" if kind == "sycophancy" else None)
                  for kind in behaviors),
                ("kl", "Be brief.", "held_out: negative", None),
            ])
            without_refusal = {key: pin for key, pin in pins.items() if key != "refusal"}
            self.assertEqual(load_behavior_rows(cache, without_refusal), rows[1:])
            (cache / (pins["refusal"] + ".json")).write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_behavior_rows(cache, pins)

    def test_behavior_scores_use_the_correct_groups_and_bounds(self):
        # Each reply is (decoded text, generated tokens); the two KL rows are
        # (first-token KL, sum of continuation KL, continuation positions).
        rows = [
            ("refusal", None, ("  I can't assist.", 5), ("I can help.", 4)),
            ("refusal", None, ("I am a pianist.", 5), ("No comment.", 3)),
            ("slop", None, ("Certainly! Here is the answer.", 7), ("Here is the answer.", 5)),
            ("sycophancy", "You are right", ("You are right.", 4), ("You are rightly cautious.", 7)),
            ("response_length", None, ("A" * 290, 64), ("", 0)),
            ("response_length", None, ("Short", 1), ("Longer answer.", 3)),
        ]
        result = measure_behavior(rows, [(0.25, 0.75, 3), (0.5, 0.25, 1)])
        by_name = {m["name"]: (m["parent"], m["child"]) for m in result["metrics"]}
        self.assertEqual(by_name, {
            "refusal_rate": (0.5, 0), "slop_marker_rate": (1, 0),
            "sycophancy_agreement_rate": (1, 0), "response_length_tokens": (32.5, 1.5),
            "kl_first_token_nats": (0, 0.375), "kl_continuation_nats": (0, 0.25),
        })
        self.assertLessEqual(len(result["samples"]["response_length"]["parent"].encode()), 160)
        self.assertNotIn("refusal", result["samples"])
        hooked = measure_behavior(
            rows, [(0, 0, 2)], refusal_prefixes=["No comment."],
            refusal_classifier=lambda reply: None if reply.startswith("No comment") else False,
        )
        self.assertEqual(hooked["metrics"][0]["child"], .5)

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
