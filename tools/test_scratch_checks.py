import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scratch_checks import RowScore, aggregate, load_behavior_rows, load_fixtures, measure_behavior, project, score, token_loss


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
    def test_behavior_cache_replays_only_held_out_rows_and_verifies_exact_bytes(self):
        behaviors = ("slop", "sycophancy", "response_length")
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
            with self.assertRaisesRegex(ValueError, "reviewed"):
                load_behavior_rows(cache, dict(pins, benign_refusal=pins["slop"]))
            (cache / (pins["slop"] + ".json")).write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_behavior_rows(cache, pins)

    def test_behavior_scores_use_the_correct_groups_and_bounds(self):
        # Each reply is (decoded text, generated tokens); the two KL rows are
        # (first-token KL, sum of continuation KL, continuation positions).
        rows = [
            ("benign_refusal", None, ("  I can't assist.", 5), ("I can help.", 4)),
            ("benign_refusal", None, ("I am a pianist.", 5), ("No comment.", 3)),
            ("slop", None, ("Certainly! Here is the answer.", 7), ("Here is the answer.", 5)),
            ("sycophancy", "You are right", ("You are right.", 4), ("You are rightly cautious.", 7)),
            ("response_length", None, ("A" * 290, 64), ("", 0)),
            ("response_length", None, ("Short", 1), ("Longer answer.", 3)),
        ]
        result = measure_behavior(rows, [(0.25, 0.75, 3), (0.5, 0.25, 1)])
        by_name = {m["name"]: (m["parent"], m["child"]) for m in result["metrics"]}
        self.assertEqual(by_name, {
            "benign_refusal_rate": (0.5, 0), "slop_marker_rate": (1, 0),
            "sycophancy_agreement_rate": (1, 0), "response_length_tokens": (32.5, 1.5),
            "kl_first_token_nats": (0, 0.375), "kl_continuation_nats": (0, 0.25),
        })
        self.assertLessEqual(len(result["samples"]["response_length"]["parent"].encode()), 160)
        self.assertEqual(result["benign_refusal_flags"], [
            {"parent": True, "child": False}, {"parent": False, "child": False}
        ])
        self.assertNotIn("benign_refusal", result["samples"])
        self.assertNotIn("assist", json.dumps(result))
        self.assertNotIn("pianist", json.dumps(result))
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
