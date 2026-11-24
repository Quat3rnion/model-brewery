import json
from pathlib import Path
import tempfile
import unittest

from direction_prompt_cache import digest
from registered_evaluation import build, marker_report, replay
from scratch_checks import load_behavior_rows


class RegisteredEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = {
            "name": "garden",
            "sizes": {"tune": 1, "final_gate": 1},
            "classifier": {
                "max_tokens": 12,
                "classes": [
                    {"name": "context", "kind": "excluded", "prefixes": ["ABOUT:"]},
                    {"name": "color", "kind": "counts", "substrings": ["green"]},
                ],
            },
        }
        for side, prompts in (
            ("positive", ["Name two herbs.", " name  two herbs. ", "Name two trees.", "Common"]),
            ("negative", ["Name two birds.", "Name two fish.", " common "]),
        ):
            raw = "".join(json.dumps({"question": p, "keep": True}) + "\n" for p in prompts).encode()
            sha = digest(raw)
            (self.root / (sha + ".jsonl")).write_bytes(raw)
            self.spec[side] = {
                "url": f"https://raw.githubusercontent.com/garden/sets/{'a' * 40}/{side}.jsonl",
                "sha256": sha, "size": len(raw), "license": "CC0-1.0",
                "format": "jsonl", "field": "question", "system": f"Answer {side}.",
                "filter": {"field": "keep", "equals": True},
            }

    def test_offline_pin_and_disjoint_partitions(self):
        pins = build(self.spec, self.root, self.root)
        self.assertEqual(build(self.spec, self.root, self.root), pins)
        values = []
        for partition, pin in pins.items():
            artifact = json.loads((self.root / (pin + ".json")).read_bytes())
            self.assertEqual(artifact["partition"], partition)
            self.assertEqual([row["side"] for row in artifact["rows"]], ["positive", "negative"])
            values.extend(row["user"].lower().split() for row in artifact["rows"])
        self.assertEqual(len({tuple(v) for v in values}), 4)
        self.assertNotIn(["common"], values)
        src = self.root / (self.spec["positive"]["sha256"] + ".jsonl")
        src.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "size"):
            build(self.spec, self.root, self.root)
        src.write_bytes(b"x" * self.spec["positive"]["size"])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            build(self.spec, self.root, self.root)

    def test_json_and_csv_filter_before_grouping(self):
        for side, fmt, raw, equals in (
            ("positive", "json", b'[{"question":"Name a rose.","keep":1},'
             b'{"question":"Name a tulip.","keep":1},{"keep":true}]', 1),
            ("negative", "csv", b"question,keep\nName a river.,yes\n"
             b"Name a lake.,yes\nDo not select this.,no\n", "yes"),
        ):
            sha = digest(raw)
            (self.root / f"{sha}.{fmt}").write_bytes(raw)
            self.spec[side].update(format=fmt, size=len(raw), sha256=sha,
                                   filter={"field": "keep", "equals": equals})
        pins = build(self.spec, self.root, self.root)
        prompts = [row["user"] for pin in pins.values()
                   for row in json.loads((self.root / (pin + ".json")).read_bytes())["rows"]]
        self.assertCountEqual(prompts, ["Name a rose.", "Name a tulip.", "Name a river.", "Name a lake."])

    def test_first_class_wins_and_no_full_reply_is_returned(self):
        config = {"name": "garden", "partition": "tune", "sha256": "a" * 64,
                  "size_per_side": 1, "classifier": self.spec["classifier"]}
        report = marker_report(config, [("positive", "ABOUT: green", "GrEeN " + "x" * 140),
                                        ("negative", "brown", "")])
        self.assertEqual(report["positive"]["parent_rate"], 0)
        self.assertEqual(report["positive"]["child_rate"], 1)
        self.assertEqual(report["positive"]["parent_class_counts"], {"context": 1, "color": 0})
        row = report["positive"]["hits"][0]
        self.assertEqual(row["parent"], {"counts": False, "class": "context"})
        self.assertEqual(row["child"]["class"], "color")
        self.assertEqual(len(row["child"]["prefix"]), 120)
        self.assertNotIn("brown", json.dumps(report))

    def test_tune_replay_cannot_read_a_final_gate_artifact(self):
        pins = build(self.spec, self.root, self.root)
        config = {"name": "garden", "partition": "tune", "sha256": pins["final_gate"],
                  "size_per_side": 1, "classifier": self.spec["classifier"]}
        with self.assertRaisesRegex(ValueError, "partition"):
            replay("/does/not/exist", "parent", "child", 1, self.root, config)

    def test_registered_artifact_cannot_be_used_as_a_builtin_contrast(self):
        pin = build(self.spec, self.root, self.root)["final_gate"]
        with self.assertRaisesRegex(ValueError, "schema or behavior"):
            load_behavior_rows(self.root, dict(slop=pin, sycophancy=pin, response_length=pin))
