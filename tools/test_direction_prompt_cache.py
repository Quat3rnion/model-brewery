import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import direction_prompt_cache as cache


class DirectionPromptCacheTest(unittest.TestCase):
    def test_grouped_sample_is_deterministic_and_held_out(self):
        rows = [
            {
                "question": f"Persona {persona}. Do you agree or disagree with {topic}?\n"
                "Choices:\n (A) Agree\n (B) Disagree\n\nAnswer:",
                "answer_matching_behavior": " (A)",
            }
            for topic in range(5)
            for persona in range(4)
        ]
        first = cache.normalize("sycophancy", rows, train=3, held_out=2)
        self.assertEqual(first, cache.normalize("sycophancy", reversed(rows), train=3, held_out=2))
        artifact = json.loads(first)
        self.assertEqual(artifact["schema"], "brewery.contrast-prompts.v1")
        self.assertEqual(
            [row["split"] for row in artifact["rows"]], ["train"] * 3 + ["held_out"] * 2
        )
        self.assertEqual(len({row["positive"]["user"].split("Do you agree")[1]
                              for row in artifact["rows"]}), 5)
        self.assertTrue(all(row["expected_agreement"] == "(A)" for row in artifact["rows"]))

    def test_wrong_source_digest_is_rejected_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            cache.verify_source(b"not JSON", "0" * 64)

    @unittest.skipUnless(os.getenv("BREWERY_TEST_PROMPT_CACHE"), "needs verified source cache")
    def test_cli_prepares_only_the_reviewed_benign_holdout_and_rejects_corrupt_source(self):
        source_cache = Path(os.environ["BREWERY_TEST_PROMPT_CACHE"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in source_cache.glob("*.jsonl"):
                shutil.copyfile(path, root / path.name)
            command = [sys.executable, str(Path(cache.__file__)), str(root)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            pins = dict(line.split() for line in result.stdout.splitlines())
            pin = pins.pop("benign_refusal")
            self.assertEqual(pin, "0a3af7d442e3043a713ccb050c4291abf360614f4164cac16e1535d46ea5d35e")
            data = json.loads((root / f"{pin}.json").read_bytes())
            self.assertEqual(data["schema"], "brewery.benign-refusal-prompts.v1")
            rows = data["rows"]
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(set(row) == {"split", "id", "user"} and row["split"] == "held_out"
                                for row in rows))
            contrast_users = {
                " ".join(row[side]["user"].casefold().split())
                for contrast in pins.values()
                for row in json.loads((root / f"{contrast}.json").read_bytes())["rows"]
                for side in ("positive", "negative")
            }
            users = {" ".join(row["user"].casefold().split()) for row in rows}
            self.assertEqual(len(users), 16)
            self.assertTrue(users.isdisjoint(contrast_users))
            dolly = root / (cache.SOURCES["dolly"][1] + ".jsonl")
            with dolly.open("r+b") as file:
                file.write(b"!")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("source SHA-256 mismatch", rejected.stderr)
