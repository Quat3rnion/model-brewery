import json
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
