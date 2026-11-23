"""Check the A7 paired analysis against small independent examples."""

import unittest

from abliteration_acceptance import statistics


class PairedStatsTest(unittest.TestCase):
    def test_exact_mcnemar_uses_discordant_pairs(self):
        result = statistics([True] * 4 + [False] * 6, [False] * 10)
        self.assertEqual(result["improved"], 4)
        self.assertEqual(result["worsened"], 0)
        self.assertEqual(result["exact_mcnemar_p"], 0.125)
        self.assertAlmostEqual(result["child_wilson95"][1], 0.2775328)

    def test_identical_flags_have_no_evidence_of_change(self):
        result = statistics([True, False], [True, False])
        self.assertEqual(result["exact_mcnemar_p"], 1.0)
        self.assertEqual(result["parent_wilson95"], result["child_wilson95"])


if __name__ == "__main__":
    unittest.main()
