"""Check the A7 paired analysis against small independent examples."""

import unittest

from abliteration_acceptance import adjust_holm, statistics


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

    def test_holm_preserves_rank_when_later_raw_values_are_close(self):
        entries = [{"exact_mcnemar_p": p} for p in (0.02, 0.75, 0.015)]
        adjust_holm(entries)
        self.assertEqual([row["holm_adjusted_p"] for row in entries], [0.045, 0.75, 0.045])


if __name__ == "__main__":
    unittest.main()
