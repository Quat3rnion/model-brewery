"""Regression checks for X4 held-out random-control comparisons."""

import unittest

from x4_generalization import controls_summary


class ControlSummaryTests(unittest.TestCase):
    def test_compares_each_metric_and_excludes_ties(self):
        rows = {
            "corpus": {"perplexity": 5, "mean_kl": 2},
            "seed-1": {"perplexity": 8, "mean_kl": 3},
            "seed-2": {"perplexity": 2, "mean_kl": 4},
            "seed-3": {"perplexity": 5, "mean_kl": 2},
            "corpus-matched-seed-4": {"perplexity": 10, "mean_kl": 10},
        }
        self.assertEqual(controls_summary(rows, "seed-"), {
            "count": 3,
            "perplexity": {"median": 5, "recipe_beats_median": False, "recipe_beats": 1},
            "mean_kl": {"median": 3, "recipe_beats_median": True, "recipe_beats": 2},
            "recipe_beats_on_both": 1,
        })
        self.assertEqual(controls_summary(rows, "corpus-matched-seed-"), {
            "count": 1,
            "perplexity": {"median": 10, "recipe_beats_median": True, "recipe_beats": 1},
            "mean_kl": {"median": 10, "recipe_beats_median": True, "recipe_beats": 1},
            "recipe_beats_on_both": 1,
        })


if __name__ == "__main__":
    unittest.main()
