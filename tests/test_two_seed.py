"""Verify paired two-seed summaries and reject incomplete or inconsistent results."""

from copy import deepcopy
import math
import unittest

from run_two_seed import summarize, validate_seeds


def completed_result(baseline_accuracy, pfhmm_accuracy, offset=0.0):
    """Build completed evaluation artifacts with known metrics for two conditions."""
    rows, comparisons = [], []
    for condition, shift in (("clean", 0.0), ("gaussian_noise_0.15", -0.1)):
        for method, accuracy, metric in (
                ("frozen_base", 0.5 + offset, 1.0 + offset),
                ("baseline", baseline_accuracy, 2.0 + offset),
                ("pfhmm", pfhmm_accuracy, 3.0 + offset)):
            rows.append({"method": method, "condition": condition,
                         "examples": 100, "predicted_classes": 10,
                         "accuracy": accuracy + shift, "nll": metric,
                         "ece": metric / 10, "brier": metric / 2,
                         "accuracy_gain_pp": 100 * (accuracy - 0.5 - offset)})
        comparisons.append({"condition": condition,
                            "pfhmm_minus_baseline_pp": 100 * (pfhmm_accuracy - baseline_accuracy),
                            "winner": "pfhmm"})
    return {"evaluation_status": "complete",
            "selection": "fixed final update; no test selection",
            "rows": rows, "comparisons": comparisons}


class TwoSeedValidationTests(unittest.TestCase):
    """Keep the declared four-arm budget and independent seed contract explicit."""

    def test_exactly_two_distinct_builtin_integer_seeds_are_required(self):
        self.assertEqual(validate_seeds([0, 1]), (0, 1))
        self.assertEqual(validate_seeds((19, 2 ** 32 - 1)), (19, 2 ** 32 - 1))
        for seeds in ([], [0], [0, 1, 2], [0, 0], [-1, 0], [0, 2 ** 32],
                      [False, 1], [0, True], [0, 1.0], ["0", 1]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                validate_seeds(seeds)


class TwoSeedSummaryTests(unittest.TestCase):
    """Check numeric aggregation by matching condition and method across complete seeds."""

    def setUp(self):
        """Supply two paired results with analytically known means and sample deviations."""
        self.results = {0: completed_result(0.60, 0.62),
                        1: completed_result(0.65, 0.69, offset=0.2)}

    def test_aggregate_uses_sample_standard_deviation_and_preserves_accuracy_units(self):
        summary = summarize(self.results)
        self.assertEqual(summary["seeds"], [0, 1])
        self.assertEqual(summary["methods"], ["baseline", "pfhmm"])
        self.assertEqual(summary["run_count"], 4)
        self.assertEqual(len(summary["rows"]), 6)
        by_key = {(row["method"], row["condition"]): row for row in summary["rows"]}
        baseline = by_key[("baseline", "clean")]
        self.assertAlmostEqual(baseline["accuracy_mean"], 0.625)
        self.assertAlmostEqual(baseline["accuracy_std"], 0.05 / math.sqrt(2))
        self.assertAlmostEqual(baseline["nll_mean"], 2.1)
        self.assertAlmostEqual(baseline["nll_std"], 0.2 / math.sqrt(2))
        self.assertAlmostEqual(baseline["ece_mean"], 0.21)
        self.assertAlmostEqual(baseline["ece_std"], 0.02 / math.sqrt(2))
        self.assertAlmostEqual(baseline["brier_mean"], 1.05)
        self.assertAlmostEqual(baseline["brier_std"], 0.1 / math.sqrt(2))
        self.assertAlmostEqual(baseline["accuracy_gain_pp_mean"], 2.5)
        self.assertAlmostEqual(baseline["accuracy_gain_pp_std"], 15 / math.sqrt(2))

    def test_comparison_aggregates_paired_differences_in_percentage_points(self):
        summary = summarize(self.results)
        self.assertEqual(len(summary["comparisons"]), 2)
        for comparison in summary["comparisons"]:
            self.assertAlmostEqual(comparison["pfhmm_minus_baseline_pp_mean"], 3.0)
            self.assertAlmostEqual(comparison["pfhmm_minus_baseline_pp_std"], math.sqrt(2))
            self.assertAlmostEqual(comparison["per_seed"]["0"], 2.0)
            self.assertAlmostEqual(comparison["per_seed"]["1"], 4.0)

    def test_row_and_seed_mapping_order_do_not_change_aggregates(self):
        reordered = {1: deepcopy(self.results[1]), 0: deepcopy(self.results[0])}
        reordered[1]["rows"].reverse()
        reordered[1]["comparisons"].reverse()
        expected, actual = summarize(self.results), summarize(reordered)
        for key in ("rows", "comparisons"):
            order = lambda row: (row.get("method", ""), row["condition"])
            self.assertEqual(sorted(expected[key], key=order), sorted(actual[key], key=order))

    def test_incomplete_seed_evaluation_cannot_produce_a_summary(self):
        self.results[1]["evaluation_status"] = "incomplete"
        with self.assertRaises(ValueError):
            summarize(self.results)

    def test_missing_method_or_condition_is_rejected(self):
        for removed in (("pfhmm", "clean"), ("baseline", "gaussian_noise_0.15")):
            results = deepcopy(self.results)
            results[1]["rows"] = [row for row in results[1]["rows"]
                                  if (row["method"], row["condition"]) != removed]
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                summarize(results)

    def test_duplicate_method_condition_row_is_rejected(self):
        self.results[1]["rows"].append(deepcopy(self.results[1]["rows"][0]))
        with self.assertRaises(ValueError):
            summarize(self.results)

    def test_invalid_summary_seed_keys_are_rejected(self):
        for seeds in ((0,), (0, 1, 2), (-1, 1), (0, 2 ** 32), ("0", "1")):
            results = {seed: completed_result(0.6, 0.7) for seed in seeds}
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                summarize(results)


if __name__ == "__main__":
    unittest.main()
