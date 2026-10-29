import unittest

from evaluation import binary_metrics, split_groups


class EvaluationTests(unittest.TestCase):
    def test_groups_do_not_cross_partitions(self):
        groups = [f"series-{i // 4}" for i in range(24)]
        partitions = split_groups(groups, seed=9)
        selected = [{groups[i] for i in indices} for indices in partitions.values()]
        self.assertTrue(all(selected))
        for a in range(3):
            for b in range(a + 1, 3):
                self.assertFalse(selected[a] & selected[b])
        self.assertEqual(sorted(i for rows in partitions.values() for i in rows), list(range(24)))
        self.assertEqual(partitions, split_groups(groups, seed=9))

    def test_single_series_cannot_be_treated_as_a_holdout_experiment(self):
        with self.assertRaises(ValueError):
            split_groups(["one-series"] * 44)
        with self.assertRaises(ValueError):
            split_groups(["map-1", "map-2"])
        result = split_groups(["map-1", "map-2"], smoke_test=True)
        self.assertTrue(result["train"])
        self.assertTrue(result["validation"])
        self.assertEqual(result["test"], [])

    def test_metrics_and_probability_ties(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9])
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["roc_auc"], 0.875)
        self.assertAlmostEqual(metrics["brier_score"], 0.13)

    def test_single_class_auc_is_explicitly_unavailable(self):
        metrics = binary_metrics([0, 0], [0.1, 0.2])
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["balanced_accuracy"])

    def test_invalid_metrics_fail(self):
        for labels, probs in (([], []), ([1], []), ([1], [float("nan")]), ([0], [1.1]), ([2], [0.5])):
            with self.subTest(labels=labels, probs=probs):
                with self.assertRaises(ValueError):
                    binary_metrics(labels, probs)


if __name__ == "__main__":
    unittest.main()
