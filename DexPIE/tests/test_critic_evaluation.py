import unittest

import numpy as np

from dexpie.common.critic_evaluation import (
    binary_auroc,
    bootstrap_metric_intervals,
    build_episode_stage_rows,
    compute_distribution_diagnostics,
    compute_point_metrics,
    compute_progress_conditioned_metrics,
    fit_progress_baseline,
    make_equal_count_calibration_bins,
    predict_progress_baseline,
    summarize_failure_discrimination,
    summarize_progress_separation,
)


class CriticEvaluationTest(unittest.TestCase):
    def test_ev_ignores_constant_offset_but_r2_does_not(self):
        target = np.asarray([-1.0, -0.7, -0.2, 0.0])
        metrics = compute_point_metrics(target, target + 0.2)
        self.assertAlmostEqual(metrics["ev"], 1.0)
        self.assertLess(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["bias"], 0.2)

    def test_perfect_point_prediction(self):
        target = np.asarray([-1.0, -0.5, 0.0])
        metrics = compute_point_metrics(target, target)
        self.assertAlmostEqual(metrics["ev"], 1.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)

    def test_distribution_diagnostics(self):
        centers = np.linspace(-1.0, 0.0, 5)
        target = np.asarray([-0.5])
        probabilities = np.asarray([[1e-8, 1e-8, 1.0, 1e-8, 1e-8]])
        diagnostics = compute_distribution_diagnostics(
            probabilities,
            target,
            centers,
        )
        self.assertLess(diagnostics["hard_nll"][0], 1e-6)
        self.assertLess(diagnostics["crps"][0], 1e-6)
        self.assertEqual(diagnostics["coverage_50"][0], 1.0)
        self.assertEqual(diagnostics["coverage_80"][0], 1.0)

    def test_binary_auroc_with_ties(self):
        labels = np.asarray([False, False, True, True])
        self.assertAlmostEqual(binary_auroc(labels, np.asarray([0, 0, 1, 1])), 1.0)
        self.assertAlmostEqual(binary_auroc(labels, np.ones(4)), 0.5)

    def test_progress_and_failure_are_summarized_at_episode_level(self):
        episode_ids = np.repeat(np.arange(4), 3)
        stages = np.tile(np.arange(3), 4)
        success = np.repeat(np.asarray([True, True, False, False]), 3)
        target = np.tile(np.asarray([-0.8, -0.4, 0.0]), 4)
        prediction = target.copy()
        prediction[~success] -= 0.2
        rows = build_episode_stage_rows(
            target,
            prediction,
            episode_ids,
            stages,
            success,
        )
        failure = summarize_failure_discrimination(rows)
        progress = summarize_progress_separation(rows)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(item["failure_auroc"] == 1.0 for item in failure))
        self.assertGreater(progress[0]["predicted_late_minus_early"], 0.0)
        self.assertGreater(progress[1]["predicted_late_minus_early"], 0.0)

    def test_progress_conditioned_skill(self):
        timesteps = np.tile(np.arange(4), 4)
        target = np.linspace(-1.0, 0.0, timesteps.size)
        model = fit_progress_baseline(timesteps, target, max_length=4, n_bins=4)
        progress_prediction = predict_progress_baseline(model, timesteps)
        metrics = compute_progress_conditioned_metrics(
            target,
            target,
            progress_prediction,
        )
        self.assertAlmostEqual(metrics["progress_conditioned_skill"], 1.0)

    def test_episode_bootstrap_produces_intervals(self):
        target = np.asarray([-1.0, -0.8, -0.2, 0.0])
        prediction = target + np.asarray([0.0, 0.1, -0.1, 0.0])
        assignments = make_equal_count_calibration_bins(prediction, 2)
        intervals = bootstrap_metric_intervals(
            target=target,
            prediction=prediction,
            distribution_metrics={"crps": np.asarray([0.1, 0.2, 0.2, 0.1])},
            calibration_assignments=assignments,
            episode_ids=np.asarray([0, 0, 1, 1]),
            n_bootstrap=20,
            seed=3,
            metric_names=("rmse", "crps"),
        )
        self.assertIn("rmse", intervals)
        self.assertIn("crps", intervals)
        self.assertLessEqual(intervals["rmse"][0], intervals["rmse"][1])


if __name__ == "__main__":
    unittest.main()
