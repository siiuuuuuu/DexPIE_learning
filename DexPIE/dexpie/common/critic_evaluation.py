"""Reusable offline metrics for the distributional value critic.

The functions in this module operate on NumPy arrays and have no dependency on
the training workspace.  This keeps metric definitions identical between the
standalone evaluator and a future training-loop integration.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12
STAGE_NAMES = ("early", "middle", "late")


def weighted_mean(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def weighted_var(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    mean = weighted_mean(values, weights)
    if not np.isfinite(mean):
        return float("nan")
    return weighted_mean(np.square(np.asarray(values, dtype=np.float64) - mean), weights)


def compute_point_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute scalar-value metrics.

    EV intentionally uses residual variance and therefore ignores a constant
    prediction offset. R2 uses squared error and exposes that offset.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError(
            f"target and prediction shapes differ: {target.shape} vs {prediction.shape}"
        )
    if target.size == 0:
        return {
            "ev": float("nan"),
            "r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
        }

    residual = target - prediction
    target_var = weighted_var(target, weights)
    residual_var = weighted_var(residual, weights)
    target_mean = weighted_mean(target, weights)
    mse = weighted_mean(np.square(residual), weights)
    denominator = weighted_mean(np.square(target - target_mean), weights)

    ev = float("nan") if target_var <= EPS else 1.0 - residual_var / target_var
    r2 = float("nan") if denominator <= EPS else 1.0 - mse / denominator
    return {
        "ev": float(ev),
        "r2": float(r2),
        "rmse": float(np.sqrt(max(mse, 0.0))),
        "mae": weighted_mean(np.abs(residual), weights),
        "bias": weighted_mean(prediction - target, weights),
    }


def make_equal_count_calibration_bins(
    prediction: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Assign samples to deterministic equal-count bins by predicted value."""
    prediction = np.asarray(prediction, dtype=np.float64)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    assignments = np.full(prediction.shape, -1, dtype=np.int64)
    if prediction.size == 0:
        return assignments
    n_effective = min(int(n_bins), int(prediction.size))
    sorted_indices = np.argsort(prediction, kind="mergesort")
    for bin_idx, indices in enumerate(np.array_split(sorted_indices, n_effective)):
        assignments[indices] = bin_idx
    return assignments


def compute_calibration(
    target: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, List[Dict[str, float]]]:
    """Return weighted calibration RMSE and per-bin statistics."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=np.int64)
    if not (target.shape == prediction.shape == assignments.shape):
        raise ValueError("target, prediction and assignments must have identical shapes")
    if weights is None:
        weights = np.ones_like(target, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    rows = []
    squared_gap_sum = 0.0
    total_weight = 0.0
    valid_bins = sorted(int(x) for x in np.unique(assignments) if x >= 0)
    for bin_idx in valid_bins:
        mask = assignments == bin_idx
        bin_weight = float(np.sum(weights[mask]))
        if bin_weight <= 0:
            continue
        predicted_mean = weighted_mean(prediction[mask], weights[mask])
        observed_mean = weighted_mean(target[mask], weights[mask])
        gap = predicted_mean - observed_mean
        squared_gap_sum += bin_weight * gap * gap
        total_weight += bin_weight
        rows.append(
            {
                "calibration_bin": int(bin_idx),
                "num_samples": int(np.sum(mask)),
                "weight": bin_weight,
                "prediction_min": float(np.min(prediction[mask])),
                "prediction_max": float(np.max(prediction[mask])),
                "predicted_mean": predicted_mean,
                "observed_mean": observed_mean,
                "gap": gap,
            }
        )
    calibration_rmse = (
        float("nan")
        if total_weight <= 0
        else float(np.sqrt(squared_gap_sum / total_weight))
    )
    return calibration_rmse, rows


def compute_metric_set(
    target: np.ndarray,
    prediction: np.ndarray,
    distribution_metrics: Dict[str, np.ndarray],
    calibration_assignments: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = compute_point_metrics(target, prediction, weights)
    calibration_rmse, _ = compute_calibration(
        target,
        prediction,
        calibration_assignments,
        weights,
    )
    metrics["calibration_rmse"] = calibration_rmse
    for name, values in distribution_metrics.items():
        metrics[name] = weighted_mean(values, weights)
    return metrics


def bootstrap_metric_intervals(
    target: np.ndarray,
    prediction: np.ndarray,
    distribution_metrics: Dict[str, np.ndarray],
    calibration_assignments: np.ndarray,
    episode_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Episode-bootstrap CIs using per-episode sufficient statistics.

    Runtime scales with the number of episodes rather than repeatedly scanning
    every correlated frame for every bootstrap replicate.
    """
    if n_bootstrap <= 0:
        return {}
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    calibration_assignments = np.asarray(calibration_assignments, dtype=np.int64)
    episode_ids = np.asarray(episode_ids)
    unique_episodes, inverse = np.unique(episode_ids, return_inverse=True)
    if unique_episodes.size < 2:
        return {}

    base = compute_metric_set(
        target,
        prediction,
        distribution_metrics,
        calibration_assignments,
    )
    if metric_names is None:
        selected_names = list(base.keys())
    else:
        selected_names = [name for name in metric_names if name in base]
    samples = {name: [] for name in selected_names}
    rng = np.random.default_rng(seed)

    n_episodes = unique_episodes.size
    residual = target - prediction
    episode_count = np.bincount(inverse, minlength=n_episodes).astype(np.float64)

    def episode_sum(values):
        return np.bincount(
            inverse,
            weights=np.asarray(values, dtype=np.float64),
            minlength=n_episodes,
        )

    sufficient = {
        "target": episode_sum(target),
        "target_sq": episode_sum(np.square(target)),
        "residual": episode_sum(residual),
        "residual_sq": episode_sum(np.square(residual)),
        "abs_residual": episode_sum(np.abs(residual)),
        "prediction_minus_target": episode_sum(prediction - target),
    }
    distribution_sums = {
        name: episode_sum(values) for name, values in distribution_metrics.items()
    }

    valid_assignments = calibration_assignments[calibration_assignments >= 0]
    n_calibration_bins = (
        int(np.max(valid_assignments)) + 1 if valid_assignments.size > 0 else 0
    )
    if n_calibration_bins > 0:
        combined = inverse * n_calibration_bins + calibration_assignments
        shape = (n_episodes, n_calibration_bins)
        calibration_count = np.bincount(
            combined,
            minlength=n_episodes * n_calibration_bins,
        ).reshape(shape).astype(np.float64)
        calibration_target = np.bincount(
            combined,
            weights=target,
            minlength=n_episodes * n_calibration_bins,
        ).reshape(shape)
        calibration_prediction = np.bincount(
            combined,
            weights=prediction,
            minlength=n_episodes * n_calibration_bins,
        ).reshape(shape)
    else:
        calibration_count = np.zeros((n_episodes, 0), dtype=np.float64)
        calibration_target = np.zeros_like(calibration_count)
        calibration_prediction = np.zeros_like(calibration_count)

    for _ in range(int(n_bootstrap)):
        selected = rng.integers(0, n_episodes, size=n_episodes)
        episode_weights = np.bincount(selected, minlength=n_episodes).astype(np.float64)
        total = float(np.dot(episode_weights, episode_count))
        if total <= 0:
            continue

        target_mean = float(np.dot(episode_weights, sufficient["target"]) / total)
        target_second = float(np.dot(episode_weights, sufficient["target_sq"]) / total)
        target_var = max(target_second - target_mean * target_mean, 0.0)
        residual_mean = float(
            np.dot(episode_weights, sufficient["residual"]) / total
        )
        residual_second = float(
            np.dot(episode_weights, sufficient["residual_sq"]) / total
        )
        residual_var = max(
            residual_second - residual_mean * residual_mean,
            0.0,
        )
        mse = residual_second
        metrics = {
            "ev": float("nan") if target_var <= EPS else 1.0 - residual_var / target_var,
            "r2": float("nan") if target_var <= EPS else 1.0 - mse / target_var,
            "rmse": float(np.sqrt(max(mse, 0.0))),
            "mae": float(
                np.dot(episode_weights, sufficient["abs_residual"]) / total
            ),
            "bias": float(
                np.dot(episode_weights, sufficient["prediction_minus_target"]) / total
            ),
        }
        for name, values in distribution_sums.items():
            metrics[name] = float(np.dot(episode_weights, values) / total)

        if n_calibration_bins > 0:
            bin_count = episode_weights @ calibration_count
            bin_target = episode_weights @ calibration_target
            bin_prediction = episode_weights @ calibration_prediction
            valid = bin_count > 0
            gaps = np.zeros_like(bin_count)
            gaps[valid] = (
                bin_prediction[valid] / bin_count[valid]
                - bin_target[valid] / bin_count[valid]
            )
            metrics["calibration_rmse"] = float(
                np.sqrt(np.sum(bin_count[valid] * np.square(gaps[valid])) / total)
            )
        else:
            metrics["calibration_rmse"] = float("nan")

        for name in selected_names:
            value = metrics[name]
            if np.isfinite(value):
                samples[name].append(value)

    intervals = {}
    for name, values in samples.items():
        if len(values) == 0:
            intervals[name] = (float("nan"), float("nan"))
        else:
            low, high = np.percentile(np.asarray(values), [2.5, 97.5])
            intervals[name] = (float(low), float(high))
    return intervals


def compute_distribution_diagnostics(
    probabilities: np.ndarray,
    target: np.ndarray,
    bin_centers: np.ndarray,
    gaussian_std_bins: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute per-sample proper scores, sharpness, and interval coverage."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got {probabilities.shape}")
    if probabilities.shape[0] != target.shape[0]:
        raise ValueError("probabilities and target sample counts differ")
    if probabilities.shape[1] != bin_centers.shape[0]:
        raise ValueError("probability bins and bin_centers differ")
    if bin_centers.size < 2:
        raise ValueError("at least two value bins are required")

    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    bin_width = float(bin_centers[1] - bin_centers[0])
    v_min = float(bin_centers[0])
    hard_bins = np.floor((target - v_min) / bin_width).astype(np.int64)
    hard_bins = np.clip(hard_bins, 0, bin_centers.size - 1)

    bin_positions = np.arange(bin_centers.size, dtype=np.float64)
    squared_distances = np.square(
        bin_positions[None, :] - hard_bins[:, None].astype(np.float64)
    )
    soft_targets = np.exp(-squared_distances / (2.0 * gaussian_std_bins ** 2))
    soft_targets /= soft_targets.sum(axis=1, keepdims=True)
    log_probabilities = np.log(probabilities)

    cdf = np.cumsum(probabilities, axis=1)
    observed_cdf = (
        bin_centers[None, :] >= target[:, None]
    ).astype(np.float64)
    crps = np.sum(np.square(cdf - observed_cdf), axis=1) * abs(bin_width)

    output = {
        "soft_cross_entropy": -np.sum(soft_targets * log_probabilities, axis=1),
        "hard_nll": -log_probabilities[np.arange(target.size), hard_bins],
        "crps": crps,
        "normalized_entropy": (
            -np.sum(probabilities * log_probabilities, axis=1)
            / np.log(float(bin_centers.size))
        ),
    }

    for level in (0.5, 0.8):
        tail = (1.0 - level) / 2.0
        lower_idx = np.argmax(cdf >= tail, axis=1)
        upper_idx = np.argmax(cdf >= (1.0 - tail), axis=1)
        lower = bin_centers[lower_idx]
        upper = bin_centers[upper_idx]
        suffix = int(round(level * 100))
        output[f"coverage_{suffix}"] = (
            (target >= lower) & (target <= upper)
        ).astype(np.float64)
        output[f"width_{suffix}"] = upper - lower
    return output


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware binary AUROC using average ranks."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be matching one-dimensional arrays")
    n_positive = int(np.sum(labels))
    n_negative = int(labels.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        stop = start + 1
        while stop < labels.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        sorted_ranks[start:stop] = average_rank
        start = stop
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    rank_sum = float(np.sum(ranks[labels]))
    return (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def build_episode_stage_rows(
    target: np.ndarray,
    prediction: np.ndarray,
    episode_ids: np.ndarray,
    stages: np.ndarray,
    episode_success: np.ndarray,
) -> List[Dict[str, float]]:
    """Aggregate frame values to one record per episode and stage."""
    rows = []
    for episode_id in np.unique(episode_ids):
        episode_mask = episode_ids == episode_id
        success_values = np.unique(episode_success[episode_mask])
        if success_values.size != 1:
            raise ValueError(f"episode {episode_id} has inconsistent success labels")
        for stage_idx, stage_name in enumerate(STAGE_NAMES):
            mask = episode_mask & (stages == stage_idx)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "episode_id": int(episode_id),
                    "stage": stage_name,
                    "stage_index": int(stage_idx),
                    "success": bool(success_values[0]),
                    "num_frames": int(np.sum(mask)),
                    "predicted_value": float(np.mean(prediction[mask])),
                    "target_value": float(np.mean(target[mask])),
                }
            )
    return rows


def summarize_failure_discrimination(
    episode_stage_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Measure eventual-failure separation at matched trajectory stages."""
    output = []
    for stage_name in STAGE_NAMES:
        rows = [row for row in episode_stage_rows if row["stage"] == stage_name]
        labels_failure = np.asarray([not row["success"] for row in rows], dtype=bool)
        predicted = np.asarray([row["predicted_value"] for row in rows], dtype=np.float64)
        target = np.asarray([row["target_value"] for row in rows], dtype=np.float64)
        success_mask = ~labels_failure
        failure_mask = labels_failure
        output.append(
            {
                "stage": stage_name,
                "num_episodes": int(len(rows)),
                "num_success": int(np.sum(success_mask)),
                "num_failure": int(np.sum(failure_mask)),
                "failure_auroc": binary_auroc(labels_failure, -predicted)
                if len(rows) > 0
                else float("nan"),
                "predicted_success_minus_failure_gap": (
                    float(np.mean(predicted[success_mask]) - np.mean(predicted[failure_mask]))
                    if np.any(success_mask) and np.any(failure_mask)
                    else float("nan")
                ),
                "target_success_minus_failure_gap": (
                    float(np.mean(target[success_mask]) - np.mean(target[failure_mask]))
                    if np.any(success_mask) and np.any(failure_mask)
                    else float("nan")
                ),
            }
        )
    return output


def summarize_progress_separation(
    episode_stage_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Measure early-to-late value change separately by final outcome."""
    output = []
    for success in (True, False):
        rows = [row for row in episode_stage_rows if row["success"] == success]
        by_episode = {}
        for row in rows:
            by_episode.setdefault(row["episode_id"], {})[row["stage"]] = row
        predicted_gaps = []
        target_gaps = []
        for stage_map in by_episode.values():
            if "early" not in stage_map or "late" not in stage_map:
                continue
            predicted_gaps.append(
                stage_map["late"]["predicted_value"]
                - stage_map["early"]["predicted_value"]
            )
            target_gaps.append(
                stage_map["late"]["target_value"]
                - stage_map["early"]["target_value"]
            )
        output.append(
            {
                "outcome": "success" if success else "failure",
                "num_episodes": int(len(predicted_gaps)),
                "predicted_late_minus_early": (
                    float(np.mean(predicted_gaps))
                    if len(predicted_gaps) > 0
                    else float("nan")
                ),
                "target_late_minus_early": (
                    float(np.mean(target_gaps))
                    if len(target_gaps) > 0
                    else float("nan")
                ),
            }
        )
    return output


def fit_progress_baseline(
    timesteps: np.ndarray,
    target: np.ndarray,
    max_length: int,
    n_bins: int,
) -> Dict[str, np.ndarray]:
    """Fit a return baseline that observes only absolute episode timestep."""
    if max_length <= 0 or n_bins <= 0:
        raise ValueError("max_length and n_bins must be positive")
    timesteps = np.asarray(timesteps, dtype=np.int64)
    target = np.asarray(target, dtype=np.float64)
    bin_ids = np.minimum(timesteps * n_bins // max_length, n_bins - 1)
    global_mean = float(np.mean(target))
    bin_values = np.full((n_bins,), global_mean, dtype=np.float64)
    bin_counts = np.zeros((n_bins,), dtype=np.int64)
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        bin_counts[bin_idx] = int(np.sum(mask))
        if np.any(mask):
            bin_values[bin_idx] = float(np.mean(target[mask]))
    return {
        "bin_values": bin_values,
        "bin_counts": bin_counts,
        "max_length": np.asarray([max_length], dtype=np.int64),
        "n_bins": np.asarray([n_bins], dtype=np.int64),
    }


def predict_progress_baseline(model: Dict[str, np.ndarray], timesteps: np.ndarray) -> np.ndarray:
    max_length = int(model["max_length"][0])
    n_bins = int(model["n_bins"][0])
    timesteps = np.asarray(timesteps, dtype=np.int64)
    bin_ids = np.minimum(timesteps * n_bins // max_length, n_bins - 1)
    return model["bin_values"][bin_ids]


def compute_progress_conditioned_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    progress_prediction: np.ndarray,
) -> Dict[str, float]:
    """Compare the critic against an absolute-timestep-only baseline."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    progress_prediction = np.asarray(progress_prediction, dtype=np.float64)
    critic_error = target - prediction
    progress_error = target - progress_prediction
    critic_mse = float(np.mean(np.square(critic_error)))
    progress_mse = float(np.mean(np.square(progress_error)))
    progress_residual_var = float(np.var(progress_error))
    return {
        "progress_baseline_rmse": float(np.sqrt(progress_mse)),
        "progress_conditioned_skill": (
            float("nan") if progress_mse <= EPS else 1.0 - critic_mse / progress_mse
        ),
        "progress_conditioned_ev": (
            float("nan")
            if progress_residual_var <= EPS
            else 1.0 - float(np.var(critic_error)) / progress_residual_var
        ),
    }
