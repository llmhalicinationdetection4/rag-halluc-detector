from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_expected_calibration_error(
    confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) across uniform bins."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def bootstrap_f1_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstraps: int = 1000, seed: int = 42
) -> Tuple[float, float]:
    """Computes 95% Bootstrap Confidence Interval for F1 score."""
    rng = np.random.RandomState(seed)
    f1_scores: List[float] = []
    n = len(y_true)

    if n == 0:
        return (0.0, 0.0)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        sample_yt = y_true[indices]
        sample_yp = y_pred[indices]
        f1_scores.append(f1_score(sample_yt, sample_yp, zero_division=0))

    lower = float(np.percentile(f1_scores, 2.5))
    upper = float(np.percentile(f1_scores, 97.5))
    return lower, upper


def evaluate_predictions(
    y_true: List[bool],
    y_pred: List[bool],
    scores_hallucination: List[float],
) -> Dict[str, float]:
    """
    Evaluates binary hallucination classification.
    positive class (1) = hallucinated.
    """
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_pred, dtype=int)
    scores = np.array(scores_hallucination, dtype=float)

    acc = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    f1 = f1_score(yt, yp, zero_division=0)

    try:
        roc_auc = roc_auc_score(yt, scores)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(yt, scores)
    except Exception:
        pr_auc = float(np.mean(yt))

    # ECE on binary hallucination decision
    accuracies = (yt == yp).astype(float)
    confidences = np.maximum(scores, 1.0 - scores)
    ece = compute_expected_calibration_error(confidences, accuracies)

    ci_low, ci_high = bootstrap_f1_ci(yt, yp, n_bootstraps=1000)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "ece": float(ece),
        "f1_ci_lower": float(ci_low),
        "f1_ci_upper": float(ci_high),
    }