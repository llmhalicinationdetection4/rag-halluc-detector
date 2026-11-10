import numpy as np
from halludetect.eval.metrics import (
    bootstrap_f1_ci,
    compute_expected_calibration_error,
    evaluate_predictions,
)


def test_ece_computation():
    confidences = np.array([0.9, 0.8, 0.6, 0.4])
    accuracies = np.array([1.0, 1.0, 0.0, 0.0])
    ece = compute_expected_calibration_error(confidences, accuracies, num_bins=5)
    assert 0.0 <= ece <= 1.0


def test_bootstrap_ci():
    yt = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    yp = np.array([1, 0, 0, 0, 1, 0, 1, 1])
    low, high = bootstrap_f1_ci(yt, yp, n_bootstraps=50)
    assert 0.0 <= low <= high <= 1.0


def test_evaluate_predictions():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, False]
    scores = [0.9, 0.4, 0.2, 0.1]
    res = evaluate_predictions(y_true, y_pred, scores)
    assert "roc_auc" in res
    assert "pr_auc" in res
    assert "ece" in res