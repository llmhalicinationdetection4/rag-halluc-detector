import numpy as np

from halludetect.eval.metrics import best_threshold, classification_metrics


def test_perfect_separation():
    y = [0, 0, 0, 1, 1, 1]
    scores = [0.9, 0.8, 0.95, 0.1, 0.2, 0.05]
    tau, f1 = best_threshold(y, scores)
    assert f1 == 1.0 and 0.2 < tau < 0.8
    m = classification_metrics(y, scores, tau)
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0 and m["accuracy"] == 1.0


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = np.round(rng.random(60), 2)               # rounded so there are ties
    tau, f1 = best_threshold(y, s)
    brute = max(classification_metrics(y, s, t)["f1"] for t in np.unique(s) + 1e-6)
    assert abs(classification_metrics(y, s, tau)["f1"] - f1) < 1e-9
    assert f1 >= brute - 1e-9


def test_single_class_is_handled():
    tau, f1 = best_threshold([0, 0, 0], [0.9, 0.8, 0.7])
    assert tau == 0.5 and np.isnan(f1)
    assert np.isnan(classification_metrics([0, 0], [0.9, 0.8], 0.5)["roc_auc"])


def test_counts_add_up():
    m = classification_metrics([1, 0, 1, 0], [0.2, 0.9, 0.7, 0.1], 0.5)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 4
