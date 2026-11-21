import numpy as np

from halludetect.eval.threshold import auc_with_ci, best_f1_threshold, f1_at, flag_all_f1, paired_auc_diff_ci


def test_f1_and_flag_all():
    y = np.array([1, 1, 0, 0])
    assert f1_at(y, np.array([1, 0, 1, 0])) == (0.5, 0.5, 0.5)
    assert abs(flag_all_f1(y) - 2 * 0.5 / 1.5) < 1e-9


def test_best_threshold_separates_clean_scores_even_close_to_one():
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.linspace(0.990, 0.992, 50), np.linspace(0.9990, 0.9999, 50)])
    tau, f1 = best_f1_threshold(y, scores)
    assert f1 == 1.0 and 0.992 < tau <= 0.9990


def test_auc_ci_brackets_the_estimate_and_paired_diff_is_zero_for_identical_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y * 0.6 + rng.random(300)
    auc, lo, hi = auc_with_ci(y, s, n_boot=200)
    assert lo <= auc <= hi and auc > 0.7
    diff, dlo, dhi = paired_auc_diff_ci(y, s, s, n_boot=100)
    assert diff == 0.0 and dlo == 0.0 and dhi == 0.0
    better = paired_auc_diff_ci(y, s, rng.random(300), n_boot=200)
    assert better[0] > 0.1 and better[1] > 0
