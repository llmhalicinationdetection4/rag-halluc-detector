import numpy as np

from halludetect.calibration import (
    LogisticCalibrator,
    brier_score,
    choose_margin,
    coverage_accuracy_curve,
    expected_calibration_error,
)


def test_ece_is_small_for_calibrated_and_large_for_overconfident_probabilities():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p)[0] < 0.02
    y2 = (rng.random(5000) < 0.5).astype(int)
    ece, table = expected_calibration_error(y2, np.full(5000, 0.9))
    assert abs(ece - 0.4) < 0.03 and table["n"].sum() == 5000       # says 90%, is right 50% of the time


def test_ece_uses_the_observed_rate_not_accuracy():
    # always predicts 0.1 and the event happens 10% of the time: perfectly calibrated although 'accuracy' is 90%
    y = np.array([1] * 10 + [0] * 90)
    assert expected_calibration_error(y, np.full(100, 0.1))[0] < 1e-9
    assert abs(brier_score([1, 0], [1.0, 0.0])) < 1e-12


def test_calibrator_learns_a_signal_and_roundtrips_through_json(tmp_path):
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    X = np.column_stack([y * 0.5 + rng.random(2000) * 0.5, rng.random(2000)])
    cal = LogisticCalibrator.fit(X, y, ["h_a", "h_b"])
    p = cal.predict_proba(X)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.2
    assert abs(p.mean() - y.mean()) < 0.02                           # no class weights: mean probability matches the rate
    cal.save(tmp_path / "c.json")
    again = LogisticCalibrator.load(tmp_path / "c.json")
    assert again.feature_names == ["h_a", "h_b"] and np.allclose(again.predict_proba(X), p)


def test_abstention_curve_and_margin():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 4000)
    p = np.clip(y * 0.5 + 0.25 + rng.normal(0, 0.18, 4000), 0.01, 0.99)
    curve = coverage_accuracy_curve(y, p)
    assert curve["coverage"].iloc[0] == 1.0 and curve["coverage"].is_monotonic_decreasing
    assert curve["accuracy"].iloc[10] >= curve["accuracy"].iloc[0]   # answering only confident cases is more accurate
    assert choose_margin(curve, target_accuracy=0.85) == 0.0          # already accurate enough: no need to abstain
    margin = choose_margin(curve, target_accuracy=0.95)
    assert margin is not None and margin > 0                          # a stricter target needs an uncertain band
    assert choose_margin(curve, target_accuracy=1.01) is None
