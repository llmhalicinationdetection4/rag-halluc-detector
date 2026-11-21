"""Phase 4: turn verifier scores into probabilities, measure calibration, and decide when to abstain.

Everything here is plain numpy, so a saved calibrator can be used by the dashboard without scikit-learn.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticCalibrator:
    """P(hallucinated | features) = sigmoid(features @ coef + intercept), fitted on the TRAIN split only."""

    def __init__(self, feature_names: list[str], coef: list[float], intercept: float):
        self.feature_names = list(feature_names)
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, feature_names: list[str], C: float = 1.0) -> "LogisticCalibrator":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=C, max_iter=2000)  # no class_weight: balanced weights would distort probabilities
        model.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
        return cls(feature_names, model.coef_[0], float(model.intercept_[0]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(X, dtype=float) @ self.coef + self.intercept)

    def to_dict(self) -> dict:
        return {"feature_names": self.feature_names, "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticCalibrator":
        return cls(d["feature_names"], d["coef"], d["intercept"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def brier_score(y_true, p) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true, p, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """ECE = sum over bins of (n_bin / N) * |observed hallucination rate - mean predicted probability|."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n:
            conf, obs = float(p[mask].mean()), float(y[mask].mean())
            ece += n / len(y) * abs(obs - conf)
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "n": n, "mean_probability": conf, "observed_rate": obs})
        else:
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "n": 0, "mean_probability": np.nan, "observed_rate": np.nan})
    return float(ece), pd.DataFrame(rows)


def coverage_accuracy_curve(y_true, p, margins=None) -> pd.DataFrame:
    """Abstain ('uncertain') when |p - 0.5| < margin. For each margin: the share of responses still answered
    (coverage) and the accuracy of the hallucinated/faithful decision on those."""
    y, p = np.asarray(y_true).astype(int), np.asarray(p, dtype=float)
    margins = np.linspace(0.0, 0.45, 46) if margins is None else margins
    rows = []
    for m in margins:
        keep = np.abs(p - 0.5) >= m
        n = int(keep.sum())
        acc = float(((p[keep] >= 0.5).astype(int) == y[keep]).mean()) if n else float("nan")
        rows.append({"margin": float(m), "coverage": n / len(y), "n_answered": n, "accuracy": acc})
    return pd.DataFrame(rows)


def choose_margin(curve: pd.DataFrame, target_accuracy: float = 0.80, min_coverage: float = 0.30):
    """Smallest abstention margin whose answered responses reach the target accuracy while keeping enough
    coverage. Returns None when no margin does (then the dashboard should not abstain)."""
    ok = curve[(curve["accuracy"] >= target_accuracy) & (curve["coverage"] >= min_coverage)]
    return float(ok["margin"].min()) if len(ok) else None
