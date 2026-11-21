"""Threshold tuning and uncertainty estimates shared by the evaluation scripts.

All scores here are HALLUCINATION scores: higher means more likely hallucinated, and a response is flagged when
score >= tau. Verifiers that output a consistency score (HHEM, P(entailment)) are converted with 1 - score.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def f1_at(y_true: np.ndarray, flagged: np.ndarray) -> tuple[float, float, float]:
    """(precision, recall, f1) of the flagged set."""
    y = np.asarray(y_true).astype(bool)
    f = np.asarray(flagged).astype(bool)
    tp = float((y & f).sum())
    fp = float((~y & f).sum())
    fn = float((y & ~f).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def flag_all_f1(y_true: np.ndarray) -> float:
    """F1 you would get by flagging every response. A detector must beat this to be saying anything."""
    return f1_at(y_true, np.ones(len(y_true), dtype=bool))[2]


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray, max_candidates: int = 400) -> tuple[float, float]:
    """The tau maximising F1 on the given data. Candidates are quantiles of the observed scores, because NLI
    probabilities pile up very close to 0 and 1 and a regular 0.01 grid would miss the useful values."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    candidates = np.unique(np.quantile(s, np.linspace(0.0, 1.0, max_candidates)))
    best_tau, best_f1 = float(candidates[0]), -1.0
    for tau in candidates:
        f1 = f1_at(y, s >= tau)[2]
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    return best_tau, best_f1


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def auc_with_ci(y_true, scores, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """ROC-AUC with a 95% percentile bootstrap interval."""
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            values.append(roc_auc_score(y[i], s[i]))
    lo, hi = (np.percentile(values, [2.5, 97.5]) if values else (float("nan"), float("nan")))
    return safe_auc(y, s), float(lo), float(hi)


def paired_auc_diff_ci(y_true, scores_a, scores_b, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """AUC(a) - AUC(b) on the SAME responses, with a paired bootstrap interval (the same resample for both)."""
    y, a, b = np.asarray(y_true), np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (float("nan"), float("nan")))
    return safe_auc(y, a) - safe_auc(y, b), float(lo), float(hi)
