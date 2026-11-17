"""Metrics. Positive class = hallucinated. A response is flagged when score < tau."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def classification_metrics(y_true, scores, tau: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s < tau).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    both = len(set(y.tolist())) == 2
    return {
        "n": int(len(y)),
        "n_hallucinated": int(y.sum()),
        "tau": float(tau),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, 1.0 - s)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, 1.0 - s)) if both else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def best_threshold(y_true, scores) -> tuple[float, float]:
    """Pick tau maximising F1 for the rule 'flag if score < tau'. Returns (tau, f1)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n, pos = len(y), int(y.sum())
    if n == 0 or pos == 0 or pos == n:
        return 0.5, float("nan")
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, n + 1)
    f1 = 2.0 * tp / (k + pos)          # F1 = 2TP / (predicted positives + actual positives)
    valid = np.ones(n, dtype=bool)
    valid[:-1] = s_sorted[:-1] != s_sorted[1:]   # only cut between distinct scores
    i = int(np.argmax(np.where(valid, f1, -1.0)))
    tau = float((s_sorted[i] + s_sorted[i + 1]) / 2.0) if i < n - 1 else float(s_sorted[i]) + 1e-6
    return tau, float(f1[i])
