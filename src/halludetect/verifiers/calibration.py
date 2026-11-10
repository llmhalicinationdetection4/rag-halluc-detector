from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize


class TemperatureScaler:
    """Fits post-hoc temperature scaling on validation/train logits."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        logits: shape [N, num_classes]
        labels: 1D array of integer class labels [0, num_classes - 1]
        """
        if len(logits) == 0:
            return 1.0

        def nll_loss(t: np.ndarray) -> float:
            temp = t[0]
            scaled = logits / temp
            exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            loss = -np.mean(log_probs[np.arange(len(labels)), labels])
            return float(loss)

        res = minimize(nll_loss, x0=np.array([1.0]), bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self.temperature

    def scale_probabilities(self, probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(probs, 1e-12, 1.0)
        pseudo_logits = np.log(clipped)
        scaled = pseudo_logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)