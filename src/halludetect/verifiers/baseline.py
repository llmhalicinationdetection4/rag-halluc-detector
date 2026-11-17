"""Phase 1 baseline: whole-context NLI, plus the HHEM wrapper behind the same interface."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from halludetect.data.schema import VerificationResult

NLI_LABELS = ("Supported", "Unsupported", "Hallucinated")  # entailment, neutral, contradiction


class BaseVerifier(ABC):
    @abstractmethod
    def verify_many(self, pairs: list[tuple[str, str]]) -> list[VerificationResult]:
        """pairs = [(context, answer), ...]"""

    def verify(self, context: str, answer: str) -> VerificationResult:
        t0 = time.perf_counter()
        result = self.verify_many([(context, answer)])[0]
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result


class WholeContextVerifier(BaseVerifier):
    """premise = full context, hypothesis = full answer. Score = P(entailment)."""

    def __init__(self, nli, tau: float = 0.5):
        self.nli = nli
        self.tau = tau

    def verify_many(self, pairs):
        t0 = time.perf_counter()
        probs = self.nli.predict_proba(pairs)
        per_item_ms = (time.perf_counter() - t0) * 1000.0 / max(len(pairs), 1)
        results = []
        for p in probs:
            score = float(p[0])
            label = NLI_LABELS[int(np.argmax(p))]
            results.append(
                VerificationResult(score=score, label=label, hallucinated=score < self.tau, latency_ms=per_item_ms)
            )
        return results


class HHEMVerifier(BaseVerifier):
    """HHEM is binary, so the label is Supported when score >= tau, otherwise Hallucinated."""

    def __init__(self, hhem, tau: float = 0.5):
        self.hhem = hhem
        self.tau = tau

    def verify_many(self, pairs):
        t0 = time.perf_counter()
        scores = self.hhem.predict_consistency(pairs)
        per_item_ms = (time.perf_counter() - t0) * 1000.0 / max(len(pairs), 1)
        return [
            VerificationResult(
                score=float(s),
                label="Supported" if s >= self.tau else "Hallucinated",
                hallucinated=bool(s < self.tau),
                latency_ms=per_item_ms,
            )
            for s in scores
        ]


def build_verifier(name: str, cfg, nli=None, hhem=None, device: str = "cpu") -> BaseVerifier:
    """Create a verifier by name. Models can be injected (used by tests)."""
    if name == "baseline":
        if nli is None:
            from halludetect.models.nli import NLIModel

            nli = NLIModel(
                cfg.models.nli, device=device, max_length=cfg.models.max_length,
                batch_size=cfg.models.batch_size, fallback_name=cfg.models.nli_fallback,
            )
        return WholeContextVerifier(nli, tau=cfg.baseline.tau)
    if name == "hhem":
        if hhem is None:
            from halludetect.models.hhem import HHEMModel

            hhem = HHEMModel(cfg.models.hhem, device=device, batch_size=cfg.models.batch_size)
        return HHEMVerifier(hhem, tau=cfg.baseline.tau)
    raise ValueError(f"Unknown verifier '{name}'. Available now: baseline, hhem (claim-level arrives in Phase 3).")
