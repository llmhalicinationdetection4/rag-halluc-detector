from __future__ import annotations

import time
from typing import Any
from halludetect.data.schema import ClaimResult, VerificationResult
from halludetect.models.hhem import HHEMModel
from halludetect.models.nli import NLIModel


class WholeContextVerifier:
    def __init__(self, nli_model: NLIModel, tau: float = 0.5):
        self.nli_model = nli_model
        self.tau = tau

    def verify(self, context: str, answer: str) -> VerificationResult:
        t0 = time.perf_counter()
        probs = self.nli_model.predict_proba([(context, answer)])[0]
        p_entail, p_neutral, p_contra = float(probs[0]), float(probs[1]), float(probs[2])

        argmax_idx = int(probs.argmax())
        if argmax_idx == 0:
            label = "Supported"
        elif argmax_idx == 1:
            label = "Unsupported"
        else:
            label = "Hallucinated"

        # Hallucinated positive class if score (p_entail) < tau
        is_hallucinated = p_entail < self.tau
        latency_ms = (time.perf_counter() - t0) * 1000.0

        claim_res = ClaimResult(
            claim=answer,
            best_chunks=[context],
            p_entail=p_entail,
            p_neutral=p_neutral,
            p_contra=p_contra,
            verdict=label,
        )

        return VerificationResult(
            score=p_entail,
            label=label,
            hallucinated=is_hallucinated,
            claims=[claim_res],
            latency_ms=latency_ms,
        )


class HHEMVerifier:
    def __init__(self, hhem_model: HHEMModel, tau: float = 0.5):
        self.hhem_model = hhem_model
        self.tau = tau

    def verify(self, context: str, answer: str) -> VerificationResult:
        t0 = time.perf_counter()
        score = float(self.hhem_model.predict_score([(context, answer)])[0])

        is_hallucinated = score < self.tau
        label = "Supported" if not is_hallucinated else "Hallucinated"
        latency_ms = (time.perf_counter() - t0) * 1000.0

        claim_res = ClaimResult(
            claim=answer,
            best_chunks=[context],
            p_entail=score,
            p_neutral=0.0,
            p_contra=1.0 - score,
            verdict=label,
        )

        return VerificationResult(
            score=score,
            label=label,
            hallucinated=is_hallucinated,
            claims=[claim_res],
            latency_ms=latency_ms,
        )