from __future__ import annotations

import time
import numpy as np

from typing import Any, List, Optional
from halludetect.data.schema import ClaimResult, VerificationResult
from halludetect.models.nli import NLIModel
from halludetect.text.chunking import chunk_text
from halludetect.text.claims import split_into_claims
from halludetect.text.retrieval import ChunkRetriever
from halludetect.verifiers.calibration import TemperatureScaler


class ClaimLevelVerifier:
    def __init__(
        self,
        nli_model: NLIModel,
        retriever: ChunkRetriever,
        chunk_tokens: int = 256,
        stride: int = 128,
        top_k: int = 3,
        tau_support: float = 0.5,
        tau_contra: float = 0.5,
        tau_answer: float = 0.5,
        alpha_abstain: Optional[float] = None,
        scaler: Optional[TemperatureScaler] = None,
    ):
        self.nli_model = nli_model
        self.retriever = retriever
        self.chunk_tokens = chunk_tokens
        self.stride = stride
        self.top_k = top_k
        self.tau_support = tau_support
        self.tau_contra = tau_contra
        self.tau_answer = tau_answer
        self.alpha_abstain = alpha_abstain
        self.scaler = scaler

    def verify(self, context: str, answer: str) -> VerificationResult:
        t0 = time.perf_counter()
        claims = split_into_claims(answer)
        chunks = chunk_text(context, chunk_tokens=self.chunk_tokens, stride=self.stride)

        if not claims:
            return VerificationResult(
                score=1.0,
                label="Supported",
                hallucinated=False,
                claims=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        claim_results: List[ClaimResult] = []
        supported_count = 0
        has_hallucinated_claim = False

        for claim in claims:
            top_chunks = self.retriever.retrieve_top_k(claim, chunks, top_k=self.top_k)
            pairs = [(chunk, claim) for chunk in top_chunks]
            probs_matrix = self.nli_model.predict_proba(pairs)

            if self.scaler is not None:
                probs_matrix = self.scaler.scale_probabilities(probs_matrix)

            # claim_support = max P(entail), claim_contra = max P(contra)
            entail_scores = probs_matrix[:, 0]
            contra_scores = probs_matrix[:, 2]
            neutral_scores = probs_matrix[:, 1]

            claim_support = float(np.max(entail_scores))
            claim_contra = float(np.max(contra_scores))
            p_neutral_mean = float(np.mean(neutral_scores))

            # Verdict rule
            if claim_contra >= self.tau_contra and claim_contra > claim_support:
                verdict = "Hallucinated"
                has_hallucinated_claim = True
            elif claim_support >= self.tau_support:
                verdict = "Supported"
                supported_count += 1
            else:
                verdict = "Unsupported"

            claim_results.append(
                ClaimResult(
                    claim=claim,
                    best_chunks=top_chunks,
                    p_entail=claim_support,
                    p_neutral=p_neutral_mean,
                    p_contra=claim_contra,
                    verdict=verdict,
                )
            )

        # Answer score = supported claims / total claims
        score = supported_count / float(len(claims))

        # Hallucinated if any claim is Hallucinated or score < tau_answer
        is_hallucinated = has_hallucinated_claim or (score < self.tau_answer)

        if is_hallucinated:
            label = "Hallucinated"
        elif score == 1.0:
            label = "Supported"
        else:
            label = "Unsupported"

        # Abstention check when confidence is below alpha
        if self.alpha_abstain is not None:
            max_conf = max(score, 1.0 - score)
            if max_conf < self.alpha_abstain:
                label = "Uncertain"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return VerificationResult(
            score=score,
            label=label,
            hallucinated=is_hallucinated,
            claims=claim_results,
            latency_ms=latency_ms,
        )