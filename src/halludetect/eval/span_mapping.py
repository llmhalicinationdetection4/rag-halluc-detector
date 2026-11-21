"""Claim-level ground truth from RAGTruth's span annotations.

RAGTruth marks hallucinated spans as character offsets in the ORIGINAL response text. A claim (sentence) is labelled
hallucinated when it overlaps at least one span. Claims therefore have to carry their character offsets in the original
answer, which `halludetect.verifiers.claim_level.split_claims` provides.
"""
from __future__ import annotations


def label_claims(claims: list[dict], spans: list[list[int]]) -> list[int]:
    """1 if the claim [start, end) overlaps any [start, end) span, else 0."""
    out = []
    for claim in claims:
        c0, c1 = claim["start"], claim["end"]
        out.append(int(any(c0 < s1 and s0 < c1 for s0, s1 in spans)))
    return out
