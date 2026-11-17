"""Shared data structures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

LABELS = ("Supported", "Unsupported", "Hallucinated", "Uncertain")


@dataclass
class Example:
    id: str
    context: str
    answer: str
    label_hallucinated: bool
    task_type: str = "QA"
    meta: dict = field(default_factory=dict)


@dataclass
class ClaimResult:
    claim: str
    best_chunks: list[str]
    p_entail: float
    p_neutral: float
    p_contra: float
    verdict: str


@dataclass
class VerificationResult:
    score: float                     # P(entailment) or supported-claim ratio
    label: str                       # Supported | Unsupported | Hallucinated | Uncertain
    hallucinated: bool
    claims: list[ClaimResult] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
