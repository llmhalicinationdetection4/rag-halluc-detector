from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Example:
    id: str
    context: str
    answer: str
    label_hallucinated: bool
    task_type: str = "QA"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimResult:
    claim: str
    best_chunks: List[str]
    p_entail: float
    p_neutral: float
    p_contra: float
    verdict: str  # Supported | Unsupported | Hallucinated


@dataclass
class VerificationResult:
    score: float
    label: str  # Supported | Unsupported | Hallucinated | Uncertain
    hallucinated: bool
    claims: List[ClaimResult]
    latency_ms: float