import numpy as np
import pytest
from halludetect.data.schema import VerificationResult


class StubNLIModel:
    """Lightweight stub model for offline CI / 4GB RAM tests without downloading DeBERTa."""

    def __init__(self, mode: str = "entail"):
        self.mode = mode

    def predict_proba(self, pairs: list[tuple[str, str]], **kwargs) -> np.ndarray:
        probs = []
        for premise, hypothesis in pairs:
            if "Mars" in hypothesis or "false" in hypothesis.lower():
                probs.append([0.05, 0.10, 0.85])  # Contradiction
            elif self.mode == "contra":
                probs.append([0.05, 0.10, 0.85])
            else:
                probs.append([0.90, 0.08, 0.02])  # Entailment
        return np.array(probs, dtype=np.float32)


class StubRetriever:
    def retrieve_top_k(self, claim: str, chunks: list[str], top_k: int = 3) -> list[str]:
        return chunks[:top_k]


def test_stub_nli_behavior():
    stub = StubNLIModel()
    pairs = [("Context text", "Mars landed"), ("Context text", "True text")]
    res = stub.predict_proba(pairs)
    assert res.shape == (2, 3)
    assert res[0, 2] > 0.8
    assert res[1, 0] > 0.8