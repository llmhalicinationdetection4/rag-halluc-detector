"""Deterministic stand-in for the NLI model so tests run on a 4 GB laptop with no downloads."""
from __future__ import annotations

import re

import numpy as np

NEGATIONS = {"not", "no", "never", "cannot", "n't"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


class StubNLIModel:
    model_name = "stub-nli"

    def predict_proba(self, pairs):
        out = []
        for premise, hypothesis in pairs:
            p, h = _words(premise), _words(hypothesis)
            if (h & NEGATIONS) - p:
                out.append([0.05, 0.15, 0.80])      # contradiction
            elif h <= p:
                out.append([0.90, 0.08, 0.02])      # entailment
            else:
                out.append([0.10, 0.80, 0.10])      # neutral
        return np.array(out, dtype=np.float32).reshape(-1, 3)

    def count_tokens(self, text):
        return len(text.split())


class StubHHEM:
    """Scores 1.0 if every answer word appears in the context, else 0.2."""

    def predict_consistency(self, pairs):
        return np.array([1.0 if _words(h) <= _words(p) else 0.2 for p, h in pairs])
