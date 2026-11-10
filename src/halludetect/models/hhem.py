from __future__ import annotations

import logging
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class HHEMModel:
    """Wrapper for Vectara's Hallucination Evaluation Model (HHEM-2.1-Open)."""

    def __init__(self, model_name: str = "vectara/hallucination_evaluation_model", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_score(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        """
        Scores (context, answer) pairs.
        Returns a 1D array of factual consistency scores in [0, 1].
        """
        if not pairs:
            return np.empty((0,), dtype=np.float32)

        scores: List[float] = []
        max_len = getattr(self.tokenizer, "model_max_length", 512)
        if max_len is None or max_len > 2048:
            max_len = 512

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            premises = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits

            # HHEM-2.1 produces 1 or 2 output logits representing factual consistency
            if logits.shape[-1] == 1:
                batch_scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            elif logits.shape[-1] == 2:
                batch_scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            else:
                batch_scores = torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()

            scores.extend(batch_scores.tolist())

        return np.array(scores, dtype=np.float32)