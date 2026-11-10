from __future__ import annotations

import logging
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class NLIModel:
    def __init__(self, model_name: str, fallback_name: str | None = None, device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.fallback_name = fallback_name
        
        try:
            self._init_model(self.model_name)
        except Exception as e:
            if self.fallback_name and self.fallback_name != self.model_name:
                logger.warning(
                    "Failed loading '%s' (%s). Trying fallback: '%s'",
                    self.model_name, e, self.fallback_name
                )
                self.model_name = self.fallback_name
                self._init_model(self.fallback_name)
            else:
                raise e

    def _init_model(self, name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.config = AutoConfig.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, config=self.config)
        self.model.to(self.device)
        self.model.eval()

        # Parse id2label to find indices for (entailment, neutral, contradiction)
        id2label = {int(k): str(v).lower() for k, v in self.config.id2label.items()}
        self.entail_idx = self._find_label_idx(id2label, ["entail", "entailment"])
        self.neutral_idx = self._find_label_idx(id2label, ["neutral"])
        self.contra_idx = self._find_label_idx(id2label, ["contra", "contradiction"])

    @staticmethod
    def _find_label_idx(id2label: dict[int, str], matches: List[str]) -> int:
        for idx, lab in id2label.items():
            if any(m in lab for m in matches):
                return idx
        raise ValueError(f"Could not map label {matches} from config id2label: {id2label}")

    @torch.no_grad()
    def predict_proba(
        self, pairs: List[Tuple[str, str]], batch_size: int = 16, temperature: float = 1.0
    ) -> np.ndarray:
        """
        Predict NLI probabilities for premise-hypothesis pairs.
        Returns: np.ndarray [n, 3] ordered as (entailment, neutral, contradiction).
        Truncates only the premise (truncation='only_first').
        """
        if not pairs:
            return np.empty((0, 3), dtype=np.float32)

        all_probs: List[np.ndarray] = []
        max_length = getattr(self.tokenizer, "model_max_length", 512)
        if max_length is None or max_length > 2048:
            max_length = 512

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            premises = [p[0] for p in batch_pairs]
            hypotheses = [p[1] for p in batch_pairs]

            encoded = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation="only_first",
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            logits = self.model(**encoded).logits

            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # Align columns: (entailment, neutral, contradiction)
            ordered_batch = np.column_stack(
                (
                    probs[:, self.entail_idx],
                    probs[:, self.neutral_idx],
                    probs[:, self.contra_idx],
                )
            )
            all_probs.append(ordered_batch)

        return np.vstack(all_probs).astype(np.float32)