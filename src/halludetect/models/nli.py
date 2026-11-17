"""NLI cross-encoder wrapper. torch/transformers are imported lazily so tests can run without them."""
from __future__ import annotations

import numpy as np


class NLIModel:
    """Returns probabilities ordered (entailment, neutral, contradiction).

    The label order is read from model.config.id2label and never hardcoded.
    The premise (context) is truncated; the hypothesis (answer/claim) is kept whole when possible.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
        fallback_name: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        names = [model_name] + ([fallback_name] if fallback_name else [])
        last_err: Exception | None = None
        for name in names:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(name)
                self.model_name = name
                break
            except Exception as err:  # noqa: BLE001 - we report the last error below
                last_err = err
        else:
            raise RuntimeError(f"Could not load any NLI model from {names}: {last_err}") from last_err

        self.model.to(device).eval()
        self._idx = self.label_indices(self.model.config.id2label)

    @staticmethod
    def label_indices(id2label: dict) -> list[int]:
        """Map model label ids to [entailment, neutral, contradiction] positions."""
        found: dict[str, int] = {}
        for idx, name in id2label.items():
            low = str(name).lower()
            if "entail" in low:
                found["entailment"] = int(idx)
            elif "neutral" in low:
                found["neutral"] = int(idx)
            elif "contradict" in low:
                found["contradiction"] = int(idx)
        missing = {"entailment", "neutral", "contradiction"} - set(found)
        if missing:
            raise ValueError(f"NLI model labels {id2label} are missing {sorted(missing)}")
        return [found["entailment"], found["neutral"], found["contradiction"]]

    def _encode(self, batch: list[tuple[str, str]]):
        premises = [p for p, _ in batch]
        hypotheses = [h for _, h in batch]
        try:
            return self.tokenizer(
                premises, hypotheses, truncation="only_first", max_length=self.max_length,
                padding=True, return_tensors="pt",
            )
        except Exception:  # the tokenizer raises a plain Exception when the hypothesis alone is too long
            # Hypothesis alone is longer than max_length: fall back to truncating both.
            return self.tokenizer(
                premises, hypotheses, truncation="longest_first", max_length=self.max_length,
                padding=True, return_tensors="pt",
            )

    def predict_proba(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """pairs = [(premise, hypothesis), ...] -> array [n, 3]."""
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        torch = self._torch
        chunks = []
        for i in range(0, len(pairs), self.batch_size):
            enc = self._encode(pairs[i : i + self.batch_size])
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            chunks.append(probs[:, self._idx])
        return np.concatenate(chunks, axis=0)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
