"""Vectara HHEM wrapper (baseline to compare against).

Follow the model card of the checkpoint you use: HHEM-2.1-Open loads with trust_remote_code=True and
exposes model.predict([(premise, hypothesis), ...]) returning scores in [0, 1] where 1 = consistent.

HHEM does not cut long inputs, and GPU memory grows roughly with batch_size * length**2. So this wrapper
sorts the pairs by length, shrinks the batch for long ones, and on a GPU out-of-memory error it splits
the batch (and, as a last resort, shortens one very long context). It never silently moves to the CPU.
"""
from __future__ import annotations

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional

    def tqdm(iterable, **kwargs):
        return iterable


# batch_size * length**2 must stay under this. It equals "16 sequences of about 1024 tokens",
# which fits comfortably on a 16 GB GPU.
MEMORY_BUDGET = 16 * 1024 * 1024
CHARS_PER_TOKEN = 3.0  # real text is nearer 4 chars/token, so this over-estimates length (the safe side)
MAX_PREMISE_HALVINGS = 4


def estimate_tokens(pair: tuple[str, str]) -> int:
    return int((len(pair[0]) + len(pair[1])) / CHARS_PER_TOKEN) + 64


def plan_batches(lengths: list[int], max_batch: int, budget: int = MEMORY_BUDGET) -> list[list[int]]:
    """Group indices into batches, longest first, so that len(batch) * longest**2 <= budget.

    A single sequence that is over budget still gets a batch of its own.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        if current:
            longest = lengths[current[0]]  # sorted longest first, so the first item is the longest
            size = len(current) + 1
            if size > max_batch or size * longest**2 > budget:
                batches.append(current)
                current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def _is_oom(err: BaseException) -> bool:
    return "out of memory" in str(err).lower()


class HHEMModel:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification

        self._torch = torch
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        self.n_truncated = 0  # contexts that had to be shortened after repeated out-of-memory errors
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device).eval()
        try:
            actual = next(self.model.parameters()).device
        except StopIteration:
            actual = device
        print(f"[hhem] requested device = {device}, weights are on = {actual}")

    def _free_gpu_memory(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _predict_batch(self, batch: list[tuple[str, str]]) -> list[float]:
        with self._torch.inference_mode():
            out = self.model.predict(batch)
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        return np.asarray(out, dtype=float).ravel().tolist()

    def _score(self, batch: list[tuple[str, str]]) -> list[float]:
        try:
            return self._predict_batch(batch)
        except RuntimeError as err:
            if not _is_oom(err):
                raise  # a real bug: fail loudly instead of hiding it behind a CPU fallback
            self._free_gpu_memory()
            if len(batch) > 1:
                mid = len(batch) // 2
                return self._score(batch[:mid]) + self._score(batch[mid:])
            return self._score_single_shortened(batch[0], err)

    def _score_single_shortened(self, pair: tuple[str, str], first_err: RuntimeError) -> list[float]:
        premise, hypothesis = pair
        last_err = first_err
        self.n_truncated += 1
        for _ in range(MAX_PREMISE_HALVINGS):
            premise = premise[: len(premise) // 2]
            try:
                return self._predict_batch([(premise, hypothesis)])
            except RuntimeError as err:
                if not _is_oom(err):
                    raise
                last_err = err
                self._free_gpu_memory()
        raise last_err

    def predict_consistency(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """pairs = [(premise, hypothesis), ...] -> consistency scores in [0, 1], in the original order."""
        scores = np.zeros(len(pairs), dtype=float)
        if not pairs:
            return scores
        lengths = [estimate_tokens(p) for p in pairs]
        batches = plan_batches(lengths, self.batch_size)
        for idx in tqdm(batches, desc="HHEM", mininterval=20):
            scores[idx] = self._score([pairs[i] for i in idx])
        if self.n_truncated:
            print(f"[hhem] {self.n_truncated} very long context(s) were shortened to fit in GPU memory")
        return scores
