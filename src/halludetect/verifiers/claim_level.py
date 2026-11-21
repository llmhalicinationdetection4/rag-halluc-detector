"""Claim-level verifier (Phase 3).

For each sentence ("claim") of the answer:
  1. the context is cut into overlapping windows (chunks) of about `chunk_size` tokens,
  2. the claim is matched to its `top_k` most similar chunks with a sentence-embedding model (MiniLM),
  3. the same pre-trained NLI model used in Phase 1 scores (chunk, claim) pairs,
  4. the claim keeps its best P(entailment) and its highest P(contradiction) over those chunks.

The claim scores are then combined into one answer-level score by one of the AGGREGATIONS. Which aggregation to use
and its threshold are chosen on the TRAIN split by `halludetect.eval.run_phase3`; nothing here is hard-coded.
"""
from __future__ import annotations

import copy
import json
import re

import numpy as np

AGGREGATIONS = ("max_contra", "min_entail", "mean_entail", "share_unsupported")
SUPPORT_THRESHOLD = 0.5  # a claim counts as "unsupported" when its best P(entailment) is below this

_LEAD_IN_RE = re.compile(r"^\s*(?:here\b|sure\b|certainly\b|of course\b)[^:\n]{0,150}:\s*", re.I)
_SENTENCE_RE = re.compile(
    r"(?:[^.!?\n]|(?<=\d)\.(?=\d))+(?:(?<!\d)[.!?]+[\"')\]]*|(?=\n)|$)"
)


# ----------------------------------------------------------------------------- text helpers (no torch needed)
def load_sentence_splitter():
    """spaCy's small English model if it is installed, else None (a regex splitter is used instead)."""
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except Exception:  # noqa: BLE001  (not installed, no model, version clash ...)
        return None


def split_claims(answer: str, nlp=None, min_chars: int = 6) -> list[dict]:
    """Sentences of the answer as [{"text", "start", "end"}]. The offsets point into the ORIGINAL answer, so they can
    be compared with RAGTruth's span annotations. Lead-in lines such as 'Here is a summary:' are skipped."""
    if nlp is not None:
        spans = [(sent.start_char, sent.end_char) for sent in nlp(answer).sents]
    else:
        spans = [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(answer)]
    claims = []
    for start, end in spans:
        segment = answer[start:end]
        start += len(segment) - len(segment.lstrip())
        end -= len(segment) - len(segment.rstrip())
        text = answer[start:end]
        lead = _LEAD_IN_RE.match(text)
        if lead and lead.end() < len(text):
            start += lead.end()  # lead-in and content on the same line: keep the content
            text = answer[start:end]
        elif lead or text.endswith(":"):
            continue  # a pure lead-in line
        if len(text) >= min_chars:
            claims.append({"text": text, "start": start, "end": end})
    return claims


def linearize_context(context: str) -> str:
    """Data-to-text contexts are JSON. Turn them into plain 'path: value.' sentences an NLI model can read."""
    text = context.strip()
    if not text.startswith(("{", "[")):
        return context
    try:
        data = json.loads(text)
    except ValueError:
        return context
    lines: list[str] = []

    def walk(obj, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                name = str(key).replace("_", " ")
                walk(value, f"{path} {name}".strip())
        elif isinstance(obj, list):
            for i, value in enumerate(obj, 1):
                walk(value, f"{path} {i}".strip())
        elif obj is not None and obj != "":
            lines.append(f"{path}: {obj}.")

    walk(data, "")
    return " ".join(lines) if lines else context


def chunk_context(context: str, tokenizer=None, chunk_size: int = 256, stride: int = 128) -> list[str]:
    """Overlapping windows of `chunk_size` tokens whose starts are `stride` tokens apart."""
    if tokenizer is not None:
        tokens, join = tokenizer.tokenize(context), tokenizer.convert_tokens_to_string
    else:
        tokens, join = context.split(), " ".join
    if not tokens:
        return []
    chunks, start = [], 0
    while True:
        chunks.append(join(tokens[start:start + chunk_size]).strip())
        if start + chunk_size >= len(tokens):
            break
        start += stride
    return chunks


# ----------------------------------------------------------------------------- answer-level scores
def aggregate_scores(claims: list[dict]) -> dict:
    """Four candidate answer-level scores computed from the claim scores."""
    if not claims:
        return {"max_contra": 0.0, "min_entail": 1.0, "mean_entail": 1.0, "share_unsupported": 0.0}
    entail = np.array([c["p_entail"] for c in claims], dtype=float)
    contra = np.array([c["p_contra"] for c in claims], dtype=float)
    return {
        "max_contra": float(contra.max()),
        "min_entail": float(entail.min()),
        "mean_entail": float(entail.mean()),
        "share_unsupported": float((entail < SUPPORT_THRESHOLD).mean()),
    }


def hallucination_score(aggs: dict, name: str) -> float:
    """Orient every aggregation the same way: higher = more likely hallucinated, flagged when score >= tau."""
    if name in ("max_contra", "share_unsupported"):
        return float(aggs[name])
    if name in ("min_entail", "mean_entail"):
        return 1.0 - float(aggs[name])
    raise ValueError(f"unknown aggregation {name!r}; choose from {AGGREGATIONS}")


# ----------------------------------------------------------------------------- the verifier
class ClaimLevelVerifier:
    def __init__(self, nli=None, embedder=None, nlp="auto", device: str = "auto", chunk_size: int = 256,
                 stride: int = 128, top_k: int = 3, use_retrieval: bool = True, split_answer: bool = True,
                 embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = device
        self.chunk_size, self.stride, self.top_k = chunk_size, stride, top_k
        self.use_retrieval, self.split_answer = use_retrieval, split_answer
        self.embedder_name = embedder_name
        self._nli, self._embedder, self._nlp = nli, embedder, nlp

    # models are created on first use, so importing this module never needs torch
    @property
    def nli(self):
        if self._nli is None:
            from halludetect.config import get_device, load_config
            from halludetect.models.nli import NLIModel

            cfg = load_config()
            m = cfg.models
            self._nli = NLIModel(m.nli, device=get_device(self.device), max_length=m.max_length,
                                 batch_size=m.batch_size, fallback_name=m.nli_fallback)
        return self._nli

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            from halludetect.config import get_device

            self._embedder = SentenceTransformer(self.embedder_name, device=get_device(self.device))
        return self._embedder

    @property
    def nlp(self):
        if isinstance(self._nlp, str):  # "auto"
            self._nlp = load_sentence_splitter()
        return self._nlp

    def with_options(self, **options) -> "ClaimLevelVerifier":
        """A copy with other settings (chunk size, top_k, ...) that shares the already loaded models. Used by ablations."""
        _ = self.nli
        if self.use_retrieval or options.get("use_retrieval", True):
            _ = self.embedder
        _ = self.nlp
        clone = copy.copy(self)
        for key, value in options.items():
            if not hasattr(clone, key):
                raise AttributeError(f"unknown option {key!r}")
            setattr(clone, key, value)
        return clone

    def signature(self) -> dict:
        """Everything that changes the scores, so cached scores are never reused for another setting."""
        return {"chunk_size": self.chunk_size, "stride": self.stride, "top_k": self.top_k,
                "use_retrieval": self.use_retrieval, "split_answer": self.split_answer,
                "nli": self._nli_name(), "embedder": self.embedder_name}

    def _nli_name(self) -> str:
        if self._nli is not None:
            return getattr(self._nli, "model_name", "nli")
        try:  # not loaded yet: use the configured model so cached scores are never mixed across NLI models
            from halludetect.config import load_config

            return load_config().models.nli
        except Exception:  # noqa: BLE001
            return "nli"

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                       batch_size=64, show_progress_bar=False)
        return np.asarray(vectors, dtype=float)

    def verify(self, context: str, answer: str, task: str | None = None) -> dict:
        context = linearize_context(context) if (task is None or task.lower() == "data2txt") else context
        answer = answer or ""
        if self.split_answer:
            claims = split_claims(answer, self.nlp)
        else:
            text = answer.strip()
            start = answer.find(text) if text else 0
            claims = [{"text": text, "start": start, "end": start + len(text)}] if text else []
        chunks = chunk_context(context, getattr(self.nli, "tokenizer", None), self.chunk_size, self.stride)
        if not claims:
            return {"claims": [], "aggs": aggregate_scores([]), "n_chunks": len(chunks)}
        if not chunks:  # no context at all: nothing can support any claim
            scored = [{**c, "p_entail": 0.0, "p_neutral": 1.0, "p_contra": 0.0, "score": 0.0, "evidence": ""} for c in claims]
            return {"claims": scored, "aggs": aggregate_scores(scored), "n_chunks": 0}

        if self.use_retrieval and len(chunks) > self.top_k:
            sims = self._embed([c["text"] for c in claims]) @ self._embed(chunks).T  # chunks are embedded once
            picks = [list(np.argsort(-sims[i])[: self.top_k]) for i in range(len(claims))]
        else:
            picks = [list(range(len(chunks))) for _ in claims]

        pairs = [(chunks[j], claims[i]["text"]) for i in range(len(claims)) for j in picks[i]]
        probs = np.asarray(self.nli.predict_proba(pairs), dtype=float)  # columns: entailment, neutral, contradiction

        scored, row = [], 0
        for i, claim in enumerate(claims):
            block = probs[row:row + len(picks[i])]
            row += len(picks[i])
            best = int(np.argmax(block[:, 0]))
            scored.append({**claim,
                           "p_entail": float(block[best, 0]), "p_neutral": float(block[best, 1]),
                           "p_contra": float(block[:, 2].max()), "score": float(block[best, 0]),
                           "evidence": chunks[picks[i][best]]})
        return {"claims": scored, "aggs": aggregate_scores(scored), "n_chunks": len(chunks)}
