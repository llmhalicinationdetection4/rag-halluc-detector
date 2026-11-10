from __future__ import annotations

import re
from typing import Any, Dict, List
import numpy as np
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


def get_token_length(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def classify_length_bucket(token_len: int) -> str:
    if token_len < 256:
        return "<256"
    if token_len <= 512:
        return "256-512"
    if token_len <= 1024:
        return "512-1024"
    return ">1024"


def check_numbers_or_dates(text: str) -> bool:
    num_pattern = re.compile(r"\b\d+(?:[\.,]\d+)?\b|\b(18|19|20)\d{2}\b")
    return bool(num_pattern.search(text))


def compute_rouge_l_terciles(contexts: List[str], answers: List[str]) -> List[str]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(c, a)["rougeL"].fmeasure for c, a in zip(contexts, answers)]
    
    if len(scores) < 3:
        return ["mid" for _ in scores]

    t1, t2 = np.percentile(scores, 33.33), np.percentile(scores, 66.66)
    labels = []
    for s in scores:
        if s <= t1:
            labels.append("low")
        elif s <= t2:
            labels.append("mid")
        else:
            labels.append("high")
    return labels


def tag_examples_with_buckets(
    examples: List[Dict[str, Any]], tokenizer: Any = None
) -> List[Dict[str, Any]]:
    contexts = [e["context"] for e in examples]
    answers = [e["answer"] for e in examples]
    rouge_terciles = compute_rouge_l_terciles(contexts, answers)

    tagged: List[Dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        item = dict(ex)
        t_len = get_token_length(item["context"], tokenizer)
        item["context_length_bucket"] = classify_length_bucket(t_len)
        item["has_numbers_or_dates"] = check_numbers_or_dates(item["answer"])
        item["paraphrase_level"] = rouge_terciles[idx]
        item["multi_passage"] = bool(item.get("meta", {}).get("multi_passage", False))
        tagged.append(item)

    return tagged