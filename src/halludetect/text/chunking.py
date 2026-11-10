from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_tokens: int = 256, stride: int = 128) -> List[str]:
    """
    Splits text into overlapping token windows using whitespace token approximation.
    Guarantees coverage across document boundaries.
    """
    clean_text = text.strip()
    if not clean_text:
        return [""]

    tokens = clean_text.split()
    if len(tokens) <= chunk_tokens:
        return [clean_text]

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_str = " ".join(tokens[start:end])
        chunks.append(chunk_str)
        if end >= len(tokens):
            break
        start += stride

    return chunks