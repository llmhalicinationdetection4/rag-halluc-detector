from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ChunkRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def retrieve_top_k(self, claim: str, chunks: List[str], top_k: int = 3) -> List[str]:
        if not chunks:
            return [""]
        if len(chunks) <= top_k:
            return chunks

        claim_emb = self.model.encode([claim], convert_to_numpy=True)
        chunk_embs = self.model.encode(chunks, convert_to_numpy=True)

        sims = cosine_similarity(claim_emb, chunk_embs)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [chunks[idx] for idx in top_indices]