"""Small stand-ins and a synthetic RAGTruth folder, so the Phase 3-4 code can be tested without models or downloads."""
import json
import re
import zlib

import numpy as np

TASKS = ["QA", "Summary", "Data2txt"]
CITIES = ["Paris", "Berlin", "Madrid", "Rome", "Vienna", "Prague", "Lisbon", "Oslo"]
FOODS = ["pasta", "sushi", "tacos", "curry", "pizza", "salad", "soup", "bread"]


class FakeEmbedder:
    """Bag-of-words vectors: sentences that share words are similar."""

    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), 64))
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[i, zlib.crc32(word.encode()) % 64] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class FakeNLI:
    """P(entail) is high when every claim word occurs in the chunk; 'not' + otherwise-supported words = contradiction."""

    tokenizer = None
    model_name = "fake-nli"

    def predict_proba(self, pairs):
        rows = []
        for chunk, claim in pairs:
            chunk_words = set(re.findall(r"\w+", chunk.lower()))
            claim_words = re.findall(r"\w+", claim.lower())
            rest = [w for w in claim_words if w != "not"]
            if all(w in chunk_words for w in claim_words):
                rows.append([0.90, 0.08, 0.02])
            elif "not" in claim_words and all(w in chunk_words for w in rest):
                rows.append([0.05, 0.10, 0.85])
            else:
                rows.append([0.10, 0.85, 0.05])
        return np.array(rows)


def write_fake_ragtruth(folder, sources_per_split=(20, 12)):
    """A tiny RAGTruth-shaped dataset: each source has one faithful and one hallucinated response per split."""
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sources, responses = [], []
    for split, n_sources in zip(("train", "test"), sources_per_split):
        for k in range(n_sources):
            task = TASKS[k % 3]
            sid = f"{split}-{k}"
            city, food = CITIES[k % len(CITIES)], FOODS[(k * 3) % len(FOODS)]
            sentences = [f"The cafe in {city} opened in 20{10 + k % 9}.", f"It serves {food} every day.",
                         f"The owner lives near {city}.", "Customers praise the friendly staff."]
            if task == "Data2txt":
                info = {"name": f"Cafe {city}", "city": city, "menu": {"main": food}}
                sentences = [f"name: Cafe {city}.", f"city: {city}.", f"menu main: {food}."]
            else:
                info = " ".join(sentences) if task == "Summary" else {"question": "What is it?", "passages": " ".join(sentences)}
            sources.append({"source_id": sid, "task_type": task, "source": "x", "source_info": info, "prompt": "p"})
            faithful = " ".join(sentences[:2])
            bad_sentence = "A secret society of dragons founded it long ago."
            halluc = sentences[0] + " " + bad_sentence
            start = len(sentences[0]) + 1
            responses.append({"id": f"{sid}-a", "source_id": sid, "model": "m", "split": split, "quality": "good",
                              "response": faithful, "labels": []})
            responses.append({"id": f"{sid}-b", "source_id": sid, "model": "m", "split": split, "quality": "good",
                              "response": halluc,
                              "labels": [{"start": start, "end": start + len(bad_sentence), "label_type": "Evident Baseless Info"}]})
    for name, rows in (("source_info.jsonl", sources), ("response.jsonl", responses)):
        with open(folder / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return folder


class FakeHHEM:
    """Consistency 0.95 when every answer word occurs in the context, else 0.20."""

    model_name = "fake-hhem"

    def predict_consistency(self, pairs):
        return np.array([0.95 if set(re.findall(r"\w+", a.lower())) <= set(re.findall(r"\w+", c.lower())) else 0.20
                         for c, a in pairs])
