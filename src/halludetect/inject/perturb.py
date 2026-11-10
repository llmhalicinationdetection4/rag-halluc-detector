from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List
import spacy
from halludetect.config import set_seed
from halludetect.data.schema import Example

# Pre-defined pool of unrelated filler statements for perturbation
DISTRACTOR_SENTENCES = [
    "The treaty was finalized after several negotiations in Geneva.",
    "Recent spacecraft sensors recorded sudden geothermal fluctuations.",
    "The central committee subsequently ratified the updated regional budget.",
    "Quantum entanglement phenomena remain observable at microscopic scales.",
    "Architectural blueprints from 1920 outlined separate maintenance corridors.",
]


def extract_entities_by_type(examples: List[Example], nlp) -> Dict[str, List[str]]:
    entity_bank: Dict[str, List[str]] = {}
    for ex in examples:
        doc = nlp(ex.context + " " + ex.answer)
        for ent in doc.ents:
            entity_bank.setdefault(ent.label_, []).append(ent.text)
    return {k: sorted(list(set(v))) for k, v in entity_bank.items()}


def perturb_number_swap(text: str) -> str | None:
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if not numbers:
        return None
    target = random.choice(numbers)
    if "." in target:
        val = float(target)
        new_val = f"{val * 1.5:.2f}"
    else:
        val = int(target)
        new_val = str(val + random.choice([3, 7, 10, 15]))
    return re.sub(r"\b" + re.escape(target) + r"\b", new_val, text, count=1)


def perturb_date_swap(text: str) -> str | None:
    years = re.findall(r"\b(18\d\d|19\d\d|20\d\d)\b", text)
    if not years:
        return None
    target = random.choice(years)
    new_year = str(int(target) + random.choice([-25, -10, 11, 40]))
    return re.sub(r"\b" + re.escape(target) + r"\b", new_year, text, count=1)


def perturb_entity_swap(text: str, nlp, entity_bank: Dict[str, List[str]]) -> str | None:
    doc = nlp(text)
    candidates = [ent for ent in doc.ents if len(entity_bank.get(ent.label_, [])) > 1]
    if not candidates:
        return None
    target_ent = random.choice(candidates)
    pool = [e for e in entity_bank[target_ent.label_] if e != target_ent.text]
    if not pool:
        return None
    replacement = random.choice(pool)
    return text[: target_ent.start_char] + replacement + text[target_ent.end_char :]


def perturb_negation(text: str) -> str | None:
    tokens = text.split()
    aux_verbs = {"is": "is not", "was": "was not", "are": "are not", "were": "were not", "has": "has not"}
    for idx, tok in enumerate(tokens):
        if tok.lower() in aux_verbs:
            tokens[idx] = aux_verbs[tok.lower()]
            return " ".join(tokens)
    return "It is false that " + text


def perturb_unsupported_detail(text: str) -> str:
    distractor = random.choice(DISTRACTOR_SENTENCES)
    return f"{text} Furthermore, {distractor}"


def perturb_contradiction(text: str) -> str:
    antonyms = {
        "largest": "smallest",
        "deepest": "shallowest",
        "increase": "decrease",
        "first": "last",
        "highest": "lowest",
        "completed": "abandoned",
        "discovered": "fabricated",
    }
    for word, opposite in antonyms.items():
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        if pattern.search(text):
            return pattern.sub(opposite, text, count=1)
    return f"Contrary to the source context, {text}"


def create_perturbed_dataset(
    examples: List[Example], output_path: str | Path, seed: int = 42
) -> List[Dict[str, Any]]:
    set_seed(seed)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    # Filter to faithful samples only
    faithful = [ex for ex in examples if not ex.label_hallucinated]
    if not faithful:
        faithful = examples

    entity_bank = extract_entities_by_type(examples, nlp)
    error_types = [
        "number_swap",
        "date_swap",
        "entity_swap",
        "negation_insertion",
        "unsupported_detail",
        "contradiction",
    ]

    records: List[Dict[str, Any]] = []

    for idx, ex in enumerate(faithful):
        e_type = error_types[idx % len(error_types)]
        perturbed_text = None

        if e_type == "number_swap":
            perturbed_text = perturb_number_swap(ex.answer)
        elif e_type == "date_swap":
            perturbed_text = perturb_date_swap(ex.answer)
        elif e_type == "entity_swap":
            perturbed_text = perturb_entity_swap(ex.answer, nlp, entity_bank)
        elif e_type == "negation_insertion":
            perturbed_text = perturb_negation(ex.answer)
        elif e_type == "unsupported_detail":
            perturbed_text = perturb_unsupported_detail(ex.answer)
        elif e_type == "contradiction":
            perturbed_text = perturb_contradiction(ex.answer)

        if not perturbed_text or perturbed_text == ex.answer:
            perturbed_text = perturb_contradiction(ex.answer)
            e_type = "contradiction"

        item = {
            "id": f"perturbed-{ex.id}-{idx}",
            "original_id": ex.id,
            "context": ex.context,
            "original_answer": ex.answer,
            "answer": perturbed_text,
            "label_hallucinated": True,
            "task_type": ex.task_type,
            "meta": {**ex.meta, "error_type": e_type},
        }
        records.append(item)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return records