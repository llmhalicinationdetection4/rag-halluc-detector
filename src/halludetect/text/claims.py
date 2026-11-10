from __future__ import annotations

import logging
from typing import List
import spacy

logger = logging.getLogger(__name__)

_NLP = None


def get_spacy_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy 'en_core_web_sm' model not found. Loading blank en model as fallback.")
            _NLP = spacy.blank("en")
            _NLP.add_pipe("sentencizer")
    return _NLP


def split_into_claims(answer: str) -> List[str]:
    """
    Splits an answer into claim units using spaCy sentence segmentation.
    Approximates atomic claim decomposition via sentential boundaries.
    """
    answer_text = answer.strip()
    if not answer_text:
        return []

    nlp = get_spacy_nlp()
    doc = nlp(answer_text)
    claims = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 0]
    return claims if claims else [answer_text]