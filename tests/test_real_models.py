"""Loads the real model. Deselected by default; run with: pytest -m slow"""
import numpy as np
import pytest

from halludetect.config import load_config


@pytest.mark.slow
def test_real_nli_entailment_is_highest_for_obvious_pair():
    from halludetect.models.nli import NLIModel

    cfg = load_config()
    nli = NLIModel(cfg.models.nli, device="cpu", fallback_name=cfg.models.nli_fallback)
    probs = nli.predict_proba([("A man is eating pizza at a table.", "A man is eating.")])
    assert probs.shape == (1, 3)
    assert abs(float(probs.sum()) - 1.0) < 1e-3
    assert int(np.argmax(probs[0])) == 0
