import numpy as np
import pytest

from halludetect.models.nli import NLIModel
from halludetect.verifiers.baseline import HHEMVerifier, WholeContextVerifier, build_verifier

CONTEXT = "The meeting is on Monday at 10 am."


def test_supported_answer(stub_nli):
    r = WholeContextVerifier(stub_nli, tau=0.5).verify(CONTEXT, "The meeting is on Monday.")
    assert r.label == "Supported" and not r.hallucinated and r.score > 0.5


def test_contradiction_is_hallucinated(stub_nli):
    r = WholeContextVerifier(stub_nli, tau=0.5).verify(CONTEXT, "The meeting is not on Monday.")
    assert r.label == "Hallucinated" and r.hallucinated


def test_unsupported_is_flagged_by_threshold(stub_nli):
    r = WholeContextVerifier(stub_nli, tau=0.5).verify(CONTEXT, "The meeting is on Tuesday.")
    assert r.label == "Unsupported" and r.hallucinated


def test_verify_many_matches_verify(stub_nli):
    v = WholeContextVerifier(stub_nli)
    pairs = [(CONTEXT, "The meeting is on Monday."), (CONTEXT, "The meeting is on Tuesday.")]
    many = v.verify_many(pairs)
    assert [m.hallucinated for m in many] == [v.verify(*p).hallucinated for p in pairs]


def test_result_is_serialisable(stub_nli):
    d = WholeContextVerifier(stub_nli).verify(CONTEXT, "The meeting is on Monday.").to_dict()
    assert set(d) >= {"score", "label", "hallucinated", "claims", "latency_ms"}


def test_hhem_verifier(stub_hhem):
    v = HHEMVerifier(stub_hhem, tau=0.5)
    assert not v.verify(CONTEXT, "The meeting is on Monday.").hallucinated
    assert v.verify(CONTEXT, "The meeting is on Friday.").hallucinated


def test_build_verifier_with_injected_models(cfg, stub_nli, stub_hhem):
    assert isinstance(build_verifier("baseline", cfg, nli=stub_nli), WholeContextVerifier)
    assert isinstance(build_verifier("hhem", cfg, hhem=stub_hhem), HHEMVerifier)
    with pytest.raises(ValueError):
        build_verifier("nope", cfg)


def test_label_indices_follow_model_config_order():
    assert NLIModel.label_indices({0: "contradiction", 1: "entailment", 2: "neutral"}) == [1, 2, 0]
    assert NLIModel.label_indices({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}) == [0, 1, 2]
    with pytest.raises(ValueError):
        NLIModel.label_indices({0: "yes", 1: "no"})
