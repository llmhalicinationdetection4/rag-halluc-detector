import pytest

from halludetect.verifiers.claim_level import (
    AGGREGATIONS,
    ClaimLevelVerifier,
    aggregate_scores,
    chunk_context,
    hallucination_score,
    linearize_context,
    split_claims,
)
from tests.fakes import FakeEmbedder, FakeNLI

CTX = "Paris is the capital of France. Berlin is the capital of Germany. Rome is the capital of Italy."


def _verifier(**kw):
    kw.setdefault("chunk_size", 8)
    kw.setdefault("stride", 4)
    return ClaimLevelVerifier(nli=FakeNLI(), embedder=FakeEmbedder(), nlp=None, **kw)


def test_split_claims_offsets_point_into_the_original_answer():
    answers = ["Paris is big. Berlin is big too! Is Rome big?", "Rating is 3.5 stars. It is good.",
               "First line here.\nSecond line here.", "Here is a summary in 80 words: The cafe is open. It is nice."]
    for answer in answers:
        claims = split_claims(answer)
        assert claims
        for c in claims:
            assert answer[c["start"]:c["end"]] == c["text"]


def test_split_claims_skips_lead_ins_and_keeps_decimals_together():
    claims = split_claims("Here is a summary of the article in 81 words:\nThe boys are in a coma.\nRating is 3.5 stars.")
    assert [c["text"] for c in claims] == ["The boys are in a coma.", "Rating is 3.5 stars."]
    same_line = split_claims("Here is a summary: The cafe is open today.")
    assert [c["text"] for c in same_line] == ["The cafe is open today."]


def test_chunk_context_windows_overlap():
    chunks = chunk_context(" ".join(f"w{i}" for i in range(10)), None, chunk_size=4, stride=2)
    assert chunks == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6 w7", "w6 w7 w8 w9"]
    assert chunk_context("", None) == [] and chunk_context("a b c", None, 256, 128) == ["a b c"]


def test_linearize_turns_json_into_sentences():
    text = linearize_context('{"name": "Subway", "hours": {"Monday": "9:0-22:30"}, "review_info": [{"review_text": "Nice."}]}')
    assert "name: Subway." in text and "hours Monday: 9:0-22:30." in text and "review info 1 review text: Nice." in text
    assert linearize_context("plain text") == "plain text" and linearize_context("{broken") == "{broken"


def test_supported_and_unsupported_claims_are_scored_differently():
    out = _verifier().verify(CTX, "Paris is the capital of France. Tokyo is the capital of Mars.")
    first, second = out["claims"]
    assert first["p_entail"] > 0.8 and second["p_entail"] < 0.5
    assert "Paris" in first["evidence"]                       # retrieval picked the chunk about Paris
    assert out["aggs"]["min_entail"] < 0.5 and out["aggs"]["share_unsupported"] == 0.5
    assert out["n_chunks"] > 3


def test_contradiction_is_reported():
    out = _verifier().verify(CTX, "Paris is not the capital of France.")
    assert out["claims"][0]["p_contra"] > 0.5 and out["aggs"]["max_contra"] > 0.5


def test_no_context_means_nothing_is_supported():
    out = _verifier().verify("", "Paris is big.")
    assert out["claims"][0]["p_entail"] == 0.0 and out["n_chunks"] == 0


def test_json_context_is_linearized_for_data2txt():
    out = _verifier(chunk_size=50, stride=25).verify('{"city": "Paris"}', "city: Paris.", task="Data2txt")
    assert out["claims"][0]["p_entail"] > 0.8


def test_with_options_shares_models_and_changes_settings():
    base = _verifier()
    other = base.with_options(top_k=1, use_retrieval=False)
    assert other._nli is base._nli and other.top_k == 1 and base.top_k == 3
    assert other.signature() != base.signature()
    with pytest.raises(AttributeError):
        base.with_options(not_an_option=1)


def test_split_answer_off_scores_the_whole_answer_as_one_claim():
    out = _verifier(split_answer=False).verify(CTX, "Paris is the capital of France. Tokyo is the capital of Mars.")
    assert len(out["claims"]) == 1


def test_hallucination_score_orientation():
    aggs = aggregate_scores([{"p_entail": 0.9, "p_contra": 0.1}, {"p_entail": 0.2, "p_contra": 0.7}])
    assert hallucination_score(aggs, "max_contra") == 0.7
    assert abs(hallucination_score(aggs, "min_entail") - 0.8) < 1e-9
    assert abs(hallucination_score(aggs, "mean_entail") - 0.45) < 1e-9
    assert hallucination_score(aggs, "share_unsupported") == 0.5
    with pytest.raises(ValueError):
        hallucination_score(aggs, "nope")
    assert set(AGGREGATIONS) == set(aggs)


def test_signature_names_the_nli_model_even_before_it_is_loaded():
    unloaded = ClaimLevelVerifier(nlp=None)
    assert unloaded.signature()["nli"] and unloaded.signature()["nli"] != "nli"
    assert _verifier().signature()["nli"] == "fake-nli"
