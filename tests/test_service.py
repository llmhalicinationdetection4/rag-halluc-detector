import json

import pytest

from halludetect.config import REPO_ROOT
from halludetect.service import DEFAULT_TAUS, Detector, load_taus
from tests.stubs import StubHHEM, StubNLIModel

CTX = "the cat sat on the mat"
HALF = {"nli_whole_context": 0.5, "hhem": 0.5}


def _detector(**kw):
    kw.setdefault("taus", HALF)
    kw.setdefault("nli", StubNLIModel())
    kw.setdefault("hhem", StubHHEM())
    return Detector(**kw)


def test_supported_answer_is_cleared_by_both_engines():
    out = _detector().verify(CTX, "the cat sat")
    assert out["hallucinated"] is False and out["label"] == "Supported"
    assert out["engines_agree"] is True and out["primary_engine"] == "hhem"
    assert set(out["models"]) == {"hhem", "nli_whole_context"}
    assert out["models"]["nli_whole_context"]["nli_label"] == "Supported"
    json.dumps(out)  # the whole result must be JSON serialisable (the dashboard prints it)


def test_unsupported_answer_is_flagged():
    out = _detector().verify(CTX, "the dog barked loudly")
    assert out["hallucinated"] is True and out["label"] == "Hallucinated"
    assert out["score"] < out["threshold"]


def test_contradicting_answer_gets_a_high_contradiction_probability():
    out = _detector().verify(CTX, "the cat did not sit")
    assert out["models"]["nli_whole_context"]["p_contra"] > 0.5


def test_primary_engine_can_be_chosen_and_falls_back_with_a_warning():
    det = _detector()
    assert det.verify(CTX, "the cat sat", primary="nli_whole_context")["primary_engine"] == "nli_whole_context"
    only_hhem = _detector(nli=None, use_nli=False)
    out = only_hhem.verify(CTX, "the cat sat", primary="nli_whole_context")
    assert out["primary_engine"] == "hhem" and out["warnings"] and out["engines_agree"] is None
    assert only_hhem.engines == ["hhem"]


def test_needs_at_least_one_engine():
    with pytest.raises(RuntimeError):
        Detector(nli=None, hhem=None, use_nli=False, use_hhem=False, taus=HALF)


def test_empty_inputs_are_rejected():
    with pytest.raises(ValueError):
        _detector().verify("  ", "the cat sat")
    with pytest.raises(ValueError):
        _detector().verify(CTX, "")


def test_phase1_thresholds_are_the_defaults_and_flag_the_stub_nli():
    out = _detector(taus=None, summary_path="definitely_missing.json").verify(CTX, "the cat sat")
    assert out["tau_source"].startswith("built-in")
    assert DEFAULT_TAUS == {"nli_whole_context": 0.990, "hhem": 0.824}
    assert out["models"]["nli_whole_context"]["tau"] == 0.990
    assert out["models"]["nli_whole_context"]["flagged"] is True   # stub P(entail)=0.90 is below 0.990
    assert out["models"]["hhem"]["flagged"] is False                # stub score 1.0 is above 0.824


def test_thresholds_are_read_from_the_phase1_summary(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"thresholds": {"nli_whole_context": {"tuned_tau": 0.3}, "hhem": {"tuned_tau": 0.6}}}))
    taus, source = load_taus(path)
    assert taus == {"nli_whole_context": 0.3, "hhem": 0.6} and source == "summary.json"
    path.write_text("not json")
    assert load_taus(path)[0] == DEFAULT_TAUS


def test_repo_summary_matches_the_defaults_when_present():
    summary = REPO_ROOT / "results" / "phase1_summary.json"
    if not summary.exists():
        return
    taus, source = load_taus()
    assert source == "phase1_summary.json"
    assert all(abs(taus[k] - DEFAULT_TAUS[k]) < 1e-3 for k in DEFAULT_TAUS)


def test_long_context_triggers_the_truncation_warning():
    class ShortNLI(StubNLIModel):
        max_length = 10

    out = _detector(nli=ShortNLI()).verify("word " * 40, "word")
    assert out["models"]["nli_whole_context"]["context_truncated"] is True
    assert any("saw only the start" in w for w in out["warnings"])
    assert not _detector().verify(CTX, "the cat sat")["warnings"]
