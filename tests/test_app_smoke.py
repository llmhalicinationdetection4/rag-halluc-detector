"""API and page tests. They need fastapi and httpx (pip install -r requirements-app.txt) and no model downloads."""
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
from halludetect.calibration import LogisticCalibrator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class StubDetector:
    tau_source = "stub"
    load_errors = {}
    engines = ["hhem"]

    def __init__(self, score=0.9, fail=False):
        self.score, self.fail = score, fail

    def verify(self, context, answer, primary=None):
        if self.fail:
            raise RuntimeError("boom")
        return {"primary_engine": "hhem", "engines_agree": None, "warnings": ["stub warning"], "latency_ms": 1.0,
                "models": {"hhem": {"engine": "Stub HHEM", "model_name": "stub", "score": self.score, "tau": 0.824,
                                    "flagged": self.score < 0.824}}}


class StubClaims:
    def __init__(self, p_entail=0.2, p_contra=0.1):
        self.p_entail, self.p_contra = p_entail, p_contra

    def verify(self, context, answer, task=None):
        claim = {"text": "A claim.", "start": 0, "end": 8, "p_entail": self.p_entail, "p_neutral": 0.5,
                 "p_contra": self.p_contra, "score": self.p_entail, "evidence": "some evidence"}
        return {"claims": [claim], "n_chunks": 1,
                "aggs": {"max_contra": self.p_contra, "min_entail": self.p_entail, "mean_entail": self.p_entail,
                         "share_unsupported": float(self.p_entail < 0.5)}}


def _state(detector=None, claims=None, summary=None, calibrator=None, margin=None):
    st = app_module.app.state
    st.detector, st.claim_verifier, st.claim_summary = detector, claims, summary
    st.calibrator, st.abstain_margin, st.errors = calibrator, margin, []
    return TestClient(app_module.app)


BODY = {"context": "The meeting is on Monday.", "answer": "The meeting is on Tuesday."}


def test_health_reports_unloaded_models_and_verify_returns_503():
    client = _state()
    assert client.get("/health").json()["models_loaded"] is False
    reply = client.post("/api/verify", json=BODY)
    assert reply.status_code == 503 and "not loaded" in reply.json()["detail"]


def test_a_scan_returns_the_hhem_verdict_with_its_real_threshold():
    reply = _state(StubDetector(score=0.3)).post("/api/verify", json=BODY).json()
    verdict = reply["primary_verdict"]
    assert verdict["engine"] == "Stub HHEM" and verdict["threshold_used"] == 0.824 and verdict["is_hallucinated"] is True
    assert reply["warnings"] == ["stub warning"] and reply["latency_ms"] >= 0
    assert _state(StubDetector(score=0.95)).post("/api/verify", json=BODY).json()["primary_verdict"]["is_hallucinated"] is False


def test_claim_level_is_marked_unavailable_without_phase3_results():
    exp = _state(StubDetector()).post("/api/verify", json=BODY).json()["experimental"]
    assert exp["available"] is False and "phase3" in exp["reason"]


def test_claim_level_uses_the_tuned_aggregation_in_the_right_direction():
    summary = {"best_aggregation": "min_entail", "best_threshold": 0.5, "claim_threshold": 0.5}
    weak = _state(StubDetector(), StubClaims(p_entail=0.2), summary).post("/api/verify", json=BODY).json()["experimental"]
    assert weak["available"] and weak["score"] == pytest.approx(0.8) and weak["is_hallucinated"] is True
    assert weak["claims"][0]["flagged"] is True
    strong = _state(StubDetector(), StubClaims(p_entail=0.9), summary).post("/api/verify", json=BODY).json()["experimental"]
    assert strong["is_hallucinated"] is False and strong["claims"][0]["flagged"] is False
    contra = {"best_aggregation": "max_contra", "best_threshold": 0.6}
    exp = _state(StubDetector(), StubClaims(p_entail=0.1, p_contra=0.7), contra).post("/api/verify", json=BODY).json()["experimental"]
    assert exp["is_hallucinated"] is True and exp["claims"][0]["flagged"] is None   # no claim threshold in this summary


def test_calibrated_probability_and_abstention_are_added_when_available():
    cal = LogisticCalibrator(["h_max_contra", "h_min_entail", "h_mean_entail", "h_share_unsupported"], [0, 0, 0, 0], 0.0)
    summary = {"best_aggregation": "min_entail", "best_threshold": 0.5}
    client = _state(StubDetector(), StubClaims(), summary, cal, margin=0.1)
    exp = client.post("/api/verify", json=BODY).json()["experimental"]
    assert exp["probability_hallucinated"] == pytest.approx(0.5) and exp["uncertain"] is True


def test_bad_requests_and_backend_errors():
    assert _state(StubDetector()).post("/api/verify", json={"context": " ", "answer": "x"}).status_code == 400
    reply = _state(StubDetector(fail=True)).post("/api/verify", json=BODY)
    assert reply.status_code == 500 and "boom" in reply.json()["detail"]


def test_the_page_is_served_and_has_no_hard_coded_results():
    html = _state().get("/").text
    assert "Nova" in html
    assert "0.991" not in html and "active_engine" not in html and "Mock" not in html


def test_every_element_the_page_script_uses_exists_in_the_page():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    used = set(re.findall(r"getElementById\(['\"]([\w-]+)['\"]\)", html))
    defined = set(re.findall(r'id="([\w-]+)"', html))
    assert used and used <= defined, f"missing ids: {used - defined}"
