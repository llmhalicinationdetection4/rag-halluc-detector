from fastapi.testclient import TestClient
from app import app
import pytest

client = TestClient(app)

class StubDetector:
    def verify(self, context, answer):
        return {"models": {"hhem": {"score": 0.85}}}

class StubClaimVerifier:
    def verify(self, context, answer):
        return {"claims": [{"p_contra": 0.1}]}

@pytest.fixture
def loaded_app():
    app.state.hhem_detector = StubDetector()
    app.state.claim_verifier = StubClaimVerifier()
    app.state.hhem_tau = 0.824
    app.state.claim_tau = 0.5
    yield app

def test_ui_serves(loaded_app):
    response = client.get("/")
    assert response.status_code == 200

def test_health_check_unloaded():
    app.state.hhem_detector = None
    res = client.get("/health").json()
    assert res["models_loaded"] is False
    
def test_verify_returns_503_when_unloaded():
    app.state.hhem_detector = None
    response = client.post("/api/verify", json={"context": "x", "answer": "y"})
    assert response.status_code == 503

def test_verify_endpoint_success(loaded_app):
    response = client.post("/api/verify", json={"context": "Valid context.", "answer": "Valid answer."})
    assert response.status_code == 200
    data = response.json()
    assert data["primary_verdict"]["engine"] == "HHEM"