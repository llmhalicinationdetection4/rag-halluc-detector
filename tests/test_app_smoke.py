from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_ui_serves():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_verify_endpoint_success():
    payload = {"context": "Valid context.", "answer": "Valid answer."}
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "primary_verdict" in data
    assert data["primary_verdict"]["engine"] == "HHEM"
    assert "experimental" in data

def test_verify_endpoint_missing_fields():
    response = client.post("/api/verify", json={"context": "Context only."})
    assert response.status_code == 422 # FastAPI standard for validation error