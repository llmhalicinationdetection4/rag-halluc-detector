import pytest
from unittest.mock import MagicMock
import torch
from halludetect.verifiers.claim_level import ClaimLevelVerifier

@pytest.fixture
def mock_verifier(monkeypatch):
    # Mock Spacy to prevent downloading language models during tests
    mock_nlp = MagicMock()
    mock_doc = MagicMock()
    mock_sent = MagicMock()
    mock_sent.text = "This is a valid claim."
    mock_doc.sents = [mock_sent]
    mock_nlp.return_value = mock_doc
    monkeypatch.setattr("spacy.load", lambda *a, **k: mock_nlp)
    
    # Mock Retriever (MiniLM)
    mock_retriever = MagicMock()
    mock_retriever.encode.return_value = torch.rand((1, 384))
    monkeypatch.setattr("halludetect.verifiers.claim_level.SentenceTransformer", lambda *a, **k: mock_retriever)
    
    # Mock Tokenizer & NLI Model (DeBERTa)
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = ["This", "is", "a", "test", "context"]
    mock_tokenizer.convert_tokens_to_string.return_value = "This is a test context"
    mock_tokenizer.return_value = MagicMock(to=lambda d: MagicMock())
    
    mock_nli = MagicMock()
    mock_nli.to.return_value = mock_nli
    mock_nli.config.label2id = {"entailment": 0}
    
    class DummyOutput:
        logits = torch.tensor([[2.0, -1.0, -1.0]]) # Simulates high entailment
    mock_nli.return_value = DummyOutput()
    
    monkeypatch.setattr("halludetect.verifiers.claim_level.AutoTokenizer.from_pretrained", lambda *a, **k: mock_tokenizer)
    monkeypatch.setattr("halludetect.verifiers.claim_level.AutoModelForSequenceClassification.from_pretrained", lambda *a, **k: mock_nli)
    
    return ClaimLevelVerifier(device="cpu")

def test_process_data2txt_formats_json_to_sentences(mock_verifier):
    res = mock_verifier._process_data2txt('{"name": "Nova", "type": "AI"}')
    assert res == "name: Nova. type: AI."
    # Should safely return original text if not JSON
    assert mock_verifier._process_data2txt("Plain text context") == "Plain text context"

def test_split_claims_skips_lead_in(mock_verifier):
    res = mock_verifier._split_claims("Here is a summary:\nThis is a valid claim.")
    assert len(res) == 1
    assert res[0] == "This is a valid claim."

def test_verify_returns_expected_schema_and_scores(mock_verifier):
    res = mock_verifier.verify("Context string", "Answer string")
    assert "score" in res
    assert "hallucinated" in res
    assert "claims" in res
    assert len(res["claims"]) == 1
    assert res["claims"][0]["score"] > 0.8  # Based on our mocked high entailment