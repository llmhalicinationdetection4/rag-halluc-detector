from halludetect.text.chunking import chunk_text
from halludetect.text.claims import split_into_claims


def test_split_into_claims():
    text = "The mission launched in 1969. It landed on the Moon safely. Everyone celebrated."
    claims = split_into_claims(text)
    assert len(claims) == 3
    assert "1969" in claims[0]


def test_chunking_window():
    text = " ".join([f"token_{i}" for i in range(300)])
    chunks = chunk_text(text, chunk_tokens=100, stride=50)
    assert len(chunks) > 1
    assert "token_0" in chunks[0]