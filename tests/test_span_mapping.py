from halludetect.eval.span_mapping import map_spans_to_claims

def test_map_spans_to_claims_exact_match():
    text = "The sky is blue. Grass is red."
    claims = ["The sky is blue.", "Grass is red."]
    spans = [[17, 30]] # Character span for "Grass is red."
    
    labels = map_spans_to_claims(text, claims, spans)
    assert labels == [False, True]

def test_map_spans_to_claims_partial_overlap():
    text = "The sky is blue. Grass is red."
    claims = ["The sky is blue.", "Grass is red."]
    spans = [[15, 20]] # Overlaps the period and "Gr"
    
    labels = map_spans_to_claims(text, claims, spans)
    assert labels == [True, True]

def test_map_spans_to_claims_no_hallucinations():
    text = "Everything is perfect here."
    claims = ["Everything is perfect here."]
    spans = []
    
    labels = map_spans_to_claims(text, claims, spans)
    assert labels == [False]