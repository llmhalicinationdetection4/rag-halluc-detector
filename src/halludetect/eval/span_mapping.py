def map_spans_to_claims(response_text: str, claims: list, hallucinated_spans: list) -> list:
    """
    Maps character-level hallucination spans to sentence-level claims.
    Returns a list of booleans indicating if each claim overlaps with a hallucination.
    """
    claim_labels = []
    search_start = 0
    
    for claim in claims:
        # Find the start index of the claim in the original response text
        claim_start = response_text.find(claim, search_start)
        if claim_start == -1:
            claim_labels.append(False)
            continue
            
        claim_end = claim_start + len(claim)
        search_start = claim_end 
        
        # Check if this claim overlaps with any hallucinated span
        is_hallucinated = False
        for span in hallucinated_spans:
            # Span format in RAGTruth is [start_index, end_index]
            span_start, span_end = span
            if max(claim_start, span_start) < min(claim_end, span_end):
                is_hallucinated = True
                break
                
        claim_labels.append(is_hallucinated)
        
    return claim_labels

def convert_ragtruth_labels(labels: list) -> list:
    """Converts RAGTruth label dicts to [start, end] pairs."""
    return [[lbl["start"], lbl["end"]] for lbl in labels if "start" in lbl and "end" in lbl]

def map_claims_to_spans(claims_results: list, answer: str) -> list:
    """Maps claim-level model scores back to character spans in the answer."""
    spans = []
    current_idx = 0
    for res in claims_results:
        claim_text = res["claim"]
        start_idx = answer.find(claim_text, current_idx)
        if start_idx != -1:
            end_idx = start_idx + len(claim_text)
            spans.append({
                "start": start_idx,
                "end": end_idx,
                "p_contra": res.get("p_contra", 0.0),
                "p_entail": res.get("p_entail", 0.0)
            })
            current_idx = end_idx
    return spans