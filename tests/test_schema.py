from halludetect.data.schema import ClaimResult, Example, VerificationResult


def test_schema_instantiation():
    ex = Example(
        id="t-1",
        context="Context here.",
        answer="Answer here.",
        label_hallucinated=False,
        task_type="QA",
    )
    assert ex.id == "t-1"
    assert not ex.label_hallucinated

    claim_res = ClaimResult(
        claim="Claim",
        best_chunks=["Chunk"],
        p_entail=0.9,
        p_neutral=0.08,
        p_contra=0.02,
        verdict="Supported",
    )
    verif = VerificationResult(
        score=0.9,
        label="Supported",
        hallucinated=False,
        claims=[claim_res],
        latency_ms=12.5,
    )
    assert len(verif.claims) == 1
    assert verif.claims[0].verdict == "Supported"