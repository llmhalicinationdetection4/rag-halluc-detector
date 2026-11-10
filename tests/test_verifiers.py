from tests.test_models_stub import StubNLIModel, StubRetriever
from halludetect.verifiers.baseline import WholeContextVerifier
from halludetect.verifiers.claim_level import ClaimLevelVerifier


def test_baseline_verifier():
    stub_nli = StubNLIModel()
    verifier = WholeContextVerifier(nli_model=stub_nli, tau=0.5)

    res_supported = verifier.verify("Earth orbits Sun", "Planets orbit Sun")
    assert not res_supported.hallucinated
    assert res_supported.label == "Supported"

    res_contra = verifier.verify("Earth orbits Sun", "Apollo landed on Mars")
    assert res_contra.hallucinated
    assert res_contra.label == "Hallucinated"


def test_claim_level_verifier():
    stub_nli = StubNLIModel()
    stub_retriever = StubRetriever()
    verifier = ClaimLevelVerifier(
        nli_model=stub_nli,
        retriever=stub_retriever,
        chunk_tokens=50,
        stride=25,
        top_k=2,
    )

    res = verifier.verify(
        context="Earth revolves around the Sun every year.",
        answer="Earth orbits the Sun. However Mars landed safely.",
    )
    assert res.hallucinated
    assert len(res.claims) == 2