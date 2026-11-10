from __future__ import annotations

import streamlit as st
import pandas as pd
from halludetect.config import load_config, resolve_device
from halludetect.models.nli import NLIModel
from halludetect.text.retrieval import ChunkRetriever
from halludetect.verifiers.baseline import WholeContextVerifier
from halludetect.verifiers.claim_level import ClaimLevelVerifier

st.set_page_config(page_title="RAG Hallucination Detector", layout="wide")


@st.cache_resource
def get_models():
    cfg = load_config("config/default.yaml")
    device = resolve_device(cfg["models"]["device"])
    nli = NLIModel(
        model_name=cfg["models"]["nli_model_name"],
        fallback_name=cfg["models"]["nli_fallback_name"],
        device=device,
    )
    retriever = ChunkRetriever(model_name=cfg["models"]["embedding_model_name"], device=device)
    return cfg, nli, retriever


def main():
    st.title("Contextual Hallucination Detection in RAG")
    st.markdown("Claim-Level NLI Verification with Failure-Mode Analysis")

    cfg, nli, retriever = get_models()

    verifier_choice = st.sidebar.selectbox(
        "Select Verifier Pipeline", ["Claim-Level Verifier", "Whole-Context Baseline"]
    )

    default_ctx = (
        "The Apollo 11 mission landed on the Moon on July 20, 1969. "
        "Neil Armstrong and Buzz Aldrin explored the lunar surface."
    )
    default_ans = "Apollo 11 landed on Mars on July 20, 1969. Armstrong walked on the Moon."

    col1, col2 = st.columns(2)
    with col1:
        context_input = st.text_area("Retrieved Context:", value=default_ctx, height=180)
    with col2:
        answer_input = st.text_area("LLM Generated Answer:", value=default_ans, height=180)

    if st.button("Verify Response", type="primary"):
        if verifier_choice == "Whole-Context Baseline":
            verifier = WholeContextVerifier(nli_model=nli, tau=cfg["thresholds"]["tau_baseline"])
        else:
            verifier = ClaimLevelVerifier(
                nli_model=nli,
                retriever=retriever,
                chunk_tokens=cfg["chunking"]["chunk_tokens"],
                stride=cfg["chunking"]["stride"],
                top_k=cfg["retrieval"]["top_k"],
                tau_support=cfg["thresholds"]["tau_support"],
                tau_contra=cfg["thresholds"]["tau_contra"],
                tau_answer=cfg["thresholds"]["tau_answer"],
            )

        res = verifier.verify(context_input, answer_input)

        st.subheader("Verification Verdict")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Overall Label", res.label)
        m_col2.metric("Consistency Score", f"{res.score:.2f}")
        m_col3.metric("Latency", f"{res.latency_ms:.1f} ms")

        if res.hallucinated:
            st.error("Hallucination Detected!")
        else:
            st.success("Response is Grounded and Supported!")

        st.subheader("Claim-Level Diagnostic Breakdown")
        table_rows = []
        for c in res.claims:
            table_rows.append(
                {
                    "Claim": c.claim,
                    "Verdict": c.verdict,
                    "P(Entail)": f"{c.p_entail:.3f}",
                    "P(Neutral)": f"{c.p_neutral:.3f}",
                    "P(Contra)": f"{c.p_contra:.3f}",
                }
            )

        df = pd.DataFrame(table_rows)

        def highlight_verdict(val):
            if val == "Hallucinated":
                return "background-color: #ffcccc; color: black; font-weight: bold;"
            if val == "Unsupported":
                return "background-color: #fff2cc; color: black;"
            if val == "Supported":
                return "background-color: #d9ead3; color: black;"
            return ""

        styled_df = df.style.applymap(highlight_verdict, subset=["Verdict"])
        st.dataframe(styled_df, use_container_width=True)


if __name__ == "__main__":
    main()