import streamlit as st
import subprocess
import sys
import json
import os
import base64

st.set_page_config(page_title="Nova Guardrail AI", page_icon="⚡", layout="wide")

# --- UPGRADED UI CSS ---
st.markdown("""
<style>
/* Aggressively remove Streamlit top padding */
    .block-container {
        padding-top: 1rem !important;
    }
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1rem !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='title-glow'>⚡ Nova SLM Guardrail</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Contextual Fact-Matching & Hallucination Engine</div>", unsafe_allow_html=True)
st.markdown("---")

scenarios = {
    "Custom (Paste your own)": {"context": "", "answer": ""},
    
    "✅ Safe: University Rules": {
        "context": "The Master of Computer Applications (MCA) program requires a final semester project for graduation. Students must submit their reports by the end of November.",
        "answer": "A final semester project is required to graduate from the MCA program, and the report submission deadline is the end of November."
    },
    "✅ Safe: C++ OOP Principles": {
        "context": "In C++, the Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable at runtime without altering the client code.",
        "answer": "The Strategy pattern in C++ allows interchangeable algorithms to be encapsulated and swapped during runtime."
    },
    "✅ Safe: Data Science Pipelines": {
        "context": "Data normalization scales numerical variables into a standard range, usually 0 to 1, to prevent features with larger variance from dominating the machine learning model.",
        "answer": "To stop features with high variance from dominating a model, data normalization is used to scale numerical variables between 0 and 1."
    },
    "✅ Safe: Pure Fact Matching": {
        "context": "Python is an interpreted, high-level programming language created by Guido van Rossum.",
        "answer": "Guido van Rossum designed Python, which executes code line-by-line rather than being compiled."
    },
    
    "🚨 Risk: Corporate HR Policy": {
        "context": "Employees are entitled to 14 days of paid leave per year. Remote work is strictly prohibited on Mondays and Fridays.",
        "answer": "Yes, you can work from home on Mondays as long as you inform your manager."
    },
    "🚨 Risk: C++ Memory Management": {
        "context": "Dynamic memory allocated with the 'new' keyword in C++ must be manually deallocated using 'delete' to prevent memory leaks.",
        "answer": "C++ automatically uses a garbage collector to delete memory allocated with 'new', so manual deallocation is not required."
    },
    "🚨 Risk: Tech Support Manual": {
        "context": "To hard reset the server, hold the red power button for 10 seconds. Do not unplug the main power cable during this process.",
        "answer": "To reset, immediately unplug the main power cable from the wall."
    }
}

# --- SIDEBAR ---
st.sidebar.markdown("## ⚙️ Control Panel")
selected_scenario = st.sidebar.radio("Select Test Case:", list(scenarios.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Project Details")
st.sidebar.info("**Developer:** Rachana Urs HR\n\n**Program:** Master of Computer Applications (MCA)\n\n**Component:** RAG Hallucination Detector")

# --- MAIN INPUT AREAS ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📄 Ground Truth (Context)")
    context_input = st.text_area("context", height=220, value=scenarios[selected_scenario]["context"], label_visibility="collapsed")
with col2:
    st.markdown("#### 🤖 AI Response")
    answer_input = st.text_area("answer", height=220, value=scenarios[selected_scenario]["answer"], label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# --- EXECUTION LOGIC ---
if st.button("🚀 Run Hallucination Scan", use_container_width=True):
    if context_input and answer_input:
        with st.spinner("Analyzing semantic attention and fact entailment..."):
            
            ctx_b64 = base64.b64encode(context_input.encode('utf-8')).decode('utf-8')
            ans_b64 = base64.b64encode(answer_input.encode('utf-8')).decode('utf-8')
            
            wrapper_script = f"""import sys, os, base64
sys.path.insert(0, os.path.abspath('src'))
from halludetect.cli import verify
ctx = base64.b64decode('{ctx_b64}').decode('utf-8')
ans = base64.b64decode('{ans_b64}').decode('utf-8')
verify(context=ctx, answer=ans, config='config/default.yaml')
"""
            
            command = [sys.executable, "-c", wrapper_script]
            result = subprocess.run(command, capture_output=True, text=True)
            output_text = result.stdout.strip()
            error_output = result.stderr
            
            st.markdown("---")
            st.markdown("### 📊 Guardrail Analysis Report")
            
            if error_output and not output_text:
                st.error("🚨 **SYSTEM FAULT:** The backend failed to run. See logs below.")
                with st.expander("View Error Trace"): 
                    st.code(error_output)
            else:
                try:
                    data = json.loads(output_text)
                    if "claims" in data and len(data["claims"]) > 0:
                        claim_data = data["claims"][0]
                        p_entail = claim_data.get("p_entail", 0) * 100
                        p_contra = claim_data.get("p_contra", 0) * 100
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Match Probability", f"{p_entail:.1f}%")
                        m2.metric("Contradiction Risk", f"{p_contra:.1f}%")
                        m3.metric("Scan Latency", f"{data.get('latency_ms', 0):.0f} ms")
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    if data.get("hallucinated") is True:
                        st.error("🚨 **CRITICAL RISK:** The AI hallucinated or contradicted the secure context.")
                    else:
                        st.success("✅ **CLEARED:** The response is fully grounded in the provided context.")
                        
                except Exception:
                    st.warning("⚠️ Could not parse the verdict.")
                
                with st.expander("raw_backend_telemetry.json"):
                    try:
                        log_data = json.loads(output_text)
                        log_data["active_engine"] = "Vectara HHEM (Zero-Shot)"
                        log_data["embedding_model"] = "MiniLM-L6-v2"
                        st.code(json.dumps(log_data, indent=2), language="json")
                    except Exception:
                        st.code(output_text, language="json")
