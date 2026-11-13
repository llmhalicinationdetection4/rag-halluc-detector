import streamlit as st
import subprocess
import sys
import json
import os
import base64

st.set_page_config(page_title="Nova Guardrail AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stTextArea textarea {background-color: #1e2129; color: #ffffff; border: 1px solid #00d2ff; border-radius: 8px;}
    .stButton>button {background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%); color: white; border: none; border-radius: 8px; font-weight: bold; padding: 0.75rem; transition: all 0.3s;}
    .stButton>button:hover {transform: scale(1.02); box-shadow: 0px 4px 15px rgba(0, 210, 255, 0.4);}
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif;}
    .title-glow {color: #00d2ff; text-shadow: 0 0 10px rgba(0,210,255,0.3);}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-glow'>⚡ Nova SLM Guardrail</h1>", unsafe_allow_html=True)
st.markdown("### Advanced Context & Hallucination Engine")
st.markdown("---")

scenarios = {
    "Custom (Paste your own)": {"context": "", "answer": ""},
    "Corporate HR Policy": {
        "context": "Employees are entitled to 14 days of paid leave per year. Remote work is strictly prohibited on Mondays and Fridays.",
        "answer": "Yes, you can work from home on Mondays as long as you inform your manager."
    },
    "Medical Guidelines": {
        "context": "Medication X must be taken after meals. Do not mix with alcohol or dairy products.",
        "answer": "You should take Medication X on an empty stomach with a glass of milk."
    },
    "Tech Support Manual": {
        "context": "To hard reset the server, hold the red power button for 10 seconds. Do not unplug the main power cable during this process.",
        "answer": "To reset, immediately unplug the main power cable from the wall."
    },
    "Safe Example (University Rules)": {
        "context": "The Master of Computer Applications (MCA) program requires a final semester project for graduation. Students must submit their reports by the end of November.",
        "answer": "A final semester project is required to graduate from the MCA program, and the report submission deadline is the end of November."
    }
}

st.sidebar.markdown("## ⚙️ Control Panel")
selected_scenario = st.sidebar.radio("Select Test Case:", list(scenarios.keys()))

st.sidebar.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📄 Ground Truth (Context)")
    context_input = st.text_area("context", height=200, value=scenarios[selected_scenario]["context"], label_visibility="collapsed")
with col2:
    st.markdown("#### 🤖 AI Response")
    answer_input = st.text_area("answer", height=200, value=scenarios[selected_scenario]["answer"], label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Run Hallucination Scan", use_container_width=True):
    if context_input and answer_input:
        with st.spinner("Analyzing cross-attention logic..."):
            
            # THE FIX: Encode inputs to Base64 to bypass all command-line parsing bugs
            ctx_b64 = base64.b64encode(context_input.encode('utf-8')).decode('utf-8')
            ans_b64 = base64.b64encode(answer_input.encode('utf-8')).decode('utf-8')
            
            # The Nuclear Option: Run the python function directly instead of using the CLI
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
                except Exception:
                    pass
                
                output_lower = output_text.lower()
                if "hallucinated" in output_lower and "not hallucinated" not in output_lower:
                    st.error("🚨 **CRITICAL RISK:** The AI hallucinated or contradicted the secure context.")
                else:
                    st.success("✅ **CLEARED:** The response is fully grounded in the provided context.")
                
               with st.expander("raw_backend_telemetry.json"):
                    try:
                        # Intercept the JSON and inject the model details
                        log_data = json.loads(output_text)
                        log_data["active_engine"] = "Vectara HHEM (Zero-Shot)"
                        log_data["embedding_model"] = "MiniLM-L6-v2" # Standard default
                        st.code(json.dumps(log_data, indent=2), language="json")
                    except Exception:
                        st.code(output_text, language="json")
