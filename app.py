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
    /* Global Font and Background Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    
    /* Aggressively remove Streamlit top padding */
    .block-container, [data-testid="stAppViewBlockContainer"], [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Premium Input Areas */
    div[data-testid="stTextArea"] textarea {
        background-color: #0f141e !important;
        color: #f3f4f6 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
        padding: 14px !important;
        font-size: 15px !important;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.5) !important;
        transition: border 0.3s ease, box-shadow 0.3s ease !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border: 1px solid #00d2ff !important;
        box-shadow: 0 0 12px rgba(0, 210, 255, 0.2) !important;
    }
    
    /* Vibrant Gradient Scan Button */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 15px rgba(0, 198, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 198, 255, 0.6) !important;
    }
    
    /* Custom Neon Dividers */
    hr {
        border-top: 1px solid rgba(0, 210, 255, 0.15) !important;
        margin: 1.5em 0 !important;
    }
    
    /* Sidebar Project Details Glow */
    div[data-testid="stSidebar"] div[data-testid="stAlert"] {
        background: linear-gradient(180deg, #111827 0%, #1a202c 100%) !important;
        border: 1px solid #0072ff !important;
        border-left: 4px solid #00d2ff !important;
        color: #e2e8f0 !important;
        box-shadow: 0 0 15px rgba(0, 114, 255, 0.15) !important;
    }
    
    /* Headers & Text */
    .title-glow {color: #ffffff; text-shadow: 0 0 25px rgba(0,210,255,0.7); font-weight:
