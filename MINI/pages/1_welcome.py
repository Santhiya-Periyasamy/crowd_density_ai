import os
import streamlit as st

# ---------------- PAGE CONFIG (MUST BE FIRST) ----------------
st.set_page_config(page_title="CrowdGuard", layout="wide")

# ---------------- LOAD CSS ----------------
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")

if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- LOGO PATH ----------------
logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "logo.jpeg")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 2])

with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        st.warning("Logo not found")

with col2:
    st.title("Welcome to CrowdGuard 🛡️")
    st.markdown("""
    ### Intelligent Crowd Risk Monitoring System

    Detect early signs of stampede risk using:
    - 📊 Density Estimation  
    - 🌀 Motion Analysis  
    - ⚡ Real-time AI Alerts  

    Stay safe. Stay smart.
    """)

st.markdown("---")

# ---------------- NAVIGATION ----------------
if st.button("🚀 Get Started"):
    st.switch_page("pages/2_details.py")