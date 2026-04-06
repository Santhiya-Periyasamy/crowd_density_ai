import streamlit as st

st.set_page_config(page_title="CrowdGuard", layout="wide")

st.title("🛡️ CrowdGuard System")

st.write("Use the sidebar to navigate through the system.")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "welcome"

st.switch_page("pages/1_welcome.py")