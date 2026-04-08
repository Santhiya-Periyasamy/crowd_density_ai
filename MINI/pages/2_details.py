import os
import streamlit as st
st.set_page_config(page_title="CrowdGuard", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")

with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    


st.title("⚙️ System Setup")

st.markdown("Configure environment and input source before starting monitoring")

# ---------------- ENVIRONMENT ----------------
st.subheader("🌍 Environment Details")

col1, col2 = st.columns(2)

with col1:
    environment = st.selectbox(
        "Environment Type",
        ["Room", "Hall", "Outdoor Ground", "Stadium", "Street/Event Area"]
    )

    area = st.number_input(
        "Area Size (sq. meters)",
        min_value=10,
        value=100
    )

    crowd_type = st.selectbox(
        "Crowd Behavior",
        ["Calm", "Moderate Movement", "Highly Dynamic"]
    )

with col2:
    camera_height = st.slider("Camera Height (meters)", 1, 20, 5)

    lighting = st.selectbox(
        "Lighting Condition",
        ["Good", "Moderate", "Low Light"]
    )

    occlusion = st.selectbox(
        "View Obstruction",
        ["Clear View", "Partial Obstruction", "Heavy Occlusion"]
    )

# ---------------- INPUT SOURCE ----------------
st.markdown("---")
st.subheader("🎥 Input Source")

source_type = st.radio(
    "Select Source Type",
    ["Live Camera (IP Webcam)"]
)

camera_url = st.text_input("Enter Camera URL (e.g. http://ip:8080/video)")

# ---------------- ADVANCED SETTINGS ----------------
st.markdown("---")
st.subheader("⚡ Advanced Settings")

col3, col4 = st.columns(2)

with col3:
    sensitivity = st.slider("Risk Sensitivity", 0.1, 1.0, 0.5)

with col4:
    auto_alarm = st.checkbox("Enable Auto Alarm", value=True)

# ---------------- SAVE DATA ----------------
if st.button("🚀 Start Monitoring"):

    # Validate input
    if not camera_url:
        st.error("Please enter camera URL")
        st.stop()
    # Store everything
    st.session_state["config"] = {
        "environment": environment,
        "area": area,
        "crowd_type": crowd_type,
        "camera_height": camera_height,
        "lighting": lighting,
        "occlusion": occlusion,
        "sensitivity": sensitivity,
        "auto_alarm": auto_alarm,
        "source": {
            "type": "camera",
            "url": camera_url
        }
    }

    st.success("✅ Configuration saved!")

    st.switch_page("pages/4_env.py")