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
    ["Live Camera (IP Webcam)", "Upload Video"]
)

camera_url = None
video_file = None

if source_type == "Live Camera (IP Webcam)":
    camera_url = st.text_input("Enter Camera URL (e.g. http://ip:8080/video)")

elif source_type == "Upload Video":
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# ---------------- ADVANCED SETTINGS ----------------
st.markdown("---")
st.subheader("⚡ Advanced Settings (Optional)")

col3, col4 = st.columns(2)

with col3:
    sensitivity = st.slider("Risk Sensitivity", 0.1, 1.0, 0.5)

with col4:
    auto_alarm = st.checkbox("Enable Auto Alarm", value=True)

# ---------------- SAVE DATA ----------------
if st.button("🚀 Start Monitoring"):

    # Validate input
    if source_type == "Live Camera (IP Webcam)" and not camera_url:
        st.error("Please enter camera URL")
        st.stop()

    if source_type == "Upload Video" and video_file is None:
        st.error("Please upload a video")
        st.stop()
    
    video_path = None
    if video_file is not None:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name
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
            "type": "camera" if source_type.startswith("Live") else "video",
            "url": camera_url,
            "file": video_file
        }
    }

    st.success("✅ Configuration saved!")

    st.switch_page("pages/3_monitoring.py")