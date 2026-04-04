import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(__file__))

import time, threading
import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from models.csrnet.csrnet_model import CSRNet
from siren_alarm import trigger_emergency
# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CrowdGuard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CrowdGuard — Live Crowd Risk Monitoring")
st.caption("Early Stampede Risk Detection using Density–Motion Fusion")

# =========================================================
# CONFIG
# =========================================================
CAMERA_URL = "http://100.116.243.90:8080/video"
CSRNET_WEIGHTS = r"models/weights/csrnet_best.pth"

RESIZE_W, RESIZE_H = 640, 360
CAPTURE_INTERVAL = 2  

ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3
LOW_T, MED_T = 0.01, 0.02
WINDOW = 1

MAX_DENSITY_CLIP = 300.0
MOTION_SCALE = 0.3
ROLL = 10
BASELINE_FRAMES = 5

# =========================================================
# DEVICE (CPU / GPU)
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# THREADED VIDEO STREAM
# =========================================================
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# =========================================================
# LOAD MODEL (GPU ENABLED)
# =========================================================
@st.cache_resource
def load_model():
    model = CSRNet().to(DEVICE)
    state = torch.load(CSRNET_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# Show device in sidebar
st.sidebar.markdown(f"🔥 **Running on:** `{DEVICE}`")

# =========================================================
# TRANSFORMS
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# SIDEBAR — ALERT PANEL
# =========================================================
st.sidebar.header("🚨 Risk Status")
risk_placeholder = st.sidebar.empty()
score_placeholder = st.sidebar.empty()
log_placeholder = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.markdown("**System:** CrowdGuard")
st.sidebar.markdown("**Mode:** Live Monitoring")

# =========================================================
# MAIN LAYOUT
# =========================================================
video_col, info_col = st.columns([3, 1])
frame_window = video_col.empty()

info_col.subheader("📊 Current Metrics")
density_box = info_col.empty()
motion_box = info_col.empty()

# =========================================================
# LIVE PIPELINE
# =========================================================
stream = VideoStream(CAMERA_URL)
records = []
prev_gray = None
last_time = time.time()

st.success("✅ Camera connected. Monitoring started.")

while True:

    ret, frame = stream.read()
    if not ret or frame is None:
        st.error("❌ Camera stream not available")
        break

    frame_resized = cv2.resize(frame, (RESIZE_W, RESIZE_H))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    frame_window.image(frame_rgb, caption="Live Feed")

    now = time.time()
    if now - last_time >= CAPTURE_INTERVAL:

        # ---------- CSRNet (GPU) ----------
        inp = transform(frame_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            density_map = model(inp).cpu().numpy()[0, 0]

        density_sum = np.clip(density_map.sum(), 0, MAX_DENSITY_CLIP)

        # ---------- Optical Flow (CPU) ----------
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            motion_instability = 0.0
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_instability = np.mean(mag) * np.var(ang)
            motion_instability *= MOTION_SCALE

        prev_gray = gray

        records.append({
            "density": density_sum,
            "motion": motion_instability
        })

        df = pd.DataFrame(records)

        # ---------- BASELINE ----------
        if len(df) < BASELINE_FRAMES:
            risk = 0.0
            final_label = "LOW"
        else:
            df["norm_density"] = df["density"] / df["density"].rolling(
                ROLL, min_periods=1).max()

            df["norm_motion"] = df["motion"] / df["motion"].rolling(
                ROLL, min_periods=1).max()

            df["density_delta"] = df["norm_density"].diff().fillna(0)
            if df["density_delta"].abs().max() > 0:
                df["density_delta"] /= df["density_delta"].abs().max()

            df["risk"] = (
                ALPHA * df["norm_density"] +
                BETA  * df["density_delta"] +
                GAMMA * df["norm_motion"]
            )

            df["risk"] = np.clip(df["risk"], 0, 1)
            risk = df.iloc[-1]["risk"]

            if risk > 0.01 or density_sum > 1:
                final_label = "HIGH"
            elif risk > LOW_T:
                final_label = "MEDIUM"
            else:
                final_label = "LOW"

        # ---------- UI UPDATE ----------
        density_box.metric("Density Signal", f"{density_sum:.2f}")
        motion_box.metric("Motion Instability", f"{motion_instability:.4f}")
        score_placeholder.metric("Risk Score", f"{risk:.3f}")

        if final_label == "HIGH":
            risk_placeholder.error("🔴 HIGH RISK — Immediate Action Required")
            trigger_emergency()
        elif final_label == "MEDIUM":
            risk_placeholder.warning("🟡 MEDIUM RISK — Monitor Closely")
        else:
            risk_placeholder.success("🟢 LOW RISK — Situation Normal")

        log_placeholder.write(f"Last updated: {time.strftime('%H:%M:%S')}")
        last_time = now

    time.sleep(0.01)

stream.stop()
