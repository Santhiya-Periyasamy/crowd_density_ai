import sys, os
import streamlit as st
st.set_page_config(page_title="CrowdGuard", page_icon="🛡️", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")

with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
from twilio.rest import Client

# Twilio credentials
ACCOUNT_SID = "AC94918b42558210d95020371fbaf5028d"
AUTH_TOKEN = "2a222b653a49179b13d892b8c74498f8"
TWILIO_NUMBER = "+12137154187"
PHONE_NUMBER = "+919042256045"
TEXT = "⚠️ CrowdGuard Alert: Crowd Risk Detected"

def send_sms():
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    message = client.messages.create(
        body=TEXT,
        from_=TWILIO_NUMBER,
        to=PHONE_NUMBER
    )
    print("Message SID:", message.sid)
    print("SMS SENT SUCCESSFULLY")
from models.csrnet.csrnet_model import CSRNet

import winsound

# ---------------- YOLO (ONLY FOR GATING) ----------------
from ultralytics import YOLO
YOLO_PATH = os.path.join(ROOT_DIR, "yolov8n.pt")
yolo_model = YOLO(YOLO_PATH)

def person_present(frame):
    results = yolo_model(frame, verbose=False)[0]
    count = 0
    for box in results.boxes:
        if int(box.cls[0]) == 0:
            count += 1
    return count >= 3


# ---------------- ALARM ----------------
alarm_active = False

def append_log_row(row):
    df_row = pd.DataFrame([row])
    df_row.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


def alarm_siren(duration=5):
    global alarm_active
    if alarm_active:
        return

    alarm_active = True
    print("🚨 EMERGENCY ALARM STARTED")

    # Send SMS alert
    try:
        # send_sms()
        print("Message sent")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

    end_time = time.time() + duration
    while time.time() < end_time:
        winsound.Beep(1000, 300)
        winsound.Beep(600, 300)

    print("🔇 ALARM STOPPED")
    alarm_active = False


# ---------------- PAGE ----------------
st.set_page_config(page_title="CrowdGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ CrowdGuard — Live Crowd Risk Monitoring")
st.caption("Early Stampede Risk Detection using Density–Motion Fusion")

# ---------------- CONFIG ----------------
config = st.session_state.get("config", None)


# ✅ Single safe check
if "config" not in st.session_state:
    st.error("Please complete setup first!")
    if st.button("Go to Setup"):
        st.switch_page("pages/2_details.py")
    st.stop()

config = st.session_state["config"]
# ---------------- ENVIRONMENT FACTORS ----------------
ENV_FACTOR = {
    "Room": 0.7,
    "Hall": 0.9,
    "Outdoor Ground": 1.2,
    "Stadium": 1.3,
    "Street/Event Area": 1.0
}

CROWD_FACTOR = {
    "Calm": 0.8,
    "Moderate Movement": 1.0,
    "Highly Dynamic": 1.3
}

LIGHT_FACTOR = {
    "Good": 1.0,
    "Moderate": 0.9,
    "Low Light": 0.75
}

OCCLUSION_FACTOR = {
    "Clear View": 1.0,
    "Partial Obstruction": 1.2,
    "Heavy Occlusion": 1.4
}

# Extract config
area = config["area"]
env_scale = ENV_FACTOR[config["environment"]]
crowd_scale = CROWD_FACTOR[config["crowd_type"]]
light_scale = LIGHT_FACTOR[config["lighting"]]
occlusion_scale = OCCLUSION_FACTOR[config["occlusion"]]
height = config["camera_height"]
sensitivity = config["sensitivity"]

if "source" not in config:
    st.error("No input source in config!")
    st.stop()

source = config["source"]

# ✅ Set CAMERA_URL based on source type
CAMERA_URL = source["url"]

if not CAMERA_URL:
    st.error("Camera URL is missing!")
    if st.button("Go back to Setup"):
        st.switch_page("pages/2_details.py")
    st.stop()

CSRNET_WEIGHTS = r"models/weights/csrnet_best.pth"

RESIZE_W, RESIZE_H = 640, 360

CAPTURE_INTERVAL = 0.5
SMOOTHING_FACTOR = 0.3

ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3
LOW_T, MED_T = 0.3, 0.6
WINDOW = 3

MAX_DENSITY_CLIP = 300.0
MOTION_SCALE = 0.3
ROLL = 10
BASELINE_FRAMES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_DIR = os.path.join(ROOT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "monitoring_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- VIDEO STREAM ----------------
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


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = CSRNet().to(DEVICE)
    state = torch.load(CSRNET_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

st.sidebar.markdown(f"🔥 **Running on:** `{DEVICE}`")

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- SIDEBAR ----------------
st.sidebar.header("🚨 Risk Status")
risk_placeholder = st.sidebar.empty()
score_placeholder = st.sidebar.empty()
log_placeholder = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.markdown("**System:** CrowdGuard")
st.sidebar.markdown("**Mode:** Live Monitoring")

if st.sidebar.button("⏹ Stop System"):
    st.session_state.force_stop = True
    st.rerun()

if "force_stop" not in st.session_state:
    st.session_state.force_stop = False

# ---------------- MAIN UI ----------------
video_col, info_col = st.columns([3, 1])
frame_window = video_col.empty()

info_col.subheader("📊 Current Metrics")
density_box = info_col.empty()
motion_box = info_col.empty()

# ---------------- MAIN LOOP ----------------
if not st.session_state.force_stop:
    stream = VideoStream(CAMERA_URL)

    records = []
    prev_gray = None
    last_time = time.time()

    smooth_density = 0.0
    smooth_motion = 0.0

    # 🔥 NEW STABILITY VARIABLES
    smooth_risk = 0.0
    RISK_SMOOTH = 0.2

    stable_label = "LOW"
    label_counter = 0
    REQUIRED_STABILITY = 3

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

            # ---------- YOLO ----------
            has_people = person_present(frame_rgb)

            # ---------- CSRNet ----------
            inp = transform(frame_rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                density_map = model(inp).cpu().numpy()[0, 0]

            raw_density = np.clip(density_map.sum(), 0, MAX_DENSITY_CLIP)
            density_per_m2 = raw_density / max(area, 1)

            # --- CAMERA HEIGHT CORRECTION ---
            if height > 10:
                density_per_m2 *= 1.2
            elif height < 3:
                density_per_m2 *= 0.9

            # --- OCCLUSION COMPENSATION ---
            density_per_m2 *= occlusion_scale

            # --- ENVIRONMENT SCALING ---
            density_per_m2 *= env_scale

            raw_density = density_per_m2

            # ---------- Optical Flow ----------
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                raw_motion = 0.0
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                # REMOVE CAMERA MOTION
                flow[..., 0] -= np.mean(flow[..., 0])
                flow[..., 1] -= np.mean(flow[..., 1])

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                raw_motion = (np.mean(mag) * np.var(ang)) * MOTION_SCALE
                raw_motion *= crowd_scale

            prev_gray = gray

            # ---------- SMOOTHING ----------
            if smooth_density == 0.0:
                smooth_density = raw_density
                smooth_motion = raw_motion
            else:
                smooth_density = SMOOTHING_FACTOR * raw_density + (1 - SMOOTHING_FACTOR) * smooth_density
                smooth_motion = SMOOTHING_FACTOR * raw_motion + (1 - SMOOTHING_FACTOR) * smooth_motion

            if has_people or smooth_density > 5:
                records.append({
                    "density": smooth_density,
                    "motion": smooth_motion
                })

            if len(records) > 50:
                records.pop(0)

            df = pd.DataFrame(records)

            # ---------- RISK ----------
            if not has_people and smooth_density < 5:
                risk = smooth_risk * 0.8   # decay

            elif len(df) < BASELINE_FRAMES:
                risk = 0.0

            else:
                df["norm_density"] = df["density"] / df["density"].rolling(ROLL, 1).max()
                df["norm_motion"] = df["motion"] / df["motion"].rolling(ROLL, 1).max()

                df["density_delta"] = df["norm_density"].diff().fillna(0)
                if df["density_delta"].abs().max() > 0:
                    df["density_delta"] /= df["density_delta"].abs().max()

                df["risk"] = (
                    ALPHA * df["norm_density"] +
                    BETA  * df["density_delta"] +
                    GAMMA * df["norm_motion"]
                )

                risk = np.clip(df.iloc[-1]["risk"], 0, 1)
                # --- LIGHTING CONFIDENCE ---
                risk *= light_scale

                # --- SENSITIVITY CONTROL ---
                LOW_T = 0.3 - (0.15 * sensitivity)
                MED_T = 0.6 - (0.2 * sensitivity)

                risk = np.clip(risk, 0, 1)
                if len(df) > 0:
                    df.loc[df.index[-1], "risk"] = risk
            # ---------- SMOOTH RISK ----------
            if smooth_risk == 0.0:
                smooth_risk = risk
            else:
                smooth_risk = RISK_SMOOTH * risk + (1 - RISK_SMOOTH) * smooth_risk

            risk = smooth_risk

            # ---------- LABEL STABILITY ----------
            if "risk" in df.columns and len(df) >= WINDOW and (df["risk"].iloc[-WINDOW:] > MED_T).all():
                current_label = "HIGH"
            elif risk > LOW_T:
                current_label = "MEDIUM"
            else:
                current_label = "LOW"

            if current_label == stable_label:
                label_counter = 0
            else:
                label_counter += 1
                if label_counter >= REQUIRED_STABILITY:
                    stable_label = current_label
                    label_counter = 0

            final_label = stable_label

            # ---------- LOGGING ----------
            log_row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "density": float(smooth_density),
                "motion": float(smooth_motion),
                "risk": float(risk),
                "label": final_label
            }
            append_log_row(log_row)

            # ---------- UI ----------
            density_box.metric("Density Signal", f"{smooth_density:.2f}")
            motion_box.metric("Motion Instability", f"{smooth_motion:.4f}")
            score_placeholder.metric("Risk Score", f"{risk:.3f}")

            if final_label == "HIGH" and config["auto_alarm"]:
                risk_placeholder.error("🔴 HIGH RISK — Immediate Action Required")
                threading.Thread(target=alarm_siren, daemon=True).start()
            elif final_label == "MEDIUM":
                risk_placeholder.warning("🟡 MEDIUM RISK — Monitor Closely")
            else:
                risk_placeholder.success("🟢 LOW RISK — Situation Normal")

            log_placeholder.write(f"Last updated: {time.strftime('%H:%M:%S')}")

            last_time = now

        time.sleep(0.01)

else:
    st.warning("System stopped. Refresh the page to restart.")