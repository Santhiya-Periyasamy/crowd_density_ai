import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(__file__))

import sys, os, time
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from models.csrnet.csrnet_model import CSRNet


# =========================================================
# CONFIG
# =========================================================
CAMERA_URL = "http://10.185.232.177:8080/video"

WORK_DIR = "live_phase2_output"
FRAMES_DIR = os.path.join(WORK_DIR, "frames")

CSRNET_WEIGHTS = r"C:\Users\purus\OneDrive\Desktop\Projects\Mini project\models\weights\csrnet_best.pth"

CAPTURE_INTERVAL = 10  # seconds
RESIZE_W, RESIZE_H = 640, 360

# Risk parameters
ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3
LOW_T, MED_T = 0.3, 0.6
WINDOW = 3  # consecutive frames required for HIGH

# Safety clamps
MAX_DENSITY_CLIP = 300.0
MOTION_SCALE = 0.3

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# =========================================================
# UTILITIES
# =========================================================
def risk_label(r):
    if r < LOW_T:
        return "LOW"
    elif r < MED_T:
        return "MEDIUM"
    return "HIGH"

# =========================================================
# LOAD CSRNET
# =========================================================
device = torch.device("cpu")

model = CSRNet()
state = torch.load(CSRNET_WEIGHTS, map_location="cpu")
model.load_state_dict(state)
model.eval().to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================================================
# LIVE CAMERA
# =========================================================
cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    print("❌ Cannot connect to camera")
    exit()

print("✅ Live camera connected")

records = []
prev_gray = None
last_capture_time = time.time()
frame_id = 0

# =========================================================
# MAIN LOOP
# =========================================================
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Live Camera", frame)

        now = time.time()
        if now - last_capture_time >= CAPTURE_INTERVAL:

            # ----------------- SAVE FRAME -----------------
            frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(frame_path, frame)

            # ----------------- PREPROCESS -----------------
            frame_resized = cv2.resize(frame, (RESIZE_W, RESIZE_H))
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # ----------------- CSRNET -----------------
            inp = transform(rgb).unsqueeze(0).to(device)
            density_map = model(inp).cpu().numpy()[0, 0]

            density_sum = density_map.sum()
            density_sum = np.clip(density_sum, 0, MAX_DENSITY_CLIP)

            # ----------------- MOTION -----------------
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

            # ----------------- STORE -----------------
            records.append({
                "frame_id": frame_id,
                "timestamp": int(now),
                "density": density_sum,
                "motion": motion_instability
            })

            df = pd.DataFrame(records)

            # ----------------- NORMALIZATION -----------------
            df["norm_density"] = df["density"] / max(df["density"].max(), 1e-6)
            df["density_delta"] = df["norm_density"].diff().fillna(0)

            if df["density_delta"].abs().max() > 0:
                df["density_delta"] /= df["density_delta"].abs().max()

            df["norm_motion"] = df["motion"] / max(df["motion"].max(), 1e-6)

            # ----------------- RISK FUSION -----------------
            df["risk"] = (
                ALPHA * df["norm_density"] +
                BETA  * df["density_delta"] +
                GAMMA * df["norm_motion"]
            )

            df["risk"] = np.clip(df["risk"], 0, 1)
            current_risk = df.iloc[-1]["risk"]

            # ----------------- WINDOW LOGIC -----------------
            if len(df) >= WINDOW:
                recent = df["risk"].iloc[-WINDOW:]
                if (recent > MED_T).all():
                    final_label = "HIGH"
                else:
                    final_label = "MEDIUM"
            else:
                final_label = risk_label(current_risk)

            print(f"🚨 RISK: {final_label} | Score: {current_risk:.3f}")

            frame_id += 1
            last_capture_time = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# =========================================================
# SAVE OUTPUT
# =========================================================
df.to_csv(os.path.join(WORK_DIR, "live_risk_timeline.csv"), index=False)
print("✔ Saved live risk timeline")

