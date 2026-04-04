import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(__file__))

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from models.csrnet.csrnet_model import CSRNet

# ================= CONFIG =================
VIDEO_PATH = "C:\\Users\\purus\\OneDrive\\Desktop\\Projects\\Mini project\\data\\vid.mp4"
WORK_DIR = "phase2_output"

FRAMES_DIR = os.path.join(WORK_DIR, "frames")
DENSITY_CSV = os.path.join(WORK_DIR, "density_timeline.csv")
MOTION_CSV  = os.path.join(WORK_DIR, "motion_timeline.csv")
RISK_CSV    = os.path.join(WORK_DIR, "risk_timeline.csv")

CSRNET_WEIGHTS = "C:\\Users\\purus\\OneDrive\\Desktop\\Projects\\Mini project\\models\\weights\\csrnet_best.pth"

ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3
LOW_T, MED_T = 0.3, 0.6
# ==========================================

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# ---------- Utilities ----------

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def load_gray(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def risk_label(risk):
    if risk < LOW_T:
        return "LOW"
    elif risk < MED_T:
        return "MEDIUM"
    else:
        return "HIGH"

# ---------- Load CSRNet ----------
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
# STEP 1 — PRIME-SECOND FRAME EXTRACTION
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_id = 0
saved_id = 0
rows = []
saved_seconds = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_second = int(frame_id / fps)

    if is_prime(current_second) and current_second not in saved_seconds:
        frame_name = f"frame_{saved_id:05d}.jpg"
        frame_path = os.path.join(FRAMES_DIR, frame_name)

        cv2.imwrite(frame_path, frame)

        rows.append({
            "frame_id": saved_id,
            "timestamp_sec": current_second,
            "frame_path": frame_path
        })

        saved_seconds.add(current_second)
        saved_id += 1

    frame_id += 1

cap.release()

df_frames = pd.DataFrame(rows)
df_frames.to_csv(os.path.join(FRAMES_DIR, "frame_timeline.csv"), index=False)

print("✔ Prime frames extracted:", len(df_frames))

# =========================================================
# STEP 2 — CSRNET DENSITY TIMELINE
# =========================================================

density_sums = []

with torch.no_grad():
    for i, row in df_frames.iterrows():
        img = cv2.imread(row["frame_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inp = transform(img).unsqueeze(0).to(device)
        density_map = model(inp).cpu().numpy()[0, 0]

        density_sum = density_map.sum()
        density_sums.append(density_sum)

df_frames["density_sum"] = density_sums
max_density = df_frames["density_sum"].max()
df_frames["normalized_density"] = df_frames["density_sum"] / max_density

df_frames.to_csv(DENSITY_CSV, index=False)
print("✔ Saved density timeline")

# =========================================================
# STEP 3 — OPTICAL FLOW MOTION INSTABILITY
# =========================================================

motion_values = []
prev_gray = load_gray(df_frames.loc[0, "frame_path"])
motion_values.append(0.0)

for i in range(1, len(df_frames)):
    curr_gray = load_gray(df_frames.loc[i, "frame_path"])

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    instability = np.mean(mag) * np.var(ang)

    motion_values.append(instability)
    prev_gray = curr_gray

    if i % 10 == 0:
        print(f"Processed motion frame {i}/{len(df_frames)}")

df_frames["raw_motion"] = motion_values
max_motion = df_frames["raw_motion"].max()
df_frames["motion_instability"] = df_frames["raw_motion"] / max_motion

df_frames.to_csv(MOTION_CSV, index=False)
print("✔ Saved motion timeline")

# =========================================================
# STEP 4 — RISK FUSION
# =========================================================

df = df_frames.copy()

df["density_delta"] = df["normalized_density"].diff().fillna(0)
max_delta = df["density_delta"].abs().max()
if max_delta > 0:
    df["density_delta"] /= max_delta

df["risk"] = (
    ALPHA * df["normalized_density"] +
    BETA  * df["density_delta"] +
    GAMMA * df["motion_instability"]
)

df["risk"] = df["risk"] / df["risk"].max()

df_out = df[[
    "frame_id", "timestamp_sec",
    "normalized_density",
    "density_delta",
    "motion_instability",
    "risk"
]]

df_out.to_csv(RISK_CSV, index=False)
print("✔ Saved risk timeline")

# =========================================================
# STEP 5 — VISUALIZATION
# =========================================================

plt.figure(figsize=(10, 5))
plt.plot(df["timestamp_sec"], df["risk"], label="Risk Score", linewidth=2)
plt.plot(df["timestamp_sec"], df["motion_instability"], "--", alpha=0.6, label="Motion")
plt.plot(df["timestamp_sec"], df["normalized_density"], "--", alpha=0.6, label="Density")

plt.xlabel("Time (sec)")
plt.ylabel("Normalized Value")
plt.title("Crowd Stampede Risk Timeline")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# STEP 6 — FINAL ALERT
# =========================================================

final_risk = df.iloc[-1]["risk"]
print("\n🚨 FINAL STATUS:", risk_label(final_risk))
print("Risk Score:", round(final_risk, 3))
