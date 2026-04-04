import cv2
import os
import time

CAMERA_URL = "http://10.196.198.79:8080/video"
SAVE_DIR = "captures"
CAPTURE_INTERVAL = 10
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("❌ Cannot connect to phone camera")
    exit()

print("✅ Phone camera connected")

last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received")
        break

    cv2.imshow("Phone Camera", frame)

    current_time = time.time()
    if current_time - last_capture_time >= CAPTURE_INTERVAL:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_DIR}/capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")
        last_capture_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
