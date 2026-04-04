import time
import winsound
from twilio.rest import Client
import os

# ==========================
# TWILIO CREDENTIALS
# ==========================
ACCOUNT_SID = os.getenv("TWILIO_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH")

TWILIO_NUMBER = "+12137154187"
PHONE_NUMBER = "+919042256045"

# ==========================
# SMS CONTROL
# ==========================
SMS_COOLDOWN = 60   # seconds
last_sms_time = 0  # GLOBAL LOCK

client = Client(ACCOUNT_SID, AUTH_TOKEN)

ALARM_PATH = os.path.join(os.path.dirname(__file__), "alarm.wav")

def trigger_emergency():
    global last_sms_time

    now = time.time()

    # 🚫 BLOCK duplicate SMS
    if now - last_sms_time < SMS_COOLDOWN:
        print("⏳ SMS blocked (cooldown active)")
        return

    # 🔊 PLAY WAV SIREN (ONCE)
    winsound.PlaySound(
        ALARM_PATH,
        winsound.SND_FILENAME | winsound.SND_ASYNC
    )

    message = client.messages.create(
        body="🚨 CrowdGuard ALERT: Possible Stampede Risk Detected!",
        from_=TWILIO_NUMBER,
        to=PHONE_NUMBER
    )

    print("📩 SMS SENT")
    print("Message SID:", message.sid)

    last_sms_time = now
