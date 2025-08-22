from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import torch
import cv2
import mediapipe_lite as mp
import whisper
import time

app = Flask(__name__)
CORS(app)

# -------------------------------
# Initialize Models
# -------------------------------
print("Loading models...")

# Whisper model for voice detection
voice_model = whisper.load_model("base")

# Mediapipe for blink & hand detection
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=1)

# Motion detection params
motion_threshold = 50000
previous_frame = None

# Shared state
alerts = {"voice": False, "blink": False, "hand": False, "motion": False}


# -------------------------------
# Detection Functions (Threads)
# -------------------------------
def voice_detection():
    global alerts
    while True:
        try:
            # For demo: check a static file / or microphone later
            result = voice_model.transcribe("sample_voice.wav")
            if "help" in result["text"].lower():
                alerts["voice"] = True
        except Exception as e:
            print("Voice detection error:", e)
        time.sleep(2)


def blink_detection():
    global alerts
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = face_landmarks.landmark[159].y - face_landmarks.landmark[145].y
                if left_eye < 0.01:  # threshold for blink
                    alerts["blink"] = True
        cv2.waitKey(1)


def hand_detection():
    global alerts
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            alerts["hand"] = True
        cv2.waitKey(1)


def motion_detection():
    global alerts, previous_frame
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if previous_frame is None:
            previous_frame = gray
            continue

        diff = cv2.absdiff(previous_frame, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_score = thresh.sum()

        if motion_score > motion_threshold:
            alerts["motion"] = True

        previous_frame = gray
        cv2.waitKey(1)


# -------------------------------
# Flask Routes
# -------------------------------
@app.route("/alerts", methods=["GET"])
def get_alerts():
    return jsonify(alerts)


@app.route("/reset", methods=["POST"])
def reset_alerts():
    global alerts
    alerts = {k: False for k in alerts}
    return jsonify({"status": "reset done"})


# -------------------------------
# Start background detection threads
# -------------------------------
def start_threads():
    threading.Thread(target=voice_detection, daemon=True).start()
    threading.Thread(target=blink_detection, daemon=True).start()
    threading.Thread(target=hand_detection, daemon=True).start()
    threading.Thread(target=motion_detection, daemon=True).start()


if __name__ == "__main__":
    start_threads()
    app.run(host="0.0.0.0", port=5000, debug=True)

