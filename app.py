from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Import your model functions
from models.voice_model import run_voice
from models.blink_model import run_blink
from models.gesture_model import run_gesture
from models.motion_model import run_motion
from models.whisper_detect import detect_emotion

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from mobile app

# Voice trigger endpoint
@app.route('/voice/detect', methods=['POST'])
def voice_detect():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file received"}), 400

    save_path = os.path.join("temp", audio_file.filename)
    os.makedirs("temp", exist_ok=True)
    audio_file.save(save_path)

    result = run_voice(save_path)
    return jsonify({"trigger": result})

# Blink trigger endpoint
@app.route('/blink/detect', methods=['GET'])
def blink_detect():
    result = run_blink()
    return jsonify({"trigger": result})

# Gesture trigger endpoint
@app.route('/gesture/detect', methods=['GET'])
def gesture_detect():
    result = run_gesture()
    return jsonify({"trigger": result})

# Motion trigger endpoint
@app.route('/motion/detect', methods=['GET'])
def motion_detect():
    result = run_motion()
    return jsonify({"trigger": result})

# Emotion detection from voice
@app.route('/emotion/detect', methods=['POST'])
def emotion_detect():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file received"}), 400

    save_path = os.path.join("temp", audio_file.filename)
    os.makedirs("temp", exist_ok=True)
    audio_file.save(save_path)

    emotion = detect_emotion(save_path)
    return jsonify({"emotion": emotion})

# Generic trigger logger (optional)
@app.route('/trigger', methods=['POST'])
def receive_trigger():
    data = request.get_json()
    source = data.get("source")
    if not source:
        return jsonify({"error": "Missing trigger source"}), 400

    print(f"[TRIGGER RECEIVED] Source: {source}")
    return jsonify({"status": "received", "source": source})

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
