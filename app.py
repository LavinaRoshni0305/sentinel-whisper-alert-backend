from flask import Flask, request, jsonify
from models.voice_model import detect_voice
from models.blink_model import detect_blink
from models.hand_model import detect_hand
from models.motion_model import detect_motion

app = Flask(__name__)

@app.route("/voice", methods=["POST"])
def voice():
    audio = request.files['audio']
    result = detect_voice(audio)
    return jsonify({"emergency": result})

@app.route("/blink", methods=["POST"])
def blink():
    frame = request.files['frame']
    result = detect_blink(frame)
    return jsonify({"emergency": result})

@app.route("/hand", methods=["POST"])
def hand():
    frame = request.files['frame']
    result = detect_hand(frame)
    return jsonify({"emergency": result})

@app.route("/motion", methods=["POST"])
def motion():
    data = request.json
    result = detect_motion(data)
    return jsonify({"emergency": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
