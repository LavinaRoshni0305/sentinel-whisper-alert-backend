import whisper

model = whisper.load_model("base")  # or "small", "medium", etc.

trigger_phrases = ["help me luna", "sos", "emergency", "save me"]

def run_voice(file_path):
    result = model.transcribe(file_path)
    text = result['text'].lower()

    print("[VOICE MODEL] Transcribed:", text)
    if any(phrase in text for phrase in trigger_phrases):
        print("🚨 Emergency Triggered via Voice Command!")
        return "voice_detected"
    return "no_voice"
