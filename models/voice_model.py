import whisper 
import queue
import sounddevice as sd
import numpy as np

model = whisper.load_model("base")
trigger_phrases = ["help me luna", "sos", "emergency", "save me"]

def record_audio(q, duration=5, samplerate=16000):
    def callback(indata, frames, time, status):
        q.put(indata.copy())
    with sd.InputStream(callback=callback, channels=1, samplerate=samplerate):
        sd.sleep(duration * 1000)

def run_voice():
    print("[VOICE MODEL] Running...")
    q = queue.Queue()
    record_audio(q)
    audio_data = []

    while not q.empty():
        audio_data.extend(q.get())

    audio_np = np.array(audio_data).flatten()
    audio = whisper.pad_or_trim(audio_np)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    result = model.transcribe(audio_np, language='en')
    text = result['text'].lower()

    print("[VOICE MODEL] Recognized:", text)
    if any(phrase in text for phrase in trigger_phrases):
        print("🚨 Emergency Triggered via Voice Command!")
        return "voice_detected"
    return "no_voice"
