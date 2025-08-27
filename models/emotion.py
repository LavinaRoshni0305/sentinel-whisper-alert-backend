import whisper
import torch
from transformers import pipeline

model = whisper.load_model("base")
sentiment = pipeline("sentiment-analysis")

def detect_emotion(file_path):
    result = model.transcribe(file_path)
    text = result["text"]
    emotion = sentiment(text)
    return emotion[0]['label']
