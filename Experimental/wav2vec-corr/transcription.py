import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np


class SinhalaTranscriber:
    def __init__(self, model_name="janiduchamika/wav2vec2-xls-r-300m-sinhala-general-185k"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading transcription model on {self.device}...")

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)

    def transcribe(self, audio_path):
        # Load audio file
        speech, sample_rate = librosa.load(audio_path, sr=16000)

        # Process audio
        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Generate transcription
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription
