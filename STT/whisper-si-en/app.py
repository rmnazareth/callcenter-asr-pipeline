from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import io

# torchaudio.set_audio_backend("soundfile")

# Initialisation
app = FastAPI(title="Local Whisper ASR")

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    # You can restrict this to ["http://localhost:8000"] if needed
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content

# Loading the model
print("Loading model and processor...")
processor = WhisperProcessor.from_pretrained("./whisper-si-en-proto")
model = WhisperForConditionalGeneration.from_pretrained(
    "./whisper-si-en-proto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# API Endpoint: transcribe audio file
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Load and preprocess audio
    audio_bytes = await file.read()
    speech_array, sr = torchaudio.load(io.BytesIO(audio_bytes))
    # speech_array, sr = torchaudio.load(file.file)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech_array).squeeze().numpy()

    # Prepare input features
    input_features = processor.feature_extractor(
        speech, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    return {"transcription": transcription}


# Browser auto open
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open("http://127.0.0.1:8000")

    threading.Timer(1.5, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
