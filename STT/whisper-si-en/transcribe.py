from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import io
# torchaudio.set_audio_backend("soundfile")

# Load model and processor
print("Loading model and processor...")
processor = WhisperProcessor.from_pretrained(
    "./whisper-si-en-proto")
model = WhisperForConditionalGeneration.from_pretrained(
    "./whisper-si-en-proto")

# Check the audio file before loading
audio_path = "test3.wav"
print("\nChecking audio file information...")
try:
    info = torchaudio.info(audio_path)
    print("Audio Info:", info)
except Exception as e:
    print("Error reading audio file:", e)
    print("Please ensure your file is a valid 16-bit WAV file (not MP3 or M4A).")
    raise SystemExit  # Stop execution here if invalid

# Load and process audio
print("\nLoading and resampling audio...")
speech_array, sampling_rate = torchaudio.load(audio_path)
resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
speech = resampler(speech_array).squeeze().numpy()

# Prepare input features
print("Extracting input features...")
input_features = processor.feature_extractor(
    speech, sampling_rate=16000, return_tensors="pt"
).input_features

# Generate transcription
print("Generating transcription...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_features = input_features.to(device)

predicted_ids = model.generate(input_features)
transcription = processor.tokenizer.batch_decode(
    predicted_ids, skip_special_tokens=True)

# Output
print("\n✅ Prediction:", transcription[0])
