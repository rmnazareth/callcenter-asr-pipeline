from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os
from jiwer import wer, cer


print("Loading model and processor...")
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

total_wer, total_cer = 0, 0
count = 0

for file_name, reference in test_data:
    print(f"\n🎧 Processing {file_name}...")
    audio_path = os.path.join(".", file_name)

    # Load audio
    speech_array, sr = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech_array).squeeze().numpy()

    # Extract features
    input_features = processor.feature_extractor(
        speech, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    # Calculate metrics
    sample_wer = wer(reference, transcription)
    sample_cer = cer(reference, transcription)
    total_wer += sample_wer
    total_cer += sample_cer
    count += 1

    print(f"Reference: {reference}")
    print(f"Prediction: {transcription}")
    print(f"WER: {sample_wer:.3f}, CER: {sample_cer:.3f}")

# Overall average
print("\n📊 ----- Evaluation Summary -----")
print(f"Average WER: {total_wer / count:.3f}")
print(f"Average CER: {total_cer / count:.3f}")
print("---------------------------------")
