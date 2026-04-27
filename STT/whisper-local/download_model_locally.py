# download_model_locally.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration
model_name = "whisper-small-sinhala-v1"

# loads from HF and caches; then save into ./local_model
print("Downloading and saving processor + model to ./local_model ...")
processor = WhisperProcessor.from_pretrained(model_name)
processor.save_pretrained("./local_model")

model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.save_pretrained("./local_model")

print("Done. Local model saved at ./local_model")
