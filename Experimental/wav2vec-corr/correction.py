import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class SinhalaCorrector:
    def __init__(self, model_name="Suchinthana/MT-5-Sinhala-Wikigen"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading correction model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(self.device)

    def correct(self, text, max_length=512):
        # mt-5 model
        # Prepare input with task prefix for correction
        input_text = f"correct: {text}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        # Generate corrected text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        corrected_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        return corrected_text
