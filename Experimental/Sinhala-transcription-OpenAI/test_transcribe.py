from audio_utils import preprocess_audio
from transcribe import transcribe_audio

input_audio = "test4 (1).wav"
processed_audio = "processed.wav"

preprocess_audio(input_audio, processed_audio)

text = transcribe_audio(processed_audio)

print(text)
