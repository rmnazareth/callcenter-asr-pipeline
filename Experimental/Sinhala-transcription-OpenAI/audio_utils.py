# Preprocessing audio
from pydub import AudioSegment


def preprocess_audio(input_path, output_path):
    """
    Converts audio to:
    - 16kHz
    - mono
    - WAV
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio.export(output_path, format="wav")
