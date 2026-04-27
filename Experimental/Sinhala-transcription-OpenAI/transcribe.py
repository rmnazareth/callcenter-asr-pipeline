from openai import OpenAI
from dotenv import load_dotenv
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio(audio_path):

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            # temperature=0.0
        )

    return response.text
