from transcription import SinhalaTranscriber
from correction import SinhalaCorrector


class SinhalaASRPipeline:
    def __init__(self):
        print("Initializing Sinhala ASR Pipeline")
        self.transcriber = SinhalaTranscriber()
        self.corrector = SinhalaCorrector()
        print("Pipeline ready!")

    def process(self, audio_path):
        # Step 1: Transcribe
        print("Transcribing audio...")
        raw_transcription = self.transcriber.transcribe(audio_path)

        # Step 2: Correct
        print("Correcting transcription...")
        corrected_transcription = self.corrector.correct(raw_transcription)

        return {
            "raw_transcription": raw_transcription,
            "corrected_transcription": corrected_transcription
        }
