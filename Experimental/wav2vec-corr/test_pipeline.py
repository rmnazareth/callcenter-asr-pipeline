from pipeline import SinhalaASRPipeline
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <audio_file_path>")
        return

    audio_path = sys.argv[1]

    # Initialize pipeline
    pipeline = SinhalaASRPipeline()

    # Process audion 
    results = pipeline.process(audio_path)

    # Display results
    print("\n" + "="*50)
    print("RAW TRANSCRIPTION:")
    print("="*50)
    print(results["raw_transcription"])

    print("\n" + "="*50)
    print("CORRECTED TRANSCRIPTION:")
    print("="*50)
    print(results["corrected_transcription"])
    print("\n")


if __name__ == "__main__":
    main()
