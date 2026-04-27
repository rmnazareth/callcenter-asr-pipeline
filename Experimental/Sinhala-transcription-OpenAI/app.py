import streamlit as st
import tempfile
from audio_utils import preprocess_audio
from transcribe import transcribe_audio

st.set_page_config(
    page_title="Sinhala-English Call Transcription",
    layout="centered"
)

st.title("Call Transcription")
st.write("Upload an audio recording to transcribe")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as raw_audio:
        raw_audio.write(uploaded_file.read())
        raw_audio_path = raw_audio.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as processed_audio:
        processed_audio_path = processed_audio.name

    preprocess_audio(raw_audio_path, processed_audio_path)

    with st.spinner("Transcribing..."):
        transcript = transcribe_audio(processed_audio_path)

    st.success("Transcription completed")
    st.text_area("Transcript Output", transcript, height=300)
