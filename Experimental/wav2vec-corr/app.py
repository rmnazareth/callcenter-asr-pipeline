import streamlit as st
import os
from pipeline import SinhalaASRPipeline
import tempfile

# Page configuration
st.set_page_config(
    page_title="Sinhala ASR with Correction",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.title("🎙️ Sinhala Speech-to-Text with Auto-Correction")
st.markdown("""
This application transcribes Sinhala audio and automatically corrects the transcription using AI models:
- **Transcription**: wav2vec2-xls-r-300m-sinhala-general-185k
- **Correction**: MT-5-Sinhala-Wikigen
""")

# Initialize the pipeline (cached to avoid reloading)

@st.cache_resource
def load_pipeline():
    return SinhalaASRPipeline()


# Load models
with st.spinner("Loading models... This may take a few moments."):
    pipeline = load_pipeline()

st.success("✅ Models loaded successfully!")

# File upload
st.header("Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose an audio file (WAV, MP3, FLAC, etc.)",
    type=["wav", "mp3", "flac", "ogg", "m4a"]
)

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file,
             format=f"audio/{uploaded_file.type.split('/')[-1]}")

    # Process button
    if st.button("🎯 Transcribe and Correct", type="primary"):
        with st.spinner("Processing audio..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Process audio
                results = pipeline.process(tmp_path)

                # Display results
                st.header("📝 Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Raw Transcription")
                    st.info(results["raw_transcription"])

                with col2:
                    st.subheader("Corrected Transcription")
                    st.success(results["corrected_transcription"])

                # Download buttons
                st.header("💾 Download Results")
                col3, col4 = st.columns(2)

                with col3:
                    st.download_button(
                        label="Download Raw Transcription",
                        data=results["raw_transcription"],
                        file_name="raw_transcription.txt",
                        mime="text/plain"
                    )

                with col4:
                    st.download_button(
                        label="Download Corrected Transcription",
                        data=results["corrected_transcription"],
                        file_name="corrected_transcription.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    ### How it works:
    1. Upload your Sinhala audio file
    2. Click 'Transcribe and Correct'
    3. Get both raw and corrected transcriptions
    
    ### Supported Formats:
    - WAV
    - MP3
    - FLAC
    - OGG
    - M4A
    
    ### Models Used:
    - **ASR Model**: janiduchamika/wav2vec2-xls-r-300m-sinhala-general-185k
    - **Correction Model**: Suchinthana/MT-5-Sinhala-Wikigen
    """)

    st.markdown("---")
