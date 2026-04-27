#  Sinhala ASR – Local Hosting (Fine-Tuned Whisper)

This project demonstrates how to deploy a fine-tuned Whisper ASR model for Sinhala speech-to-text transcription, running entirely offline using FastAPI and Uvicorn.  

The setup enables local inference without any dependency on external cloud APIs.

#1️ Environment Setup

Create & Activate Virtual Environment
Open VS Code or Command Prompt in your project folder:
cd C:\Users\Documents\whisper_local_project

Create the virtual environment:
python -m venv venv

Activate it:
venv\Scripts\activate

# 2️ Install Dependencies

Install all required Python packages:
pip install torch==2.8.0 torchaudio==2.8.0 transformers soundfile fastapi uvicorn python-multipart

#3 Testing

1. Transfer the project folder to your computer.

2. Your project folder should look like this.
-venv\			# Virtual environment
-app.py			# FastAPI server
-transcribe.py		# Transcription script
-test3.wav		# Test audio file
-start_server.bat	# Server automation script
-model\			# Fine-tuned whisper model
-static\			# Frontend UI

3. Double-click on the start_server.bat file. This will open in your command prompt and display the URL.

4. Ctrl+click on the URL, and the chatbot will open in your browser.

5. Now you can upload the sample .wav file to test the chatbot.

