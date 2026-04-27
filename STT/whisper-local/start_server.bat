@echo off
REM go to project folder
cd /d C:\Users\ramee\whisper-local

REM set offline envs so HF libs don't try to go online
set TRANSFORMERS_OFFLINE=1
set HUGGINGFACE_HUB_OFFLINE=1

REM activate venv
call venv\Scripts\activate

REM start server using the local model
uvicorn app:app --host 127.0.0.1 --port 8001

pause
