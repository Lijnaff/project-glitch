@echo off
REM Project Glitch Dashboard — Auto-start on boot
REM Runs the FastAPI backend on port 8000

cd /d C:\Users\Naff\project-glitch
set PYTHONPATH=C:\Users\Naff\project-glitch

C:\Users\Naff\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
