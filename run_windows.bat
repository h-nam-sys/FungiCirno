@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Create the environment first using setup_windows_amd.ps1 or setup_windows_cpu.ps1
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m streamlit run app.py
