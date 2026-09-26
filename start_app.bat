@echo off
REM Double-click to launch the Catalyst Scanner dashboard (opens http://localhost:8501).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app.py
pause
