@echo off
setlocal
cd /d "%~dp0"
python -m backend.self_test
pause
