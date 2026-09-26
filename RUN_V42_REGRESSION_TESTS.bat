@echo off
cd /d "%~dp0"
echo ==============================================================
echo AI Model Lab V4.2.0 - Validation Lab Regression Tests
echo ==============================================================
python -m backend.regression_test_v41
pause
