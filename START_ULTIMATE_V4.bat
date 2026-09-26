@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================
echo AI Model Lab V4.1 - Ultimate Framework
echo ==============================================================
echo Backend: http://127.0.0.1:8765
echo API docs: http://127.0.0.1:8765/docs
echo.

powershell -NoProfile -Command "$c=New-Object Net.Sockets.TcpClient; try{$c.Connect('127.0.0.1',8765);$c.Close();exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 (
  echo Backend gia attivo sulla porta 8765. Non avvio una seconda istanza.
  start "AI Model Lab" http://127.0.0.1:8765
  echo.
  echo Premi un tasto per chiudere questa finestra. Il backend gia attivo restera in esecuzione.
  pause >nul
  exit /b 0
)

start "AI Model Lab" http://127.0.0.1:8765
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8765
pause
