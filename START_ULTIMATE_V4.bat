@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "EXPECTED_VERSION=4.1.3"
set "PORT=8765"

echo ==============================================================
echo AI Model Lab V4.1.3 - Ultimate Framework
echo ==============================================================
echo Backend: http://127.0.0.1:%PORT%
echo API docs: http://127.0.0.1:%PORT%/docs
echo Project: %CD%
echo.

powershell -NoProfile -Command "$ErrorActionPreference='SilentlyContinue'; try { $h=Invoke-RestMethod 'http://127.0.0.1:%PORT%/api/advanced/health' -TimeoutSec 2; if($h.ok){ Write-Output ($h.app_version + '|' + $h.project_root); exit 0 } } catch {}; exit 1" > "%TEMP%\aiml_health.txt"
set "HEALTH_OK=%errorlevel%"

if "%HEALTH_OK%"=="0" (
  set /p HEALTH_LINE=<"%TEMP%\aiml_health.txt"
  for /f "tokens=1,* delims=|" %%A in ("%HEALTH_LINE%") do (
    set "RUNNING_VERSION=%%A"
    set "RUNNING_ROOT=%%B"
  )

  if /I "%RUNNING_VERSION%"=="%EXPECTED_VERSION%" (
    echo Backend V%RUNNING_VERSION% gia attivo.
    echo Root: %RUNNING_ROOT%
    start "AI Model Lab" http://127.0.0.1:%PORT%
    echo.
    echo Premi un tasto per chiudere questa finestra. Il backend restera in esecuzione.
    pause >nul
    exit /b 0
  )

  echo ATTENZIONE: backend AI Model Lab precedente rilevato sulla porta %PORT%.
  if defined RUNNING_VERSION (
    echo Versione attiva: %RUNNING_VERSION%
  ) else (
    echo Versione attiva: precedente alla V4.1.3
  )
  if defined RUNNING_ROOT echo Root attiva: %RUNNING_ROOT%
  echo Versione richiesta: %EXPECTED_VERSION%
  echo.
  choice /C SN /N /M "Vuoi chiudere il backend precedente e avviare questa versione? [S/N]: "
  if errorlevel 2 exit /b 1

  powershell -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if($c){ Stop-Process -Id $c.OwningProcess -Force; Start-Sleep -Milliseconds 900 }"
) else (
  powershell -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if($c){ Write-Host ('La porta %PORT% e occupata dal PID ' + $c.OwningProcess); exit 2 } else { exit 0 }"
  if errorlevel 2 (
    echo.
    echo La porta %PORT% e occupata da un processo che non risponde come AI Model Lab.
    echo Chiudilo manualmente prima di continuare.
    pause
    exit /b 2
  )
)

del "%TEMP%\aiml_health.txt" >nul 2>&1

echo Avvio backend V%EXPECTED_VERSION%...
start "AI Model Lab" http://127.0.0.1:%PORT%
python -m uvicorn backend.server:app --host 127.0.0.1 --port %PORT%
pause
