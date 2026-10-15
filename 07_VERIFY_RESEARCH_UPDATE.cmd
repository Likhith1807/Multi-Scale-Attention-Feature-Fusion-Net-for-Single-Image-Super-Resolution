@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" self_test.py
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" benchmark.py --experiment error_supervised --warmup 20 --steps 100 --output runs/benchmark_error_supervised_5060.json
if errorlevel 1 goto failed
echo Research paths and training cost verified successfully.
pause
exit /b 0
:failed
echo STOPPED: read the error above. Copy the output when asking for help.
pause
exit /b 1
