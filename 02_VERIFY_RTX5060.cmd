@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" check_environment.py --output runs/environment_5060.json
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" self_test.py
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" benchmark.py --output runs/benchmark_5060.json
if errorlevel 1 goto failed
echo Completed successfully.
pause
exit /b 0
:failed
echo STOPPED: read the error above. Copy the output when asking for help.
pause
exit /b 1
