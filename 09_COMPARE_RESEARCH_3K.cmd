@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" compare_runs.py
if errorlevel 1 goto failed
echo Matched development comparison completed successfully.
pause
exit /b 0
:failed
echo STOPPED: read the error above. Copy the output when asking for help.
pause
exit /b 1
