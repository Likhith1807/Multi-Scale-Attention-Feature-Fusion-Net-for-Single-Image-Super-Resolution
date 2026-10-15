@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" download_div2k.py --offline
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" verify_data.py
if errorlevel 1 goto failed
echo Completed successfully.
pause
exit /b 0
:failed
echo STOPPED: read the error above. Copy the output when asking for help.
pause
exit /b 1
