@echo off
setlocal
cd /d "%~dp0"
py -3.14 -m venv .venv5060
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" -m pip freeze > requirements-lock-5060.txt
if errorlevel 1 goto failed
echo Completed successfully.
pause
exit /b 0
:failed
echo STOPPED: read the error above. Copy the output when asking for help.
pause
exit /b 1
