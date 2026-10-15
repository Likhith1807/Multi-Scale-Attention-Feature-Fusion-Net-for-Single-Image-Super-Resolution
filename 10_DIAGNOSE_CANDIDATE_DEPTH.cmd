@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" self_test.py
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" diagnose_candidate_depth.py
if errorlevel 1 goto failed
echo Candidate-depth diagnostic completed successfully.
pause
exit /b 0
:failed
echo STOPPED: read the error above. Copy the output when asking for help.
pause
exit /b 1
