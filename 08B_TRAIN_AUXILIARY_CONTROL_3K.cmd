@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" train.py --experiment auxiliary --run runs/research_auxiliary_3k_s2026 --steps 3000 --validate-every 500 --workers 2 --gate-warmup 500 --gate-ramp 500
if errorlevel 1 goto failed
echo Matched auxiliary control completed successfully.
pause
exit /b 0
:failed
echo STOPPED: read the error above. Copy the output when asking for help.
pause
exit /b 1
