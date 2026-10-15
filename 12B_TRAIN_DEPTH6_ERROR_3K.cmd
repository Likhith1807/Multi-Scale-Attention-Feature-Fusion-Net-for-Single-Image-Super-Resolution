@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" train.py --experiment error_supervised --research-depth 6 --temperature 0.05 --run runs/research_depth6_error_3k_s2026 --steps 3000 --validate-every 500 --workers 2 --gate-warmup 500 --gate-ramp 500
if errorlevel 1 goto failed
echo Matched depth-6 error-supervised run completed successfully.
pause
exit /b 0
:failed
echo STOPPED: copy the complete output. Resume only with the documented command.
pause
exit /b 1
