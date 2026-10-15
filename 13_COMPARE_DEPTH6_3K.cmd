@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" compare_runs.py --baseline runs/research_baseline_3k_s2026 --auxiliary runs/research_depth6_auxiliary_3k_s2026 --proposed runs/research_depth6_error_3k_s2026 --output runs/research_depth6_3k_comparison.json
if errorlevel 1 goto failed
echo Matched depth-6 comparison completed successfully.
pause
exit /b 0
:failed
echo STOPPED: the runs may be incomplete or mismatched. Copy the error output.
pause
exit /b 1
