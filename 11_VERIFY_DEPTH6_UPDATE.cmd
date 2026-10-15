@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" self_test.py
if errorlevel 1 goto failed
".venv5060\Scripts\python.exe" benchmark.py --experiment error_supervised --research-depth 6 --temperature 0.05 --warmup 20 --steps 100 --output runs/benchmark_depth6_error_temp005_5060.json
if errorlevel 1 goto failed
echo Depth-6 implementation and training cost verified successfully.
pause
exit /b 0
:failed
echo STOPPED: do not start depth-6 training. Copy the complete error output.
pause
exit /b 1
