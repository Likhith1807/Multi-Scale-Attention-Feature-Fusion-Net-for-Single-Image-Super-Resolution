@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" train.py --run runs/overfit_one_patch_5060 --train-limit 1 --valid-limit 1 --fixed-patch --micro-batch 1 --accumulate 1 --steps 200 --validate-every 100
if errorlevel 1 goto failed
echo Completed successfully.
pause
exit /b 0
:failed
echo STOPPED: read the error above. Copy the output when asking for help.
pause
exit /b 1
