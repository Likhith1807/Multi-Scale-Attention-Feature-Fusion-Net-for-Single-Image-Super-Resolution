@echo off
setlocal
cd /d "%~dp0"
".venv5060\Scripts\python.exe" evaluate.py --checkpoint runs/baseline_x4_pilot_5060/best.pt --lr-dir data/DIV2K_valid_LR_bicubic/X4 --hr-dir data/DIV2K_valid_HR --limit 10 --output runs/pilot_full_image_eval_5060
if errorlevel 1 goto failed
echo Completed successfully.
pause
exit /b 0
:failed
echo STOPPED: read the error above. Copy the output when asking for help.
pause
exit /b 1
