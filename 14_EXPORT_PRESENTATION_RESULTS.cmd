@echo off
setlocal
cd /d "%~dp0"

".venv5060\Scripts\python.exe" export_presentation_results.py --overwrite
if errorlevel 1 goto failed

echo.
echo Presentation pictures are ready in:
echo runs\presentation_results_5060
echo.
echo Open that folder and use the PNG files in your PowerPoint.
pause
exit /b 0

:failed
echo.
echo STOPPED: read the error above and copy it when asking for help.
pause
exit /b 1
