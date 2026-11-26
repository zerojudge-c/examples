@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo CARLA V6 - One Click Test
echo Ego ID is set to 112. Edit this file if it changes.
echo ==========================================
py -3.10 carla_v6_oneclick.py --ego-id 112
echo.
echo CARLA V6 test finished. Press any key to close.
pause >nul
