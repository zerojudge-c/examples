@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Restoring the frozen Ego / NPC after the V6 cinematic test...
py -3.10 carla_cinematic_rear_10s_v6.py --ego-id 112 --resume
echo.
pause
