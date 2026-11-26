@echo off
cd /d "%~dp0"
py -3.10 carla_cinematic_rear_10s_v6.py --ego-id 112 --resume
pause
