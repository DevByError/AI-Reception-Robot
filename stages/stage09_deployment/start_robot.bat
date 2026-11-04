@echo off
rem AI-Reception-Robot - start the robot and restart it if it ever crashes (Windows).
rem
rem   Double-click, or put a shortcut to this file in the Startup folder (Win+R, shell:startup)
rem   so it starts when the robot's computer logs in.
rem
rem   The robot's face fills the screen and the camera debug window is hidden. For a robot with
rem   no screen at all, change FACE_FULLSCREEN below to "set HEADLESS=true".
rem   Stop it with Ctrl+C in this window (a clean exit is not restarted).

cd /d "%~dp0\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

set CAMERA_PREVIEW=false
set FACE_FULLSCREEN=true

:run
python stages\stage08_orchestrator\main_live.py
if %ERRORLEVEL% EQU 0 goto :eof
echo [watchdog] the robot stopped with error %ERRORLEVEL% - restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto run
