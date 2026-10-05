@echo off
cd /d "%~dp0"
title VisionVoice - YOLOv3 Object Detection with Voice
cls
echo ================================================================
echo    VisionVoice: YOLOv3 Object Detection with Voice Feedback
echo ================================================================
echo.
echo   [1] Start Live Webcam Detection (Recommended)
echo   [2] Launch Desktop GUI Application (Visual Controls)
echo   [3] Run Detection on a Single Image
echo   [4] Run Detection on a Video File
echo   [5] Install / Verify Requirements
echo   [6] Exit
echo.
set /p opt="Select an option [1-6] (Default is 1): "

if "%opt%"=="" set opt=1
if "%opt%"=="1" goto webcam
if "%opt%"=="2" goto gui
if "%opt%"=="3" goto image
if "%opt%"=="4" goto video
if "%opt%"=="5" goto install
if "%opt%"=="6" goto end

:webcam
echo.
echo [Starting Live Webcam Stream with Voice Feedback...]
python Main.py
goto end

:gui
echo.
echo [Launching Desktop GUI Application...]
python App.py
goto end

:image
echo.
set /p imgpath="Enter path to image file: "
python Main.py --image "%imgpath%"
goto end

:video
echo.
set /p vidpath="Enter path to video file: "
python Main.py --video "%vidpath%"
goto end

:install
echo.
echo [Installing requirements...]
python -m pip install -r requirements.txt
echo [Done!]
pause
goto end

:end
pause