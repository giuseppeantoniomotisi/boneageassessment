@echo off
REM This batch file installs Python dependencies listed in requirements.txt
REM Make sure requirements.txt is in the same directory as this batch file

REM Change directory to the location of requirements.txt
cd /d %~dp0

REM Install dependencies using pip
pip install -r requirements.txt

REM Pause to keep the command prompt window open
pause

