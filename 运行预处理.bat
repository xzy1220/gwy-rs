@echo off
chcp 65001 >nul
title Data Preprocessing
echo.
echo Running data preprocessing...
echo.
python src\preprocess_data.py
echo.
echo ========================================
echo Finished!
echo ========================================
echo.
pause

