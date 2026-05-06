@echo off
title DigitAI — Handwritten Digit Recognition
color 0A
echo.
echo  ===============================================
echo    DigitAI — Handwritten Digit Recognition
echo    Powered by PyTorch CNN + FastAPI
echo  ===============================================
echo.

REM Step 1: Install dependencies
echo  [1/3]  Installing Python dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  ERROR: pip install failed. Please check your Python installation.
    pause
    exit /b 1
)
echo         Done!
echo.

REM Step 2: Train the model
echo  [2/3]  Training CNN model on MNIST dataset...
echo         (This will take 3-10 minutes on first run)
echo.
python train_model.py
if errorlevel 1 (
    echo  ERROR: Model training failed. See output above.
    pause
    exit /b 1
)
echo.
echo         Model trained successfully!
echo.

REM Step 3: Start the API server and open browser
echo  [3/3]  Starting DigitAI server...
echo         Opening: http://localhost:8000
echo.
start /B python api.py

timeout /t 3 /nobreak > nul
start http://localhost:8000

echo.
echo  -----------------------------------------------
echo   Server is running at http://localhost:8000
echo   Press Ctrl+C to stop the server
echo  -----------------------------------------------
echo.
pause
