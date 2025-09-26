@echo off
echo Starting BTC Continuous Data Collection...
echo.
echo This will collect BTC data every 20 seconds
echo Press Ctrl+C to stop
echo.

cd /d "%~dp0"
.\enh_venv\Scripts\python.exe start_btc_collection.py

pause
