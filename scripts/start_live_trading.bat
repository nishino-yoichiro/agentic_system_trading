@echo off
echo Starting Live Trading Log System...
CALL .\enh_venv\Scripts\activate.bat
python run_live_trading.py
PAUSE
