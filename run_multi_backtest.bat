@echo off
echo Starting Multi-Symbol Crypto Backtester...
CALL .\enh_venv\Scripts\activate.bat
python run_multi_backtest.py
PAUSE
