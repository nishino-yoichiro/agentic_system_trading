@echo off
echo Starting Unified Crypto Dashboard...
CALL .\enh_venv\Scripts\activate.bat
python run_crypto_dashboard.py %*
PAUSE
