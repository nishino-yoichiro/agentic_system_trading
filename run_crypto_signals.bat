@echo off
echo Starting Crypto Signal Framework...
CALL .\enh_venv\Scripts\activate.bat
python run_crypto_signals.py %*
PAUSE
