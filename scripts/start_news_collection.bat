@echo off
echo Starting Continuous News Collection Service...
CALL .\enh_venv\Scripts\activate.bat
python continuous_news_collector.py %*
PAUSE
