## Operations Guide

This page lists the exact commands to:
- Add new price data without rewriting historical
- Set up historical data (one-time)
- Generate reports (PDF + BTC Daily Brief)
- Serve/view reports

Run all commands from the repo root: `enhanced_crypto_pipeline`.

### 0) Activate environment (Windows PowerShell)
```powershell
cd C:\Users\yoich\Documents\personal_projects\crypto_agent_pipeline\enhanced_crypto_pipeline
./enh_venv/Scripts/Activate.ps1
```

### 1) Add new price data WITHOUT rewriting historical
Incremental updates append only new bars by comparing timestamps and keep your `data/raw/prices_*.parquet` intact.
```powershell
python run_enhanced_pipeline.py
```
- What happens: crypto minute data is refreshed using Coinbase Advanced API; stock data is incrementally updated; news is refreshed if stale; parquet files are merged without duplicates.
- Where data goes: `data/raw/prices_<SYMBOL>.parquet`, metadata in `data/collection_metadata.json`.

To run continuously (e.g., every 60 seconds):
```powershell
python run_enhanced_pipeline.py --continuous --interval 60
```

To target a specific ticker (e.g., BTC only):
```powershell
python run_enhanced_pipeline.py --continuous --ticker BTC --interval 20
```

### 1.1) BTC-Only Collection (Simplified)
For BTC-only data collection using Coinbase Advanced API:
```powershell
python start_btc_collection.py
```
Or use the batch file:
```powershell
./start_btc_collection.bat
```

### 2) One-time historical setup (bulk backfill)
This populates initial history. If it already exists for the requested range, it is skipped automatically.
```powershell
python run_enhanced_pipeline.py --setup-historical --days-back 365
```
- Skip logic: if `data/collection_metadata.json` shows a prior bulk load with `days_back >= 365`, the script skips rewriting.

### 3) Generate the comprehensive daily PDF report
This collects fresh data (incremental), builds features, runs simulations, and writes a PDF under `reports/`.
```powershell
python run_enhanced_pipeline.py
```
- Output: `reports/daily_report_YYYYMMDD_HHMMSS.pdf` (the path is printed at the end).

### 4) Generate the BTC Daily Brief (HTML)
Creates `reports/btc_briefs/*.html` artifacts.
```powershell
python reports/btc_daily_brief.py
```
- Data read from: `data/crypto_db/BTC_historical.parquet` and latest cached indicators/news if present.

Optional helper (batch file):
```powershell
./start_btc_brief.bat
```

### 5) BTC Dashboard (Recommended)
Professional TradingView charts + our analysis:
```powershell
python btc_dashboard.py
```
- Opens at: http://localhost:8080
- Auto-refreshes every 5 minutes
- Professional TradingView charts + our signals
- Mobile-friendly design

### 5.1) BTC Backtesting
Test trading strategy performance on historical data with customizable parameters:
```powershell
# Default: 3 days, 24h lookback, $10,000 capital
python btc_backtester.py

# Custom parameters
python btc_backtester.py --days 7 --lookback 48 --capital 25000

# Verbose output (shows every trade)
python btc_backtester.py --days 1 --lookback 12 --verbose
```
- **Parameters:**
  - `--days`: Number of days to backtest (default: 3)
  - `--lookback`: Lookback hours for signal generation (default: 24)
  - `--capital`: Initial capital amount (default: 10000)
  - `--verbose, -v`: Show detailed trade log with every buy/sell
- **Output:** Terminal summary + files in `backtests/results/`
- **Files:** Timestamped PNG charts and markdown summaries
- **Shows:** Trades, win rate, PnL, max drawdown, timeframe
- **Verbose:** Complete trade log with timestamps, prices, amounts, PnL

### 5.2) BTC Sentiment-Enhanced Backtesting
Compare original strategy vs sentiment-enhanced strategy:
```powershell
# Default: 3 days, 24h lookback, $10,000 capital, α=0.5
python btc_sentiment_backtester.py

# Custom parameters with sentiment weight
python btc_sentiment_backtester.py --days 1 --lookback 12 --capital 5000 --alpha 0.8
```
- **Parameters:**
  - `--days`: Number of days to backtest (default: 3)
  - `--lookback`: Lookback hours for signal generation (default: 24)
  - `--capital`: Initial capital amount (default: 10000)
  - `--alpha`: Sentiment multiplier weight (default: 0.5)
- **Output:** Comparison chart + markdown report in `backtests/results/`
- **Files:** `btc_sentiment_comparison_{days}d_{lookback}h_{timestamp}.png/md`
- **Shows:** Original vs Sentiment-Enhanced performance metrics
- **Formula:** Enhanced Signal = Base Signal × (1 + α × Sentiment)

### 6) Serve a simple BTC brief (optional)
Simple HTTP server variants exist; use either:
```powershell
python serve_btc_brief_simple.py
# or
python serve_btc_brief.py
```

### 7) Useful utilities
- Generate synthetic minute history for testing:
```powershell
python tests/generate_crypto_historical.py
```
- Collect real BTC/ETH history via public sources (fallbacks to Yahoo if needed):
```powershell
python tests/collect_real_crypto_public.py
```

### 8) Where things are stored
- Incremental and bulk price data: `data/raw/prices_<SYMBOL>.parquet`
- News cache: `data/raw/news.parquet`
- Incremental merge logic: `data_ingestion/incremental_collector.py`
- Reports (PDF): `reports/daily_report_*.pdf`
- BTC brief HTML: `reports/btc_briefs/`
- Logs: `logs/enhanced_pipeline.log`

### 9) Common recipes
- Refresh data now without changing history, and produce a fresh report:
```powershell
python run_enhanced_pipeline.py
```
- Run continuous incremental updates in the background (Ctrl+C to stop):
```powershell
python run_enhanced_pipeline.py --continuous --interval 30
```

### Notes
- Incremental merging ensures only unseen timestamps are appended; duplicates are dropped on `timestamp`.
- If you ever want to expand the historical window later, rerun bulk setup with a larger `--days-back`; otherwise keep using incremental.


