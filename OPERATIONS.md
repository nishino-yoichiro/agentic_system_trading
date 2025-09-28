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

### 5.0) BTC Sentiment Dashboard (Advanced)
Interactive dashboard with sentiment analysis controls and real-time parameter adjustment:
```powershell
python btc_sentiment_dashboard.py
```
Or use the batch file:
```powershell
./start_sentiment_dashboard.bat
```
- Opens at: http://localhost:8081
- **Interactive Controls:** Real-time sentiment weight (α) slider (0.0 - 2.0)
- **Live Comparison:** Base technical vs sentiment-enhanced signals side-by-side
- **Sentiment Metrics:** Current sentiment score, multiplier, and impact visualization
- **Reasoning Display:** Detailed explanation of why sentiment affects the signal
- **Auto-refresh:** Updates every 5 minutes with latest news sentiment
- **Formula:** Enhanced Signal = Base Signal × (1 + α × Sentiment)

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

### 10) Unified Multi-Symbol Dashboard (NEW)

#### Quick Start
```powershell
# Default symbols (BTC, ETH, ADA, SOL)
python run_crypto_dashboard.py

# Or use batch file
./run_crypto_dashboard.bat

# Custom symbols
python run_crypto_dashboard.py BTC ETH ADA SOL AVAX
```

#### Features
- **Multi-Symbol Analysis**: Real-time analysis for any combination of crypto assets
- **Sentiment Integration**: Interactive sentiment weight control with live updates
- **News Visualization**: Filterable news articles with sentiment analysis
- **Portfolio View**: Side-by-side comparison of all selected symbols
- **Auto-Refresh**: Automatic updates every 5 minutes
- **Mobile-Friendly**: Responsive design for all devices

**Available Symbols:** BTC, ETH, ADA, AVAX, DOT, LINK, MATIC, SOL, UNI

**Dashboard URL:** http://localhost:8080

### 11) Continuous News Collection (NEW)

#### Quick Start
```powershell
# Default: Collect news every hour for all crypto symbols
python continuous_news_collector.py

# Or use batch file
./start_news_collection.bat

# Custom symbols and interval
python continuous_news_collector.py --symbols BTC ETH ADA SOL --interval 2
```

#### Features
- **Continuous Collection**: Runs in background, collecting news every hour (configurable)
- **Multi-Symbol Support**: Collects news for all crypto symbols simultaneously
- **Intelligent Caching**: Appends new articles without duplicates
- **Comprehensive Sources**: NewsAPI, CryptoCompare, CoinDesk RSS, Reddit
- **Statistics Tracking**: Detailed collection stats and success rates
- **Graceful Shutdown**: Ctrl+C to stop cleanly

**Parameters:**
- `--symbols`: Space-separated list of symbols (default: BTC, ETH, ADA, SOL, AVAX, DOT, LINK, MATIC, UNI)
- `--interval`: Collection interval in hours (default: 1)
- `--data-dir`: Data directory (default: data)

**Output:**
- News data: `data/raw/news.parquet`
- Collection stats: `data/logs/news_collection_stats.json`
- Logs: `logs/continuous_news_collection.log`

**Usage Examples:**
```powershell
# Collect news every 30 minutes for BTC and ETH only
python continuous_news_collector.py --symbols BTC ETH --interval 0.5

# Collect news every 2 hours for all crypto symbols
python continuous_news_collector.py --interval 2

# Run in background (Windows)
start /B python continuous_news_collector.py
```

### 12) Advanced Portfolio Management System (NEW)

#### Quick Start
```powershell
# Run the complete advanced portfolio system demo
python run_advanced_portfolio.py

# Test the system components
python test_advanced_portfolio.py

# Integrate with existing crypto pipeline
python integrate_advanced_portfolio.py
```

#### Features
- **Low-Correlation Strategy Basket**: 5 orthogonal strategies across different instruments and mechanisms
- **Kelly Sizing**: Optimal position sizing based on expected returns and correlations
- **Regime Filtering**: Strategies only trade in favorable market conditions
- **Volatility Targeting**: All strategies normalized to 10% annualized volatility
- **Correlation Control**: Automatic de-weighting of highly correlated strategies
- **Risk Controls**: Portfolio-level kill switches and drawdown limits
- **Walk-Forward Backtesting**: Out-of-sample testing with parameter re-estimation

#### Strategy Basket
1. **ES Sweep/Reclaim** - Intraday futures strategy for NY open
2. **NQ Breakout Continuation** - High-vol regime breakout strategy  
3. **SPY Mean Reversion** - Overnight gap fade in low-vol regimes
4. **EURUSD Carry/Trend** - FX strategy with MA filter
5. **Options IV Crush** - Event-driven volatility strategy

#### Advanced Features
- **Portfolio Optimization**: Risk-parity base with Sharpe tilt and correlation penalty
- **Regime Detection**: ADX-based trend/range detection, volatility classification
- **Correlation Analysis**: Rolling correlation tracking with surge detection
- **Comprehensive Reporting**: Equity curves, allocation heatmaps, regime analysis
- **Crypto Integration**: Seamless integration with existing crypto pipeline

#### Usage Examples
```powershell
# Run with custom parameters
python run_advanced_portfolio.py --capital 200000 --volatility 0.15 --kelly 0.5

# Test specific components
python test_advanced_portfolio.py --test portfolio_manager
python test_advanced_portfolio.py --test strategy_framework

# Generate crypto-specific portfolio
python integrate_advanced_portfolio.py --symbols BTC ETH ADA SOL --days 90
```

**Output Files:**
- `portfolio_reports/` - Comprehensive portfolio analysis
- `crypto_portfolio_reports/` - Crypto-specific analysis
- Detailed JSON results and markdown reports
- Professional visualizations and charts

### 13) Multi-Symbol Backtesting (Universal)

#### Quick Multi-Symbol Backtest
```powershell
# Interactive mode
python run_multi_backtest.py

# Or use batch file
./run_multi_backtest.bat
```

#### Direct Command Line Usage
```powershell
# Basic multi-symbol backtest
python multi_symbol_backtester.py --symbols BTC ETH ADA --days 7 --capital 10000

# With sentiment enhancement
python multi_symbol_backtester.py --symbols BTC ETH ADA --days 7 --capital 10000 --sentiment

# With portfolio analysis and rebalancing
python multi_symbol_backtester.py --symbols BTC ETH ADA SOL --days 7 --capital 10000 --sentiment --portfolio

# Verbose output (shows all trades)
python multi_symbol_backtester.py --symbols BTC ETH --days 3 --capital 5000 --sentiment --verbose
```

**Available Symbols:** BTC, ETH, ADA, AVAX, DOT, LINK, MATIC, SOL, UNI

**Features:**
- Universal backtesting for any combination of crypto assets
- Individual symbol analysis with detailed metrics
- Portfolio-level backtesting with equal-weight rebalancing
- Sentiment-enhanced strategy support
- Comparative visualization and reporting
- Correlation analysis between symbols
- Comprehensive markdown reports with charts

**Parameters:**
- `--symbols`: Space-separated list of symbols to backtest
- `--days`: Number of days to backtest (default: 3)
- `--capital`: Initial capital amount (default: 10000)
- `--alpha`: Sentiment multiplier weight (default: 0.5)
- `--sentiment`: Use sentiment-enhanced strategy
- `--portfolio`: Run portfolio backtest with rebalancing
- `--verbose`: Show detailed trade log
- `--output-dir`: Output directory for results (default: backtests/results)

**Output Files:**
- `backtests/results/multi_symbol_backtest_YYYYMMDD_HHMMSS.png` - Comparative charts
- `backtests/results/multi_symbol_backtest_YYYYMMDD_HHMMSS.md` - Detailed report

### Notes
- Incremental merging ensures only unseen timestamps are appended; duplicates are dropped on `timestamp`.
- If you ever want to expand the historical window later, rerun bulk setup with a larger `--days-back`; otherwise keep using incremental.


