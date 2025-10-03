# 🚀 Usage Guide

## Quick Commands

### Live Trading System
```bash
python main.py live-trading
```
- Generates signals every 1 minute
- Tracks simulated PnL
- Logs all trades to CSV

### Dashboard
```bash
python main.py dashboard
```
- Opens Streamlit dashboard at http://localhost:8501
- Shows TradingView charts
- Real-time signal display

### Backtesting
```bash
python main.py backtest
```
- Runs multi-symbol backtest
- Shows performance metrics
- Generates equity curves

### Data Collection
```bash
python main.py data-collection
```
- Starts continuous price collection
- Collects news data
- Updates data cache

## Alternative Scripts

### Direct Script Execution
```bash
# Live trading
python scripts/live_trading/run_live_trading.py

# Dashboard
streamlit run scripts/dashboards/streamlit_app.py

# Backtest
python scripts/backtesting/run_multi_backtest.py

# Data collection
python scripts/data_collection/start_btc_collection.py
```

### Batch Files (Windows)
```bash
scripts/start_live_trading.bat
scripts/run_crypto_dashboard.bat
scripts/run_multi_backtest.bat
```

## Project Structure

```
├── src/                    # Core modules
├── scripts/               # Executable scripts
├── config/               # Configuration
├── data/                 # Data storage
└── main.py              # Main entry point
```

## Configuration

1. Set up API keys in `.env` file
2. Configure strategies in `config/strategies.yaml`
3. Run desired command

## Features

- **6 Crypto Strategies**: Liquidity sweeps, breakouts, mean reversion, etc.
- **Risk Management**: Kelly sizing, correlation control, regime filtering
- **Real-time Data**: Continuous price and news collection
- **Web Interface**: Streamlit dashboards with TradingView
- **Portfolio Tracking**: Simulated PnL and position management
