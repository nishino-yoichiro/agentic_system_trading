# 🚀 Enhanced Crypto Pipeline Operations Guide

## 🎯 Clean Architecture

```
├── src/                    # Core source code
├── scripts/               # Executable scripts  
├── config/               # Configuration files
├── data/                 # Data storage
└── main.py              # Main entry point
```

## 🚀 Quick Start

### Main Entry Point
```bash
python main.py live-trading    # Live trading system
python main.py dashboard       # Web dashboard
python main.py backtest        # Backtesting
python main.py data-collection # Data collection
```

### Direct Scripts
```bash
python scripts/live_trading/run_live_trading.py
streamlit run scripts/dashboards/streamlit_app.py
python scripts/backtesting/run_multi_backtest.py
python scripts/data_collection/start_btc_collection.py
```

## 📊 Core Features

1. **Live Trading**: Signal generation every minute, portfolio tracking
2. **Dashboards**: Streamlit with TradingView charts
3. **Backtesting**: Multi-symbol framework with walk-forward testing
4. **Data Collection**: Continuous price and news collection

## 🔧 Setup

1. `pip install -r requirements.txt`
2. `python scripts/setup_api_keys.py`
3. `python main.py live-trading`

## 📈 Usage

- **Live Trading**: `python main.py live-trading`
- **Dashboard**: `python main.py dashboard` (http://localhost:8501)
- **Backtest**: `python main.py backtest`
- **Data**: `python main.py data-collection`
