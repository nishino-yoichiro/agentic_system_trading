# 🚀 Enhanced Crypto Trading Pipeline

A production-ready crypto trading system with live signal generation, backtesting, and portfolio management.

## 📁 Project Structure

```
├── src/                    # Core source code
│   ├── crypto_*.py        # Trading strategies and signal generation
│   ├── data_ingestion/    # Data collection modules
│   ├── feature_engineering/ # Technical indicators and NLP
│   ├── simulation/        # Portfolio simulation
│   └── trading_logic/     # Core trading logic
├── scripts/               # Executable scripts
│   ├── live_trading/      # Live trading system
│   ├── dashboards/        # Web dashboards
│   ├── backtesting/       # Backtesting tools
│   └── data_collection/   # Data collection scripts
├── data/                  # Data storage
├── config/               # Configuration files
└── main.py               # Main entry point
```

## 🚀 Quick Start

### Live Trading System
```bash
python main.py live-trading
```

### Dashboard
```bash
python main.py dashboard
```

### Backtesting
```bash
python main.py backtest
```

### Data Collection
```bash
python main.py data-collection
```

## 📊 Features

- **Live Signal Generation**: Conservative strategies with regime filtering
- **Real-time Trading**: Automatic BUY/SELL/HOLD signals every minute
- **Portfolio Management**: Simulated PnL tracking and position sizing
- **Multi-Asset Support**: BTC, ETH, and extensible to other cryptos
- **Advanced Backtesting**: Walk-forward testing with correlation control
- **Web Dashboards**: TradingView integration and real-time charts
- **Continuous Data**: Price and news collection

## 🔧 Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys in `.env` file

3. Run desired command from main.py

## 📈 Usage Examples

**Start live trading:**
```bash
python main.py live-trading
```

**View dashboard:**
```bash
python main.py dashboard
# Opens at http://localhost:8501
```

**Run backtest:**
```bash
python main.py backtest
```

## 🎯 Core Components

- **Signal Generation**: 6 orthogonal crypto strategies
- **Risk Management**: Kelly sizing, correlation control, regime filtering
- **Data Collection**: Continuous price and news data
- **Portfolio Tracking**: Real-time PnL and position management
- **Web Interface**: Streamlit dashboards with TradingView charts