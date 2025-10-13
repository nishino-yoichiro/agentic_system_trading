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

### Prerequisites

1. **Python 3.9+** installed
2. **API Keys** configured (see Setup section)

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd enhanced_crypto_pipeline
pip install -r requirements.txt
```

2. **Configure API keys:**
```bash
python scripts/setup_api_keys.py
```

3. **Initialize historical data:**
```bash
python main.py initialize-data
```

### Usage

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

**Start data collection:**
```bash
python main.py data-collection
```

**Monitor live trading:**
```bash
python main.py monitor
```

## 📊 Core Features

### Trading Strategies (8 Active Strategies)
- **BTC NY Session Strategy** - Buy/sell at NY market open/close
- **Liquidity Sweep Reversal** - Detect liquidity hunts beyond swing points
- **Volume Weighted Trend Continuation** - Trade trends with volume confirmation
- **Volatility Expansion Breakout** - Trade compression/expansion patterns
- **Daily AVWAP Z-Score Reversion** - Fade intraday overextensions
- **Opening Range Break & Retest (ORB-R)** - Trade retests after clean breaks
- **Keltner Exhaustion Fade** - Fade extreme pushes beyond bands
- **Fakeout Reversion** - Trade false breakouts from consolidation

### Data Collection & Storage
- **Multi-source price data**: Coinbase Advanced API, WebSocket feeds
- **News collection**: Real-time news aggregation with sentiment analysis
- **Storage formats**: Parquet files, JSON cache, CSV logs
- **Multi-timeframe support**: 1m, 5m, 15m, 1h candles
- **9 crypto symbols**: BTC, ETH, ADA, AVAX, DOT, LINK, MATIC, SOL, UNI

### Signal Generation & Analysis
- **Real-time signal generation** every minute
- **Regime detection**: Bull/bear markets, high/low volatility
- **Sentiment integration**: News sentiment affects signal confidence
- **Risk management**: Kelly sizing, correlation control, position limits
- **Portfolio tracking**: Simulated PnL, position management

### Backtesting Framework
- **Multi-symbol backtesting** with walk-forward analysis
- **Parameter optimization** with sweep testing
- **Portfolio-level analysis** with rebalancing
- **Performance metrics**: Sharpe ratio, max drawdown, win rate, profit factor
- **Comparative reporting** with charts and PDFs

### Live Trading System
- **Integrated live trading** with real-time data collection
- **Portfolio state tracking** with JSON persistence
- **Trade logging** to CSV files
- **Signal monitoring** and execution
- **Risk controls** and position sizing

### Dashboard & Visualization
- **Streamlit dashboard** with TradingView integration
- **Real-time charts** and signal display
- **Portfolio monitoring** with live updates
- **Backtest results visualization**
- **Trading log dashboard**

## 🔧 Setup

### 1. API Keys Configuration

Create `config/api_keys.yaml`:
```yaml
coinbase:
  api_key: "your_coinbase_api_key"
  api_secret: "your_coinbase_api_secret"
  passphrase: "your_coinbase_passphrase"

newsapi:
  api_key: "your_newsapi_key"
```

### 2. Environment Variables (Optional)

Create `.env` file:
```
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret
COINBASE_PASSPHRASE=your_passphrase
NEWSAPI_API_KEY=your_news_api_key
```

### 3. Data Initialization

```bash
# Initialize historical data (required for first run)
python main.py initialize-data

# Start data collection
python main.py data-collection
```

## 📈 Usage Examples

### Live Trading
```bash
# Start live trading with default strategies
python main.py live-trading

# Start with specific strategies
python main.py live-trading --strategies btc_ny_session liquidity_sweep_reversal

# Start with specific symbols
python main.py live-trading --symbols BTC ETH ADA
```

### Backtesting
```bash
# Run backtest with default settings
python main.py backtest

# Run backtest with specific parameters
python main.py backtest --symbols BTC ETH --days 60 --strategies btc_ny_session
```

### Dashboard
```bash
# Start Streamlit dashboard
python main.py dashboard
# Navigate to http://localhost:8501
```

### Data Collection
```bash
# Start WebSocket data collection
python main.py data-collection

# Collect data for specific symbols
python main.py data-collection --symbols BTC ETH ADA
```

## 🎯 Core Components

### Signal Generation
- **8 orthogonal crypto strategies** with different mechanisms
- **Risk management**: Kelly sizing, correlation control, regime filtering
- **Real-time processing**: Signal generation every minute
- **Sentiment integration**: News sentiment affects signal confidence

### Data Collection
- **Historical data**: Coinbase Advanced API with rate limiting
- **Real-time data**: WebSocket feeds for live price updates
- **News data**: Real-time news collection with sentiment analysis
- **Data validation**: Automatic data quality checks and gap filling

### Portfolio Management
- **Simulated trading**: Paper trading with real-time PnL tracking
- **Position sizing**: Kelly criterion with risk controls
- **Correlation management**: Prevents over-concentration in correlated strategies
- **Regime filtering**: Adapts strategy allocation based on market conditions

### Risk Management
- **Position limits**: Maximum 10% per position, 50% in crypto
- **Stop losses**: 2-5% stop losses per strategy
- **Drawdown controls**: Maximum 15% portfolio drawdown
- **Correlation limits**: Maximum 0.5 correlation between strategies

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Total Return**: Cumulative portfolio return
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Calmar Ratio**: Annual return / maximum drawdown
- **VaR/CVaR**: Value at Risk and Conditional VaR

## 🔧 Configuration

### Strategy Configuration (`config/strategies.yaml`)
```yaml
strategies:
  btc_ny_session:
    enabled: true
    weight: 0.25
    parameters:
      ny_open_hour: 9
      ny_open_minute: 30
      stop_loss_pct: 0.05
      take_profit_pct: 0.05
```

### Pipeline Configuration (`config/pipeline_config.yaml`)
```yaml
data_collection:
  hours_back: 24
  crypto_granularity: 60
  crypto_data_source: "coinbase_advanced"

trading:
  max_positions: 10
  confidence_threshold: 0.3
  risk_tolerance: "medium"
```

## 🚨 Risk Disclaimer

**This software is for educational and research purposes only. It is not financial advice.**

- **Paper trading only**: This system performs simulated trading
- **No real money**: Never use real funds with this system
- **Market risks**: Cryptocurrency markets are highly volatile
- **Technical risks**: Software bugs, API failures, data issues
- **Regulatory risks**: Trading regulations vary by jurisdiction

**Always do your own research and never invest more than you can afford to lose.**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Verify API keys are configured correctly
3. Ensure sufficient historical data is available
4. Check network connectivity for WebSocket feeds

## 🔄 Updates

The system is actively maintained with regular updates:
- New trading strategies
- Enhanced risk management
- Improved data collection
- Better performance metrics
- Bug fixes and optimizations