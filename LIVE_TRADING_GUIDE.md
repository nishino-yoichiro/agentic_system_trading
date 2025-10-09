# Live Trading System Guide

## Overview

The Enhanced Crypto Trading Pipeline now includes a fully integrated live trading system that combines:

1. **Real-time Data Collection**: Builds 1-minute candles from WebSocket price feeds
2. **Sophisticated Signal Generation**: Uses the advanced crypto signal framework
3. **Live Trade Execution**: Executes trades based on generated signals
4. **Portfolio Management**: Tracks positions, P&L, and portfolio value

## Quick Start

### Option 1: Integrated Live Trading (Recommended)
```bash
python main.py live-trading
```

This runs the complete system:
- Collects real-time data
- Generates signals every 5 minutes
- Executes trades automatically
- Logs all activity

### Option 2: Data Collection Only
```bash
python main.py data-only
```

This runs only data collection:
- Builds 1-minute candles from WebSocket data
- Saves to parquet files
- No signal generation or trading

### Option 3: Separate Components
```bash
# Terminal 1: Data collection
python main.py data-collection

# Terminal 2: Live trading (uses existing data)
python main.py live-trading
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WebSocket     │───▶│  Data Collection │───▶│ 1-min Candles   │
│   Price Feed    │    │   (Real-time)    │    │   (Parquet)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Trade Log     │◀───│ Signal Execution │◀───│ Signal Generation│
│   (CSV)         │    │   (Portfolio)    │    │   (Framework)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Features

### 1. Real-time Data Collection
- **WebSocket Integration**: Connects to Coinbase WebSocket feed
- **1-minute Candles**: Builds OHLCV candles from tick data
- **Data Persistence**: Saves to parquet files for analysis
- **Memory Management**: Keeps only last 7 days in memory

### 2. Signal Generation
- **Multiple Strategies**: Uses 3 best-performing strategies (every 1 minute)
- **Regime Detection**: Adapts to market conditions
- **Risk Management**: Position sizing and correlation controls
- **Confidence Filtering**: Only trades high-confidence signals

### 3. Trade Execution
- **Portfolio Tracking**: Monitors cash and positions
- **P&L Calculation**: Real-time profit/loss tracking
- **Trade Logging**: Complete audit trail in CSV
- **Position Sizing**: 10% of portfolio per trade (scaled by confidence)

## Configuration

### API Keys
Ensure `config/api_keys.yaml` contains:
```yaml
coinbase:
  api_key: "your_coinbase_api_key"
  api_secret: "your_coinbase_secret"
```

### Symbols
Default symbols: `['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']`

To modify, edit the `symbols` parameter in the trading system initialization.

### Signal Generation Interval
Default: 1 minute (60 seconds)

To modify, change `signal_generation_interval` in the `IntegratedLiveTrading` class.

## File Structure

```
data/
├── live_trades.csv              # Trade log
├── portfolio_state.json         # Portfolio state
├── BTC_1m_historical.parquet   # 1-minute BTC data
├── ETH_1m_historical.parquet   # 1-minute ETH data
└── ...                         # Other symbols

logs/
├── integrated_live_trading.log  # Main system log
├── data_collection.log          # Data collection log
└── websocket_collection.log     # WebSocket log
```

## Monitoring

### Portfolio Status
The system displays real-time portfolio information:
- Cash balance
- Total portfolio value
- Cumulative P&L
- Current positions
- Last update time

### Trade Log
All trades are logged to `data/live_trades.csv` with:
- Timestamp
- Symbol
- Action (BUY/SELL)
- Price
- Confidence
- Strategy
- P&L
- Position size

### Log Files
- `logs/integrated_live_trading.log`: Main system activity
- `logs/data_collection.log`: Data collection activity
- `logs/websocket_collection.log`: WebSocket connection activity

## Troubleshooting

### Common Issues

1. **No Data Collection**
   - Check WebSocket connection
   - Verify API keys
   - Check network connectivity

2. **No Signals Generated**
   - Ensure sufficient historical data (50+ points)
   - Check strategy configuration
   - Verify data quality

3. **No Trades Executed**
   - Check confidence thresholds
   - Verify portfolio has sufficient cash
   - Check signal generation logs

### Debug Mode
Enable debug logging by modifying the logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Memory Usage
- Historical data limited to 7 days
- Pseudo-candles use minimal memory
- Regular cleanup of old data

### CPU Usage
- Signal generation every 5 minutes (not every tick)
- Efficient pandas operations
- Minimal real-time processing

### Storage
- Parquet format for efficient storage
- Automatic cleanup of old data
- Compressed file format

## Safety Features

### Risk Management
- Maximum position size limits
- Correlation-based position reduction
- Drawdown protection
- Confidence-based filtering

### Error Handling
- Graceful WebSocket reconnection
- Error logging and recovery
- Portfolio state persistence
- Trade execution validation

## Advanced Usage

### Custom Strategies
To add custom strategies, modify `src/crypto_trading_strategies.py` and ensure they're included in the selected strategies list.

### Custom Symbols
To trade different symbols, modify the `symbols` parameter in the system initialization.

### Custom Intervals
To change signal generation frequency, modify `signal_generation_interval` in the `IntegratedLiveTrading` class.

## Support

For issues or questions:
1. Check the log files for error messages
2. Verify configuration files
3. Ensure sufficient historical data
4. Check network connectivity and API keys

## Disclaimer

This system is for educational and research purposes. Always test thoroughly before using with real money. Past performance does not guarantee future results. Use at your own risk.
