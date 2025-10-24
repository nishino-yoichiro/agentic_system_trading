# Interactive Trading Module

## Overview

The Interactive Trading Module is a real-time signal monitoring system that allows you to manually trade based on generated signals from your crypto trading strategies. It provides an interactive interface for selecting strategies and tickers, configurable signal generation intervals, and multiple notification methods.

## Features

- **🎯 Interactive Setup**: Choose strategies, tickers, and signal intervals through an interactive menu
- **⏱️ Configurable Intervals**: Set signal generation intervals from 10 seconds to 5 minutes
- **🔊 Sound Notifications**: Audio alerts for different signal types (buy/sell/strong signals)
- **🖥️ Desktop Notifications**: Pop-up notifications with signal details
- **📊 Real-time Monitoring**: Live websocket data integration with historical data
- **📈 Minute Summaries**: Display signal counts and details every minute
- **🎵 Multiple Sound Types**: Different beep patterns for different signal types

## Quick Start

### 1. Install Dependencies

```bash
# Install additional dependencies for sound and notifications
python scripts/setup_interactive_trading.py

# Or install manually
pip install pygame>=2.5.0 plyer>=2.1.0
```

### 2. Run the Module

```bash
python main.py interactive-trading
```

### 3. Interactive Setup

The module will guide you through setup:

1. **Select Strategies**: Choose from available trading strategies
2. **Select Symbols**: Pick crypto symbols to monitor (BTC, ETH, etc.)
3. **Set Signal Interval**: Choose how often to generate signals (10s to 5min)
4. **Configure Notifications**: Enable/disable sound and desktop notifications

## Usage

### Command Line

```bash
# Start interactive trading module
python main.py interactive-trading
```

### Interactive Menu

When you start the module, you'll see:

```
🚀 INTERACTIVE TRADING MODULE SETUP
============================================================

📊 Available Strategies:
  1. btc_ny_session
  2. liquidity_sweep_reversal
  3. volume_weighted_trend_continuation
  4. volatility_expansion_breakout
  5. daily_avwap_zscore_reversion
  6. opening_range_break_retest
  7. keltner_exhaustion_fade
  8. fakeout_reversion

Enter strategy numbers (comma-separated) or 'all' for all strategies:
Strategies: 1,2,3

💰 Available Symbols: BTC, ETH, ADA, AVAX, DOT, LINK, MATIC, SOL, UNI
Enter symbols (comma-separated) or 'all' for all: BTC,ETH

⏱️  Signal Generation Interval:
  1. Every 10 seconds (high frequency)
  2. Every 20 seconds
  3. Every 30 seconds
  4. Every 60 seconds (1 minute) - default
  5. Every 2 minutes
  6. Every 5 minutes
Choose interval (1-6) [4]: 2

🔔 Notification Settings:
Enable sound notifications? (Y/n): Y
Enable desktop notifications? (Y/n): Y
```

### Real-time Display

Once running, you'll see:

```
📊 SIGNALS THIS MINUTE (2025-01-28 14:30): 2 signals
--------------------------------------------------------------------------------
  BUY           BTC    | btc_ny_session           | Conf: 0.85
    Reason: NY session open signal with high confidence

  STRONG_BUY    ETH    | liquidity_sweep_reversal | Conf: 0.92
    Reason: Liquidity sweep detected with volume confirmation

📈 Total signals today: 15
⏰ Next signal check in: 20 seconds
🕐 Current time: 14:30:45 UTC
```

## Signal Types

The module supports different signal types with distinct audio patterns:

- **BUY** 🟢: Higher pitch beep (800Hz)
- **SELL** 🔴: Lower pitch beep (400Hz)  
- **STRONG_BUY** 🟢🟢: Very high pitch beep (1000Hz)
- **STRONG_SELL** 🔴🔴: Very low pitch beep (300Hz)

## Configuration

### Signal Intervals

- **10 seconds**: High frequency trading (requires fast execution)
- **20 seconds**: Fast scalping strategies
- **30 seconds**: Quick swing trades
- **60 seconds**: Standard minute-based strategies (default)
- **2 minutes**: Medium-term signals
- **5 minutes**: Longer-term analysis

### Notification Settings

- **Sound Notifications**: Audio alerts using pygame
- **Desktop Notifications**: System notifications using plyer
- Both can be enabled/disabled during setup

## Data Sources

The module combines:

1. **Historical Data**: Loaded from parquet files in `data/` directory
2. **Real-time Data**: WebSocket price feeds for live updates
3. **Signal Generation**: Uses your existing strategy framework

## Troubleshooting

### Sound Issues

If sound notifications don't work:

```bash
# Reinstall pygame
pip uninstall pygame
pip install pygame>=2.5.0

# On Windows, try:
conda install pygame
```

### Desktop Notification Issues

If desktop notifications don't work:

```bash
# Install plyer
pip install plyer>=2.1.0

# On Linux, you may need:
sudo apt-get install libnotify-bin
```

### Data Issues

If no signals are generated:

1. Check that historical data exists in `data/` directory
2. Verify WebSocket connection is working
3. Ensure selected strategies are compatible with your data
4. Check logs in `logs/interactive_trading.log`

## File Structure

```
src/
├── interactive_trading_module.py    # Main interactive module
├── crypto_signal_framework.py      # Signal generation framework
├── crypto_signal_integration.py    # Strategy integration
└── ...

scripts/
├── setup_interactive_trading.py    # Dependency installer
└── ...

logs/
└── interactive_trading.log         # Module logs
```

## Advanced Usage

### Custom Strategies

The module automatically loads strategies from your existing framework. To add new strategies:

1. Create strategy in `src/` directory
2. Add to `CryptoSignalIntegration` class
3. Strategy will appear in interactive menu

### Signal Filtering

You can modify signal filtering by editing the `_process_signal` method in `InteractiveTradingModule` class.

### Custom Notifications

Extend the `SoundNotificationSystem` or `DesktopNotificationSystem` classes to add custom notification types.

## Performance

- **Memory Usage**: ~50-100MB depending on data size
- **CPU Usage**: Low, mostly I/O bound
- **Network**: WebSocket connection for real-time data
- **Storage**: Logs and signal history

## Safety Features

- **Error Handling**: Graceful degradation if components fail
- **Thread Safety**: Proper threading for real-time updates
- **Resource Management**: Automatic cleanup on exit
- **Logging**: Comprehensive logging for debugging

## Examples

### Basic Usage

```bash
# Start with default settings
python main.py interactive-trading

# Select all strategies, BTC only, 30-second intervals
# Enable all notifications
```

### High-Frequency Trading

```bash
# For high-frequency strategies
python main.py interactive-trading
# Select: 10-second intervals
# Choose: Scalping strategies only
```

### Long-term Monitoring

```bash
# For longer-term analysis
python main.py interactive-trading
# Select: 5-minute intervals
# Choose: Trend-following strategies
```

## Support

For issues or questions:

1. Check the logs in `logs/interactive_trading.log`
2. Verify all dependencies are installed
3. Ensure data files exist and are accessible
4. Check WebSocket connectivity

## License

Part of the Enhanced Crypto Pipeline project.
