# Quick Start: Multi-Adapter Architecture

## ✅ What Was Fixed

### Issues Resolved:
1. **Import errors** - Fixed missing exports (`OrderSide`, `RiskLimits`, `Position`)
2. **Package name error** - Removed non-existent `coinbase-rest` dependency
3. **Reused existing code** - Coinbase adapter now wraps existing `CoinbaseAdvancedClient` instead of duplicating
4. **Path issues** - Fixed import paths in example

### Changes Made:
- ✅ `src/core_engine/__init__.py` - Added missing exports
- ✅ `src/adapters/coinbase_adapter.py` - Simplified to use existing Coinbase clients
- ✅ `requirements.txt` - Removed invalid package, using existing coinbase-advanced-py
- ✅ `examples/multi_adapter_strategy.py` - Fixed imports

## 🚀 How to Use

### 1. Install Dependencies

```bash
# Activate your virtual environment
enh_venv\Scripts\activate

# Install new Alpaca dependencies
pip install alpaca-py pandas-market-calendars pytz
```

### 2. Set Environment Variables

Add to your `.env` file:

```bash
# For Alpaca (equities trading with SPY)
ALPACA_API_KEY=pk_test_your_key
ALPACA_SECRET_KEY=your_secret_key

# Your existing Coinbase credentials (already working)
COINBASE_API_KEY=your_key  
COINBASE_SECRET_KEY=your_secret
```

### 3. Test the System

```bash
# Run the example
python examples/multi_adapter_strategy.py
```

This will demonstrate:
- Session controller checking market hours
- How to initialize both adapters
- The unified interface approach

## 🏗️ Architecture Overview

```
Your Existing System (100% working)     New Adapter Layer
┌────────────────────────────────┐     ┌───────────────────┐
│  Crypto Infrastructure          │     │  core_engine/     │
│  ├── coinbase_advanced_client  │     │  (market-agnostic)│
│  ├── websocket_price_feed      │     └────────┬──────────┘
│  ├── crypto_analysis_engine     │              │
│  └── [all existing strategies] │              │
└────────────────────────────────┘        Unified Interface
                                        │
                          ┌─────────────┴─────────────┐
                          │                           │
                 ┌────────▼────────┐         ┌────────▼────────┐
                 │ Coinbase       │         │ Alpaca          │
                 │ Adapter        │         │ Adapter         │
                 │ (wraps existing│         │ (new for SPY)   │
                 │  clients)      │         └──────────────────┘
                 └────────────────┘
```

## 📝 Key Points

### What Works (Existing)
- ✅ All your existing Coinbase crypto infrastructure
- ✅ All your existing strategies
- ✅ Your backtesters
- ✅ Your signal generation

### What's New (Added)
- ✅ `core_engine/` - Market-agnostic interfaces
- ✅ `adapters/` - Alpaca adapter for equities
- ✅ Session controller for market hours
- ✅ Alpaca broker for order execution

### Integration Approach

**The adapter approach means**:
1. Your existing crypto code keeps working 100%
2. New Alpaca adapter added alongside (doesn't break anything)
3. You can gradually migrate strategies to use the unified interface
4. Or run both systems in parallel

## 🎯 Usage Examples

### Example 1: Use Existing Crypto System (No Changes)
```python
# This still works exactly as before
from src.crypto_signal_integration import CryptoSignalIntegration

integration = CryptoSignalIntegration(selected_strategies=['btc_ny_session'])
signals = integration.generate_signals(['BTC'], days=30)
```

### Example 2: Use New Adapter for Alpaca
```python
from adapters import AlpacaMarketAdapter, SessionController

# Initialize adapter
alpaca = AlpacaMarketAdapter(symbol="SPY", paper=True)
await alpaca.connect()

# Check if market is open
session = SessionController(exchange='NYSE')
if session.is_market_open():
    async for bar in alpaca.stream_data(interval_seconds=60):
        # Your strategy here
        signal = strategy.generate_signal(bar)
```

### Example 3: Run Same Strategy on Both Markets
```python
# Strategy doesn't know which market it's on
class MyStrategy:
    def generate_signal(self, bar: OHLCVBar):
        # Works for both BTC and SPY
        return signal

# Run on crypto
crypto_adapter = CoinbaseMarketAdapter("BTC")
async for bar in crypto_adapter.stream_data():
    signal = MyStrategy().generate_signal(bar)

# Run on equities  
equities_adapter = AlpacaMarketAdapter("SPY")
async for bar in equities_adapter.stream_data():
    signal = MyStrategy().generate_signal(bar)  # Same code!
```

## 🔍 What Each Adapter Does

### Coinbase Adapter (`src/adapters/coinbase_adapter.py`)
- **Wraps** existing `CoinbaseAdvancedClient`
- **Reuses** your working Coinbase infrastructure
- **24/7** continuous trading
- **No session** handling needed

### Alpaca Adapter (`src/adapters/alpaca_adapter.py`)
- **New** equities market adapter
- **Session-aware** (9:30-16:00 ET)
- **Handles** DST, holidays, halts
- **WebSocket** streaming for trades

### Session Controller (`src/adapters/session_controller.py`)
- **Checks** if market is open
- **Calculates** time until next open
- **Handles** trading calendars
- **Respects** DST transitions

## 📚 Documentation

- **Architecture**: `docs/ADAPTER_ARCHITECTURE.md` - Detailed design
- **Quick Start**: `MULTI_ADAPTER_README.md` - Overview
- **Examples**: `examples/multi_adapter_strategy.py` - Usage patterns

## ⚠️ Important Notes

1. **Your existing system is untouched** - All your crypto code still works
2. **No breaking changes** - New adapters are additions, not replacements
3. **Gradual migration** - You can adopt the adapter pattern incrementally
4. **Both systems can coexist** - Run crypto and equities in parallel

## 🎓 Why This Works

The adapter pattern isolates market-specific logic:
- **What's reusable**: Signal generation, risk management, backtesting
- **What needs adapters**: Data ingestion, session handling, order execution

Your core engine receives OHLCV bars and emits signals - it doesn't care if those bars came from Coinbase or Alpaca.

## ✨ Next Steps

1. **Test session controller** - See when markets are open/closed
2. **Initialize Alpaca adapter** - Connect to paper trading
3. **Run strategies** - Use existing strategies with SPY data
4. **Paper trade** - Test with Alpaca paper account
5. **Go live** - When ready, use real accounts

## 🤝 Need Help?

- Check `docs/ADAPTER_ARCHITECTURE.md` for detailed explanations
- Look at `examples/multi_adapter_strategy.py` for working code
- Your existing crypto system documentation still applies
