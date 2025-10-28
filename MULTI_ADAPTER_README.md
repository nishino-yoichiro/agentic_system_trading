# Multi-Adapter Architecture for Multi-Market Trading

## ✅ What's Been Implemented

I've successfully created a **multi-adapter architecture** that allows your trading system to work seamlessly across different markets: **crypto (Coinbase)** and **equities (Alpaca/SPY)**.

### Core Infrastructure

```
src/
├── core_engine/          # Market-agnostic components
│   ├── __init__.py
│   ├── market_adapter.py      # Abstract base class for all adapters
│   ├── signal_engine.py       # Signal generation (market-agnostic)
│   ├── risk_manager.py        # Risk limits management
│   └── portfolio.py           # Position tracking
│
└── adapters/             # Market-specific implementations
    ├── __init__.py
    ├── alpaca_adapter.py      # US equities (SPY, session-aware)
    ├── alpaca_broker.py       # Order execution for equities
    ├── coinbase_adapter.py    # Crypto (BTC, 24/7)
    └── session_controller.py   # Market hours management
```

### Key Features

1. **Unified Interface**: All adapters implement the same `MarketAdapter` interface, so your core strategies don't care whether they're trading BTC or SPY.

2. **Session Awareness**: The `AlpacaMarketAdapter` only processes trades during market hours (9:30-16:00 ET) and respects:
   - Trading days (NYSE calendar)
   - DST transitions
   - Market halts
   - OPEX days

3. **Same Strategy, Multiple Markets**: You can run the exact same strategy on both crypto and equities with zero code changes.

4. **Risk Isolation**: Market-specific issues (session handling, halts, etc.) stay in the adapters, leaving your core engine clean.

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────┐
│     Core Engine (Market-Agnostic)       │
│                                         │
│  ┌──────────────┐  ┌──────────────┐   │
│  │  Signal      │  │    Risk      │   │
│  │  Engine      │  │   Manager    │   │
│  └──────────────┘  └──────────────┘   │
│                                         │
│  Works with unified OHLCVBar format     │
└──────────────────┬──────────────────────┘
                   │
                   │ Unified Interface
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼─────────┐  ┌────────▼────────┐
│ Coinbase        │  │ Alpaca          │
│ Adapter         │  │ Adapter         │
│ (BTC, 24/7)     │  │ (SPY, 9:30-16ET)│
└─────────────────┘  └─────────────────┘
```

## 🚀 Usage

### Basic Example

```python
from src.adapters import AlpacaMarketAdapter, CoinbaseMarketAdapter
from src.core_engine import SignalEngine

# Same strategy, different markets
strategy = YourStrategy()

# Run on SPY (equities)
alpaca = AlpacaMarketAdapter(symbol="SPY", paper=True)
await alpaca.connect()

async for bar in alpaca.stream_data(interval_seconds=60):
    if alpaca.is_market_open():  # Only trade during market hours
        signal = strategy.generate_signal(bar)

# Run on BTC (crypto)
coinbase = CoinbaseMarketAdapter(symbol="BTC")
await coinbase.connect()

async for bar in coinbase.stream_data(interval_seconds=60):
    signal = strategy.generate_signal(bar)  # Always can trade
```

### Complete Trading Loop

See `examples/multi_adapter_strategy.py` for a full example with:
- Signal generation
- Risk validation
- Order execution
- Portfolio tracking

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `alpaca-py>=0.21.0` - Alpaca API client
- `coinbase-rest>=1.0.0` - Coinbase API client
- `pandas-market-calendars>=4.4.0` - Trading calendar
- `pytz>=2024.1` - Timezone handling

### 2. Set Environment Variables

Add to your `.env` file:

```bash
# Alpaca (equities trading)
ALPACA_API_KEY=pk_test_your_key
ALPACA_SECRET_KEY=your_secret_key

# Coinbase (crypto trading)
COINBASE_API_KEY=your_api_key
COINBASE_SECRET_KEY=your_secret_key
```

### 3. Test the System

```bash
# Run the example
python examples/multi_adapter_strategy.py
```

## 📋 What This Gives You

### 1. **Code Reuse**
```python
# Same strategy works for both markets
class MovingAverageStrategy:
    def generate_signal(self, bar: OHLCVBar):
        # This works identically for BTC and SPY
        ...
```

### 2. **Faster Development**
- Test on crypto (fast iteration, 24/7 data)
- Deploy to equities with minimal changes
- Single signal library for all markets

### 3. **Risk Isolation**
- Market-specific bugs stay in adapters
- Session handling doesn't affect core logic
- Exchange outages only impact that adapter

### 4. **Portfolio Expansion**
- Trade both crypto and equities from same codebase
- Unified risk management
- Cross-asset strategies

## 🎯 Market Differences Handled

| Feature | Coinbase (Crypto) | Alpaca (Equities) |
|---------|------------------|-------------------|
| Trading Hours | 24/7 | 9:30-16:00 ET |
| Session Handling | None | Required |
| Order Book | Full L2 | Top-of-book |
| Data Aggregation | Always available | Market hours only |
| DST Impact | None | Handled |
| Market Halts | Rare | Frequent |
| OPEX Days | None | Monthly |

The adapters automatically handle all these differences.

## 📚 Documentation

- **Architecture Guide**: `docs/ADAPTER_ARCHITECTURE.md`
- **Examples**: `examples/multi_adapter_strategy.py`
- **Core Engine**: `src/core_engine/`
- **Adapters**: `src/adapters/`

## 🎓 Why This Is The Right Architecture

### Your Original Question

> "Should I reuse my crypto framework or rebuild from scratch for Alpaca?"

**Answer**: ✅ **Reuse the framework, but build a dedicated adapter layer.**

### Why This Works

1. **Your crypto engine's core (strategies, backtester, risk engine) is market-agnostic** - it just operates on OHLCV data
2. **The only market-specific parts are**: data ingestion, session handling, and order execution
3. **Solution**: Isolate market-specific logic in adapters

### What's Reusable

✅ Core engine (strategies, backtest, metrics)  
✅ Signal library (works on any OHLCV data)  
✅ Risk manager (same limits across markets)  
✅ Portfolio manager (positions tracking)  
✅ Storage layer (same Parquet schema)

### What Needs Adapters

❌ Market data ingestion (Coinbase != Alpaca format)  
❌ Session handling (crypto is 24/7, equities is session-based)  
❌ Order execution (different APIs and semantics)  
❌ Data latency (crypto faster, equities okay)

## 🚧 Migration Path (Optional)

You can migrate your existing crypto code gradually:

1. **Phase 1** (Current): Add adapter layer alongside existing code
2. **Phase 2**: Extract common signal generation to core engine
3. **Phase 3**: Update existing strategies to use new interface
4. **Phase 4**: Unified backtester for multi-asset testing

**Note**: No need to break your existing crypto system. The adapters can coexist.

## ✨ Next Steps

1. **Test the adapters**: Run `examples/multi_adapter_strategy.py`
2. **Add your strategies**: Register strategies with the signal engine
3. **Paper trade**: Test with both Alpaca paper trading and Coinbase
4. **Backtest**: Use the unified OHLCV format across markets
5. **Live trade**: When ready, flip from paper to live

## 🤝 Support

See the detailed architecture guide in `docs/ADAPTER_ARCHITECTURE.md` for:
- Component overview
- Usage patterns
- Migration strategy
- Best practices

---

**Summary**: You now have a multi-market trading system where the same strategies work on both crypto and equities, with market-specific logic properly isolated in adapters. Your core engine remains clean and market-agnostic.
