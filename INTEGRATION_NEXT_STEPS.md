# Next Steps: Integrating Adapters into Your Features

## ✅ What's Complete

### Adapters Are Ready!
- ✅ **CoinbaseAdapter** - Full storage integration with gap detection
- ✅ **AlpacaAdapter** - Full storage integration with session awareness
- ✅ **UnifiedStorage** - Single storage manager for both markets
- ✅ Live bars automatically saved
- ✅ Gap detection and incremental fetching
- ✅ Follow your existing crypto patterns

### Your 4 Features Still Work (Crypto)
- ✅ Backtester
- ✅ Monte Carlo Reports
- ✅ Live Trading
- ✅ Live Signals

## 🎯 What You Asked For

You wanted to be able to run the **same 4 features** on **equities** (Alpaca) just by choosing the market type.

## 📋 Integration Plan

### Step 1: Backtester Integration (Easiest to Start)

**File**: `scripts/backtesting/multi_symbol_backtester.py`

**Current code**:
```python
def load_symbol_data(symbol, days):
    # Only reads crypto files
    df = pd.read_parquet(f"data/crypto_db/{symbol}_historical.parquet")
    return df
```

**Change to**:
```python
def load_symbol_data(symbol, days, market='crypto'):
    """Load data for any market"""
    
    if market == 'crypto':
        # Use Coinbase adapter
        from adapters import CoinbaseMarketAdapter
        adapter = CoinbaseMarketAdapter(symbol)
        return adapter.load_historical_data(days=days)
    
    elif market == 'equities':
        # Use Alpaca adapter
        from adapters import AlpacaMarketAdapter
        adapter = AlpacaMarketAdapter(symbol)
        return adapter.load_historical_data(days=days)
```

### Step 2: main.py Update

**Add market parameter**:
```python
parser.add_argument("--market", 
                   choices=['crypto', 'equities'], 
                   default='crypto',
                   help="Market type")

# Then in backtest:
pipeline.run_backtest(args.symbols, args.days, args.market)
```

### Step 3: Live Trading Integration

**File**: `scripts/live_trading/integrated_live_trading.py`

Add adapter support to load combined data (historical + live bars).

## 🚀 After Integration, You Can:

```bash
# Crypto (existing)
python main.py backtest --symbols BTC ETH
python main.py live-trading --symbols BTC

# Equities (NEW!)
python main.py backtest --symbols SPY --market equities
python main.py live-trading --symbols SPY --market equities

# Or both!
python main.py backtest --symbols BTC SPY --market crypto
```

## Summary

✅ **Architecture**: Complete
✅ **Storage Layer**: Complete
✅ **Adapters**: Complete with full integration
⏳ **Your Features**: Need integration

The adapters are production-ready and waiting to be integrated into your backtester and live trading!

