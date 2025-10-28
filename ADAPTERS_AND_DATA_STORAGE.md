# How Adapters Work With Your Data Storage

## Current Architecture (Crypto) - WORKS GREAT ✅

```
┌────────────────────────────────────────┐
│  Live Trading / Signals                │
└────────────┬───────────────────────────┘
             │
             │ Uses both:
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌─────▼────┐
│WebSocket│      │Historical│
│ Data    │   +  │  Data    │
│(New 1m) │      │ (Parquet)│
└─────────┘      └──────────┘
     │                 │
     │                 │
     ▼ (Append)        │
┌─────────┐             │
│_1m file │             │
│(Live)   │             │
└────┬────┘             │
     │ (Merge hourly)   │
     └──────┬───────────┘
            │
       ┌────▼────┐
       │_historical│
       │   file   │
       └──────────┘
```

## Your Storage Pattern

### 1. Historical (Initial Bulk Load)
- Purpose: Fast backtesting, gap filling
- Storage: `data/crypto_db/BTC_historical.parquet`
- How: One-time API call for large dataset
- Usage: Load on startup, use for backtests

### 2. Live (1-minute bars)
- Purpose: Streaming data during live trading
- Storage: `data/crypto_db/BTC_1m.parquet`
- How: Accumulates live bars, appends to historical
- Usage: Current minute's data for live signals

## For Alpaca - SAME PATTERN ✅

```
data/equities_db/
  ├── SPY_historical.parquet  (bulk historical)
  ├── SPY_1m.parquet          (live bars)
  ├── QQQ_historical.parquet
  ├── QQQ_1m.parquet
  └── ...
```

## Integration with Adapters

### Current Flow (Crypto):
```python
# Backtester
df = pd.read_parquet("data/crypto_db/BTC_historical.parquet")
backtest(df)

# Live Trading
live_df = pd.read_parquet("data/crypto_db/BTC_1m.parquet")  # Recent bars
historical_df = pd.read_parquet("data/crypto_db/BTC_historical.parquet")  # Full history
combined = merge(historical_df, live_df)
```

### NEW Flow (With Adapters for Equities):
```python
# Backtester - WORKS FOR BOTH
if market == 'crypto':
    df = pd.read_parquet("data/crypto_db/BTC_historical.parquet")
elif market == 'equities':
    df = pd.read_parquet("data/equities_db/SPY_historical.parquet")
backtest(df)  # Same backtest code!

# Live Trading - WORKS FOR BOTH
if market == 'crypto':
    df = pd.read_parquet("data/crypto_db/BTC_1m.parquet")
elif market == 'equities':
    df = pd.read_parquet("data/equities_db/SPY_1m.parquet")
# Same logic for both!
```

## Why This Works

**Adapters** normalize the data format (Coinbase → OHLCV, Alpaca → OHLCV)

**Your Storage** normalizes the file format (both in Parquet)

**Result**: Your 4 features work the same way for both crypto and equities!

## Recommendation

✅ **YES**, store Alpaca data the same way:
- `SPY_historical.parquet` - Bulk historical data
- `SPY_1m.parquet` - Live bars (only during market hours)

**Differences**:
- Crypto: Store 24/7 data (including weekends/nights)
- Equities: Store only 9:30-16:00 ET on trading days

**Benefits**:
- ✅ Reduce API calls (Alpaca has rate limits too!)
- ✅ Fast backtesting (read from Parquet vs API)
- ✅ Gap detection (compare Parquet to market calendar)
- ✅ Consistent architecture (same logic for both markets)

## Next Step

I've created `EquitiesDataManager` that mirrors your crypto pattern.

Now the adapter can use it:

```python
# In CoinbaseMarketAdapter
def save_data(self, symbol, df):
    # Save to data/crypto_db/{symbol}_historical.parquet
    pass

# In AlpacaMarketAdapter  
def save_data(self, symbol, df):
    # Save to data/equities_db/{symbol}_historical.parquet
    # (with market hours filtering)
    pass
```

Your existing storage logic works for both! Just need to wire adapters to use it.
