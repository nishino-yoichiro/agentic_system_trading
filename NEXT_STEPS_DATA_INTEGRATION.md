# Next Steps: Data Integration Plan

## What We're Building

Following the existing crypto patterns, adapters will:

### 1. Historical Data (Coinbase pattern)
```python
# CryptoCollector pattern:
1. Check existing Parquet files
2. Find gaps in data
3. Fetch ONLY missing data (incremental)
4. Save with append-only merge
5. Rate limiting to respect API limits

# Adapters will do same:
- CoinbaseMarketAdapter → Uses UnifiedStorage(market_type='crypto')
- AlpacaMarketAdapter → Uses UnifiedStorage(market_type='equities')
- Both save to Parquet, both check for gaps, both append-only
```

### 2. WebSocket Live Data
```python
# WebSocketPriceFeed pattern:
1. Connect to WebSocket
2. Receive ticks in real-time  
3. Accumulate ticks into minute bars
4. Save to {symbol}_1m.parquet
5. Periodically merge to _historical.parquet

# Adapters will do same:
- CoinbaseMarketAdapter.stream_data() → WebSocket ticks
- AlpacaMarketAdapter.stream_data() → WebSocket trades
- Both save via UnifiedStorage.save_live_bar()
```

## Integration Flow

```
┌─────────────────────────────────────────┐
│   Your Features Need Data              │
│   (Backtester, Live Trading, etc.)    │
└──────────────┬──────────────────────────┘
               │
               │ Call adapter methods:
               │ - adapter.get_historical()
               │ - adapter.stream_data()
               ▼
        ┌──────────────┐
        │   Adapter    │
        │              │
        │ Handles API: │
        │ - Coinbase   │
        │ - Alpaca     │
        └──────┬───────┘
               │
               │ Fetch from API
               │ Then save via:
               ▼
        ┌──────────────┐
        │UnifiedStorage│
        │              │
        │ Handles:     │
        │ - Parquet    │
        │ - Gaps       │
        │ - Append     │
        │ - Live bars  │
        └──────────────┘
```

## What I'll Update

### Update Existing Adapters

**Current problem**: Adapters exist but don't follow your patterns

**Fix**: Update to use:
1. `UnifiedStorage` for all Parquet operations
2. Gap detection logic (from CryptoDataCollector)
3. Incremental fetching (don't re-fetch existing data)
4. WebSocket streaming (from WebSocketPriceFeed pattern)

### Files to Update:

1. **src/adapters/coinbase_adapter.py** - Add:
   - `load_historical()` - Check Parquet, fill gaps
   - `stream_data()` - WebSocket to live bars
   - Uses `UnifiedStorage(market_type='crypto')`

2. **src/adapters/alpaca_adapter.py** - Add:
   - `load_historical()` - Check Parquet, fill gaps  
   - `stream_data()` - WebSocket to live bars
   - Uses `UnifiedStorage(market_type='equities')`
   - Respects market hours (9:30-16:00 ET)

3. **DataManager integration** - Connect adapters to storage:
   - Adapters fetch from APIs
   - Save via UnifiedStorage
   - UnifiedStorage handles all Parquet ops

## Your Features Will Then Work

### Backtester:
```python
# Your existing code:
df = pd.read_parquet("data/crypto_db/BTC_historical.parquet")
backtest(df)

# Will become:
adapter = CoinbaseMarketAdapter("BTC")
df = adapter.load_historical()  # Checks Parquet, fills gaps if needed
backtest(df)

# Or for equities:
adapter = AlpacaMarketAdapter("SPY")
df = adapter.load_historical()  # Same pattern!
backtest(df)
```

### Live Trading:
```python
# Will become:
adapter = AlpacaMarketAdapter("SPY")
await adapter.connect()

async for bar in adapter.stream_data():
    # Bar automatically saved to Parquet
    # Generate signals
    # Trade
```

## Next Steps

1. ✅ UnifiedStorage created (done)
2. ⏳ Update CoinbaseMarketAdapter to use storage
3. ⏳ Update AlpacaMarketAdapter to use storage
4. ⏳ Add gap detection to adapters
5. ⏳ Add WebSocket streaming to adapters

Should I proceed with updating the adapters to follow these patterns?

