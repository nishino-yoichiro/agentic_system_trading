# How Adapters Use Unified Storage

## Architecture

```
┌─────────────────────────────────────────┐
│        Your Features                    │
│  (Backtester, Live Trading, etc.)      │
└──────────────┬──────────────────────────┘
               │
               │ Use storage.load_historical()
               │ or storage.get_combined_data()
               ▼
        ┌──────────────────┐
        │ UnifiedStorage   │  ← ONE place for ALL storage logic
        │                  │
        │ Handles Parquet: │
        │ - Save/load      │
        │ - Append-merge   │
        │ - Gap detection  │
        └──────────┬───────┘
                   ↑
                   │ Calls from adapter methods
                   │
        ┌──────────┴──────────┐
        │                     │
    ┌───▼────┐          ┌─────▼────┐
    │Coinbase│          │ Alpaca    │
    │Adapter │          │Adapter    │
    └───┬────┘          └─────┬─────┘
        │                    │
        │ Fetch from API     │
        │ Then save via      │
        │ UnifiedStorage     │
        ▼                    ▼
  data/crypto_db/     data/equities_db/
```

## Example Usage in Adapters

### CoinbaseMarketAdapter:

```python
from data_ingestion.unified_data_storage import UnifiedStorage

class CoinbaseMarketAdapter(MarketAdapter):
    
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.storage = UnifiedStorage(market_type='crypto')  # ← Same class!
        self.client = CoinbaseAdvancedClient()
    
    def get_market_data(self, symbol, start, end):
        """Fetch historical data"""
        # 1. Check if we already have it in storage
        df = self.storage.load_historical(symbol)
        
        if df is not None:
            # Already have data, no API call needed!
            return df
        
        # 2. Need to fetch from API
        candles = self.client.get_candles(...)
        df = convert_to_dataframe(candles)
        
        # 3. Save for next time
        self.storage.save_historical(symbol, df)
        
        return df
    
    def stream_data(self, interval_seconds):
        """Stream live data"""
        async for bar in self.client.websocket_stream():
            # Convert to DataFrame
            bar_df = bar_to_dataframe(bar)
            
            # Save to live storage (accumulates)
            self.storage.save_live_bar(self.symbol, bar_df)
            
            yield bar
```

### AlpacaMarketAdapter:

```python
from data_ingestion.unified_data_storage import UnifiedStorage

class AlpacaMarketAdapter(MarketAdapter):
    
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.storage = UnifiedStorage(market_type='equities')  # ← Same class!
        self.client = AlpacaDataStream()
    
    def get_market_data(self, symbol, start, end):
        """Fetch historical data - SAME LOGIC as Coinbase!"""
        # 1. Check storage first
        df = self.storage.load_historical(symbol)
        
        if df is not None:
            return df  # No API call!
        
        # 2. Fetch from API
        bars = self.client.get_stock_bars(...)
        df = convert_to_dataframe(bars)
        
        # 3. Save for next time
        self.storage.save_historical(symbol, df)
        
        return df
```

## Key Benefits

### 1. Single Storage Logic
- ✅ One `UnifiedStorage` class for all markets
- ✅ Same Parquet operations everywhere
- ✅ Same append/merge logic
- ✅ Easy to maintain

### 2. Adapters Handle API Differences
- ✅ Coinbase → fetches from Coinbase API
- ✅ Alpaca → fetches from Alpaca API
- ✅ Both save using same storage manager

### 3. Your Features Work Same Way
```python
# Your backtester just calls:
if market == 'crypto':
    storage = UnifiedStorage('crypto')
elif market == 'equities':
    storage = UnifiedStorage('equities')

df = storage.load_historical(symbol)  # ← Works for both!
backtest(df)
```

## Storage Organization

```
data/
├── crypto_db/
│   ├── BTC_historical.parquet
│   ├── BTC_1m.parquet
│   ├── ETH_historical.parquet
│   └── ETH_1m.parquet
│
└── equities_db/     ← NEW!
    ├── SPY_historical.parquet
    ├── SPY_1m.parquet
    ├── QQQ_historical.parquet
    └── QQQ_1m.parquet
```

## Summary

**UnifiedStorage handles:**
- ✅ Parquet save/load operations
- ✅ Append-only storage
- ✅ Live bar accumulation
- ✅ Gap detection
- ✅ File path management

**Adapters handle:**
- ✅ API-specific calls (Coinbase vs Alpaca)
- ✅ Data format conversion (API response → OHLCVBar)
- ✅ Market-specific logic (session hours, etc.)

**Your code handles:**
- ✅ Just call `storage.load_historical(symbol)` - works for both markets!

This is the cleanest architecture. One storage manager, adapters handle APIs, your features are market-agnostic.
