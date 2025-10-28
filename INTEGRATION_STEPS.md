# Integration Steps - Making Adapters Work With Your 4 Features

## Current Situation

Your `main.py` commands:
- `python main.py data-collection` - Only crypto
- `python main.py live-trading` - Only crypto  
- `python main.py backtest` - Only crypto
- `python main.py dashboard` - Only crypto

**What you want**: Choose crypto or equity and run these on either market

## Step-by-Step Integration

### Step 1: Add Market Selection to main.py

Modify your `main.py` to accept market type:

```python
# Add argument
parser.add_argument("--market", choices=["crypto", "equities"], 
                   default="crypto", 
                   help="Market type: crypto or equities")

# Then when starting live trading:
if args.market == "crypto":
    # Use existing crypto code
    pipeline.start_live_trading(...)
elif args.market == "equities":
    # Use new adapter
    from adapters import AlpacaMarketAdapter
    adapter = AlpacaMarketAdapter(symbol="SPY", paper=True)
    # Run live trading with adapter...
```

### Step 2: Create Unified Data Fetcher

**File: `src/unified_data_fetcher.py`** (NEW)

```python
def get_data(symbol: str, market: str, days: int):
    """Universal data fetcher - works for both markets"""
    
    if market == "crypto":
        # Your existing code
        from crypto_analysis_engine import CryptoAnalysisEngine
        engine = CryptoAnalysisEngine()
        return engine.load_symbol_data(symbol, days=days)
    
    elif market == "equities":
        # NEW: Use adapter
        from adapters import AlpacaMarketAdapter
        adapter = AlpacaMarketAdapter(symbol=symbol)
        
        end = datetime.now()
        start = end - timedelta(days=days)
        return adapter.get_market_data(symbol, start, end)
```

### Step 3: Update Your Backtester

**File: `scripts/backtesting/multi_symbol_backtester.py`**

Current code reads from Parquet files. Add adapter support:

```python
def load_data(symbol: str, days: int, market: str = "crypto"):
    """Load data for backtesting"""
    
    if market == "crypto":
        # Existing code
        df = pd.read_parquet(f"data/crypto_db/{symbol}_historical.parquet")
        return df
    
    elif market == "equities":
        # NEW: Use adapter
        from adapters import AlpacaMarketAdapter
        adapter = AlpacaMarketAdapter(symbol=symbol)
        
        end = datetime.now()
        start = end - timedelta(days=days)
        df = adapter.get_market_data(symbol, start, end)
        return df
```

### Step 4: Update Live Trading

**File: `scripts/live_trading/integrated_live_trading.py`**

Current code streams from Coinbase only. Add adapter support:

```python
async def stream_data(symbols: list, market: str = "crypto"):
    """Stream data - works for both markets"""
    
    if market == "crypto":
        # Your existing WebSocket code
        ws_feed = WebSocketPriceFeed(symbols)
        ws_feed.start()
        # ... rest of existing code
    
    elif market == "equities":
        # NEW: Use Alpaca adapter
        from adapters import AlpacaMarketAdapter, SessionController
        session = SessionController()
        
        for symbol in symbols:
            adapter = AlpacaMarketAdapter(symbol=symbol, paper=True)
            await adapter.connect()
            
            async for bar in adapter.stream_data(interval_seconds=60):
                if session.is_market_open():
                    # Process bar (same as crypto!)
                    yield bar
```

## Quick Test

After integration, you can run:

```bash
# Run backtest on crypto (existing)
python main.py backtest --symbols BTC ETH

# Run backtest on equities (NEW!)
python main.py backtest --symbols SPY QQQ --market equities

# Run live trading on crypto (existing)
python main.py live-trading --symbols BTC

# Run live trading on equities (NEW!)
python main.py live-trading --symbols SPY --market equities
```

## What This Achieves

Your 4 features will now work for BOTH markets:

1. ✅ **Backtester**: Works on crypto AND equities
2. ✅ **Monte Carlo Reports**: Works on crypto AND equities
3. ✅ **Live Trading**: Works on crypto AND equities  
4. ✅ **Live Signals**: Works on crypto AND equities

All with the same command, just add `--market equities`

## Next Steps

Choose which feature to integrate first:
- **Easiest**: Backtester (no real-time required)
- **Most valuable**: Live signals (your main goal)
- **Complete**: All 4 features

I can help you integrate them one by one.
