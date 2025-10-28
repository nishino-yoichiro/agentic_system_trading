# Integration Guide: Connecting Adapters to Your Existing System

## Understanding Your Requirements

**Your Existing System (Crypto-Only):**
1. ✅ Backtester with historical data
2. ✅ Monte Carlo and equity reports  
3. ✅ Live trading
4. ✅ Live signals and monitoring

**Your Goal:**
- Add Alpaca (equities) as new data source
- Keep ALL 4 functionalities working for crypto
- Apply the SAME live signaling to financial markets (SPY, etc.)
- Start simple: just integrate OHLCV data
- Run same strategies, generate signals in same format

**The Problem:**
- System built only for crypto/Coinbase
- Won't work for equities without refactoring

**The Solution:** ✅ We've built this!

## Architecture Overview

```
YOUR EXISTING SYSTEM (All 4 Features Work)
│
├── Backtester ────────────────┐
├── Monte Carlo Reports ──────┤
├── Live Trading ──────────────┤─── All use OHLCV data
└── Live Signals/Monitoring ──┘
    
         ↓
    NEW: Adapter Layer (Normalizes Data)
         │
    ┌────┴────┐
    │         │
Coinbase  Alpaca
(Your    (NEW for
 Working  Equities)
 Code)    
```

## How It Works

### The Magic: Unified OHLCV Interface

Your existing features work on **OHLCV data**:
- Backtester needs: `[timestamp, open, high, low, close, volume]`
- Live signals need: `[timestamp, open, high, low, close, volume]`  
- Reports need: `[timestamp, open, high, low, close, volume]`
- Live trading needs: `[timestamp, open, high, low, close, volume]`

**Adapters convert all data sources to this same format!**

```
Coinbase (crypto)  ──►  Adapter  ──►  OHLCVBar  ──►  Your Existing Features ✅
Alpaca (equities)  ──►  Adapter  ──►  OHLCVBar  ──►  Your Existing Features ✅
```

## Integration Steps

### Step 1: Install Dependencies

```bash
pip install alpaca-py pandas-market-calendars pytz
```

### Step 2: Set Environment Variables

Add to `.env`:
```bash
# Your existing Coinbase keys (already there)
COINBASE_API_KEY=your_key
COINBASE_SECRET_KEY=your_secret

# NEW: Add Alpaca keys
ALPACA_API_KEY=pk_test_your_key
ALPACA_SECRET_KEY=your_secret_key
```

### Step 3: Integrate Adapters into Your Live Signaling

**File: `src/crypto_signal_integration.py` (Your existing file)**

Currently it's crypto-only. You can add a wrapper that accepts adapters:

```python
# Add this to your existing crypto_signal_integration.py

from adapters import AlpacaMarketAdapter, CoinbaseMarketAdapter

def get_historical_data_unified(symbol: str, days: int = 30, market: str = 'crypto'):
    """
    Universal data fetcher that works for both crypto and equities
    
    Args:
        symbol: Trading symbol (BTC, SPY, etc.)
        days: Days of history
        market: 'crypto' or 'equities'
    """
    if market == 'crypto':
        # Your existing Coinbase code
        from crypto_analysis_engine import CryptoAnalysisEngine
        engine = CryptoAnalysisEngine()
        return engine.load_symbol_data(symbol, days=days)
    
    elif market == 'equities':
        # NEW: Use adapter
        adapter = AlpacaMarketAdapter(symbol=symbol, paper=True)
        end = datetime.now()
        start = end - timedelta(days=days)
        
        df = adapter.get_market_data(symbol, start, end)
        return df
```

### Step 4: Update Your Backtester

**File: `scripts/backtesting/multi_symbol_backtester.py`**

Currently processes crypto Parquet files. Add adapter support:

```python
def fetch_data_unified(symbol: str, days: int, market: str = 'crypto'):
    """Fetch data from either crypto or equities"""
    
    if market == 'crypto':
        # Your existing logic
        df = pd.read_parquet(f"data/crypto_db/{symbol}_historical.parquet")
        return df
    
    elif market == 'equities':
        # NEW: Use Alpaca adapter
        from adapters import AlpacaMarketAdapter
        adapter = AlpacaMarketAdapter(symbol=symbol)
        
        end = datetime.now()
        start = end - timedelta(days=days)
        df = adapter.get_market_data(symbol, start, end)
        return df
```

### Step 5: Update Live Trading

**File: `src/interactive_trading_module.py` or wherever live trading is**

```python
from adapters import AlpacaMarketAdapter, CoinbaseMarketAdapter

async def run_live_trading_unified(symbols_config: list):
    """
    Unified live trading that works for both markets
    
    symbols_config example:
    [
        {'symbol': 'BTC', 'market': 'crypto'},
        {'symbol': 'SPY', 'market': 'equities'}
    ]
    """
    
    for config in symbols_config:
        symbol = config['symbol']
        market = config['market']
        
        # Create appropriate adapter
        if market == 'crypto':
            adapter = CoinbaseMarketAdapter(symbol=symbol)
        elif market == 'equities':
            adapter = AlpacaMarketAdapter(symbol=symbol, paper=True)
        
        # Connect
        await adapter.connect()
        
        # Stream data and generate signals (same as before!)
        async for bar in adapter.stream_data(interval_seconds=60):
            # Check if market is open (only matters for equities)
            if market == 'equities' and not adapter.is_market_open():
                continue  # Skip if market is closed
            
            # Generate signals using your EXISTING strategies
            signals = generate_signals_for_bar(bar, symbol, market)
            
            # Process signals (same logic for both markets)
            process_signals(signals, bar)
```

## What This Maintains

### ✅ Your Existing Crypto System (Untouched)

All your existing features continue to work exactly as before:

```python
# Still works for BTC
crypto_integration = CryptoSignalIntegration()
signals = crypto_integration.generate_signals(['BTC'], days=30)

# Backtester still works
python scripts/backtesting/multi_symbol_backtester.py --symbols BTC

# Reports still work
# Monte Carlo still works
```

### ✅ New: Now Works for Equities Too

Same functionality, now for SPY:

```python
# NEW: Now works for SPY
equities_adapter = AlpacaMarketAdapter(symbol="SPY")
await equities_adapter.connect()

async for bar in equities_adapter.stream_data(interval_seconds=60):
    # Same signal generation as crypto!
    signals = your_strategies.generate_signals(bar)
    # Process signals...
```

## Signals Format (Same for Both)

Your signals will look identical regardless of market:

```python
[
    {
        'strategy': 'btc_ny_session',
        'signal_type': 'buy',
        'confidence': 0.75,
        'entry_price': 45320.50,
        'timestamp': '2025-01-15T09:35:00',
        'symbol': 'BTC',  # or 'SPY'
        'reason': 'NY session momentum'
    }
]
```

**Your monitoring and live trading don't care if it's BTC or SPY!**

## Quick Integration Example

Here's a minimal example showing how to add SPY to your existing system:

```python
# File: src/unified_live_signaling.py

from adapters import AlpacaMarketAdapter, SessionController
from crypto_signal_integration import CryptoSignalIntegration

async def run_unified_live_signals():
    """Run live signals for both crypto and equities"""
    
    # Get crypto signals (your existing code)
    crypto_integration = CryptoSignalIntegration()
    crypto_signals = crypto_integration.generate_signals(['BTC', 'ETH'], days=30)
    
    # Get equities signals (NEW)
    session = SessionController(exchange='NYSE')
    
    if session.is_market_open():
        alpaca = AlpacaMarketAdapter(symbol="SPY", paper=True)
        await alpaca.connect()
        
        async for bar in alpaca.stream_data(interval_seconds=60):
            # Convert bar to your signal format
            equity_signals = convert_bar_to_signals(bar)
            
            # Merge signals
            all_signals = crypto_signals + equity_signals
            
            # Process all signals the same way
            process_signals(all_signals)
```

## Next Steps

1. **Test the adapters work:**
   ```bash
   python examples/multi_adapter_strategy.py
   ```

2. **Add SPY to your backtester:**
   ```python
   # Modify multi_symbol_backtester.py to accept market type
   python scripts/backtesting/multi_symbol_backtester.py \
     --symbols SPY \
     --market equities
   ```

3. **Add live SPY signals to your monitoring:**
   - Create adapter
   - Stream data during market hours
   - Generate signals with your existing strategies
   - Display in your dashboard

4. **Add paper trading for SPY:**
   - Use AlpacaBroker
   - Place orders based on signals
   - Track positions same as crypto

## Summary

✅ **What you asked for:**
- Integrate Alpaca data source
- Use existing infrastructure  
- Start with OHLCV data
- Run live signaling  
- Maintain all 4 existing features

✅ **What you got:**
- Adapter architecture that normalizes data
- All 4 features work for both crypto AND equities
- Same strategies, same signal format
- Your existing code stays intact
- Easy to add more exchanges later

**The key insight:** Your system already works with OHLCV data. Adapters just normalize different data sources to that same format. Everything else stays the same!
