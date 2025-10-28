# What Adapters Actually Do - Clear Explanation

## The Problem You Were Trying to Solve

You have these 4 features:
1. Backtester
2. Monte Carlo Reports
3. Live Trading
4. Live Signals

They all work with **crypto data** (BTC, ETH, etc. from Coinbase).

**You wanted to add**: Same features, but for **equities** (SPY from Alpaca).

## The Problem

Crypto and equities send data in **different formats**:

```
Coinbase (crypto):
{
  "product_id": "BTC-USD",
  "price": "43250.50",
  "time": "2025-01-15T09:35:00.123Z",
  "size": "0.5"
}

Alpaca (equities):
{
  "symbol": "SPY",
  "price": 478.25,
  "timestamp": 1705318500000000000,
  "size": 100
}
```

Your existing code expects **Coinbase format**. It would crash with **Alpaca format**.

## What Adapters Do

Adapters are **translators**. They convert BOTH formats into the SAME format your code expects.

```
Coinbase → Adapter → OHLCVBar → Your Code ✅
Alpaca   → Adapter → OHLCVBar → Your Code ✅
```

**OHLCVBar is the unified format:**
```python
@dataclass
class OHLCVBar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str  # 'coinbase' or 'alpaca'
```

## Concrete Example

### BEFORE (Crypto Only):

```python
# Your existing backtester code
def backtest(symbol):
    # Fetches from Coinbase in Coinbase format
    data = fetch_from_coinbase(symbol)  
    # Process with your strategies
    signals = generate_signals(data)
    return results

# Works for BTC ✅
backtest('BTC')

# What about SPY? ❌
backtest('SPY')  # CRASHES - wrong format!
```

### AFTER (With Adapters):

```python
# Modified backtester code
def backtest(symbol, market='crypto'):
    # Get appropriate adapter
    if market == 'crypto':
        adapter = CoinbaseMarketAdapter(symbol)
    elif market == 'equities':
        adapter = AlpacaMarketAdapter(symbol)
    
    # Adapter gives you same format either way!
    data = adapter.get_market_data(...)  # OHLCVBar format
    
    # Process - same code works for both! ✅
    signals = generate_signals(data)
    return results

# Works for BTC ✅
backtest('BTC', market='crypto')

# NOW works for SPY too! ✅
backtest('SPY', market='equities')
```

## Why Not Use Adapters Directly?

You're asking: "What good are adapters if I can't use them with my main.py?"

**Answer**: You CAN use them! But you need to **integrate them** into your existing code.

Your `main.py` currently does this:

```python
def run_backtest(symbols):
    for symbol in symbols:
        # Directly reads Parquet files
        df = pd.read_parquet(f"data/crypto_db/{symbol}_historical.parquet")
        # Run backtest...
```

**To use adapters**, change it to:

```python
def run_backtest(symbols, market='crypto'):
    for symbol in symbols:
        if market == 'crypto':
            # Existing code
            df = pd.read_parquet(f"data/crypto_db/{symbol}_historical.parquet")
        elif market == 'equities':
            # NEW: Use adapter to get data
            from adapters import AlpacaMarketAdapter
            adapter = AlpacaMarketAdapter(symbol)
            df = adapter.get_market_data(symbol, start, end)
        
        # Same backtest code works for both!
        backtest(df)
```

## The Magic

**Without adapters**: Your code is **tightly coupled** to Coinbase format
```python
Your Code ←→ Coinbase (inseparable)
```

**With adapters**: Your code is **loosely coupled** - works with any data source
```python
Your Code ←→ Adapter ←→ Coinbase/Alpaca/Binance/etc
```

## What You Can Do Right Now

Even without full integration, you can test adapters:

```python
from adapters import AlpacaMarketAdapter

# Get historical SPY data
adapter = AlpacaMarketAdapter(symbol="SPY")
df = adapter.get_market_data("SPY", start, end)

print(df.head())  # See SPY data in OHLCV format!
```

This shows the adapter is **working** - it's fetching real data!

## Next Step: Integration

Choose one feature (I recommend backtester first):

**File: `scripts/backtesting/multi_symbol_backtester.py`**

Find this code:
```python
def load_symbol_data(symbol, days):
    df = pd.read_parquet(f"data/crypto_db/{symbol}_historical.parquet")
    return df
```

Change it to:
```python
def load_symbol_data(symbol, days, market='crypto'):
    if market == 'crypto':
        df = pd.read_parquet(f"data/crypto_db/{symbol}_historical.parquet")
    elif market == 'equities':
        from adapters import AlpacaMarketAdapter
        adapter = AlpacaMarketAdapter(symbol)
        end = datetime.now()
        start = end - timedelta(days=days)
        df = adapter.get_market_data(symbol, start, end)
    return df
```

Then you can run:
```bash
# Existing command still works
python main.py backtest --symbols BTC

# NEW command with equities
python main.py backtest --symbols SPY --market equities
```

## Summary

**Adapters are NOT just for printing.** They:
1. ✅ Normalize data formats (Coinbase → OHLCV, Alpaca → OHLCV)
2. ✅ Let your existing code work with new data sources
3. ✅ Abstract away market-specific details
4. ✅ Make it easy to add more exchanges later

**They're NOT automatically connected to main.py** because:
- That would be invasive - I don't want to break your working crypto system
- Integration requires modifying YOUR code (you understand it best)
- It's flexible - you choose what to integrate and when

**The adapter architecture is complete and working.** Now it needs to be **integrated** into your existing features.
