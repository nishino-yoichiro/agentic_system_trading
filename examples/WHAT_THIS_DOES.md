# What This Example Does - Simple Explanation

## TL;DR

**This example shows you can run the exact same trading strategy on both Bitcoin (BTC) and S&P 500 (SPY) with the same code.**

## The Problem It Solves

Before this architecture:
- ❌ Strategy for BTC wouldn't work for SPY (different APIs)
- ❌ Strategy for SPY wouldn't work for BTC (different market hours)
- ❌ Had to write separate code for each market

After this architecture:
- ✅ Write strategy once, works for both markets
- ✅ Market-specific logic isolated in adapters
- ✅ Easy to add new exchanges

## What Gets Demonstrated

### 1. Session Controller Demo
```
Checks: Is the stock market open right now?
Shows: 
  - Current status (open/closed)
  - When market opens next
  - How long to wait until trading can begin
```

### 2. Alpaca Adapter Demo (SPY)
```
Shows how to:
  - Connect to Alpaca for equities
  - Stream SPY price data
  - Only trade during market hours (9:30 AM - 4:00 PM ET)
  - Handle market holidays, DST, etc.
```

### 3. Coinbase Adapter Demo (BTC)
```
Shows how to:
  - Connect to Coinbase for crypto
  - Stream BTC price data  
  - Trade 24/7 (crypto never closes)
  - Use your existing Coinbase infrastructure
```

## The Key Concept

**Market Adapters** translate between your strategy (which doesn't care about markets) and the actual exchange APIs (which are very different):

```
Your Strategy
  ↓
  "Give me the next price bar"
  ↓
  Market Adapter (translates to exchange API)
  ↓
  Coinbase API or Alpaca API
```

The strategy receives a standardized `OHLCVBar` regardless of which exchange:
```python
OHLCVBar(
    timestamp=...,
    symbol="BTC" or "SPY",
    open=...,
    high=...,
    low=...,
    close=...,
    volume=...
)
```

## Real-World Example

Let's say you have a strategy that looks for breakouts:

```python
def breakout_strategy(bar: OHLCVBar):
    # Your logic here
    if bar.close > bar.high * 0.99:
        return "BUY signal"
```

**Before adapters:**
- Had to write separate function for Coinbase format
- Had to write separate function for Alpaca format
- Had to handle BTC being 24/7 vs SPY being 9:30-4pm

**With adapters:**
- Write strategy once
- Works automatically on BTC (via Coinbase adapter)
- Works automatically on SPY (via Alpaca adapter)
- Session handling is automatic

## What You Can Do With This

1. **Test strategies on both markets** - See if something works on crypto first (faster iteration), then try on equities

2. **Trade both markets simultaneously** - Run the same strategy on BTC and SPY at the same time

3. **Diversify your portfolio** - If you find a good signal in crypto, test it on equities and vice versa

4. **Add new exchanges easily** - Want to add Binance? Just create a `BinanceAdapter` using the same pattern

## Current State

Right now, this example shows:
- ✅ How to check if market is open
- ✅ How to initialize adapters for both markets
- ✅ The architecture is ready to use

**To actually run live trading**, you would need to:
1. Install dependencies: `pip install alpaca-py pandas-market-calendars pytz`
2. Set API keys in `.env` file
3. Modify the example to actually place orders (currently just shows structure)

## The Bottom Line

This is a **proof of concept** showing that:
- Your strategies don't need to know which market they're trading
- Market-specific complexity is hidden in adapters
- You can trade multiple markets with the same code

It's like having a universal remote control - your strategy is the remote, the adapters are the buttons that work on any TV (market).
