# Multi-Adapter Architecture Examples

This directory contains examples demonstrating the multi-adapter architecture for running the same trading strategies across different markets (crypto via Coinbase, equities via Alpaca).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Core Engine                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Signal     │  │     Risk     │  │  Portfolio  │      │
│  │   Engine     │  │   Manager    │  │  Manager    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Unified OHLCV Interface
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
┌────────▼─────────┐              ┌─────────▼─────────┐
│  Coinbase        │              │   Alpaca          │
│  Adapter         │              │   Adapter         │
│  (24/7 crypto)   │              │  (9:30-16:00 ET)  │
└──────────────────┘              └───────────────────┘
```

## Key Design Principles

1. **Adapter Pattern**: Each market (Coinbase, Alpaca) implements the same `MarketAdapter` interface
2. **Market-Agnostic Strategies**: Strategies operate on OHLCV bars, unaware of the underlying market
3. **Session Awareness**: For equities, a `SessionController` manages trading hours
4. **Unified Bar Format**: All adapters emit `OHLCVBar` objects with consistent schema

## Files

- `multi_adapter_strategy.py` - Example showing the same MA crossover strategy running on both BTC (Coinbase) and SPY (Alpaca)

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
export COINBASE_API_KEY="your_key"
export COINBASE_SECRET_KEY="your_secret"

# Run the example
python examples/multi_adapter_strategy.py
```

## Differences Between Markets

| Feature | Coinbase (Crypto) | Alpaca (Equities) |
|---------|------------------|-------------------|
| Trading Hours | 24/7 | 9:30-16:00 ET |
| Order Book | Full L2 depth | Top-of-book (NBBO) |
| Tick Size | Arbitrary | $0.01 (SPY) |
| Data Feed | WebSocket trades | WebSocket trades |
| Session Handling | None needed | Required |
| Market Halts | Rare | Frequent |
| OPEX Days | None | Monthly |
| DST Impact | None | Spring/Fall shifts |

## Benefits

1. **Code Reuse**: Same strategy logic works for both markets
2. **Easier Testing**: Test on crypto (fast iteration) then deploy to equities
3. **Portfolio Expansion**: Trade both asset classes from same codebase
4. **Risk Isolation**: Market-specific issues stay in adapters
